import json
import random
import glob
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
from torchvision.models.video import r3d_18, R3D_18_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4
NUM_FRAMES = 16
CHECKPOINT_DIR = Path("./artifacts/r3d_checkpoints")

# 0 = no_shot, 1 = shot
CLASS_NAMES = ["no_shot", "shot"]

K_SHOTS = [10, 20, 40, 50]

random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

train_list_pass = sorted(glob.glob("./data/train/no_shot/*.mp4"))
train_list_shot = sorted(glob.glob("./data/train/shot/*.mp4"))

test_list_pass = sorted(glob.glob("./data/test/no_shot/*.mp4"))
test_list_shot = sorted(glob.glob("./data/test/shot/*.mp4"))

val_list_pass = sorted(glob.glob("./data/val/no_shot/*.mp4"))
val_list_shot = sorted(glob.glob("./data/val/shot/*.mp4"))

labels_train_pass = [0] * len(train_list_pass)
labels_train_shot = [1] * len(train_list_shot)

labels_test_pass = [0] * len(test_list_pass)
labels_test_shot = [1] * len(test_list_shot)

labels_val_pass = [0] * len(val_list_pass)
labels_val_shot = [1] * len(val_list_shot)

videos_test = test_list_pass + test_list_shot
labels_test = labels_test_pass + labels_test_shot

videos_val = val_list_pass + val_list_shot
labels_val = labels_val_pass + labels_val_shot

train_shot = list(zip(train_list_shot, labels_train_shot))
train_pass = list(zip(train_list_pass, labels_train_pass))

random.shuffle(train_shot)
random.shuffle(train_pass)

weights = R3D_18_Weights.KINETICS400_V1
transform = weights.transforms()


class ShotDataset(Dataset):
    def __init__(self, videos, labels, transform):
        self.videos = videos
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video, _, _ = read_video(
            self.videos[idx],
            pts_unit="sec",
            output_format="TCHW",
        )

        # video: (T, C, H, W)
        t = video.shape[0]

        if t >= NUM_FRAMES:
            indices = torch.linspace(0, t - 1, NUM_FRAMES).long()
            video = video[indices]
        else:
            pad = NUM_FRAMES - t
            last_frame = video[-1:].repeat(pad, 1, 1, 1)
            video = torch.cat([video, last_frame], dim=0)

        # weights.transforms() devuelve (C, T, H, W), que es lo que R3D espera en batch
        video = transform(video)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return video, label

pin_memory = torch.cuda.is_available()

test_dataset = ShotDataset(videos_test, labels_test, transform)
val_dataset = ShotDataset(videos_val, labels_val, transform)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=pin_memory,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=pin_memory,
)
def build_model(mode):
    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)

    if mode == "linear_probe":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    elif mode == "finetune":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True

    return model.to(DEVICE)

def checkpoint_path(mode: str, n_shots: int) -> Path:
    return CHECKPOINT_DIR / f"{mode}_n{n_shots}_best.pt"

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    mode: str,
    n_shots: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "mode": mode,
            "n_shots": n_shots,
            "class_names": CLASS_NAMES,
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, optimizer=None) -> dict:
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint

def train_model(model, train_loader, val_loader, mode: str, n_shots: int):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
    )

    best_val_acc = -1.0
    best_epoch = 0
    best_path = checkpoint_path(mode, n_shots)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)
        val_metrics = evaluate(model, val_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"train_loss={train_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1_macro={val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch + 1
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch + 1,
                val_metrics,
                mode,
                n_shots,
            )
            print(f"  -> mejor checkpoint guardado (val_acc={best_val_acc:.4f})")

    if best_path.exists():
        checkpoint = load_checkpoint(best_path, model, optimizer)
        print(
            f"Checkpoint restaurado: epoch={checkpoint['epoch']} "
            f"val_acc={checkpoint['metrics']['accuracy']:.4f}"
        )
    else:
        print("Aviso: no se guardó ningún checkpoint.")

    return model, {"best_epoch": best_epoch, "best_val_acc": best_val_acc}

@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    y_true = []
    y_pred = []

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        pred = model(x).argmax(dim=1)

        y_true.extend(y.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0, labels=[0, 1]
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0, labels=[0, 1]
        ),
        "f1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0, labels=[0, 1]
        ),
        "precision_shot": precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "recall_shot": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "confusion_matrix": cm.tolist(),
        "n_samples": len(y_true),
        "n_correct": int((cm[0, 0] + cm[1, 1])),
    }


def print_metrics(split_name: str, metrics: dict) -> None:
    cm = metrics["confusion_matrix"]
    print(
        f"{split_name}: acc={metrics['accuracy']:.4f} "
        f"f1_macro={metrics['f1_macro']:.4f} "
        f"({metrics['n_correct']}/{metrics['n_samples']} correctos)"
    )
    print(
        f"  confusion [rows=true no_shot/shot, cols=pred]: "
        f"no_shot->{cm[0]} shot->{cm[1]}"
    )

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Train pool: {len(train_pass)} no_shot, {len(train_shot)} shot | "
        f"Val: {len(videos_val)} | Test: {len(videos_test)}"
    )

    all_results = []

    for mode in ["linear_probe", "finetune"]:
        print("\n==========================")
        print(mode)
        print("==========================")

        for n_shots in K_SHOTS:
            subset = train_shot[:n_shots] + train_pass[:n_shots]
            random.shuffle(subset)

            videos_train = [x[0] for x in subset]
            labels_train = [x[1] for x in subset]

            train_dataset = ShotDataset(videos_train, labels_train, transform)
            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                pin_memory=pin_memory,
            )

            model = build_model(mode)

            print(f"\n{mode}  N={n_shots}  (train clips: {len(videos_train)})")

            model, train_info = train_model(
                model,
                train_loader,
                val_loader,
                mode,
                n_shots,
            )

            val_metrics = evaluate(model, val_loader)
            test_metrics = evaluate(model, test_loader)

            print_metrics("Val (best checkpoint)", val_metrics)
            print_metrics("Test", test_metrics)

            result = {
                "mode": mode,
                "n_shots": n_shots,
                "best_epoch": train_info["best_epoch"],
                "best_val_acc": train_info["best_val_acc"],
                "val": val_metrics,
                "test": test_metrics,
                "checkpoint": str(checkpoint_path(mode, n_shots)),
            }
            all_results.append(result)

    summary_path = CHECKPOINT_DIR / "results_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResumen guardado en: {summary_path}")
