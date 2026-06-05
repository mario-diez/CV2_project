from __future__ import annotations

import argparse
import ast
import json
import os
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

SHOT_ACTIONS = {'2PT Shot', '3PT Shot', 'Free Throw'}
CLASS_NAMES = ['pass', 'shot']


@dataclass
class PrepareConfig:
    bard_data_dir: Path = Path('BARD/data')
    labels_csv: Path = Path('BARD/dataset/dataset.csv')
    output_dir: Path = Path('data')
    test_ratio: float = 0.15
    val_ratio: float = 0.15
    seed: int = 42
    copy_videos: bool = False


def infer_shot_pass_label(actions: list[dict]) -> Optional[str]:
    shot_events = [action for action in actions if action.get('action') in SHOT_ACTIONS]
    if not shot_events:
        return None
    last_shot = shot_events[-1]
    if last_shot.get('assisted') is True:
        return 'pass'
    return 'shot'


def build_catalog(bard_data_dir: Path, labels_csv: Path) -> tuple[pd.DataFrame, dict]:
    labels_df = pd.read_csv(labels_csv, sep=';')
    rows = []
    stats = {
        'annotations_total': len(labels_df),
        'downloaded_videos': 0,
        'skipped_no_shot': 0,
        'skipped_missing_video': 0,
    }

    for row_idx, row in labels_df.iterrows():
        video_id = row_idx + 1
        source_path = bard_data_dir / f'{video_id}.mp4'
        if source_path.exists():
            stats['downloaded_videos'] += 1

        actions = ast.literal_eval(row['actions'])
        label = infer_shot_pass_label(actions)
        if label is None:
            stats['skipped_no_shot'] += 1
            continue
        if not source_path.exists():
            stats['skipped_missing_video'] += 1
            continue

        rows.append(
            {
                'video_id': video_id,
                'label': label,
                'source_path': str(source_path.resolve()),
                'url': row['urls'],
                'actions': row['actions'],
                'numerosity': row.get('numerosity'),
            }
        )

    catalog = pd.DataFrame(rows)
    stats['usable_clips'] = len(catalog)
    return catalog, stats


def assign_splits(catalog: pd.DataFrame, test_ratio: float, val_ratio: float, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    catalog = catalog.copy()
    catalog['split'] = 'train'

    for label in CLASS_NAMES:
        label_mask = catalog['label'] == label
        indices = catalog.index[label_mask].tolist()
        rng.shuffle(indices)
        n = len(indices)

        if n <= 1:
            continue

        n_test = max(1, int(round(n * test_ratio))) if n >= 3 else 0
        n_val = max(1, int(round(n * val_ratio))) if n - n_test >= 2 else 0

        if n_test + n_val >= n:
            n_test = min(n_test, max(0, n - 2))
            n_val = min(n_val, max(1, n - n_test - 1))

        test_idx = indices[:n_test]
        val_idx = indices[n_test:n_test + n_val]

        catalog.loc[test_idx, 'split'] = 'test'
        catalog.loc[val_idx, 'split'] = 'val'

    return catalog


def link_or_copy_video(source: Path, destination: Path, force_copy: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()

    source = source.resolve()

    if force_copy:
        shutil.copy2(source, destination)
        return

    try:
        os.symlink(source, destination)
        return
    except OSError:
        pass

    try:
        os.link(source, destination)
        return
    except OSError:
        shutil.copy2(source, destination)


def export_dataset(catalog: pd.DataFrame, cfg: PrepareConfig) -> dict:
    output_dir = cfg.output_dir
    splits_dir = output_dir / 'splits'
    splits_dir.mkdir(parents=True, exist_ok=True)

    catalog = catalog.sort_values(['split', 'label', 'video_id']).reset_index(drop=True)
    catalog['filename'] = catalog.apply(
        lambda row: f"{row['video_id']}.mp4",
        axis=1,
    )
    catalog['dataset_path'] = catalog.apply(
        lambda row: str((output_dir / row['split'] / row['label'] / row['filename']).resolve()),
        axis=1,
    )

    catalog.to_csv(output_dir / 'catalog.csv', index=False)
    for split_name in ['train', 'val', 'test']:
        split_df = catalog[catalog['split'] == split_name]
        split_df.to_csv(splits_dir / f'{split_name}.csv', index=False)

    for _, row in catalog.iterrows():
        destination = Path(row['dataset_path'])
        link_or_copy_video(Path(row['source_path']), destination, force_copy=cfg.copy_videos)

    split_counts = (
        catalog.groupby(['split', 'label']).size().unstack(fill_value=0).astype(int).to_dict()
    )
    summary = {
        'class_names': CLASS_NAMES,
        'class_to_idx': {name: idx for idx, name in enumerate(CLASS_NAMES)},
        'splits': ['train', 'val', 'test'],
        'counts': {
            'total': int(len(catalog)),
            'by_split': catalog['split'].value_counts().to_dict(),
            'by_label': catalog['label'].value_counts().to_dict(),
            'by_split_and_label': split_counts,
        },
        'config': {
            **{k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        },
    }

    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_args() -> PrepareConfig:
    parser = argparse.ArgumentParser(description='Prepare BARD shot/pass dataset under data/.')
    parser.add_argument('--bard-data-dir', type=Path, default=PrepareConfig.bard_data_dir)
    parser.add_argument('--labels-csv', type=Path, default=PrepareConfig.labels_csv)
    parser.add_argument('--output-dir', type=Path, default=PrepareConfig.output_dir)
    parser.add_argument('--test-ratio', type=float, default=PrepareConfig.test_ratio)
    parser.add_argument('--val-ratio', type=float, default=PrepareConfig.val_ratio)
    parser.add_argument('--seed', type=int, default=PrepareConfig.seed)
    parser.add_argument(
        '--copy-videos',
        action='store_true',
        help='Copy videos instead of creating symlinks/hardlinks.',
    )
    args = parser.parse_args()
    return PrepareConfig(
        bard_data_dir=args.bard_data_dir,
        labels_csv=args.labels_csv,
        output_dir=args.output_dir,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        copy_videos=args.copy_videos,
    )


def main() -> None:
    cfg = parse_args()

    if not cfg.labels_csv.exists():
        raise FileNotFoundError(f'Labels file not found: {cfg.labels_csv}')
    if not cfg.bard_data_dir.exists():
        raise FileNotFoundError(f'BARD data directory not found: {cfg.bard_data_dir}')

    catalog, build_stats = build_catalog(cfg.bard_data_dir, cfg.labels_csv)
    if catalog.empty:
        raise ValueError('No labeled videos found. Download videos to BARD/data/ first.')

    catalog = assign_splits(catalog, cfg.test_ratio, cfg.val_ratio, cfg.seed)
    summary = export_dataset(catalog, cfg)

    print('--- BARD dataset prepared ---')
    print(f"Annotations in CSV: {build_stats['annotations_total']}")
    print(f"Downloaded videos:  {build_stats['downloaded_videos']}")
    print(f"Usable shot/pass:   {build_stats['usable_clips']}")
    print(f"Skipped (no shot):  {build_stats['skipped_no_shot']}")
    print(f"Skipped (missing):  {build_stats['skipped_missing_video']}")
    print(f"Output directory:   {cfg.output_dir.resolve()}")
    print('Split sizes:', summary['counts']['by_split'])
    print('Label sizes:', summary['counts']['by_label'])


if __name__ == '__main__':
    main()
