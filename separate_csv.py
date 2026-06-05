import pandas as pd
import ast

def check_missing_shots(file_path):
    # This function gives back all lines from the dataset.csv that do not contain 3PT, 2PT or Free Throw
    # Read the CSV specifying the semicolon delimiter
    df = pd.read_csv(file_path, sep=';')
    
    def has_no_shots(actions_str):
        try:
            # Safely parse the string representation of the list
            actions = ast.literal_eval(actions_str)
            if not isinstance(actions, list):
                return True
            
            # Check if any action dictionary indicates a 2PT or 3PT Shot
            has_shot = any(
                act.get('action') in ['3PT Shot', '2PT Shot', 'Free Throw'] 
                for act in actions 
                if isinstance(act, dict)
            )
            
            # We want rows that DO NOT contain either shot
            return not has_shot
            
        except (ValueError, SyntaxError):
            # If the row is empty or malformed, we consider it as not having a shot
            return True

    # Apply the helper function to filter the DataFrame
    entries_without_shots = df[df['actions'].apply(has_no_shots)]
    
    # Check if there are any such entries
    if not entries_without_shots.empty:
        print(f"Found {len(entries_without_shots)} entry/entries without a 2PT or 3PT Shot.")
    else:
        print("All entries contain at least one 2PT or 3PT Shot.")
        
    return entries_without_shots


def check_shots(file_path):
    # This function gives back all lines from the dataset.csv that do contains 3PT, 2PT or Free Throw
    df = pd.read_csv(file_path, sep=';')
    
    def has_no_shots(actions_str):
        try:
            # Safely parse the string representation of the list
            actions = ast.literal_eval(actions_str)
            if not isinstance(actions, list):
                return True
            
            # Check if any action dictionary indicates a 3PT Shot or Free Throw
            has_shot = any(
                act.get('action') in ['3PT Shot', 'Free Throw'] 
                for act in actions 
                if isinstance(act, dict)
            )
            
            # We want rows that contain either shot
            return has_shot
            
        except (ValueError, SyntaxError):
            # If the row is empty or malformed, we consider it as not having a shot
            return True

    # Apply the helper function to filter the DataFrame
    entries_without_shots = df[df['actions'].apply(has_no_shots)]
    
    # Check if there are any such entries
    if not entries_without_shots.empty:
        print(f"Found {len(entries_without_shots)} entry/entries without a 2PT or 3PT Shot.")
    else:
        print("All entries contain at least one 2PT or 3PT Shot.")
        
    return entries_without_shots


no_shot = check_missing_shots('./BARD/dataset/dataset.csv')
shot = check_shots('./BARD/dataset/dataset.csv')

shot.sample(frac = 1, random_state=42).iloc[:225].to_csv('./BARD/dataset/shot_dataset.csv', index = False, sep=';')
no_shot.sample(frac = 1, random_state=42).iloc[:225].to_csv('./BARD/dataset/no_shot_dataset.csv', index = False, sep=';')