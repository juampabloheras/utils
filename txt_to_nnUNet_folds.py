import json
import os

def create_train_val_splits(fold_dir):
    folds = []
    for i in range(1, 6):
        fold_file = os.path.join(fold_dir, f"fold_{i}.txt")
        with open(fold_file, "r") as file:
            folds.append([line.strip() for line in file.readlines()])

    splits = []
    for val_index in range(5):
        train = []
        val = folds[val_index]
        for i in range(5):
            if i != val_index:
                train.extend(folds[i])
        splits.append({
            "train": train,
            "val": val
        })

    return splits

def save_splits_to_json(splits, output_file):
    # Save the splits to a JSON file
    with open(output_file, "w") as json_file:
        json.dump(splits, json_file, indent=4)

if __name__ == "__main__":
    fold_dir = "/mmfs1/gscratch/kurtlab/brats2024/experiments/brats-goat/smw/CV-folds/" 
    output_file = "train_val_splits.json"
    
    splits = create_train_val_splits(fold_dir)
    save_splits_to_json(splits, output_file)

    print(f"Train/Val splits saved to {output_file}")
