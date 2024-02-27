import os
from tqdm import tqdm
from pathlib import Path
import pandas as pd

DATAPATH = Path(__file__).resolve().parent.parent.parent / "data" / "train"


def load_and_save_train_images(dir: str):
    df = pd.DataFrame(columns=["path", "label"])
    for subdir in tqdm(os.listdir(dir)):
        for file in tqdm(os.listdir(os.path.join(dir, subdir))):
            if file.endswith(".png") or file.endswith(".jpg"):
                new_record = pd.DataFrame(
                    {"relative_path": [os.path.join(subdir, file)], "label": [subdir]}
                )
                df = pd.concat([df, new_record])
    return df


def main():
    df = load_and_save_train_images(DATAPATH)
    df.to_csv(DATAPATH / "train.csv", index=False)


if __name__ == "__main__":
    main()
