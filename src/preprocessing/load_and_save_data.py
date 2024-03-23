import os
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd

DATAPATH = Path(__file__).resolve().parent.parent.parent / "data"


def load_and_save_train_images(dir: str) -> pd.DataFrame:
    """ "
    Load and save images from a directory to a pandas Data Frame.
    Args:
        dir: str, directory where the images are stored.
    Returns:
        df: pd.DataFrame, a pandas Data Frame with the following columns:
            - relative_path: str, relative path of the image.
            - label: str, label of the image.
            - image: list, list of pixels representing the image.
    Warning:
        To load the images from the image column in obtained Data Frame, use the following code:
        `df['image'].apply(lambda x: np.array(eval(x)))`
    """
    df = pd.DataFrame(columns=["relative_path", "label", "image"])
    for subdir in tqdm(os.listdir(dir)):
        for file in tqdm(os.listdir(os.path.join(dir, subdir))):
            if file.endswith(".png") or file.endswith(".jpg"):
                img = Image.open(dir / subdir / file)
                img.info.pop("icc_profile", None)
                new_record = pd.DataFrame(
                    {
                        "relative_path": [os.path.join(subdir, file)],
                        "label": [subdir],
                        "image": [np.array(img).tolist()],
                    }
                )
                df = pd.concat([df, new_record])
    return df


def load_and_save_test_images(dir: str) -> pd.DataFrame:
    """ "
    Load and save images from a directory to a pandas Data Frame.
    Args:
        dir: str, directory where the test images are stored.
    """
    df = pd.DataFrame(columns=["relative_path", "image"])
    for file in tqdm(os.listdir(os.path.join(dir, dir))):
        if file.endswith(".png") or file.endswith(".jpg"):
            img = Image.open(dir / file)
            img.info.pop("icc_profile", None)
            new_record = pd.DataFrame(
                {
                    "relative_path": [os.path.join(dir, file)],
                    "image": [np.array(img).tolist()],
                }
            )
            df = pd.concat([df, new_record])
    return df


def main():
    df_train = load_and_save_train_images(DATAPATH / "train")
    df_test = load_and_save_test_images(DATAPATH / "test")
    df_train.to_csv(DATAPATH / "df_train.csv", index=False)
    df_test.to_csv(DATAPATH / "df_test.csv", index=False)


if __name__ == "__main__":
    main()
