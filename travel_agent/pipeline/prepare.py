import argparse
import yaml
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def preprocess_df(df):
    processed_df = df.dropna(subset=["name_ru"])

    logger.info("Removed %d NaN rows", processed_df.shape[0] - df.shape[0])

    return processed_df


def save_split_dataset(df, split_ratio, seed, out_dir):
    train_df, test_df = train_test_split(df, test_size=split_ratio, random_state=seed)

    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    logger.info("Saved train, test to: %s", out_dir)


def save_moscow_dataset(df, out_dir):
    moscow_df = df[df["address"].str.startswith("Москва", na=False)]

    moscow_df.to_csv(out_dir / "moscow.csv", index=False)

    logger.info("Saved Moscow-only data to %s", out_dir)


def main():
    params = yaml.safe_load(open("params.yaml"))["prepare"]
    logger.info("Loaded params: %s", params)

    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument("dataset", type=str, help="Path to dataset YAML file")
    parser.add_argument(
        "-o",
        "--out",
        required=True,
        type=str,
        help="Output directory for processed files",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)

    df = preprocess_df(df)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_split_dataset(df, params["split"], params["seed"], out_dir)
    save_moscow_dataset(df, out_dir)


if __name__ == "__main__":
    main()
