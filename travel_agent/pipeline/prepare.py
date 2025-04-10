import yaml
import logging
import pandas as pd
from pathlib import Path
from transliterate import slugify


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def preprocess_df(df):
    processed_df = df.dropna(subset=["name_ru"])

    logger.info("Removed %d NaN rows", processed_df.shape[0] - df.shape[0])

    return processed_df


def save_city_prefix_dataset(df, city, out_dir):
    logger.info("Creating '%s'-prefix dataset", city)

    city_df = df[df["address"].str.startswith(city, na=False)]

    filename = f"{slugify(city)}.csv"
    city_df.to_csv(out_dir / filename, index=False)

    logger.info("Created %s", filename)


def main():
    params = yaml.safe_load(open("params.yaml"))["prepare"]
    logger.debug("Loaded params: %s", params)

    df = pd.read_csv(params["dataset"])
    df = preprocess_df(df)

    out_dir = Path(params["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for city in params["cities"]:
        save_city_prefix_dataset(df, city, out_dir)


if __name__ == "__main__":
    main()
