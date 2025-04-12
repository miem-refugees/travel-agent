from pathlib import Path

import pandas as pd
import yaml
from loguru import logger
from pydantic import BaseModel
from transliterate import slugify


class Params(BaseModel):
    dataset: str
    out_dir: str
    cities: list[str]


def preprocess_df(df):
    processed_df = df.dropna(subset=["name_ru"])

    logger.info("Removed {} NaN rows", processed_df.shape[0] - df.shape[0])

    return processed_df


def save_city_prefix_dataset(df, city, out_dir):
    logger.info("Creating '{}'-prefix dataset", city)

    city_df = df[df["address"].str.startswith(city, na=False)]

    filename = f"{slugify(city)}.csv"
    city_df.to_csv(out_dir / filename, index=False)

    logger.info("Created {}", filename)


def main():
    params = Params.model_validate(yaml.safe_load(open("params.yaml"))["prepare"])
    logger.debug("Loaded params: {}", params)

    df = pd.read_csv(params.dataset)
    df = preprocess_df(df)

    out_dir = Path(params.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for city in params.cities:
        save_city_prefix_dataset(df, city, out_dir)


if __name__ == "__main__":
    main()
