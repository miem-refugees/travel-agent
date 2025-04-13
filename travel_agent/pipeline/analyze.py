from pathlib import Path

import pandas as pd
import yaml
from loguru import logger
from pydantic import BaseModel


class Params(BaseModel):
    dataset: str
    out_dir: str


def extract_unique_rubrics(df) -> pd.DataFrame:
    df["rubrics_list"] = df["rubrics"].str.split(";")

    exploded = df.explode("rubrics_list")
    exploded["rubrics"] = exploded["rubrics_list"].str.strip()

    unique_rubrics_df = exploded["rubrics"].value_counts().reset_index()

    return unique_rubrics_df


def main():
    params = Params.model_validate(yaml.safe_load(open("params.yaml"))["analyze"])
    logger.debug("Loaded params: {}", params)

    df = pd.read_csv(params.dataset)
    logger.debug("Loaded dataset: {}", params.dataset)

    unique_rubrics_df = extract_unique_rubrics(df)
    logger.info("Unique rubrics: {}", len(unique_rubrics_df))

    out_dir = Path(params.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = f"{params.out_dir}/unique_rubrics.csv"

    unique_rubrics_df.to_csv(output_path, index=False)
    logger.info("Saved unique rubrics to {}", output_path)


if __name__ == "__main__":
    main()
