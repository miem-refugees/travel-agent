import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger
from pydantic import BaseModel


class Params(BaseModel):
    out_dir: str
    phrases_to_find: list[str]


def main():
    params = Params.model_validate(
        yaml.safe_load(open("params.yaml"))["generate_embed_questions"]
    )
    logger.debug("Loaded params: {}", params)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="csv-dataset path")

    args = parser.parse_args()
    input_data_path = args.dataset

    df = pd.read_csv(input_data_path)

    df["text"] = (
        df["text"]
        .str.lower()
        .apply(lambda text: re.sub(r"[^а-яa-z0-9ё\s.,!?-]", " ", text))
    )

    query_to_places = defaultdict(set)

    for phrase in params.phrases_to_find:
        for _, row in df.iterrows():
            if phrase in row["text"]:
                query_to_places[phrase].add(row["name_ru"])

        if len(query_to_places[phrase]) == 0:
            logger.warning("No expected places for phrase: {}", phrase)

    rows = [
        {"query": phrase, "expected_places": ";".join(sorted(list(places)))}
        for phrase, places in query_to_places.items()
        if places
    ]

    eval_df = pd.DataFrame(rows)

    out_dir = Path(params.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = f"{params.out_dir}/{Path(input_data_path).name}"
    eval_df.to_csv(output_path, index=False)
    logger.info("Saved to {}", output_path)


if __name__ == "__main__":
    main()
