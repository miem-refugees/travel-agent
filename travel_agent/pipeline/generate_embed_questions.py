import argparse
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger
from pydantic import BaseModel


class Params(BaseModel):
    out_dir: str
    rubric_to_query: dict[str, str]


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

    df = df[df["text"].notna() & (df["text"].str.len() > 50)].copy()
    df["rubric_list"] = df["rubrics"].str.split(";")

    rows = []

    for rubric, query in params.rubric_to_query.items():
        matched = df[
            df["rubric_list"].apply(
                lambda lst: any(rubric.lower() in r.lower() for r in lst)
            )
        ]
        top_names = matched["name_ru"].dropna().unique()

        if len(top_names) == 0:
            logger.warning("Rubric {} has no matches", rubric)
            continue

        logger.info("Rubric {} matched {} places", rubric, len(top_names))

        if len(top_names):
            rows.append({"query": query, "expected_places": ";".join(top_names)})

    eval_df = pd.DataFrame(rows)

    out_dir = Path(params.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = f"{params.out_dir}/{Path(input_data_path).name}"
    eval_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
