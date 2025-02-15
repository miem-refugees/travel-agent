from dotenv import load_dotenv
import logging
from argparse import ArgumentParser
from os import getenv

from qdrant_client.qdrant_fastembed import SUPPORTED_EMBEDDING_MODELS

from travel_agent.qdrant.client import Qdrant

log = logging.getLogger(__name__)


def main():
    load_dotenv()

    parser = ArgumentParser(
        prog="qload", description="Tool to embed and fill qdrant database"
    )
    parser.add_argument(
        "--loglevel",
        default="debug",
        help="Log level",
        choices=["debug", "info", "warning", "error", "critical"],
    )
    parser.add_argument(
        "--model", help="HF model name", choices=SUPPORTED_EMBEDDING_MODELS
    )
    parser.add_argument("--qdrant-url", help="Qrant URL")

    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel.upper())

    upload(
        model=getattr(args, "model") or getenv("MODEL_NAME"),
        qdrant_url=getattr(args, "qdrant_url")
        or getenv("QRANT_URL")
        or "http://localhost:6433",
    )


def upload(model: str, qdrant_url: str):
    client = Qdrant(url=qdrant_url, model_name=model)

    log.debug("loading model: %s", model)
    client.set_model(model)
    log.info("loaded qdrant model: %s", model)

    # client.add_places_collection()
    # TODO


if __name__ == "__main__":
    main()
