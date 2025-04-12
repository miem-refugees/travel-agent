import yaml
from loguru import logger


def main():
    params = yaml.safe_load(open("params.yaml"))["embed"]
    logger.debug("Loaded params: {}", params)

    logger.error("Not implemented stage")


if __name__ == "__main__":
    main()
