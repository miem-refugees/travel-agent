from loguru import logger

from agent import create_agent
from tracing import setup_langfuse_tracing
from ui import TravelGradioUI


def main():
    setup_langfuse_tracing()

    agent = create_agent()
    if agent is None:
        logger.error("Agent initialization failed")
        exit(1)

    TravelGradioUI(agent).launch(server_name=None)


if __name__ == "__main__":
    main()
