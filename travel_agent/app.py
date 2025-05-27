import argparse

from loguru import logger

from travel_agent.smolagents import create_agent
from travel_agent.tracing import setup_langfuse_tracing
from travel_agent.ui.gradio import TravelGradioUI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama-model", type=str, default=None)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--share", type=bool, default=False)
    args = parser.parse_args()

    setup_langfuse_tracing()

    agent = create_agent(args.ollama_model)
    if agent is None:
        logger.error("Agent initialization failed")
        exit(1)

    logger.success("agent ready")

    TravelGradioUI(agent).launch(args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
