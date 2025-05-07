from loguru import logger
from smolagents import LiteLLMModel, ToolCallingAgent

from tool import GetExistingAvailableRubricsTool, TravelReviewQueryTool
from ui import TravelGradioUI
from qdrant import create_client


@logger.catch
def init_agent():
    logger.info("Init app dependencies...")

    qdrant_client = create_client()

    llm = LiteLLMModel(
        model_id="deepseek/deepseek-chat",
    )

    return ToolCallingAgent(
        model=llm,
        tools=[
            GetExistingAvailableRubricsTool(),
            TravelReviewQueryTool(
                "intfloat/multilingual-e5-base",
                qdrant_client,
                "moskva_intfloat_multilingual_e5_base",
            ),
        ],
        max_steps=7,
        verbosity_level=1,
        planning_interval=3,
    )


if __name__ == "__main__":
    agent = init_agent()
    if agent is None:
        logger.error("Agent initialization failed")
        exit(1)

    TravelGradioUI(agent).launch()
