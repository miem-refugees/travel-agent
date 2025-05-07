from loguru import logger
from smolagents import LiteLLMModel, ToolCallingAgent

from travel_agent.retrieval.smolagents.tool import GetExistingAvailableRubricsTool, TravelReviewQueryTool
from travel_agent.ui.gradio import TravelGradioUI


@logger.catch
def init_agent():
    logger.info("Init app dependencies...")

    from travel_agent.qdrant import client as qdrant_client

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
