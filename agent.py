import os

from loguru import logger
from smolagents import (
    DuckDuckGoSearchTool,
    LiteLLMModel,
    ToolCallingAgent,
    VisitWebpageTool,
)

from prompts import PROMPT_TEMPLATES
from tool import GetExistingAvailableRubricsTool, TravelReviewQueryTool


@logger.catch
def create_agent() -> ToolCallingAgent:
    logger.info("init agent...")

    if os.getenv("DEEPSEEK_API_KEY"):
        logger.info("Using Deepseek as LLM")
        llm = LiteLLMModel(model_id="deepseek/deepseek-chat")
    else:
        logger.error("No DEEPSEEK_API_KEY")
        return

    return ToolCallingAgent(
        model=llm,
        tools=[
            GetExistingAvailableRubricsTool(),
            TravelReviewQueryTool(),
            DuckDuckGoSearchTool(),
            VisitWebpageTool(),
        ],
        max_steps=os.getenv("MAX_STEPS", 7),
        prompt_templates=PROMPT_TEMPLATES,
        verbosity_level=os.getenv("VERBOSITY_LEVEL", 1),
        planning_interval=os.getenv("PLANNING_INTERVAL", 5),
    )
