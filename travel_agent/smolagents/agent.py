import os
from typing import Optional

from loguru import logger
from smolagents import (
    DuckDuckGoSearchTool,
    LiteLLMModel,
    ToolCallingAgent,
    VisitWebpageTool,
)

from travel_agent.smolagents.prompts import PROMPT_TEMPLATES
from travel_agent.smolagents.tool import GetExistingAvailableRubricsTool, TravelReviewQueryTool


@logger.catch
def create_agent(ollama_model: Optional[str]) -> ToolCallingAgent:
    logger.info("init agent...")

    if ollama_model:
        logger.info("using ollama: {}", ollama_model)

        llm = LiteLLMModel(
            model_id=f"ollama_chat/{ollama_model}",
            api_base=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
            num_ctx=8192,
        )
    elif os.getenv("DEEPSEEK_API_KEY"):
        logger.info("Using Deepseek as LLM")
        llm = LiteLLMModel(model_id="deepseek/deepseek-chat")
    elif os.getenv("LITELLM_MODEL_ID"):
        model_id = os.getenv("LITELLM_MODEL_ID")

        logger.info("Using custom LiteLLMModel {}", model_id)
        llm = LiteLLMModel(model_id=model_id)
    else:
        logger.error("No LLM model info provided. Set DEEPSEEK_API_KEY or LITELLM_MODEL_ID or --ollama-model=...")
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
