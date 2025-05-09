from loguru import logger
from smolagents import (
    DuckDuckGoSearchTool,
    LiteLLMModel,
    MultiStepAgent,
    ToolCallingAgent,
    VisitWebpageTool,
)

from travel_agent.retrieval.smolagents.tool import GetExistingAvailableRubricsTool, TravelReviewQueryTool
from travel_agent.ui.gradio import TravelGradioUI


@logger.catch
def init_agent(deepseek: bool) -> MultiStepAgent:
    logger.info("Init app dependencies...")

    from travel_agent.qdrant import client as qdrant_client

    if deepseek:
        logger.info("Using Deepseek as LLM")
        llm = LiteLLMModel(model_id="deepseek/deepseek-chat")
    else:
        model_name = "hf.co/IlyaGusev/saiga_nemo_12b_gguf:Q4_0"

        logger.info("Using ollama {} as LLM", model_name)

        llm = LiteLLMModel(
            model_id=f"ollama_chat/{model_name}",
            api_base="http://127.0.0.1:11434",
            num_ctx=8192,
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
            DuckDuckGoSearchTool(),
            VisitWebpageTool(),
        ],
        max_steps=7,
        verbosity_level=1,
        planning_interval=3,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--deepseek", type=bool, default=False)
    parser.add_argument("--tracing", type=bool, default=False)
    parser.add_argument("--share", type=bool, default=False)
    args = parser.parse_args()

    if args.tracing:
        from openinference.instrumentation.smolagents import SmolagentsInstrumentor
        from phoenix.otel import register

        # creates a tracer provider to capture OTEL traces
        tracer_provider = register(project_name="travel-agent-local", verbose=False)
        # automatically captures any smolagents calls as traces
        SmolagentsInstrumentor().instrument(tracer_provider=tracer_provider)
    else:
        logger.debug("Tracing disabled (launthing with --tracing flag)")

    agent = init_agent(args.deepseek)
    if agent is None:
        logger.error("Agent initialization failed")
        exit(1)

    TravelGradioUI(agent).launch(share=args.share)


if __name__ == "__main__":
    main()
