from loguru import logger
from smolagents import LiteLLMModel, ToolCallingAgent

from travel_agent.retrieval.smolagents.tool import GetExistingAvailableRubricsTool, TravelReviewQueryTool
from travel_agent.ui.gradio import TravelGradioUI


@logger.catch
def init_agent(model_name: str):
    logger.info("Init app dependencies...")

    from travel_agent.qdrant import client as qdrant_client

    # To support llm via api uncomment:
    # llm = InferenceClientModel("mistralai/Mistral-Small-3.1-24B-Instruct-2503")

    llm = LiteLLMModel(
        model_id=f"ollama_chat/{model_name}",
        api_base="http://127.0.0.1:11434",
        num_ctx=8192,
    )

    llm.create_client()

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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="hf.co/IlyaGusev/saiga_nemo_12b_gguf:Q4_0")
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
        logger.info("Tracing disabled (launthing with --tracing flag)")

    agent = init_agent(args.model_name)
    if agent is None:
        logger.error("Agent initialization failed")
        exit(1)

    TravelGradioUI(agent).launch(share=args.share)
