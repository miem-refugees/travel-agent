import argparse

from loguru import logger
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from phoenix.otel import register
from smolagents import GradioUI, LiteLLMModel, ToolCallingAgent

from travel_agent.qdrant import client as qdrant_client
from travel_agent.retrieval.smolagents.tool import GetExistingAvailableRubricsTool, TravelReviewQueryTool


@logger.catch
def init_agent(model_name: str):
    logger.info("Init app dependencies...")

    qdrant_client.info()

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
        max_steps=3,
        verbosity_level=2,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="hf.co/IlyaGusev/saiga_nemo_12b_gguf:Q4_0")
    parser.add_argument("--tracing", type=bool, default=False)
    parser.add_argument("--share", type=bool, default=False)
    args = parser.parse_args()

    if args.tracing:
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

    GradioUI(agent).launch(share=args.share)
