import os

from loguru import logger
from smolagents import (
    DuckDuckGoSearchTool,
    LiteLLMModel,
    ToolCallingAgent,
    VisitWebpageTool,
)

from travel_agent.retrieval.smolagents.tool import GetExistingAvailableRubricsTool, TravelReviewQueryTool
from travel_agent.ui.gradio import TravelGradioUI


@logger.catch
def init_agent(ollama_model: str):
    logger.info("Init app dependencies...")

    from travel_agent.qdrant import client as qdrant_client

    if ollama_model:
        logger.info("Using ollama {} as LLM", ollama_model)

        llm = LiteLLMModel(
            model_id=f"ollama_chat/{ollama_model}",
            api_base=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
            num_ctx=8192,
        )
    elif os.getenv("DEEPSEEK_API_KEY"):
        logger.info("Using Deepseek as LLM")
        llm = LiteLLMModel(model_id="deepseek/deepseek-chat")
    else:
        logger.error("No LLM model info provided")
        return

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
        planning_interval=5,
    )


def try_setup_langfuse_tracing():
    if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        logger.info("Tracing disabled (no OTEL env vars)")
        return

    from openinference.instrumentation.smolagents import SmolagentsInstrumentor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

    SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama-model", type=str, default=None)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--share", type=bool, default=False)
    args = parser.parse_args()

    try_setup_langfuse_tracing()

    agent = init_agent(args.ollama_model)
    if agent is None:
        logger.error("Agent initialization failed")
        exit(1)

    TravelGradioUI(agent).launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
