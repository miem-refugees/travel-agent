import argparse
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
            TravelReviewQueryTool(retrieve_limit=10),
            DuckDuckGoSearchTool(),
            VisitWebpageTool(),
        ],
        max_steps=os.getenv("MAX_STEPS", 7),
        verbosity_level=os.getenv("VERBOSITY_LEVEL", 1),
        planning_interval=os.getenv("PLANNING_INTERVAL", 5),
    )


def setup_langfuse_tracing(trace: bool):
    if not trace or not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        logger.info("Tracing disabled")
        return

    from openinference.instrumentation.smolagents import SmolagentsInstrumentor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

    SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama-model", type=str, default=None)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--trace", type=bool, default=False)
    parser.add_argument("--share", type=bool, default=False)
    args = parser.parse_args()

    setup_langfuse_tracing(args.trace)

    agent = init_agent(args.ollama_model)
    if agent is None:
        logger.error("Agent initialization failed")
        exit(1)

    logger.success("agent ready")

    TravelGradioUI(agent).launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
