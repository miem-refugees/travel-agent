from loguru import logger
from smolagents import (
    DuckDuckGoSearchTool,
    LiteLLMModel,
    ToolCallingAgent,
    VisitWebpageTool,
)

from tool import GetExistingAvailableRubricsTool, TravelReviewQueryTool
from ui import TravelGradioUI
from qdrant import create_client


@logger.catch
def init_agent():
    logger.info("Init app dependencies...")

    qdrant_client = create_client()

    return ToolCallingAgent(
        model=LiteLLMModel(model_id="deepseek/deepseek-chat"),
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
    from openinference.instrumentation.smolagents import SmolagentsInstrumentor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

    SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)


def main():
    try_setup_langfuse_tracing()

    agent = init_agent()
    if agent is None:
        logger.error("Agent initialization failed")
        exit(1)

    TravelGradioUI(agent).launch()


if __name__ == "__main__":
    main()
