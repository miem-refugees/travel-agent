import os
from loguru import logger
from smolagents import (
    DuckDuckGoSearchTool,
    LiteLLMModel,
    ToolCallingAgent,
    VisitWebpageTool,
)

from tool import GetExistingAvailableRubricsTool, TravelReviewQueryTool
from ui import TravelGradioUI


@logger.catch
def init_agent():
    logger.info("Init app dependencies...")

    return ToolCallingAgent(
        model=LiteLLMModel(model_id="deepseek/deepseek-chat"),
        tools=[
            GetExistingAvailableRubricsTool(),
            TravelReviewQueryTool(retrieve_limit=os.getenv("RETRIEVE_LIMIT", 10)),
            DuckDuckGoSearchTool(),
            VisitWebpageTool(),
        ],
        max_steps=os.getenv("MAX_STEPS", 7),
        verbosity_level=os.getenv("VERBOSITY_LEVEL", 1),
        planning_interval=os.getenv("PLANNING_INTERVAL", 5),
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
