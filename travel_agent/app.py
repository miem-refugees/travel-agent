from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from smolagents import GradioUI, LiteLLMModel, ToolCallingAgent

from travel_agent.qdrant import client as qdrant_client
from travel_agent.retrieval.smolagents.tool import GetExistingAvailableRubricsTool, TravelReviewQueryTool

trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)


try:
    qdrant_client.info()
except Exception as e:
    print("Cannot ping qdrant:", e)
    raise


# To support llm via api uncomment:
# llm = InferenceClientModel("mistralai/Mistral-Small-3.1-24B-Instruct-2503")

MODEL_NAME = "hf.co/IlyaGusev/saiga_nemo_12b_gguf:Q4_0"

llm = LiteLLMModel(
    model_id=f"ollama_chat/{MODEL_NAME}",
    api_base="http://127.0.0.1:11434",
    num_ctx=8192,
)

review_search_tool = TravelReviewQueryTool(
    "intfloat/multilingual-e5-base",
    qdrant_client,
    "moskva_intfloat_multilingual_e5_base",
)
get_avaiable_rubrics_tool = GetExistingAvailableRubricsTool()


agent = ToolCallingAgent(
    model=llm,
    tools=[review_search_tool, get_avaiable_rubrics_tool],
    max_steps=3,
    verbosity_level=2,
)

if __name__ == "__main__":
    GradioUI(agent).launch(share=False)
