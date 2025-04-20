import importlib

import yaml

from smolagents import PromptTemplates

RUS_PROMPT_TEMPLATES: PromptTemplates = yaml.safe_load(
    importlib.resources.files("travel_agent.retrieval.smolagents").joinpath("toolcalling_agent.yaml").read_text()
)
