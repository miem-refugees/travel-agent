import importlib

import yaml
from smolagents import PromptTemplates

PROMPT_TEMPLATES: PromptTemplates = yaml.safe_load(
    importlib.resources.files("travel_agent.smolagents").joinpath("toolcalling_agent.yaml").read_text()
)
