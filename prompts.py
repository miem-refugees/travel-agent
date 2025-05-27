import importlib

import yaml
from smolagents import PromptTemplates

PROMPT_TEMPLATES: PromptTemplates = yaml.safe_load(
    importlib.resources.files(".").joinpath("prompts.yaml").read_text()
)
