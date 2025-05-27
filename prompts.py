import yaml
from smolagents import PromptTemplates

with open("prompts.yaml", "r") as f:
    PROMPT_TEMPLATES: PromptTemplates = yaml.safe_load(f)
