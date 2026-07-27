from .base import ReasoningStrategy
from .prompts import COD_PROMPT


class ChainOfDraft(ReasoningStrategy):
    @property
    def name(self) -> str:
        return "chain_of_draft"
    
    @property
    def prompt_template(self):
        return COD_PROMPT
