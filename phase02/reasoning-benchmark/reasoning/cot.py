from .base import ReasoningStrategy
from .prompts import COT_PROMPT


class ChainOfThought(ReasoningStrategy):
    @property
    def name(self) -> str:
        return "chain_of_thought"
    
    @property
    def prompt_template(self):
        return COT_PROMPT
