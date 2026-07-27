from pydantic import BaseModel, Field


class ReActStep(BaseModel):
    """
    A single reasoning step produced by the LLM in ReAct format.
    """
    thought: str = Field(
        description="Short reasoning about the next step."
    )
    action: str | None = Field(
        default=None,
        description="Tool name to execute, or None if no tool is needed.",
    )
    action_input: str | None = Field(
        default=None,
        description="Search query for the tool, or None.",
    )
    final_answer: str | None = Field(
        default=None,
        description="Final answer for the user when enough evidence exists.",
    )
    observation: str | None = Field(
        default=None,
        description="Observation from the action result, if any.",
    )
