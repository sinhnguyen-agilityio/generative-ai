BRAINSTORM_PROMPT = """
You are a senior software architect.

Goal

{goal}

Generate THREE different implementation approaches.

Return JSON in this format:

[
  {{
    "name": "...",
    "description": "...",
    "advantages": [],
    "disadvantages": []
  }}
]
"""


EVALUATE_PROMPT = """
Evaluate this implementation.

{approach}

Return JSON.

{{
  "simplicity": 8,
  "scalability": 9,
  "maintainability": 7,
  "risk": 4,
  "reasoning": "..."
}}
"""


FINAL_SELECTION_PROMPT = """
Goal

{goal}

Evaluations

{evaluations}

Choose ONE.

Explain why.
"""
