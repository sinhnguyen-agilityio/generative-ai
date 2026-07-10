COT_PROMPT = """
You are an expert Python programmer.

Solve the following HumanEval programming task.

Before writing the final code, think through the solution step by step.

Your response must have exactly two sections.

### Reasoning
Describe your reasoning in detail.

Include:
- Understanding the task
- Important edge cases
- Algorithm selection
- Why the algorithm is correct
- Complexity analysis
- A brief verification using one or two test cases

### Code
Provide only the final Python implementation.

Task:

{task}
"""

COD_PROMPT = """
You are an expert Python programmer.

Solve the following HumanEval programming task.

Before writing the final code, create a concise draft plan.

Your response must have exactly two sections.

### Draft
Write only short notes.

Rules:
- bullet points only
- no complete sentences
- one line per idea
- maximum 30 words total

### Code
Provide only the final Python implementation.

Task:

{task}
"""
