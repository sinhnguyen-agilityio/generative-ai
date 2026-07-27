from dataclasses import dataclass


@dataclass
class Approach:
    name: str
    description: str
    advantages: list[str]
    disadvantages: list[str]


@dataclass
class Evaluation:
    approach: str
    simplicity: int
    scalability: int
    maintainability: int
    risk: int
    reasoning: str
