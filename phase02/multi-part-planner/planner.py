import json

from prompts import (
    BRAINSTORM_PROMPT,
    EVALUATE_PROMPT,
    FINAL_SELECTION_PROMPT,
)

from models import (
    Approach,
    Evaluation,
)

from tracing import langfuse
from llm import LLM


class Planner:
    def __init__(self, llm: LLM):
        self.llm = llm

    def brainstorm(self, goal: str) -> list[Approach]:

        with langfuse.start_as_current_observation(
            name="brainstorm",
            as_type="span",
            input={"goal": goal},
        ) as observation:

            result = self.llm.generate_json(
                BRAINSTORM_PROMPT.format(goal=goal)
            )

            approaches = [
                Approach(
                    name=item["name"],
                    description=item["description"],
                    advantages=item["advantages"],
                    disadvantages=item["disadvantages"],
                )
                for item in result
            ]

            observation.update(
                output={
                    "approach_count": len(approaches),
                    "approaches": [a.name for a in approaches],
                }
            )

            return approaches

    def evaluate(self, approach: Approach) -> Evaluation:

        with langfuse.start_as_current_observation(
            name=f"evaluate:{approach.name}",
            as_type="span",
            input={
                "approach": approach.name,
                "description": approach.description,
            },
        ) as observation:

            result = self.llm.generate_json(
                EVALUATE_PROMPT.format(
                    approach=json.dumps(
                        {
                            "name": approach.name,
                            "description": approach.description,
                            "advantages": approach.advantages,
                            "disadvantages": approach.disadvantages,
                        },
                        indent=2,
                    )
                )
            )

            evaluation = Evaluation(
                approach=approach.name,
                simplicity=result["simplicity"],
                scalability=result["scalability"],
                maintainability=result["maintainability"],
                risk=result["risk"],
                reasoning=result["reasoning"],
            )

            observation.update(
                output={
                    "simplicity": evaluation.simplicity,
                    "scalability": evaluation.scalability,
                    "maintainability": evaluation.maintainability,
                    "risk": evaluation.risk,
                }
            )

            return evaluation

    def choose(
        self,
        goal: str,
        evaluations: list[Evaluation],
    ) -> str:

        with langfuse.start_as_current_observation(
            name="choose-best-approach",
            as_type="span",
            input={
                "goal": goal,
                "evaluations": [
                    vars(e)
                    for e in evaluations
                ],
            },
        ) as observation:

            design = self.llm.generate(
                FINAL_SELECTION_PROMPT.format(
                    goal=goal,
                    evaluations=json.dumps(
                        [vars(e) for e in evaluations],
                        indent=2,
                    ),
                )
            )

            observation.update(
                output=design,
            )

            return design
