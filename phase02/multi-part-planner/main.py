from llm import LLM
from planner import Planner
from tracing import langfuse


def main():

    goal = """
    Design a software feature implementation using Tree of Thoughts,
    including a brainstorm of three approaches, evaluations of each, and a final chosen path.
    """

    with langfuse.start_as_current_observation(
        name="software-design-planner",
        as_type="span",
        input={"goal": goal},
    ) as root:

        llm = LLM()
        planner = Planner(llm)

        approaches = planner.brainstorm(goal)

        print("\n=== Brainstorm ===")
        for approach in approaches:
            print(approach)

        print("\n=== Evaluation ===")

        evaluations = []

        for approach in approaches:
            evaluation = planner.evaluate(approach)
            evaluations.append(evaluation)
            print(evaluation)

        print("\n=== Final Design ===")

        design = planner.choose(goal, evaluations)

        print(design)

        root.update(
            output=design,
        )

    langfuse.flush()


if __name__ == "__main__":
    main()
