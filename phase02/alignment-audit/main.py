from agent import LangChainAgent
from alignment import AlignmentChecker
from pipeline import AlignmentAuditPipeline
from scenarios import SCENARIOS


def print_menu() -> list:
    print("\n" + "=" * 60)
    print("Alignment Audit Demo")
    print("=" * 60)

    scenarios = list(SCENARIOS.items())

    for index, (_, scenario) in enumerate(scenarios, start=1):
        print(f"{index}. {scenario.name} - {scenario.description}")

    print("\nType the scenario number to run.")
    print("Type 'quit' or 'q' to exit.\n")

    return scenarios


def main() -> None:
    agent = LangChainAgent()
    checker = AlignmentChecker()

    pipeline = AlignmentAuditPipeline(
        agent=agent,
        checker=checker,
    )

    scenarios = list(SCENARIOS.items())

    while True:
        print_menu()

        choice = input("> ").strip().lower()

        if choice in ("q", "quit", "exit"):
            print("\nGoodbye!")
            break

        if not choice.isdigit():
            print("\n❌ Please enter a valid number.")
            continue

        index = int(choice)

        if index < 1 or index > len(scenarios):
            print("\n❌ Invalid scenario number.")
            continue

        _, scenario = scenarios[index - 1]

        print("\n" + "=" * 60)
        print(f"Running: {scenario.name}")
        print(f"Description: {scenario.description}")
        print("=" * 60)

        pipeline.run(scenario.prompt)

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
