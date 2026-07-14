from agent import build_agent
from callbacks import ReActLogger


def main():
    agent = build_agent()
    logger = ReActLogger()

    print("=" * 60)
    print("Self-Correcting Research Assistant")
    print("=" * 60)

    while True:
        question = input("\nQuestion (or 'exit'): ").strip()

        if question.lower() in ["exit", "quit"]:
            break

        print("\n" + "=" * 60)
        print(f"Question: {question}")
        print("=" * 60)

        try:
            for event in agent.stream(
                question,
            ):
                logger.on_event(event)

        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
