import asyncio

from rag.agent_rag import AgentRAG
from rag.dataset import (
    build_evaluation_dataset,
    load_ragbench,
)


async def main():
    dataset = load_ragbench()
    evaluation_dataset = build_evaluation_dataset(
        dataset,
        limit=5,
    )

    rag = AgentRAG()

    for index, sample in enumerate(evaluation_dataset):

        print("\n" + "=" * 80)
        print(f"QUESTION {index + 1}")
        print("=" * 80)

        print(sample["question"])

        result = await rag.ainvoke(
            sample["question"]
        )

        print("\nANSWER:")
        print(result["answer"])

        print("\nMESSAGES:")

        for message in result["messages"]:
            print(
                f"\n[{message.type}]"
            )
            print(
                str(message.content)[:500]
            )


if __name__ == "__main__":
    asyncio.run(main())
