from langchain_openai import ChatOpenAI

from meta_refiner import MetaRefiner
from executor import Executor
from pipeline import MetaPipeline
from config import settings


def main():

    llm = ChatOpenAI(
        model=settings.model,
        temperature=0,
    )

    refiner = MetaRefiner(llm)
    executor = Executor(llm)

    pipeline = MetaPipeline(
        refiner=refiner,
        executor=executor,
    )

    while True:
        query = input("\nUser question: ")

        if query.lower() in {"exit", "quit"}:
            break

        result = pipeline.run(query)

        print(result)


if __name__ == "__main__":
    main()
