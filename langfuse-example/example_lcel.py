import asyncio
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from trace_client import langfuse_handler


prompt1 = ChatPromptTemplate.from_template(
    "what is the city {person} is from?")
prompt2 = ChatPromptTemplate.from_template(
    "what country is the city {city} in? respond in {language}"
)

model = ChatOpenAI()

chain1 = prompt1 | model | StrOutputParser()
chain2 = (
    {"city": chain1, "language": itemgetter("language")}
    | prompt2
    | model
    | StrOutputParser()
)


async def test_langfuse():
    # Sync invoke
    chain2.invoke(
        {"person": "obama", "language": "english"},
        config={"callbacks": [langfuse_handler]},
    )

    # Async invoke
    await chain2.ainvoke(
        {"person": "biden", "language": "german"},
        config={"callbacks": [langfuse_handler]},
    )

    # Batch
    chain2.batch(
        [
            {"person": "elon musk", "language": "english"},
            {"person": "mark zuckerberg", "language": "english"},
        ],
        config={"callbacks": [langfuse_handler]},
    )

    # Async batch
    await chain2.abatch(
        [
            {"person": "jeff bezos", "language": "english"},
            {"person": "tim cook", "language": "english"},
        ],
        config={"callbacks": [langfuse_handler]},
    )

    # Stream (consume the iterator)
    for _ in chain2.stream(
        {"person": "steve jobs", "language": "english"},
        config={"callbacks": [langfuse_handler]},
    ):
        pass

    # Async stream (consume the async iterator)
    async for _ in chain2.astream(
        {"person": "bill gates", "language": "english"},
        config={"callbacks": [langfuse_handler]},
    ):
        pass

if __name__ == "__main__":
    asyncio.run(test_langfuse())
