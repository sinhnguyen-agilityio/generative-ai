import asyncio
from operator import itemgetter

from langfuse.langchain import CallbackHandler

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from trace_client import get_client


langfuse = get_client()

prompt1 = ChatPromptTemplate.from_template(
    "what is the city {person} is from?"
)

prompt2 = ChatPromptTemplate.from_template(
    "what country is the city {city} in? respond in {language}"
)

model = ChatOpenAI()

chain1 = prompt1 | model | StrOutputParser()

chain2 = (
    {
        "city": chain1,
        "language": itemgetter("language"),
    }
    | prompt2
    | model
    | StrOutputParser()
)


async def test_langfuse():
    with langfuse.start_as_current_observation(
        as_type="span",
        name="LangChain Runnable Demo",
    ) as root_span:

        langfuse_handler = CallbackHandler()

        # Invoke
        with langfuse.start_as_current_observation(
            as_type="span",
            name="Invoke",
        ):
            chain2.invoke(
                {
                    "person": "obama",
                    "language": "english",
                },
                config={"callbacks": [langfuse_handler]},
            )

        # Async Invoke
        with langfuse.start_as_current_observation(
            as_type="span",
            name="Async Invoke",
        ):
            await chain2.ainvoke(
                {
                    "person": "biden",
                    "language": "german",
                },
                config={"callbacks": [langfuse_handler]},
            )

        # Batch
        with langfuse.start_as_current_observation(
            as_type="span",
            name="Batch",
        ):
            chain2.batch(
                [
                    {
                        "person": "elon musk",
                        "language": "english",
                    },
                    {
                        "person": "mark zuckerberg",
                        "language": "english",
                    },
                ],
                config={"callbacks": [langfuse_handler]},
            )

        # Async Batch
        with langfuse.start_as_current_observation(
            as_type="span",
            name="Async Batch",
        ):
            await chain2.abatch(
                [
                    {
                        "person": "jeff bezos",
                        "language": "english",
                    },
                    {
                        "person": "tim cook",
                        "language": "english",
                    },
                ],
                config={"callbacks": [langfuse_handler]},
            )

        # Stream
        with langfuse.start_as_current_observation(
            as_type="span",
            name="Stream",
        ):
            for _ in chain2.stream(
                {
                    "person": "steve jobs",
                    "language": "english",
                },
                config={"callbacks": [langfuse_handler]},
            ):
                pass

        # Async Stream
        with langfuse.start_as_current_observation(
            as_type="span",
            name="Async Stream",
        ):
            async for _ in chain2.astream(
                {
                    "person": "bill gates",
                    "language": "english",
                },
                config={"callbacks": [langfuse_handler]},
            ):
                pass

        root_span.update(
            input={"example": "Runnable demo"},
            output={"status": "completed"},
        )

    langfuse.flush()


if __name__ == "__main__":
    asyncio.run(test_langfuse())
