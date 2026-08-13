from datetime import datetime

from openai import AsyncOpenAI
from ragas import experiment
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

from generation.rag_chatbot import RAGChatbot
from tracer import langfuse


# ----------------------------------------
# Faithfulness scorer
# ----------------------------------------

client = AsyncOpenAI()

llm = llm_factory(
    "gpt-4o-mini",
    client=client,
)

scorer = Faithfulness(llm=llm)


def _get_question_from_row(row):
    if isinstance(row, dict):
        return (
            row.get("user_input")
            or row.get("question")
            or row.get("input")
        )

    for field in (
        "user_input",
        "question",
        "input",
    ):
        value = getattr(row, field, None)

        if value is not None:
            return value

    return None


def create_baseline_experiment(
    chatbot: RAGChatbot,
):
    @experiment()
    async def baseline_experiment(row):
        question = _get_question_from_row(row)

        if not question:
            raise ValueError(
                f"Missing question from row: {row}"
            )

        result = await chatbot.ainvoke(question)
        answer = result["answer"]
        response = (
            answer.content
            if hasattr(answer, "content")
            else str(answer)
        )

        documents = result.get("documents", [])

        retrieved_contexts = [
            doc.page_content
            if hasattr(doc, "page_content")
            else str(doc)
            for doc in documents
        ]

        with langfuse.start_as_current_observation(
            as_type="span",
            name="ragas_faithfulness",
            input={
                "question": question,
                "response": response,
                "retrieved_contexts": retrieved_contexts,
            },
        ) as evaluation:
            faithfulness = await scorer.ascore(
                user_input=question,
                response=response,
                retrieved_contexts=retrieved_contexts,
            )

            evaluation.update(
                output={
                    "faithfulness": faithfulness.value,
                }
            )

        base_row = (
            row.model_dump(exclude_none=True)
            if hasattr(row, "model_dump")
            else dict(row)
        )

        return {
            **base_row,
            "response": response,
            "faithfulness": faithfulness.value,
            "retrieved_contexts": retrieved_contexts,
            "experiment_name": "baseline_v1",
            "model_version": "gpt-4o-mini",
            "timestamp": datetime.now().isoformat(),
        }

    return baseline_experiment
