from typing import Any, Dict

from dotenv import load_dotenv

from ragas import experiment
from ragas.metrics import DiscreteMetric

from rag.rag import RAG

# Load environment variables
load_dotenv(".env")


# Define correctness metric
correctness_metric = DiscreteMetric(
    name="correctness",
    prompt="""Compare the model response to the expected answer and determine if it's correct.    
        Consider the response correct if it:
        1. Contains the key information from the expected answer
        2. Is factually accurate based on the provided context
        3. Adequately addresses the question asked

        Return 'pass' if the response is correct, 'fail' if it's incorrect.

        Question: {question}
        Expected Answer: {expected_answer}
        Model Response: {response}

        Evaluation:""",
    allowed_values=["pass", "fail"],
)


@experiment()
async def evaluate_rag(row: Dict[str, Any], rag: RAG, llm) -> Dict[str, Any]:
    """
    Run RAG evaluation on a single row.
    
    Args:
        row: Dictionary containing question, context, and expected_answer
        rag: Pre-initialized RAG instance
        llm: Pre-initialized LLM client for evaluation
        
    Returns:
        Dictionary with evaluation results
    """
    question = row["question"]
    
    # Query the RAG system
    rag_response = await rag.query(question, top_k=4)
    model_response = rag_response.get("answer", "")
    
    # Evaluate correctness asynchronously
    score = await correctness_metric.ascore(
        question=question,
        expected_answer=row["expected_answer"],
        response=model_response,
        llm=llm
    )
    
    
    # Return evaluation results
    result = {
        **row,
        "model_response": model_response,
        "correctness_score": score.value,
        "correctness_reason": score.reason,
        "retrieved_documents": [
            doc.get("content", "")[:200] + "..." if len(doc.get("content", "")) > 200 else doc.get("content", "")
            for doc in rag_response.get("retrieved_documents", [])
        ]
    }
    
    return result
