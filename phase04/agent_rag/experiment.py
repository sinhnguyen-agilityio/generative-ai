import os
import os
from datetime import datetime
from typing import Optional

from openai import AsyncOpenAI

from ragas.llms import llm_factory
from rag.dataset import create_ragas_dataset
from rag.rag import RAG
from rag.retriever import BM25Retriever
from rag.evaluation import evaluate_rag


async def run_experiment(mode: str = "naive", model: str = "gpt-5-nano", name: Optional[str] = None):
    """
    Simple function to run RAG evaluation experiment.

    Args:
        mode: RAG mode - "naive" or "agentic"
        model: OpenAI model to use
        name: Optional experiment name. If None, auto-generated with timestamp

    Returns:
        List of experiment results
    """
    # Check for OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set your OpenAI API key: export OPENAI_API_KEY='your_key'"
        )

    # Prepare dataset and initialize system
    print("Initializing RAG system...")
    dataset = create_ragas_dataset()

    # Initialize RAG system with inline client creation
    openai_client = AsyncOpenAI(api_key=api_key)
    rag = RAG(
        llm_client=openai_client,
        retriever=BM25Retriever(),
        model=model,
        mode=mode
    )
    print("RAG system initialized!")

    # Run evaluation experiment
    experiment_results = await evaluate_rag.arun(
        dataset,
        name=name or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{'agenticrag' if mode == 'agentic' else 'naiverag'}",
        rag=rag,
        llm=llm_factory("gpt-4o-mini", client=openai_client,
                        temperature=1, top_p=None)
    )

    # Print basic results
    if experiment_results:
        pass_count = sum(1 for result in experiment_results if result.get(
            "correctness_score") == "pass")
        total_count = len(experiment_results)
        pass_rate = (pass_count / total_count) * 100 if total_count > 0 else 0

        print(f"Results: {pass_count}/{total_count} passed ({pass_rate:.1f}%)")

    return experiment_results
