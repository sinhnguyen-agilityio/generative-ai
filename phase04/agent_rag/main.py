import asyncio
from experiment import run_experiment
import pandas as pd

if __name__ == "__main__":
    naive_rs = asyncio.run(run_experiment(mode="naive", model="gpt-5-nano"))
    agentic_rs = asyncio.run(run_experiment(
        mode="agentic", model="gpt-5-nano"))

    naive_df = pd.DataFrame(naive_rs)
    agentic_df = pd.DataFrame(agentic_rs)

    comparison = pd.DataFrame({
        "question": naive_df["question"],
        "expected_answer": naive_df["expected_answer"],

        "naive_response": naive_df["model_response"],
        "naive_score": naive_df["correctness_score"],

        "agentic_response": agentic_df["model_response"],
        "agentic_score": agentic_df["correctness_score"],
    })

    comparison.to_csv(
        "data/compare.csv",
        index=False,
        encoding="utf-8-sig",
    )
