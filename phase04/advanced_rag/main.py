from pathlib import Path
from dotenv import load_dotenv
from ragas.backends.inmemory import InMemoryBackend

from evaluations.serializer import DatasetSerializer
from generation.rag_chatbot import RAGChatbot
from ingestion.ingest import create_retriever
from evaluations.experiment import create_baseline_experiment
from ragas import Dataset
import asyncio


import argparse


load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation"
    )

    parser.add_argument(
        "--reingest",
        action="store_true",
        help="Re-ingest documents and rebuild Chroma",
    )

    return parser.parse_args()


async def main():
    try:
        args = parse_args()

        # ----------------------------------------
        # Chroma / ingestion
        # ----------------------------------------
        retriever = create_retriever(
            reingest=args.reingest
        )

        # ----------------------------------------
        # Evaluation dataset
        # ----------------------------------------
        evaluator = DatasetSerializer()
        testset = evaluator.load_eval_dataset(
            "evaluations/dataset.json"
        )
        print(
            f"Testset size: {len(testset.samples)}"
        )

        # ----------------------------------------
        # Chatbot
        # ----------------------------------------
        chatbot = RAGChatbot(
            retriever=retriever
        )

        # ----------------------------------------
        # Experiment dataset
        # ----------------------------------------
        experiment_dataset = Dataset(
            "baseline_eval",
            backend=InMemoryBackend(),
            data=[
                {
                    "user_input": sample.user_input,
                    "reference": sample.reference,
                }
                for sample in testset.samples
                if isinstance(
                    sample.user_input,
                    str,
                )
            ],
        )

        # ----------------------------------------
        # Run experiment
        # ----------------------------------------
        baseline_experiment = (
            create_baseline_experiment(chatbot)
        )
        experiment_results = (
            await baseline_experiment.arun(
                experiment_dataset,
                name="baseline_v1",
            )
        )
        print(
            f"Experiment rows: "
            f"{len(experiment_results)}"
        )

        # ----------------------------------------
        # Export
        # ----------------------------------------
        df = experiment_results.to_pandas()
        Path("reports").mkdir(
            parents=True,
            exist_ok=True,
        )
        df.to_csv(
            "reports/ragas_experiment_report.csv",
            index=False,
        )
    except Exception as e:
        print("Error: ", e)

if __name__ == "__main__":
    asyncio.run(main())
