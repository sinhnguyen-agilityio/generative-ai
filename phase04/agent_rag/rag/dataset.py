
from pathlib import Path

import sys
from pathlib import Path
from ragas import Dataset

sys.path.insert(0, str(Path(__file__).parent))


def create_ragas_dataset() -> Dataset:
    """Create a Ragas Dataset from the downloaded CSV file."""
    dataset_path = Path("data/hf_qa_doc_eval.csv")
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        print(f"Dataset not exists at {dataset_path}")
        raise

    dataset: Dataset = Dataset(
        name='hf_qa_doc_eval', backend='local/csv', root_dir="data")

    import pandas as pd
    df = pd.read_csv(dataset_path)

    for _, row in df.iterrows():
        dataset.append(
            {"question": row["question"], "expected_answer": row["expected_answer"]})

    dataset.save()
    print(f"Created Ragas dataset with {len(df)} samples")
    return dataset
