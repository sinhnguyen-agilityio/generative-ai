import json

from ragas import EvaluationDataset


class DatasetSerializer:
    def load_eval_dataset(self, path: str) -> EvaluationDataset:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return EvaluationDataset.from_list(data)

    def create_evaluation_dataset(
        self,
        samples: list[dict],
    ) -> EvaluationDataset:
        return EvaluationDataset.from_list(samples)
