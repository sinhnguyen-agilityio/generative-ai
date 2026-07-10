from dataloader.loader import DatasetLoader
from models.task import ReasoningTask
from datasets import load_dataset


class HumanEvalLoader(DatasetLoader):

    def load(self, limit=None):
        dataset = load_dataset("openai/openai_humaneval")["test"]
        tasks = []

        for row in dataset:
            tasks.append(
                ReasoningTask(
                    task_id=row["task_id"],
                    prompt=row["prompt"],
                    canonical_solution=row["canonical_solution"],
                    test=row["test"],
                    entry_point=row["entry_point"],
                )
            )

            if limit and len(tasks) >= limit:
                break

        return tasks
