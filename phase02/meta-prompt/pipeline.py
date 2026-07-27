from meta_refiner import MetaRefiner
from executor import Executor


class MetaPipeline:
    def __init__(
        self,
        refiner: MetaRefiner,
        executor: Executor,
    ):
        self.refiner = refiner
        self.executor = executor

    def run(self, query: str) -> str:
        refined_prompt = self.refiner.refine(query)
        answer = self.executor.execute(refined_prompt)

        return answer
