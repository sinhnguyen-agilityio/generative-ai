from abc import ABC, abstractmethod
from models.task import ReasoningTask


class DatasetLoader(ABC):

    @abstractmethod
    def load(self, limit: int | None = None) -> list[ReasoningTask]:
        pass
