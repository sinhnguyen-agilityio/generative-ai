from openai import OpenAI
from abc import ABC, abstractmethod
from models.ticket import SecureTicket


class ToxicityDetector(ABC):

    @abstractmethod
    def detect(self, ticket: SecureTicket) -> SecureTicket:
        """Validate the ticket for toxic content."""
        raise NotImplementedError


class OpenAIModerationDetector(ToxicityDetector):

    def __init__(self, api_key: str):
        self._client = OpenAI(api_key=api_key)

    def detect(self, ticket: SecureTicket) -> SecureTicket:
        result = self._client.moderations.create(
            model="omni-moderation-latest",
            input=ticket.sanitized_text,
        )
        moderation = result.results[0]
        ticket.toxicity_detected = moderation.flagged

        return ticket
