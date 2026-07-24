from pydantic import BaseModel, Field
from llm_models import get_llm


class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(description="The title of the movie")
    year: int = Field(description="The year the movie was released")
    director: str = Field(description="The director of the movie")
    rating: float = Field(description="The movie's rating out of 10")


openai = get_llm()

model_with_structure = openai.with_structured_output(Movie)

response = model_with_structure.invoke("Provide details about the movie Inception")

print(response)
