from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from prompt import META_PROMPT
from tracer import get_langfuse_handler


class MetaRefiner:
    """
    Responsible for improving an underspecified user query into
    a high-quality prompt for another LLM.

    It NEVER answers the user's question.
    """

    def __init__(
        self,
        llm: ChatOpenAI,
    ):
        self.llm = llm
        self.handler = get_langfuse_handler()
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system", META_PROMPT,
                ),
                ("human", "{query}"),
            ],
        )

        self.chain = (
            self.prompt
            | self.llm
            | StrOutputParser()
        )

    def refine(self, query: str) -> str:
        """
        Convert an underspecified query into
        a high-quality executable prompt.

        Parameters
        ----------
        query : str

        Returns
        -------
        str
            Refined prompt.
        """

        return self.chain.invoke(
            {
                "query": query,
            },
            config={"callbacks": [self.handler]}
        )
