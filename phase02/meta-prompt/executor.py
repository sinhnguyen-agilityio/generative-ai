from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tracer import get_langfuse_handler


class Executor:
    """
    Executes the refined prompt produced by MetaRefiner.

    This class never modifies prompts.
    """

    def __init__(
        self,
        llm: ChatOpenAI,
    ):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                        You are a knowledgeable AI assistant.
                        Execute the provided prompt exactly as instructed.
                        Do not explain your reasoning.
                        Return only the requested answer.

                        Additional constraints:
                            - Keep the response concise.
                            - Do not exceed 200 words.
                            - If the task cannot be completed within the limit, summarize the most important information.
                            - Do not explain your reasoning.
                            - Return only the final answer.
                    """,
                ),
                ("human", "{prompt}"),
            ]
        )
        self.handler = get_langfuse_handler()
        self.chain = (
            self.prompt
            | self.llm
            | StrOutputParser()
        )

    def execute(
        self,
        prompt: str,
    ) -> str:
        """
        Execute a refined prompt.

        Parameters
        ----------
        prompt : str

        Returns
        -------
        str
            Final answer.
        """

        return self.chain.invoke(
            {
                "prompt": prompt,
            },
            config={"callbacks": [self.handler]}
        )
