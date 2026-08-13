from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()


llm = ChatOpenAI(model="gpt-5-nano")

rewriter_prompt_template = """
    Generate search query for the ChromaDB vector store
    from a user question, allowing for a more accurate
    response through semantic search.
    Just return the revised ChromaDB query, with quotes around it.
    User question: {user_question}
    Revised ChromaDB query:
"""
rewriter_prompt = ChatPromptTemplate.from_template(
    rewriter_prompt_template)

rewriter_chain = rewriter_prompt | llm | StrOutputParser()
