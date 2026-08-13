import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

EMBEDDING_MODEL = "text-embedding-3-small"

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "ragbench_hotpotqa"
LLM_MODEL = "gpt-5-nano"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4
