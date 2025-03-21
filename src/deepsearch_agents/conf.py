import os
import dotenv


dotenv.load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_API_KEY", None)
JINA_API_KEY = os.getenv("JINA_API_KEY", None)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
MY_OPENAI_API_KEY = os.getenv("MY_OPENAI_API_KEY", None)
MAX_TASK_DEPTH = os.getenv("MAX_TASK_DEPTH", 2)
MAX_SEARCH_RESULTS = 7
