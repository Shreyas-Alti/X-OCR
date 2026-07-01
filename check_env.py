from dotenv import load_dotenv
import os

load_dotenv()
print("MOCK_MODE:", os.environ.get("MOCK_MODE"))
print("LLM_MODE:", os.environ.get("LLM_MODE"))
print("GEMINI_API_KEY set:", bool(os.environ.get("GEMINI_API_KEY")))