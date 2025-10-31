import os
from dotenv import load_dotenv
import google.generativeai as genai
load_dotenv()
print("SNOWFLAKE_ACCOUNT:", os.getenv("SNOWFLAKE_ACCOUNT"))
print("SNOWFLAKE_HOST:", os.getenv("SNOWFLAKE_HOST"))
print("Pinecone:", os.getenv("PINECONE_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
for m in genai.list_models():
    print(m.name)