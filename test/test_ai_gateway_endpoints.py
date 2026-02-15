from openai import OpenAI
from dotenv import load_dotenv
import os

from utils.clnt_utils import get_openai_client, is_databricks_ai_gateway_client, get_ai_gateway_model_names

load_dotenv()

# To get a DATABRICKS_TOKEN, click the "Generate Access Token" button or follow https://docs.databricks.com/en/dev-tools/auth/pat.html
DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
AI_GATEWAY_BASE_URL = os.environ.get('AI_GATEWAY_BASE_URL')
MODELS = get_ai_gateway_model_names()
print(f"✅ Using models: {MODELS}")
USE_DATABRICKS_AI_GATEWAY = is_databricks_ai_gateway_client()
print(f"✅ Using models: {MODELS}")

if USE_DATABRICKS_AI_GATEWAY:
  client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url=AI_GATEWAY_BASE_URL
  )
else:
  client = get_openai_client()

chat_completion = client.chat.completions.create(
  messages=[
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hello! How can I assist you today?"},
    {"role": "user", "content": "What is Databricks?"},
  ],
  model=MODELS[0],
  max_tokens=1024
)

print(chat_completion.choices[0].message.content)