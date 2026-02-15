from databricks.sdk import WorkspaceClient
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from databricks_langchain import ChatDatabricks
import os
from dotenv import load_dotenv

def get_workspace_client():
    """Initialize Databricks workspace client."""
    databricks_token = os.environ.get("DATABRICKS_TOKEN")
    databricks_host = os.environ.get("DATABRICKS_HOST")
    return WorkspaceClient(
        profile="DEFAULT",
        host=databricks_host,
        token=databricks_token,
    )
def get_ai_gateway_model_names():
    """Get the model names for the AI Gateway."""
    return os.environ.get("AI_GATEWAY_MODELS").split(",")


def get_databricks_workspace_client():
    """Initialize Databricks client."""
    return get_workspace_client().serving_endpoints.get_open_ai_client()

def get_databricks_ai_gateway_client():
    return OpenAI(api_key=os.environ.get("DATABRICKS_TOKEN"),
                base_url=os.environ.get("AI_GATEWAY_BASE_URL"))

def get_openai_client():
    """Initialize OpenAI client."""
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def is_databricks_client():
    """Check if the client will a Databricks provider."""
    return os.environ.get("USE_DATABRICKS_CLIENT") == "True"

def is_databricks_ai_gateway_client():
    """Check if the client will a Databricks AI Gateway provider."""
    return os.environ.get("USE_DATABRICKS_AI_GATEWAY") == "True"

def is_openai_client():
    """Check if the client will a OpenAI provider."""
    return os.environ.get("USE_OPENAI_CLIENT") == "True"

def get_langchain_chat_openai_client(model_name: str, temperature: float = 1.0):
    """Initialize LangChain ChatOpenAI client."""
    return ChatOpenAI(model=model_name, temperature=temperature)

def get_databricks_langchain_chat_client(model_name: str, temperature: float = 1.0):
    """Initialize LangChain ChatDatabricks client."""
    return ChatDatabricks(endpoint=model_name, temperature=temperature)

def get_databricks_ai_gateway_langchain_client(model_name: str, temperature: float = 1.0):
    """Initialize LangChain client for Databricks AI Gateway."""
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=os.environ.get("DATABRICKS_TOKEN"),
        base_url=os.environ.get("AI_GATEWAY_BASE_URL")
    )
def main():
    """Main function."""
    databricks_ai_gateway_provider = is_databricks_ai_gateway_client()
    print("✅ Using", "Databricks AI Gateway" if databricks_ai_gateway_provider else "OpenAI", "as provider")
    model_names = get_ai_gateway_model_names()
    print(f"✅ Using models: {model_names}")
    
    if databricks_ai_gateway_provider:
        llm = get_databricks_ai_gateway_langchain_client(model_name=model_names[0], temperature=1.0)
    else:
        llm = get_langchain_chat_openai_client(model_name="gpt-5-2", temperature=1.0)

    print(llm.invoke([SystemMessage(content="You are a helpful assistant."), HumanMessage(content="What is the capital of France?")]))
    

if __name__ == "__main__":
    load_dotenv("../.env")
    main()