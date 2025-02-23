import getpass
import os
from dotenv import load_dotenv
from langsmith.wrappers import wrap_openai
from langsmith import traceable
import uuid
from openai import OpenAI

run_id = str(uuid.uuid4())

load_dotenv()

if not os.getenv("AZURE_OPENAI_API_KEY"):
  os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter API key for Azure: ")

from langchain_openai import AzureChatOpenAI

# llm = wrap_openai(AzureChatOpenAI(
#     azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
#     azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
#     openai_api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
# ))

openai_client = wrap_openai(OpenAI())

# This is the retriever we will use in RAG
# This is mocked out, but it could be anything we want
@traceable(run_type="retriever")
def retriever(query: str):
    results = ["Harrison worked at Kensho"]
    return results

# This is the end-to-end RAG chain.
# It does a retrieval step then calls OpenAI
@traceable(metadata={"llm": "gpt-4o-mini"})
def rag(question):
    docs = retriever(question)
    system_message = """Answer the users question using only the provided information below:
    
    {docs}""".format(docs="\n".join(docs))
    return openai_client.chat.completions.create(messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": question},

    ], model="gpt-4o-mini")

print(rag(
    "where did harrison work",
    langsmith_extra={"run_id": run_id, "metadata": {"user_id": "harrison"}}
))

from langsmith import Client
ls_client = Client()

ls_client.create_feedback(
    run_id,
    key="user-score",
    score=1.0,
)