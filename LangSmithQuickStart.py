import getpass
import os
from dotenv import load_dotenv
from langsmith.wrappers import wrap_openai
from langsmith.wrappers.base import ModuleWrapper

load_dotenv()

if not os.getenv("AZURE_OPENAI_API_KEY"):
  os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter API key for Azure: ")

from langchain_openai import AzureChatOpenAI

llm = ModuleWrapper(AzureChatOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
    openai_api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
))


# This is the retriever we will use in RAG
# This is mocked out, but it could be anything we want
def retriever(query: str):
    results = ["Harrison worked at Kensho"]
    return results

# This is the end-to-end RAG chain.
# It does a retrieval step then calls OpenAI
def rag(question):
    docs = retriever(question)
    system_message = """Answer the users question using only the provided information below:
    
    {docs}""".format(docs="\n".join(docs))
    messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ]
    
    return llm.invoke(messages).content

print(rag("Where did Harrison work?"))