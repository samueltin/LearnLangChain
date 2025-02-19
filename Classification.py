import getpass
import os
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("AZURE_OPENAI_API_KEY"):
  os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter API key for Azure: ")

from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
    openai_api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    temperature=0, 
    model="gpt-4o-mini"
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)


class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text", enum=["happy", "neutral", "sad"])
    aggressiveness: int = Field(
        description="describes how aggressive the statement is, the higher the number the more aggressive", enum=[1, 2, 3, 4, 5]
    )
    language: str = Field(description="The language the text is written in", enum=["spanish", "english", "french", "german", "italian"])


# LLM
llm = llm.with_structured_output(
    Classification
)

# inp = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
# prompt = tagging_prompt.invoke({"input": inp})
# response = llm.invoke(prompt)

# inp = "Estoy muy enojado con vos! Te voy a dar tu merecido!"
# prompt = tagging_prompt.invoke({"input": inp})
# response = llm.invoke(prompt)

inp = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
inp = "Estoy muy enojado con vos! Te voy a dar tu merecido!"
inp = "Weather is ok here, I can go outside without much more than a coat"
prompt = tagging_prompt.invoke({"input": inp})
response = llm.invoke(prompt)

print(response)