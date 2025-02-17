import getpass
import os
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("AZURE_OPENAI_API_KEY"):
  os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter API key for Azure: ")

from langchain_openai import AzureChatOpenAI

model = AzureChatOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
    openai_api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
)

from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]

# response = model.invoke(messages)
# print(response.content)

# userMesaage = input("Enter your message: ")
# print(model.invoke([HumanMessage(userMesaage)]).content)

# userMesaage = input("Enter your message: ")
# print(model.invoke([HumanMessage(userMesaage)]).content)

for token in model.stream(messages):
    print(token.content, end="|")