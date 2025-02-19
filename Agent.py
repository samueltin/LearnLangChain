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

from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults(max_results=2)
search_results = search.invoke("what is the weather in London?")
# print(search_results)
# If we want, we can create other tools.
# Once we have all the tools we want, we can put them in a list that we will reference later.
tools = [search]

from langchain_core.messages import HumanMessage

# response = model.invoke([HumanMessage(content="hi!")])
# print(response.content)

model_with_tools = model.bind_tools(tools)
# response = model_with_tools.invoke([HumanMessage(content="Hi!")])

# print(f"ContentString: {response.content}")
# print(f"ToolCalls: {response.tool_calls}")

# response = model_with_tools.invoke([HumanMessage(content="What's the weather in SF?")])

# print(f"ContentString: {response.content}")
# print(f"ToolCalls: {response.tool_calls}")

from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(model, tools)

# response = agent_executor.invoke({"messages": [HumanMessage(content="hi there!")]})

# print(response["messages"])

# response = agent_executor.invoke(
#     {"messages": [HumanMessage(content="whats the weather in Hong Kong?")]}
# )
# print(response["messages"])

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the weather in Hong Kong?")]}
):
    print(chunk)
    print("----")