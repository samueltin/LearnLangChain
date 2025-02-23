from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
# print(db.dialect)
# print(db.get_usable_table_names())
# result = db.run("SELECT * FROM Artist LIMIT 10;")
# print(result)

from typing_extensions import TypedDict


class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

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
)

from langchain import hub

query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")

assert len(query_prompt_template.messages) == 1
# query_prompt_template.messages[0].pretty_print()

from typing_extensions import Annotated

class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]


def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}


query = write_query({"question": "How many Employees are there?"})
print(query)

from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool


def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}

# result = execute_query(query)

# print(result)


def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

from langgraph.graph import START, StateGraph

# graph_builder = StateGraph(State).add_sequence(
#     [write_query, execute_query, generate_answer]
# )
# graph_builder.add_edge(START, "write_query")
# graph = graph_builder.compile()

# # for step in graph.stream(
# #     {"question": "How many employees are there?"}, stream_mode="updates"
# # ):
# #     print(step)

# from langgraph.checkpoint.memory import MemorySaver

# memory = MemorySaver()
# graph = graph_builder.compile(checkpointer=memory, interrupt_before=["execute_query"])

# # Now that we're using persistence, we need to specify a thread ID
# # so that we can continue the run after review.
# config = {"configurable": {"thread_id": "1"}}

# for step in graph.stream(
#     {"question": "How many employees are there?"},
#     config,
#     stream_mode="updates",
# ):
#     print(step)

# try:
#     user_approval = input("Do you want to go to execute query? (yes/no): ")
# except Exception:
#     user_approval = "no"

# if user_approval.lower() == "yes":
#     # If approved, continue the graph execution
#     for step in graph.stream(None, config, stream_mode="updates"):
#         print(step)
# else:
#     print("Operation cancelled by user.")

# from langchain_community.agent_toolkits import SQLDatabaseToolkit

# toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# tools = toolkit.get_tools()

# from langchain import hub

# prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")

# assert len(prompt_template.messages) == 1
# # prompt_template.messages[0].pretty_print()

# system_message = prompt_template.format(dialect="SQLite", top_k=5)

# from langchain_core.messages import HumanMessage
# from langgraph.prebuilt import create_react_agent

# agent_executor = create_react_agent(llm, tools, prompt=system_message)

# question = "Which country's customers spent the most?"

# for step in agent_executor.stream(
#     {"messages": [{"role": "user", "content": question}]},
#     stream_mode="values",
# ):
#     step["messages"][-1].pretty_print()

# question = "Describe the playlisttrack table"

# for step in agent_executor.stream(
#     {"messages": [{"role": "user", "content": question}]},
#     stream_mode="values",
# ):
#     step["messages"][-1].pretty_print()

# import ast
# import re


# def query_as_list(db, query):
#     res = db.run(query)
#     res = [el for sub in ast.literal_eval(res) for el in sub if el]
#     res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
#     return list(set(res))


# artists = query_as_list(db, "SELECT Name FROM Artist")
# albums = query_as_list(db, "SELECT Title FROM Album")
# # albums[:5]

# # print(albums)

# from langchain_openai import AzureOpenAIEmbeddings

# embeddings = AzureOpenAIEmbeddings(
#     azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
#     azure_deployment="text-embedding-ada-002",
#     openai_api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
# )

# from langchain_core.vectorstores import InMemoryVectorStore

# vector_store = InMemoryVectorStore(embeddings)

# from langchain.agents.agent_toolkits import create_retriever_tool

# _ = vector_store.add_texts(artists + albums)
# retriever = vector_store.as_retriever(search_kwargs={"k": 5})
# description = (
#     "Use to look up values to filter on. Input is an approximate spelling "
#     "of the proper noun, output is valid proper nouns. Use the noun most "
#     "similar to the search."
# )
# retriever_tool = create_retriever_tool(
#     retriever,
#     name="search_proper_nouns",
#     description=description,
# )

# print(retriever_tool.invoke("Alice Chains"))

# # Add to system message
# suffix = (
#     "If you need to filter on a proper noun like a Name, you must ALWAYS first look up "
#     "the filter value using the 'search_proper_nouns' tool! Do not try to "
#     "guess at the proper name - use this function to find similar ones."
# )

# system = f"{system_message}\n\n{suffix}"

# tools.append(retriever_tool)

# agent = create_react_agent(llm, tools, prompt=system)

# question = "How many albums does alis in chain have?"

# for step in agent.stream(
#     {"messages": [{"role": "user", "content": question}]},
#     stream_mode="values",
# ):
#     step["messages"][-1].pretty_print()