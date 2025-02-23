from langchain_community.utilities import SQLDatabase
import getpass
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain import hub
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from typing_extensions import Annotated
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

load_dotenv()

db = SQLDatabase.from_uri(database_uri="sqlite:///Chinook.db")
# print(db.get_usable_table_names())

class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str



llm = AzureChatOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
    openai_api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")
# assert len(query_prompt_template.messages) == 1
# query_prompt_template.messages[0].pretty_print()

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

def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}

def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question list of dict \n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

def write_query_and_execute_query(state: State):
    query = write_query(state)
    result = execute_query(query)

    return {**query, **result}

graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)
graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile()

# for step in graph.stream(
#     {"question": "Please show 10 employees"}, stream_mode="updates"
# ):
#     print(step)