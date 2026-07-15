from __future__ import annotations

import os
import sqlite3
import tempfile

from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader

from langchain_community.tools import DuckDuckGoSearchRun

from langchain_community.vectorstores import FAISS

from langchain_core.messages import (
    BaseMessage,
    SystemMessage
)

from langchain_core.tools import tool


from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)


from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.graph import (
    START,
    StateGraph
)

from langgraph.graph.message import add_messages

from langgraph.prebuilt import (
    ToolNode,
    tools_condition
)

import requests



# -------------------
# Environment
# -------------------

load_dotenv()



# -------------------
# Gemini LLM
# -------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)



embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)



# -------------------
# PDF Retriever Store
# -------------------

_THREAD_RETRIEVERS: Dict[str, Any] = {}

_THREAD_METADATA: Dict[str, list] = {}



def _get_retriever(thread_id: Optional[str]):

    if (
        thread_id
        and thread_id in _THREAD_RETRIEVERS
    ):
        return _THREAD_RETRIEVERS[thread_id]

    return None




def ingest_pdf(
        file_bytes: bytes,
        thread_id: str,
        filename: Optional[str] = None
):

    """
    Upload multiple PDFs and create
    a common FAISS retriever.
    """

    if not file_bytes:
        raise ValueError(
            "Empty file"
        )


    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(file_bytes)

        temp_path = temp_file.name



    try:

        loader = PyPDFLoader(
            temp_path
        )


        docs = loader.load()



        splitter = RecursiveCharacterTextSplitter(

            chunk_size=1000,

            chunk_overlap=200,

            separators=[
                "\n\n",
                "\n",
                " ",
                ""
            ]
        )


        chunks = splitter.split_documents(
            docs
        )



        # Add filename metadata

        for doc in chunks:

            doc.metadata["source"] = (
                filename
                or
                "uploaded_file.pdf"
            )



        # If first PDF

        if thread_id not in _THREAD_RETRIEVERS:


            vector_store = FAISS.from_documents(

                chunks,

                embeddings

            )


        else:

            # Add new documents
            vector_store = (
                _THREAD_RETRIEVERS[thread_id]
                .vectorstore
            )


            vector_store.add_documents(
                chunks
            )



        retriever = vector_store.as_retriever(

            search_type="similarity",

            search_kwargs={
                "k":4
            }

        )


        _THREAD_RETRIEVERS[thread_id] = retriever



        if thread_id not in _THREAD_METADATA:

            _THREAD_METADATA[thread_id] = []



        _THREAD_METADATA[thread_id].append(

            {
                "filename":
                filename,

                "pages":
                len(docs),

                "chunks":
                len(chunks)
            }

        )



        return {

            "filename": filename,

            "pages": len(docs),

            "chunks": len(chunks)

        }



    finally:


        try:

            os.remove(
                temp_path
            )

        except OSError:

            pass


# -------------------
# 3. Tools
# -------------------

search_tool = DuckDuckGoSearchRun(
    region="us-en"
)



@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str
) -> dict:
    """
    Basic calculator tool.
    """

    try:

        if operation == "add":
            result = first_num + second_num

        elif operation == "sub":
            result = first_num - second_num

        elif operation == "mul":
            result = first_num * second_num

        elif operation == "div":

            if second_num == 0:
                return {
                    "error":
                    "Cannot divide by zero"
                }

            result = first_num / second_num

        else:
            return {
                "error":
                "Invalid operation"
            }


        return {
            "result": result
        }


    except Exception as e:

        return {
            "error": str(e)
        }





@tool
def rag_tool(
    query: str,
    thread_id: Optional[str] = None
) -> dict:
    """
    Retrieve information from uploaded PDF.
    """

    retriever = _get_retriever(
        thread_id
    )


    if retriever is None:

        return {

            "error":
            "No PDF uploaded",

            "query":
            query
        }



    documents = retriever.invoke(
        query
    )



    results = []


    for doc in documents:

        results.append(

            {
                "content":
                doc.page_content,

                "source":
                doc.metadata.get(
                    "source",
                    "unknown"
                ),

                "page":
                doc.metadata.get(
                    "page",
                    "unknown"
                )
            }

        )



    return {

        "query":
        query,

        "documents":
        results

    }





tools = [

    search_tool,

    calculator,

    rag_tool

]


llm_with_tools = llm.bind_tools(
    tools
)

# -------------------
# 4. State
# -------------------

class ChatState(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]



# -------------------
# 5. Nodes
# -------------------

def chat_node(
    state: ChatState,
    config=None
):

    """
    Main LLM node.
    Model decides whether to answer
    or call a tool.
    """


    thread_id = None


    if config and isinstance(config, dict):

        thread_id = (
            config
            .get("configurable", {})
            .get("thread_id")
        )



    system_message = SystemMessage(

        content=f"""

You are an AI Document Assistant.

Rules:

1. If user asks anything related to uploaded documents,
always use rag_tool.

2. Always pass thread_id:
{thread_id}

3. Use calculator only for calculations.

4. Use web search only when information is not available
in uploaded documents.

5. Give clear answers with sources if available.

6. If no PDF is uploaded, tell user to upload a document.

"""
    )



    messages = [

        system_message,

        *state["messages"]

    ]

    response = llm_with_tools.invoke(
        messages,
        config=config
    )

    return {

        "messages":
        [
            response
        ]

    }

# Tool execution node

tool_node = ToolNode(
    tools
)

# -------------------
# 6. Checkpointer
# -------------------

conn = sqlite3.connect(

    database="chatbot.db",

    check_same_thread=False

)

checkpointer = SqliteSaver(
    conn=conn
)

# -------------------
# 7. Graph
# -------------------

graph = StateGraph(
    ChatState
)

graph.add_node(
    "chat_node",
    chat_node
)

graph.add_node(
    "tools",
    tool_node
)

graph.add_edge(
    START,
    "chat_node"
)


graph.add_conditional_edges(

    "chat_node",

    tools_condition

)

graph.add_edge(

    "tools",

    "chat_node"

)

chatbot = graph.compile(

    checkpointer=checkpointer

)

# -------------------
# 8. Helpers
# -------------------

def retrieve_all_threads():

    all_threads = set()

    for checkpoint in checkpointer.list(None):

        all_threads.add(

            checkpoint
            .config["configurable"]
            ["thread_id"]

        )

    return list(all_threads)

def thread_has_document(
    thread_id: str
):

    return (

        str(thread_id)

        in _THREAD_RETRIEVERS

    )




def thread_document_metadata(
    thread_id: str
):

    return (

        _THREAD_METADATA

        .get(
            str(thread_id),
            {}
        )

    )