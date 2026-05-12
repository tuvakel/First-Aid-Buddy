from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage,  AIMessage
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from concurrent.futures import ThreadPoolExecutor
from langchain.schema import Document
from jinja2 import Template
import time
import json
#from langgraph.checkpoint.memory import MemorySaver
import re
import pickle
from langchain_core.messages import AIMessage    
from src.utils import init_chat_LLM
llm_70b = init_chat_LLM(api_key=st.secrets["GROQ"]["GROQ_API_KEY"])


def process_pages(pages:List[Document]):
    
    for doc in pages:
        # Remove ESI handbook repeating chapter headers
        doc.page_content = doc.page_content.replace(
            "Chapter 1. Introduction to the Emergency Severity Index: A Research-Based Triage Tool", ""
        )
        doc.page_content = doc.page_content.replace(
            "Chapter 2. Overview of the Emergency Severity Index", ""
        )
        doc.page_content = doc.page_content.replace(
            "Chapter 1. Introduction to the Emergency Severity Index: A Research-Based Triage Tools", ""
        )
        # Remove stray page numbers
        doc.page_content = re.sub(r'^\d+\s*\n\s*\n', '', doc.page_content)
        # Fix capitalisation artefact from Word-to-PDF conversion (paTienT → patient)
        doc.page_content = re.sub(r'(?<=[a-z])T(?=[a-z])', 't', doc.page_content)
    return pages


def process_pdf_triage(file_path:str):
    print('process_pdf_triage')
    # Load the PDF pages
    loader = PyPDFLoader(file_path)
    pages = loader.load()[10:]

    # Split the pages into subgroups for each core
    num_cores = os.cpu_count() or 1  # fallback to 1 in containers
    chunk_size = len(pages) // num_cores + (len(pages) % num_cores > 0)
    chunks = [pages[i:i + chunk_size] for i in range(0, len(pages), chunk_size)]

    # Parallelize the work with ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        processed_chunks = list(executor.map(process_pages, chunks))

    # Combine the results
    documents = [page for chunk in processed_chunks for page in chunk]
    return documents


def create_bm25_retriever_triage(pdf_file_path: str, bm25_index_path):
    """
    Create or load BM25 retriever safely.
    """

    # ✅ Ensure directory exists BEFORE anything else
    os.makedirs(os.path.dirname(bm25_index_path), exist_ok=True)

    if os.path.exists(bm25_index_path):
        with open(bm25_index_path, "rb") as f:
            bm25_retriever = pickle.load(f)
            bm25_retriever.k = 3
            documents = []
    else:
        documents = process_pdf_triage(pdf_file_path)
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 3

        # ✅ Now safe to save
        with open(bm25_index_path, "wb") as f:
            pickle.dump(bm25_retriever, f)

    return bm25_retriever, documents


def create_triage_retriever(pdf_file_path:str, bm25_index_path:str, faiss_path:str):
    # Step 1: Configure the BM25 index for titles
    bm25_retriever, documents = create_bm25_retriever_triage(pdf_file_path, bm25_index_path)
    # Step 2: Configure FAISS for the content
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    faiss_index_file = os.path.join(faiss_path, "index.faiss")
    if os.path.exists(faiss_index_file):
        vectorstore = FAISS.load_local(faiss_path, embeddings=embedding, allow_dangerous_deserialization=True)
        print('load triage retriever')
    else:
        if documents:
            vectorstore = FAISS.from_documents(documents, embedding=embedding)
            vectorstore.save_local(faiss_path)
        else:
            documents = process_pdf_triage(pdf_file_path)
            vectorstore = FAISS.from_documents(documents, embedding=embedding)
            vectorstore.save_local(faiss_path)
    similarity_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4})

    # Step 3: Configure a MultiRetriever
    ensemble_retriever = EnsembleRetriever(retrievers=[
        bm25_retriever,
        similarity_retriever
    ], weights=[0.3, 0.7])
    class SafeRetriever:
        def __init__(self, retriever):
            self.retriever = retriever

        def invoke(self, query):
            if not isinstance(query, str):
                print("⚠️ FIXING QUERY TYPE:", type(query))
                query = str(query)
            return self.retriever.invoke(query)

    return SafeRetriever(ensemble_retriever)


severity_to_color = {
    1: "#00FF00",  
    2: "#ADFF2F",  
    3: "#FFFF00",  
    4: "#FFA500",  
    5: "#FF0000"   
}



class TriageState(TypedDict):
    ensemble_retriever_triage: Any
    severity: int
    questions: Annotated[list, add_messages]
    messages: Annotated[list, add_messages]
    full_query: str


def start_emergency_bot(state:TriageState):
    # Initial coordination node, returns the state unchanged
    return {}


def log_state(node_name, state:TriageState):
    print(f"Node '{node_name}' State: {state}")


def extract_json_from_response(response_text: str) -> dict:
    cleaned_resp = response_text.strip()
    
    # 1. Try to find a JSON object first (existing logic)
    match = re.search(r'(\{.*\})', cleaned_resp, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass  # fall through to plain-text parsing

    # 2. Fallback: parse plain "Key: Value" format
    result = {}
    for line in cleaned_resp.splitlines():
        if ':' in line:
            key, _, value = line.partition(':')
            result[key.strip()] = value.strip()
    
    if 'Score' in result or 'Question' in result:
        return result

    raise ValueError(
        f"JSON decoding error: Could not parse response. "
        f"Response received: {response_text}"
    )


def triage_evaluation(state: TriageState):
    messages = state['messages']

    # Convert messages to plain text
    clean_messages = []
    for m in messages:
        if hasattr(m, "content"):
            clean_messages.append(m.content)
        else:
            clean_messages.append(str(m))
    messages_str = "\n".join(clean_messages)


    contextualize_q_system_prompt = f"""You are an AI assistant specialized in medical triage. Your task is to analyze the conversation history between the user and the AI, understand the user's current medical concerns, and summarize the key information. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.

    ### Instructions:
    1. **Triage Context**:
    - Review the conversation history to understand the user's medical concerns and symptoms.

    2. **Focus on Current Query**:
    - Pay special attention to the user's latest messages to ensure the summary reflects their current problem or question.

    3. **Be Concise and Relevant**:
    - Provide a clear and concise summary (1-3 sentences) of the user's current medical concern.
    - Highlight the symptoms and context provided by the user that are essential for triage.

    ### Input:
    Conversation History:
    {messages_str}

    ### Output:"""
    if isinstance(state, dict):
        print("STATE KEYS:", state.keys())
    response_obj = llm_70b.invoke([HumanMessage(content=contextualize_q_system_prompt)])

    full_query = response_obj.content

    if not isinstance(full_query, str):
        print("⚠️ full_query was not string:", type(full_query))
        full_query = str(full_query)
    
    if not isinstance(full_query, str):
        full_query = str(full_query)
    print("DEBUG full_query TYPE:", type(full_query))
    print("DEBUG full_query VALUE:", full_query)
    print(f"full_query: {full_query}")
    
    
    ensemble_retriever_triage = state.get('ensemble_retriever_triage')

    retrieved_docs = []
    if ensemble_retriever_triage is None:
        print("⚠️ No retriever found in state")
        retrieved_docs = []
    else:
        retrieved_docs = []

        if ensemble_retriever_triage:
            try:
                if not isinstance(full_query, str):
                    full_query = str(full_query)
                    
                print("FINAL QUERY:", full_query)
                print("TYPE:", type(full_query))
                
                print("🔍 QUERY TYPE BEFORE RETRIEVER:", type(full_query))
                retrieved_docs = ensemble_retriever_triage.invoke(full_query)

            except Exception as e:
                print("🚨 RETRIEVER FAILED:", e)
                print("🚨 QUERY VALUE:", full_query)

    retrieved_info = [
        doc.page_content for doc in retrieved_docs[:2]
        if hasattr(doc, "page_content")
    ]
    full_retrieved_info = " ".join(retrieved_info) if retrieved_info else ""
    system_prompt = Template("""
    You are a highly skilled professional in emergency medicine, specializing in Triage. Your task is to assess the severity of the user's situation by providing a score from 1 to 5, or ask a concise question to obtain further information if necessary.

    ### Instructions:
    1. **Severity Assessment**:
        - Analyze the user's situation to determine its severity using the information provided in the documents.
        - The score is defined as:
        - `1`: Minimal severity, no immediate danger.
        - `5`: Critical and potentially life-threatening emergency, requires immediate intervention.

    2. **Request for Further Information**:
        - If the available information is not sufficient, ask a direct and specific question to clarify.

    3. **Response Format**:
    - The response must be a JSON with one of the following formats:
     - **If you have enough information**:
       {
         "Reasoning": "Briefly explain your assessment.",
         "Score": "Score between 1 and 5"
       }
     - **If you need further information**:
       {
         "Reasoning": "Explain why more information is needed.",
         "Question": "Direct and specific question."
       }

    You must return the output in the specified format, without adding backticks and json string in response.                         
    
    ### Example Outputs:
    #### Scenario 1:
    You have enough information to assess the severity.
    Output: {"Reasoning": "Based on the information I have, the cut doesn't seem severe, so the severity of the situation is relatively low.", "Score" : "2"} 

    #### Scenario 2:
    You need further information.
    Output: {"Reasoning": "I don't have enough information to determine the severity of the situation. I need to ask another question.", "Question" : "Have you ever had allergic reactions in your life?"} 

    ### Documents:
    {{full_retrieved_info}}

    ### User's Medical Situation:
    {{full_query}}
    
    """)
    system_prompt = system_prompt.render(full_retrieved_info=full_retrieved_info, full_query=full_query)
    updated_prompt = [HumanMessage(system_prompt)]
    start_time = time.time()
    response = llm_70b.invoke(updated_prompt).content
    end_time = time.time()
    print(f"Time taken for LLM invoke: {end_time - start_time:.2f} seconds\n")
    print(f"response: {response}")
    response = extract_json_from_response(response)
    # Analyze the type of response
    if 'Score' in response:
        try:
            raw_score = float(response['Score'])       # float handles "3.5" safely
            clamped = max(1, min(5, int(raw_score)))   # clamp to 1–5
        except (ValueError, TypeError):
            clamped = 3                                 # safe middle-ground fallback
        return {
            "severity": clamped,
            "full_query": full_query
        }

    else:
        return {
            "questions": [AIMessage(content=response['Question'])],
            "full_query": full_query  # ✅ preserve it
        }


def create_triage_agent():
    graph = StateGraph(TriageState)
    graph.add_node("start_emergency_bot", start_emergency_bot)
    graph.set_entry_point("start_emergency_bot")
    graph.add_node("triage_evaluation", triage_evaluation)
    graph.add_edge("start_emergency_bot", "triage_evaluation")
    graph.set_finish_point("triage_evaluation")

    app = graph.compile() #checkpointer=memory

    #img_bytes = app.get_graph().draw_mermaid_png()
    #with open('presentation/agents/triage.png', 'wb') as f:
        #f.write(img_bytes)

    return app
