from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from concurrent.futures import ThreadPoolExecutor
from langchain.schema import Document
import re
from jinja2 import Template
import time
import json
#from langgraph.checkpoint.memory import MemorySaver
#memory = MemorySaver()
import pickle

llm_70b = ChatGroq(model="llama-3.1-70b-versatile", api_key=st.secrets["GROQ"]["GROQ_API_KEY"])
llm_8b = ChatGroq(model="llama-3.1-8b-instant", api_key=st.secrets["GROQ"]["GROQ_API_KEY"])

def process_pages(pages:List[Document]):
    import re
    for doc in pages:
        doc.page_content = doc.page_content.replace(
            "Manuale regionale Triage intra-ospedaliero modello Lazio a cinque codici \n \n", ""
        )
        doc.page_content = re.sub(r'^\d+\s*\n\s*\n', '', doc.page_content)
    return pages

def process_pdf_triage(file_path:str):
    print('process_pdf_triage')
    # Carica le pagine del PDF
    loader = PyPDFLoader(file_path)
    pages = loader.load()[11:]

    # Dividi le pagine in sottogruppi per ogni core
    num_cores = os.cpu_count()  # Numero di core disponibili
    chunk_size = len(pages) // num_cores + (len(pages) % num_cores > 0)
    chunks = [pages[i:i + chunk_size] for i in range(0, len(pages), chunk_size)]

    # Parallelizza il lavoro con ProcessPoolExecutor
    with ThreadPoolExecutor() as executor:
        processed_chunks = list(executor.map(process_pages, chunks))

    # Combina i risultati
    documents = [page for chunk in processed_chunks for page in chunk]
    return documents


def create_bm25_retriever_triage(pdf_file_path:str, bm25_index_path):
    """
    Crea o carica un retriever BM25.

    Args:
        documents (list): Lista di documenti da indicizzare.
        bm25_index_path (str): Percorso per salvare o caricare l'indice BM25.

    Returns:
        BM25Retriever: Un retriever BM25.
    """
    # Se esiste un file salvato, carica il retriever
    if os.path.exists(bm25_index_path):
        #print("Caricamento retriever BM25 esistente.")
        with open(bm25_index_path, "rb") as f:
            bm25_retriever : BM25Retriever = pickle.load(f)
            bm25_retriever.k = 3
            documents = []
    else:
        #print("Creazione di un nuovo retriever BM25.")
        # Creazione del retriever BM25
        documents = process_pdf_triage(pdf_file_path)
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 3
        # Salva il retriever
        with open(bm25_index_path, "wb") as f:
            pickle.dump(bm25_retriever, f)
    
    return bm25_retriever, documents


def create_triage_retriever(pdf_file_path:str, bm25_index_path:str, faiss_path:str):
    # Step 1: Configura l'indice BM25 per i titoli
    bm25_retriever, documents = create_bm25_retriever_triage(pdf_file_path, bm25_index_path)
    # Step 2: Configura FAISS per i contenuti
    embedding = OpenAIEmbeddings(api_key=st.secrets["OPENAI"]["OPENAI_API_KEY"])
    if os.path.exists(faiss_path):
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

    # Step 3: Configura un MultiRetriever
    ensemble_retriever = EnsembleRetriever(retrievers=[
        bm25_retriever,
        similarity_retriever
    ], weights=[0.3, 0.7])
    return ensemble_retriever



severity_to_color = {
    1: "#00FF00",  # Verde
    2: "#ADFF2F",  # Giallo-verde
    3: "#FFFF00",  # Giallo
    4: "#FFA500",  # Arancione
    5: "#FF0000"   # Rosso
}


class TriageState(TypedDict):
    ensemble_retriever_triage : EnsembleRetriever
    severity : int
    questions : Annotated[list, add_messages]
    messages: Annotated[list, add_messages]
    full_query : str

def start_emergency_bot(state:TriageState):
    # Nodo di coordinamento iniziale, ritorna lo stato invariato
    return state

def log_state(node_name, state:TriageState):
    print(f"Node '{node_name}' State: {state}")

def extract_json_from_response(response_text: str) -> dict:
    cleaned_resp = response_text.strip()
    match = re.search(r'json\s*(\{.*?\})\s*', cleaned_resp, re.DOTALL)
    if match:
        cleaned_resp = match.group(1)
    try:
        return json.loads(cleaned_resp)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON decoding error: {e}. Response received: {response_text}")
    
def triage_evaluation(state:TriageState):
    messages = state['messages']


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
    {messages}

    ### Output:"""
    full_query = llm_70b.invoke(contextualize_q_system_prompt).content

    print(f"full_query: {full_query}")
    ensemble_retriever_triage = state['ensemble_retriever_triage']
    retrieved_docs = ensemble_retriever_triage.invoke(full_query)
    print(f"len_retrieved_docs: {len(retrieved_docs)}")
    retrieved_info = [doc.page_content for doc in retrieved_docs]
    full_retrieved_info = " ".join([message for message in retrieved_info[:2]])
    system_prompt = Template("""Sei un professionista altamente esperto in medicina d'urgenza, specializzato in Triage. Il tuo compito è valutare la gravità della situazione dell'utente fornendo un punteggio da 1 a 5 oppure chiedere una domanda concisa per ottenere ulteriori informazioni, se necessario.

    ### Istruzioni:
    1. **Valutazione della gravità**:
        - Analizza la situazione dell'utente per determinarne la gravità utilizzando le informazioni fornite nei documenti.
        - Il punteggio è definito come:
        - `1`: Situazione di minima gravità, nessun pericolo immediato.
        - `5`: Emergenza critica e potenzialmente letale, richiede intervento immediato.

    2. **Richiesta di ulteriori informazioni**:
        - Se le informazioni disponibili non sono sufficienti, chiedi una domanda diretta e specifica per ottenere chiarimenti.

    3. **Formato della risposta**:
    - Rispondi esclusivamente in italiano. La risposta deve essere un JSON con uno dei seguenti formati:
     - **Se hai abbastanza informazioni**:
       {
         "Reasoning": "Spiega brevemente la tua valutazione.",
         "Score": "Punteggio tra 1 e 5"
       }
     - **Se hai bisogno di ulteriori informazioni**:
       {
         "Reasoning": "Spiega perché sono necessarie più informazioni.",
         "Question": "Domanda diretta e specifica."
       }

    ### Esempi di output:
    #### Scenario 1:
    Hai abbastanza informazioni per valutare la gravità.
    Output: {"Reasoning": "Sulla base delle informazioni che ho, il taglio non mi sembra grave dunque la gravità della situazione è piuttosto bassa", "Score" : "2"} 

    #### Scenario 2:
    Hai bisogno di ulteriori informazioni.
    Output: {"Reasoning": "Non ho ancora abbastanza informazioni per determinare la gravità della situazione. Mi occorre effettuare un'altra domanda.", "Question" : "Hai mai avuto reazioni allergiche nella tua vita?"} 


    ### Documenti:
    {{full_retrieved_info}}

    ### Situazione medica dell'utente:
    {{full_query}}
    """)
    system_prompt = system_prompt.render(full_retrieved_info=full_retrieved_info, full_query=full_query)
    updated_prompt = [HumanMessage(system_prompt)]
    #print(f"updated_prompt: {updated_prompt}")
    start_time = time.time()
    response = llm_70b.invoke(updated_prompt).content
    end_time = time.time()
    print(f"Time taken for LLM invoke: {end_time - start_time:.2f} seconds\n")
    print(f"response: {response}")
    response = extract_json_from_response(response)
    # Analizza il tipo di risposta
    if 'Score' in response:
        return {"severity": response['Score'], 'full_query': full_query}  # Restituisce il numero, 'next_node' : 'end'
    else:
        return {"questions": response['Question']}  # , 'next_node' : 'new_question', Restituisce la domanda


def create_triage_agent():
    # Creazione del grafo
    graph = StateGraph(TriageState)

    # Nodo iniziale per avviare i flussi paralleli
    graph.add_node("start_emergency_bot", start_emergency_bot)
    graph.set_entry_point("start_emergency_bot")

    graph.add_node("triage_evaluation", triage_evaluation)
    graph.add_edge("start_emergency_bot", "triage_evaluation")
    #graph.add_edge("ask_user_question", "triage_evaluation")

    # graph.add_conditional_edges(
    #         "triage_evaluation",
    #         should_ask,
    #         {
    #             "new_question": "ask_user_question",
    #             "end": END,
    #         }
    #     )
    graph.set_finish_point("triage_evaluation")

    # Compilazione del grafo
    app = graph.compile() #checkpointer=memory

    # Store the image in memory using BytesIO
    img_bytes = app.get_graph().draw_mermaid_png()
    with open('presentation/agents/triage.png', 'wb') as f:
        f.write(img_bytes)

    return app