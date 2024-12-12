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
#from langgraph.checkpoint.memory import MemorySaver
#memory = MemorySaver()
import pickle

llm_70b = ChatGroq(model="llama-3.1-70b-versatile", api_key=st.secrets["GROQ"]["GROQ_API_KEY"])
llm_8b = ChatGroq(model="llama-3.1-8b-instant", api_key=st.secrets["GROQ"]["GROQ_API_KEY"])

def process_pages(pages):
    import re
    for doc in pages:
        doc.page_content = doc.page_content.replace(
            "Manuale regionale Triage intra-ospedaliero modello Lazio a cinque codici \n \n", ""
        )
        doc.page_content = re.sub(r'^\d+\s*\n\s*\n', '', doc.page_content)
    return pages

def process_pdf(file_path):
    # Carica le pagine del PDF
    loader = PyPDFLoader(file_path)
    pages = loader.load()[11:]

    # Dividi le pagine in sottogruppi per ogni core
    num_cores = os.cpu_count()  # Numero di core disponibili
    print(f"num_cores: {num_cores}")
    chunk_size = len(pages) // num_cores + (len(pages) % num_cores > 0)
    print(f"chunk_size: {chunk_size}")
    chunks = [pages[i:i + chunk_size] for i in range(0, len(pages), chunk_size)]

    # Parallelizza il lavoro con ProcessPoolExecutor
    with ThreadPoolExecutor() as executor:
        processed_chunks = list(executor.map(process_pages, chunks))

    # Combina i risultati
    documents = [page for chunk in processed_chunks for page in chunk]
    return documents


def create_bm25_retriever(pdf_file_path, bm25_index_path="bm25_triage_index.pkl"):
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
            bm25_retriever = pickle.load(f)
            documents = []
    else:
        #print("Creazione di un nuovo retriever BM25.")
        # Creazione del retriever BM25
        documents = process_pdf(pdf_file_path)
        bm25_retriever = BM25Retriever.from_documents(documents)
        
        # Salva il retriever
        with open(bm25_index_path, "wb") as f:
            pickle.dump(bm25_retriever, f)
    
    return bm25_retriever, documents

def create_retriever(pdf_file_path, faiss_path="faiss_triage_index"):
    # Step 1: Configura l'indice BM25 per i titoli
    bm25_retriever, documents = create_bm25_retriever(pdf_file_path)
    # Step 2: Configura FAISS per i contenuti
    embedding = OpenAIEmbeddings(api_key=st.secrets["OPENAI"]["OPENAI_API_KEY"])
    if os.path.exists(faiss_path):
        vectorstore = FAISS.load_local(faiss_path, embeddings=embedding, allow_dangerous_deserialization=True)
    else:
        if documents:
            vectorstore = FAISS.from_documents(documents, embedding=embedding)
            vectorstore.save_local(faiss_path)
        else:
            documents = process_pdf(pdf_file_path)
            vectorstore = FAISS.from_documents(documents, embedding=embedding)
            vectorstore.save_local(faiss_path)
    similarity_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2})

    # Step 3: Configura un MultiRetriever
    ensemble_retriever = EnsembleRetriever(retrievers=[
        bm25_retriever,
        similarity_retriever
    ], weights=[0.3, 0.7])
    return ensemble_retriever


class TriageState(TypedDict):
    ensemble_retriever : EnsembleRetriever
    severity : int
    questions : Annotated[list, add_messages]
    next_node : str
    messages: Annotated[list, add_messages]

def start_emergency_bot(state:TriageState):
    # Nodo di coordinamento iniziale, ritorna lo stato invariato
    return state

def log_state(node_name, state:TriageState):
    print(f"Node '{node_name}' State: {state}")


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
    ensemble_retriever = state['ensemble_retriever']
    retrieved_docs = ensemble_retriever.invoke(full_query)
    print(f"len_retrieved_docs: {len(retrieved_docs)}")
    retrieved_info = [doc.page_content for doc in retrieved_docs]
    full_retrieved_info = " ".join([message for message in retrieved_info])
    system_prompt = f""" Sei un professionista altamente esperto in medicina d'urgenza, specializzato in Triage. Il tuo compito è valutare la gravità della situazione dell'utente fornendo un punteggio da 1 a 5 oppure chiedere una domanda concisa per ottenere ulteriori informazioni, se necessario.

    ### Istruzioni:
    1. **Valutazione della gravità**:
        - Usa le informazioni fornite nei documenti e nella conversazione per determinare la gravità della situazione dell'utente.
        - Il punteggio è definito come:
        - `1`: Situazione di minima gravità, nessun pericolo immediato.
        - `5`: Emergenza critica e potenzialmente letale, richiede intervento immediato.

    2. **Richiesta di ulteriori informazioni**:
        - Se le informazioni disponibili non sono sufficienti, chiedi una domanda diretta e specifica per ottenere chiarimenti.

    ### Esempi di output:
    #### Scenario 1:
    Hai abbastanza informazioni per valutare la gravità.
    **Output**: 3

    #### Scenario 2:
    Hai bisogno di ulteriori informazioni.
    **Output**: Hai mai avuto reazioni allergiche nella tua vita?

    ### Contesto:
    Documenti forniti: {full_retrieved_info}

    ### Nota:
    - Rispondi esclusivamente in italiano.
    - Evita risposte prolisse; usa solo un punteggio o una domanda diretta.
    """
    updated_prompt = [SystemMessage(content=system_prompt), HumanMessage(full_query)]
    print(f"updated_prompt: {updated_prompt}")
    response = llm_70b.invoke(updated_prompt).content
    print(f"response: {response}")
    # Analizza il tipo di risposta
    if response.isdigit() and int(response) in range(1, 6):
        return {"severity": int(response), 'next_node' : 'end'}  # Restituisce il numero
    else:
        return {"questions": response, 'next_node' : 'new_question'}  # Restituisce la domanda
    

def should_ask(state: TriageState):
    #log_state("route_next_node", state)
    return state['next_node']  # Restituisce il nome del prossimo nodo


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
    return app