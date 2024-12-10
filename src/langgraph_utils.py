from langchain_groq import ChatGroq
from langgraph.graph import StateGraph
import pickle
from typing import TypedDict, Annotated, List
import geocoder
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_community.utilities import GoogleSerperAPIWrapper
import requests
from bs4 import BeautifulSoup
import re
import json
from dotenv import load_dotenv
load_dotenv()
import os
import requests
from geopy.geocoders import Nominatim
import streamlit as st
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema import Document


llm_70b = ChatGroq(model="llama-3.1-70b-versatile", api_key=st.secrets["GROQ"]["GROQ_API_KEY"])
llm_8b = ChatGroq(model="llama-3.1-8b-instant", api_key=st.secrets["GROQ"]["GROQ_API_KEY"])

import geocoder

def get_user_location():
    """
    Ottieni la posizione dell'utente tramite IP.
    
    Returns:
        tuple: Latitudine e longitudine dell'utente o None se non disponibile.
    """
    location = geocoder.ip('me')
    return location.latlng if location.latlng else (None, None)

def process_pdf(file_path):
    """
    Carica e processa un file PDF per estrarre il contenuto delle pagine desiderate.

    Args:
        file_path (str): Percorso al file PDF.

    Returns:
        str: Testo processato e unificato delle pagine selezionate.
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()[40:]
    full_text = "\n".join([doc.page_content for doc in pages])
    full_text = full_text.replace("MANUALE PER GLI INCARICATI DI PRIMO SOCCORSO", "")
    full_text = full_text.replace("LE POSIZIONI DI SICUREZZA", "")
    full_text = full_text.replace("APPARATO VISIVO", "")
    full_text = full_text.replace("APPARATO UDITIVO", "")
    full_text = full_text.replace("SISTEMA NERVOSO - anatomia", "")
    full_text = full_text.replace("IL SISTEMA NERVOSO\n", "")
    full_text = full_text.replace("-\n", "")
    # Espressione regolare per identificare titoli in maiuscolo (che terminano con \n)
    main_title_pattern = r'(?:\n|^)([A-Z\s\’\’]+(?:\n[A-Z\s\’\’]+)*)\n'
    sub_section_pattern = r'(?:^|\n)([a-z]\))'  # Per riconoscere sottosezioni come "a)" o "b)"
    degree_section_pattern = r'(?:^|\n)([IV]+\s+GRADO)'

    # Mappa dei numeri di pagina
    page_number_map = []
    for page in pages:
        page_number_map.append({"text": page.page_content, "page_number": page.metadata["page"]})

    # Trova tutti i titoli principali
    matches = list(re.finditer(main_title_pattern, full_text))
    documents = []
    current_content = ""
    current_title = None
    current_page = None

    for i in range(len(matches)):
        # Ottieni il titolo corrente
        title_start = matches[i].start()
        title_end = matches[i].end()
        title = full_text[title_start:title_end].strip()

        # Determina il contenuto fino al prossimo titolo principale o alla fine del testo
        if i + 1 < len(matches):
            content_start = title_end
            content_end = matches[i + 1].start()
            content = full_text[content_start:content_end].strip()
        else:
            content = full_text[title_end:].strip()

        # Trova il numero di pagina del titolo corrente
        if current_page is None:
            for page in page_number_map:
                if title in page["text"]:
                    current_page = page["page_number"]
                    break

        # Accorpa sottosezioni (es: "a)", "b)", "I GRADO", "II GRADO") al contenuto principale
        content_lines = content.split("\n")
        organized_content = []
        current_subsection = None

        for line in content_lines:
            # Riconosci sottosezioni come "a)", "b)"
            if re.match(sub_section_pattern, line):
                current_subsection = line
                organized_content.append(f"\n{line}")
            # Riconosci sezioni come "I GRADO", "II GRADO"
            elif re.match(degree_section_pattern, line):
                current_subsection = line
                organized_content.append(f"\n{line}")
            elif current_subsection:
                # Accorpa le righe successive alla sottosezione corrente
                organized_content[-1] += f" {line.strip()}"
            else:
                # Accorpa al contenuto principale
                organized_content.append(line.strip())

        content = "\n".join(organized_content)

        # Salva il documento precedente
        if current_title:
            documents.append({"title": current_title, "page_content": current_content, "page_nr": current_page})

        # Inizia un nuovo documento
        current_title = title
        current_content = content
        current_page = None  # Reset del numero di pagina

    # Salva l'ultimo documento
    if current_title:
        documents.append({"title": current_title, "page_content": current_content, "page_nr": current_page})

    # Converti in oggetti Document compatibili con LangChain
    documents = [
        Document(
            page_content=doc["page_content"],
            metadata={"title": doc["title"], "page_nr": doc["page_nr"]}
        )
        for doc in documents
    ]
    return documents

def create_bm25_retriever(documents, bm25_index_path="bm25_index.pkl"):
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
        print("Caricamento retriever BM25 esistente.")
        with open(bm25_index_path, "rb") as f:
            bm25_retriever = pickle.load(f)
    else:
        print("Creazione di un nuovo retriever BM25.")
        # Creazione del retriever BM25
        bm25_retriever = BM25Retriever.from_documents(documents)
        
        # Salva il retriever
        with open(bm25_index_path, "wb") as f:
            pickle.dump(bm25_retriever, f)
    
    return bm25_retriever

def create_retriever(documents, faiss_path="faiss_index"):
    # Step 1: Configura l'indice BM25 per i titoli
    bm25_retriever = create_bm25_retriever(documents)
    # Step 2: Configura FAISS per i contenuti
    embedding = OpenAIEmbeddings()
    if os.path.exists(faiss_path):
        vectorstore = FAISS.load_local(faiss_path, embeddings=embedding, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(documents, embedding=embedding)
        vectorstore.save_local(faiss_path)
    similarity_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})

    # Step 3: Configura un MultiRetriever
    ensemble_retriever = EnsembleRetriever(retrievers=[
        bm25_retriever,
        similarity_retriever
    ], weights=[0.3, 0.7])
    return ensemble_retriever

class AgentState(TypedDict):
    query: str
    history: List[dict]
    rag_answer : str
    ensemble_retriever : EnsembleRetriever

    keywords: str
    search_results: str
    video_title:str
    youtube_api_key : str
    retry_count_youtube: int

    google_maps_api_key : str
    google_maps_url: str
    user_location : List[str]
    hospital_name : str
    
    
    web_search_keywords : str
    retry_count_web_search : int
    web_answer : str

    final_result: List[str]

def answer_from_rag(state:AgentState):
    log_state("answer_from_rag", state)
    query = state['query']
    ensemble_retriever = state['ensemble_retriever']
    retrieved_docs = ensemble_retriever.invoke(query)
    retrieved_info = [doc.page_content for doc in retrieved_docs]
    prompt = f"""You are a highly experienced professional in emergency medicine with over 20 years of experience and certifications in Advanced Trauma Life Support (ATLS) and Prehospital Trauma Life Support (PHTLS). Your expertise lies in effectively managing life-threatening situations under pressure, with a calm demeanor and advanced skills that consistently save lives in critical moments.

    Your task is to provide immediate and accurate guidance for managing life-threatening emergencies, ensuring safety, critical intervention, and strict adherence to advanced emergency response protocols. 

    Your response must:
    - Include a clear, step-by-step guide for life-saving actions, tailored to the situation described in the query.
    - Reference the consulted materials by citing specific quotations or sections. Ensure that these citations are clear and directly support your guidance.
    - Emphasize the importance of contacting emergency services promptly and provide a brief explanation of when and why it is essential to do so.

    This is the user query: {query}
    This is the history of your conversation: {state['history']}
    These are the documents you should rely on for your response: {retrieved_info}
    
    If there are no specific information in the documents about the user query please return: 'NO INFO AVAILABLE'"""
    response = llm_70b.invoke([HumanMessage(content=prompt)])
    return {"rag_answer" : response.content.strip()}


def log_state(node_name, state:AgentState):
    print(f"Node '{node_name}' State: {state}")

def web_search(state: AgentState) -> str:
    """
    Searches the Internet to retrieve reliable and certified information related to a specific medical query.

    Args:
        query (str): A simplified Italian string, optimized for an effective Google search based on the user's query.

    Returns:
        str: A string containing useful and relevant information retrieved from certified websites related to the user's query. 
             If no pertinent information is found, it returns a message indicating the absence of results.
    """
    # Fase 1: Ricerca su Internet
    log_state("web_search", state)
    query = state['keywords']
    if not isinstance(query, str):
        return "Nessun contenuto pertinente trovato su Internet"
    compliant_links = ['my-personaltrainer', 'msdmanuals']
    serper = GoogleSerperAPIWrapper(api_key=os.environ["SERPER_API_KEY"])
    try:
        search_results = serper.results(query)['organic']
        # Filtra e seleziona un link per ciascun dominio compliant
        selected_links = []
        for domain in compliant_links:
            for result in search_results:
                if domain in result['link']:
                    selected_links.append(result['link'])
                    break  # Esci dal ciclo per passare al prossimo dominio

        general_content = []
        selected_links = [selected_links[0]]
        for url in selected_links:
            try:
                # Effettua una richiesta al sito
                response = requests.get(url)
                response.raise_for_status()  # Controlla se la richiesta è andata a buon fine
                
                # Analizza il contenuto della pagina con BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Estrai il contenuto principale della pagina (potresti dover adattare il selettore)
                page_content = soup.get_text(separator=' ', strip=True)
                
                general_content.append(page_content)

            except requests.exceptions.RequestException as e:
                print(f"NO Info")
        return {"web_answer" : general_content}
    except:
        return {"web_answer" : "NO Info"}
    
def extract_keywords_web_search(state:AgentState):
   log_state("extract_keywords_web_search", state)
   query = state['query']
   previous_keywords = state.get('web_search_keywords', '')
    # Costruisci il prompt
   prompt = f"""You are a highly skilled virtual assistant with expertise in first aid. Your task is to extract the most relevant medical keywords from the following user query: '{query}' and the history of your conversation:  {state['history']}.
   These keywords will help optimize searches for first aid guidance on various websites. Follow these rules strictly:

    1. **Focus on medical relevance:** Extract only the essential details about the medical issue or injury described in the query. For example:
    - Type of injury or symptom (e.g., "cut," "burn," "panic attack").
    - Cause of the issue, if specified (e.g., "knife," "hot water," "bee sting").
    2. **Omit redundant or irrelevant details:** Ignore unnecessary context, such as who the injury happened to or extraneous background information.
    3. **Translate into Italian:** Ensure the extracted keywords are translated into Italian, regardless of the query's original language.
    4. **Output format:** Return the result strictly as a JSON object with the key 'keywords' containing the extracted keywords.""" + \
    """
    1. Query: "I am feeling anxious, I think I am having a panic attack. What should I do?" 
       Output : {"keywords": "attacco di panico, primo soccorso"}
    2. Query: "Cosa devo fare se mi punge un'ape?"
       Output : {"keywords": "Puntura ape, primo soccorso"}
    3. Query: "Come medicare un taglio profondo fatto con un coltello?"
       Output : {"keywords": "Taglio profondo coltello, primo soccorso"}
    4. Query: "Come trattare una scottatura con acqua bollente?"
       Output : {"keywords": "Scottatura acqua bollente, primo soccorso"}
    5. Query: "Cosa fare in caso di reazione allergica improvvisa?"
       Output : {"keywords": "Reazione allergica, primo soccorso"}
    6. Query: "Un mio amico sta avendo un attacco di panico"
       Output : {"keywords": "Attacco di panico, primo soccorso"} """

   if previous_keywords:
        prompt += f" Previous search with keywords '{previous_keywords}' returned no results. Try a different search query."
        
   # Chiamata al modello LLM
   response = llm_70b.invoke([HumanMessage(content=prompt)])
   return {"web_search_keywords": json.loads(response.content)["keywords"], "retry_count_web_search" : state["retry_count_web_search"]+1}



# Funzione per controllare se continuare
def should_continue_web_search(state:AgentState):
    web_search_results = state.get('web_answer', '')
    log_state("should_continue_web_search", state)
    print(state['retry_count_web_search'])
    retry_count_web_search = state.get('retry_count_web_search', 0)
    if (not web_search_results or web_search_results == "NO Info") and retry_count_web_search <2:
        # Incrementa il contatore dei retry
        return "retry"
    return "end"


# Funzione per controllare se continuare
def should_web_search(state:AgentState):
    rag_answer = state.get('rag_answer', '')
    if not rag_answer or "no info available" in rag_answer.lower():
        return "web_search"
    return "end"


def extract_keywords_youtube(state:AgentState):
   log_state("extract_keywords_youtube", state)
   query = state['query']
   previous_keywords = state.get('keywords', '')
    # Costruisci il prompt
   prompt = f"""From the following user query: '{query}' and the history of your conversation: '{state['history']}', extract the most relevant keywords to optimize the search for a video on YouTube. Translate them into English.
    Return just a Json object with the key: 'keywords'
    Here are examples of user queries and the corresponding optimized output:""" + \
   """
    1. Query: "I am feeling anxious, I think I am having a panic attack. What should I do?" 
       Output : {"keywords": "panic attack, first aid"}
    2. Query: "Cosa devo fare se mi punge un'ape?"
       Output : {"keywords": "bee sting treatment, first aid"}
    3. Query: "Come medicare un taglio profondo fatto con un coltello?"
       Output : {"keywords": "knife deep cut treatment, first aid"}
    4. Query: "Come trattare una scottatura con acqua bollente?"
       Output : {"keywords": "boiling water burn, first aid"}
    5. Query: "Cosa fare in caso di reazione allergica improvvisa?"
       Output : {"keywords": "allergic reaction help, first aid"} 
    6. Query: "Un mio amico sta avendo un attacco di panico"
       Output : {"keywords": "panic attack, first aid"} 
       """

   if previous_keywords:
        prompt += f" Previous search with keywords '{previous_keywords}' returned no results. Try a different search query."
   
    # Chiamata al modello LLM
   response = llm_70b.invoke([HumanMessage(content=prompt)])
   print(response.content)
   return {"keywords": json.loads(response.content)["keywords"], "retry_count_youtube" : state["retry_count_youtube"]+1}



# Funzione per controllare se continuare
def should_continue_youtube(state:AgentState):
    search_results = state.get('search_results', '')
    retry_count_youtube = state.get('retry_count_youtube', 0)
    if (not search_results or "No videos found" in search_results) and retry_count_youtube <2:
        return "retry"
    return "end"


def search_youtube_videos(state:AgentState) -> str:
    """
    Cerca video su YouTube da una lista certificata di canali affidabili.

    Args:
        query (str): Una versione semplificata e in inglese, adatta per una ricerca su youtube, della query di ricerca fornita dall'utente.

    Returns:
        str: Un di link utile rispetto alla query, o un messaggio che indica che non sono stati trovati video.
    """
    log_state("search_youtube_videos", state)
    keywords = state['keywords']
    if not isinstance(keywords, str):
        return "Nessun video pertinente trovato per la query specificata nei canali consentiti."
    YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
    allowed_channels=['UCwywRelPfy7U8jAI312J_Xw', #First Aid,
                      'UCQK834Q3xqlo85LJqrEd7fw' #ChatterDocs
                      ]  #'UCTVZkcCKSqFD0TTJ8BjYLDQ' Croce Rossa, 
    max_results = 3
    prompt = """
    You are tasked with determining whether a YouTube video is relevant to a user's query. The query typically describes a **medical problem involving a person**, unless explicitly stated otherwise. Analyze the query and the video title, and decide if the video could be useful. Respond strictly with "YES" or "NO". Do not provide any explanations or additional information.

    ### Guidelines:
    1. Assume the query pertains to a medical issue affecting a person unless otherwise specified.
    2. Focus only on the **relevance** of the video to the user's query.
    3. Base your decision solely on the content of the query and the video title.
    4. Return only "YES" or "NO". No explanations.

    ### Format:
    - Input: User query and video title.
    - Output: "YES" or "NO".

    ### Examples:
    1. Input: Query: "Mi ha punto un'ape", Video title: "Allergic Reactions in Dogs".
    Output: "NO"

    2. Input: Query: "How to treat a bee sting?", Video title: "First Aid for Bee Stings".
    Output: "YES"

    3. Input: Query: "How to help a dog with an allergic reaction?", Video title: "Allergic Reactions in Dogs".
    Output: "YES"

    4. Input: Query: "How to treat a deep knife cut?", Video title: "Emergency Care for Deep Cuts".
    Output: "YES"

    ### Now process the following input:
    Query: {query}
    Video title: {video_title}
"""
    try:
        for channel_id in allowed_channels:
            params = {
                "part": "snippet",
                "q": keywords,
                "channelId": channel_id,
                "maxResults": max_results,
                "type": "video",
                "key": state['youtube_api_key'],
            }

            response = requests.get(YOUTUBE_SEARCH_URL, params=params)
            data = response.json()

            # Controlla se ci sono risultati
            if "items" in data and len(data["items"]) > 0:
                for item in data["items"]:
                    video_id = item["id"]["videoId"]
                    video_title = item["snippet"]["title"]
                
                    response = llm_70b.invoke([HumanMessage(content=prompt.format(query=state['query'], video_title=video_title))]).content
                    if response.strip().lower() == 'yes':
                        return {"search_results": f"https://www.youtube.com/watch?v={video_id}",
                                "video_title": video_title}
    except requests.exceptions.RequestException as e:
        return {"search_results": f"Error during YouTube search: {str(e)}", "video_title": None}
    return {"search_results": "No relevant videos found for the given query on the allowed channels.", "video_title": None}

def get_google_maps_url(state:AgentState):
    """
    Trova l'ospedale più vicino utilizzando la Google Places API.

    Args:
        lat (float): Latitudine dell'utente.
        lng (float): Longitudine dell'utente.
        api_key (str): Google Maps API Key.

    Returns:
        dict: Informazioni sull'ospedale più vicino o un messaggio di errore.
    """
    # URL dell'API di Google Places
    places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    lat, lng = state['user_location']
    google_maps_api_key = state['google_maps_api_key']
    # Parametri della richiesta
    params = {
        "location": f"{lat},{lng}",  # Latitudine e longitudine
        "radius": 7000,             # Raggio di ricerca in metri (es. 7km)
        "type": "hospital",         # Tipo di luogo da cercare
        "key": google_maps_api_key,             # Google API Key
    }

    try:
        # Invia la richiesta
        response = requests.get(places_url, params=params)
        data = response.json()

        # Controlla se ci sono risultati
        if "results" in data and len(data["results"]) > 0:
            nearest_hospital = data["results"][0]  # Il primo risultato è il più vicino
            print(nearest_hospital)
            hospital_name = nearest_hospital["name"]
            address = nearest_hospital.get("vicinity")
            location = nearest_hospital["geometry"]["location"]
            print(address)
            return {
                "hospital_name": hospital_name,
                "google_maps_url": f"https://www.google.com/maps?q={location['lat']},{location['lng']}",
            }
        else:
            return {"google_maps_url": "No hospitals found nearby."}
    except requests.exceptions.RequestException as e:
        return {"google_maps_url": f"Request failed: {str(e)}"}
    
def start_emergency_bot(state:AgentState):
    # Nodo di coordinamento iniziale, ritorna lo stato invariato
    return state


def combine_results(state:AgentState):
    video_result = state.get("search_results", "No video found.")
    video_title = state.get("video_title", "No video found.")
    google_maps_url = state.get("google_maps_url", "")
    hospital_name = state.get("google_maps_url", "No hospital information found.")
    if state.get("web_answer", ""):
        doc_answer = state["web_answer"]
    else:
        doc_answer = state.get("rag_answer", "")
    
    return {"final_result": [doc_answer, google_maps_url, hospital_name, video_result, video_title]}

def create_langgraph_agent():
    # Creazione del grafo
    graph = StateGraph(AgentState)

    # Nodo iniziale per avviare i flussi paralleli
    graph.add_node("start_emergency_bot", start_emergency_bot)

    # Setta "start_emergency_bot" come entry point
    graph.set_entry_point("start_emergency_bot")

    # Aggiunta dei nodi
    graph.add_node("extract_keywords_youtube", extract_keywords_youtube)
    graph.add_node("search_youtube_videos", search_youtube_videos)
    graph.add_node("answer_from_rag", answer_from_rag)
    graph.add_node("web_search", web_search)


    graph.add_edge("extract_keywords_youtube", "search_youtube_videos")
    graph.add_conditional_edges(
        "search_youtube_videos",
        should_continue_youtube,
        {
            "retry": "extract_keywords_youtube",
            "end": "combine_results",
        }
    )

    # Secondo agente (Location)
    graph.add_node("get_google_maps_url", get_google_maps_url)



    # Terzo agente (Combinazione risultati)
    graph.add_node("combine_results", combine_results)

    # Integrazione flussi paralleli
    # graph.add_edge("search_youtube_videos", "combine_results")
    graph.add_edge("get_google_maps_url", "combine_results")
    graph.add_conditional_edges(
        "answer_from_rag",
        should_web_search,
        {
            "web_search": "extract_keywords_web_search",
            "end": "combine_results",
        }
    )

    graph.add_node("extract_keywords_web_search", extract_keywords_web_search)
    graph.add_edge("extract_keywords_web_search", "web_search")
    graph.add_conditional_edges(
        "web_search",
        should_continue_web_search,
        {
            "retry": "extract_keywords_web_search",
            "end": "combine_results",
        }
    )
    # Collegamenti ai flussi paralleli
    graph.add_edge("start_emergency_bot", "extract_keywords_youtube")
    graph.add_edge("start_emergency_bot", "get_google_maps_url")
    graph.add_edge("start_emergency_bot", "answer_from_rag")

    graph.set_finish_point("combine_results")

    # Compilazione del grafo
    app = graph.compile()
    return app