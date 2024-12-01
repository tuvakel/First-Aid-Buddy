import yaml
from typing import List
import requests
from crewai import LLM, Task, Agent, Process, Crew
from crewai_tools import tool
import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage
from langchain_community.utilities import GoogleSerperAPIWrapper
import requests
from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim


llm = LLM(model="groq/llama-3.1-70b-versatile", api_key=st.secrets["GROQ"]["GROQ_API_KEY"])
YOUTUBE_API_KEY = st.secrets["YOUTUBE"]["YOUTUBE_API_KEY"]
SERPER_API_KEY = st.secrets["SERPER"]["SERPER_API_KEY"]

@tool
def youtube_channel_search_tool(query: str) -> List[str]:
    """
    Cerca video su YouTube solo da una lista specifica di canali.

    Args:
        query (str): Una versione semplificata e in inglese, adatta per una ricerca su youtube, della query di ricerca fornita dall'utente.

    Returns:
        dict: un dizionario contenenti il titolo e il link del video, o un messaggio che indica che non sono stati trovati video.
    """
    YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
    allowed_channels=['UCwywRelPfy7U8jAI312J_Xw', 'UCTVZkcCKSqFD0TTJ8BjYLDQ', 'UCQK834Q3xqlo85LJqrEd7fw'] #First Aid, Croce Rossa, ChatterDocs
    max_results = 1
    for channel_id in allowed_channels:
        params = {
            "part": "snippet",
            "q": query,
            "channelId": channel_id,
            "maxResults": max_results,
            "type": "video",
            "key": YOUTUBE_API_KEY,
        }

        response = requests.get(YOUTUBE_SEARCH_URL, params=params)
        data = response.json()

        # Controlla se ci sono risultati
        if "items" in data and len(data["items"]) > 0:
            # Restituisci il link al primo video trovato
            video_id = data["items"][0]["id"]["videoId"]
            title = data["items"][0]["snippet"]["title"]
            return {"link": f"https://www.youtube.com/watch?v={video_id}", "title" : title}
    # Se nessun video è trovato
    return "Nessun video pertinente trovato per la query specificata nei canali consentiti."

@tool
def get_google_maps_url_tool(hospital_name: str) -> str:
    """
    Returns a Google Maps URL for the hospital name.
    
    Args:
        hospital_name (str): A string indicating the name of the hospital to search for.
        
    Returns:
        str: A Google Maps URL for the location, or a message indicating the location was not found.
    """
    if not isinstance(hospital_name, str):
        return "No information available from these sources"
    geolocator = Nominatim(user_agent="MyExampleLLAMAFirstAid")
    location = geolocator.geocode(hospital_name)
    if location:
        return f"https://www.google.com/maps?q={location.latitude},{location.longitude}"
    else:
        return f"Location '{hospital_name}' could not be found."
    
@tool
def triage_pdf_search_tool(query: str) -> str:
    """
    Retrieves reliable and relevant information from certified PDF sources related to triage and risk assessment evaluations.

    Args:
        query (str): A simplified Italian string optimized for an effective RAG (Retrieval-Augmented Generation) search within a pre-built embedding database.

    Returns:
        str: A string containing the most relevant content retrieved from the sources, or a message indicating no relevant information was found.
    """
    if not isinstance(query, str):
        return "No information available from these sources"
    index_path = "../data/doc_triage/vector_db"
    try:
        # Step 1: Load the index
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context)

        # Step 2: Query the index
        query_engine = index.as_query_engine()
        response = query_engine.query(query)

        # Step 3: Return the result
        return response.response

    except Exception as e:
        return "No information available from these sources"
    
@tool
def emergency_pdf_search_tool(query: str) -> str:
    """
    Retrieves reliable and relevant information from certified PDF sources related to medical emergency situations.

    Args:
        query (str): A simplified Italian string optimized for an effective RAG (Retrieval-Augmented Generation) search within a pre-built embedding database.

    Returns:
        str: A string containing the most relevant content retrieved from the sources, or a message indicating no relevant information was found.
    """
    if not isinstance(query, str):
        return "No information available from these sources"
    index_path = "../data/doc_emergency/vector_db"
    try:
        # Step 1: Load the index
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context)

        # Step 2: Query the index
        query_engine = index.as_query_engine()
        response = query_engine.query(query)

        # Step 3: Return the result
        return response.response

    except Exception as e:
        return "No information available from these sources"
    
@tool
def everyday_pdf_search_tool(query: str) -> str:
    """
    Retrieves reliable and relevant information from certified PDF sources related to non-urgent medical situations.

    Args:
        query (str): A simplified Italian string optimized for an effective RAG (Retrieval-Augmented Generation) search within a pre-built embedding database.

    Returns:
        str: A string containing the most relevant content retrieved from the sources, or a message indicating no relevant information was found.
    """
    if not isinstance(query, str):
        return "No information available from these sources"
    index_path = "../data/doc_everyday/vector_db"
    try:
        # Step 1: Load the index
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context)

        # Step 2: Query the index
        query_engine = index.as_query_engine()
        response = query_engine.query(query)

        # Step 3: Return the result
        return str(response.response)

    except Exception as e:
        return "No information available from these sources"
    

@tool
def internet_search_tool(query: str) -> str:
    """
    Searches the Internet to retrieve reliable and certified information related to a specific medical query.

    Args:
        query (str): A simplified Italian string, optimized for an effective Google search based on the user's query.

    Returns:
        str: A string containing useful and relevant information retrieved from certified websites related to the user's query. 
             If no pertinent information is found, it returns a message indicating the absence of results.
    """
    # Fase 1: Ricerca su Internet
    
    if not isinstance(query, str):
        return "Nessun contenuto pertinente trovato su Internet"
    compliant_links = ['my-personaltrainer', 'msdmanuals']
    serper = GoogleSerperAPIWrapper(api_key=SERPER_API_KEY)
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
                print(f"Errore durante il recupero del contenuto per l'URL {url}: {e}")
        return general_content
    except:
        return "Nessun contenuto pertinente trovato su Internet"
    
# Inizializzazione dell'agente Crew
def init_crew():
    config_location = {'agents': '../agents/simple/agents.yaml',
                       'tasks': '../agents/simple/tasks.yaml'
                       }
    # Load configurations from YAML files
    configs = {}
    for config_type, file_path in config_location.items():
        with open(file_path, 'r') as file:
            configs[config_type] = yaml.safe_load(file)

    # Assign loaded configurations to specific variables
    agents_config = configs['agents']
    tasks_config = configs['tasks']

    first_aid_chatbot = Agent(config=agents_config['first_aid_chatbot'], llm=llm)
    first_aid_chatbot.tools=[youtube_channel_search_tool, get_google_maps_url_tool]

    first_aid_response_task = Task(config=tasks_config['first_aid_response_task'], agent=first_aid_chatbot)
    first_aid_response_task.tools=[youtube_channel_search_tool, get_google_maps_url_tool]

    crew = Crew(
        agents=[first_aid_chatbot],
        tasks=[first_aid_response_task],
        process=Process.sequential,
        verbose=False
    )
    return crew