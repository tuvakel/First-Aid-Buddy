import yaml
from typing import List
import requests
from crewai import LLM, Task, Agent, Process, Crew
from crewai_tools import tool
import streamlit as st


llm = LLM(model="groq/llama-3.1-70b-versatile", api_key=st.secrets["GROQ"]["GROQ_API_KEY"])
YOUTUBE_API_KEY = st.secrets["YOUTUBE"]["YOUTUBE_API_KEY"]

@tool
def youtube_channel_search_tool(query: str) -> List[str]:
    """
    Cerca video su YouTube solo da una lista specifica di canali.

    Args:
        query (str): Una versione semplificata e in inglese, adatta per una ricerca su youtube, della query di ricerca fornita dall'utente.

    Returns:
        List[str]: Una lista di link utili rispetto alla domanda, o un messaggio che indica che non sono stati trovati video.
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
            return f"https://www.youtube.com/watch?v={video_id}"
    # Se nessun video è trovato
    return "Nessun video pertinente trovato per la query specificata nei canali consentiti."


# Inizializzazione dell'agente Crew
def init_crew():
    config_location = {'agents': '../config/agents.yaml',
                       'tasks': '../config/tasks.yaml'
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
    first_aid_chatbot.tools=[youtube_channel_search_tool]

    first_aid_response_task = Task(config=tasks_config['first_aid_response_task'], agent=first_aid_chatbot)
    first_aid_response_task.tools=[youtube_channel_search_tool]

    crew = Crew(
        agents=[first_aid_chatbot],
        tasks=[first_aid_response_task],
        process=Process.sequential,
        verbose=False
    )
    return crew