import yaml
import requests
from crewai import LLM, Task, Agent, Process, Crew
from crewai_tools import tool


# Funzione per inizializzare il LLM
def init_LLM(model_name, api_key):
    llm = LLM(model=model_name, api_key=api_key)
    return llm


# Funzione per la ricerca su YouTube tramite API
@tool
def youtube_channel_search_tool(query: str, max_results: int = 1, YOUTUBE_API_KEY=None) -> str:
    YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
    allowed_channels = ['UCwywRelPfy7U8jAI312J_Xw']  # Canali consentiti
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

        if "items" in data and len(data["items"]) > 0:
            video_id = data["items"][0]["id"]["videoId"]
            return f"https://www.youtube.com/watch?v={video_id}"

    return "Nessun video pertinente trovato per la query specificata nei canali consentiti."


# Caricamento delle configurazioni da file YAML
def load_configs(config_location):
    with open(config_location, 'r') as file:
        return yaml.safe_load(file)


# Inizializzazione dell'agente Crew
def init_crew(config_location, model_name, GROQ_API_KEY, SERPER_API_KEY, YOUTUBE_API_KEY):
    # Carica le configurazioni (agents e tasks)
    configs = load_configs(config_location)
    agents_config = configs['agents']
    tasks_config = configs['tasks']

    # Inizializza LLM
    llm = init_LLM(model_name, GROQ_API_KEY)

    # Inizializza gli agenti e i task
    first_aid_chatbot = Agent(config=agents_config['first_aid_chatbot'], llm=llm)
    first_aid_chatbot.tools = [lambda query, max_results=1: youtube_channel_search_tool(query, max_results, YOUTUBE_API_KEY)]

    first_aid_response_task = Task(config=tasks_config['first_aid_response_task'], agent=first_aid_chatbot)
    first_aid_response_task.tools = [lambda query, max_results=1: youtube_channel_search_tool(query, max_results, YOUTUBE_API_KEY)]

    # Crea il processo Crew
    crew = Crew(
        agents=[first_aid_chatbot],
        tasks=[first_aid_response_task],
        process=Process.sequential,
    )

    return crew