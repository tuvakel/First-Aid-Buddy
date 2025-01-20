import streamlit as st
from src.utils import *
import tempfile
import hashlib
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from src.triage_utils import create_triage_retriever, create_triage_agent, severity_to_color
from src.emergency_utils import create_emergency_retriever, create_emergency_agent
from streamlit_js_eval import get_geolocation
import time


st.set_page_config(page_title="llama-first-aid", page_icon="presentation/logo/logo.png", layout="wide", initial_sidebar_state="expanded")

app_version = generate_app_id(
    github_repo="Amatofrancesco99/llama-first-aid",
    last_commit_file="data/app_version/last_commit.txt",
    version_file="data/app_version/version.txt"
)
print(f"App version: {app_version}")


# Hash session ID using hashlib
if 'session_id' not in st.session_state:
    session_id = hashlib.sha256(str(datetime.now()).encode()).hexdigest()
    st.session_state.session_id = session_id
else:
    session_id = st.session_state.session_id


if st.sidebar.checkbox("Use my current location", value=False):
    with st.spinner("Searching for location..."):
        user_location_info = None
        while user_location_info is None:
            user_location_info = get_geolocation()
            if user_location_info is None:
                time.sleep(0.5)
    user_location_info = user_location_info.get('coords', None)
    user_location = (user_location_info['latitude'], user_location_info['longitude'])
else:
    user_location = (None, None)

language, detailed_location = get_language(user_location)

# Initialize the LLM with the Google API key from secrets
llm = init_LLM(API_KEY=st.secrets["GROQ"]["GROQ_API_KEY"])
YOUTUBE_API_KEY = st.secrets["YOUTUBE"]["YOUTUBE_API_KEY"]
GOOGLE_MAPS_API_KEY = st.secrets["GOOGLE_MAPS"]["GOOGLE_MAPS_API_KEY"]
llm_text_model_name = "llama3-70b-8192"
llm_audio_model_name = "whisper-large-v3"
file_path_triage = "data/doc_triage/pdf/Manuale-Triage.pdf"
file_path_emergency = "data/doc_emergency/pdf/manuale_primo_soccorso.pdf"
prompt_emergency_file_path = "src/templates/emergency_prompt.jinja"
prompt_everyday_file_path = "src/templates/everyday_prompt.jinja"
prompt_emergency = load_template(prompt_emergency_file_path)
prompt_everyday = load_template(prompt_everyday_file_path)
ensemble_retriever_emergency = None
ensemble_retriever_triage = None
triage_agent=None

# Funzione per creare il retriever
@st.cache_resource
def load_triage_retriever(file_path, bm25_path, faiss_path):
    return create_triage_retriever(file_path, bm25_path, faiss_path)

# Funzione per creare il retriever
@st.cache_resource
def load_emergency_retriever(file_path, bm25_path, faiss_path):
    return create_emergency_retriever(file_path, bm25_path, faiss_path)

# Funzione per creare l'triage_agente
@st.cache_resource
def load_triage_agent():
    return create_triage_agent()

@st.cache_resource
def load_emergency_agent():
    return create_emergency_agent()


ensemble_retriever_triage = load_triage_retriever(file_path_triage, bm25_path="data/bm_25/bm25_triage_index.pkl", faiss_path="data/faiss/faiss_triage_index")
ensemble_retriever_emergency = load_emergency_retriever(file_path_emergency, bm25_path="data/bm_25/bm25_emergency_index.pkl", faiss_path="data/faiss/faiss_emergency_index")
triage_agent = load_triage_agent()
emergency_agent = load_emergency_agent()

# GCS client to store session data
gcs_client = initialize_gcs_client(SERVICE_ACCOUNT_KEY=st.secrets["GCP"]["SERVICE_ACCOUNT_KEY"])


# Main function
def main():
    st.sidebar.markdown(f"**Location details:** {detailed_location}" if language != "it" else f"**Dettagli posizione:** {detailed_location}")
    
    get_sidebar(language)

    st.title("LLAMA FIRST AID")

    # User query input
    query = ""
    image_base64 = ""
    audio_value = ""

    query = st.chat_input("Describe your issue or emergency" if language != "it" 
                         else "Descrivi il problema o la situazione di emergenza")
    audio_value = st.audio_input("Speak with your assistant (optional)" if language != "it" else "Parla col tuo assistente (opzionale)")

    #if allow_images:
    #    captured_image = st.file_uploader("Carica un'immagine (opzionale)", type=["jpg", "jpeg", "png"])
    #    if captured_image:
    #        image_base64 = convert_image_to_base64(captured_image, resize=50)

    if (query or (query and image_base64)) or (audio_value or (audio_value and image_base64)):
        trscb_message_template = load_template("src/templates/trscb_message_template.jinja")
        trscb_message = trscb_message_template.render()
        
        if audio_value:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                temp_audio_path = save_uploaded_audio(audio_value.getvalue(), temp_audio_file.name)
            query = transcribe_audio(llm, llm_audio_model_name, temp_audio_path, trscb_message, language)
            
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [HumanMessage(content=query)]
    
            if image_base64:
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": f"data:image/jpeg;base64,{image_base64}"
                })
        else:
            st.session_state.chat_history.append(HumanMessage(content=query))

        # Mostra la cronologia della conversazione
        for message in st.session_state.chat_history:
            if not isinstance(message, SystemMessage):
                if isinstance(message, HumanMessage):
                    role = "user"
                elif isinstance(message, AIMessage):
                    role = "assistant"
                with st.chat_message(role):
                    st.markdown(message.content)
            
        with st.spinner("Assessing emergency severity" if language != "it" else "Sto pensando per capire la gravità della situazione..."):
            # Call the LLM with the Jinja prompt and DataFrame context
            with st.chat_message("assistant"):
                input = {
                    "messages": st.session_state.chat_history,
                    "ensemble_retriever_triage": ensemble_retriever_triage,
                    "questions" : []
                }
                start_time = time.time()
                output = triage_agent.invoke(input)
                end_time = time.time()
                severity = output.get('severity', None)
                severity = int(severity) if severity is not None else None
                if severity:
                    color = severity_to_color[severity]
                    st.markdown(
                        (f"<span style='font-size: 16px;'>Emergency has <strong>severity {severity}</strong></span>" if language != "it"
                        else f"<span style='font-size: 16px;'>Al tuo codice è stato affidato <strong>codice {severity}</strong></span>"),
                        # f"<div style='display: inline-block; width: 20px; height: 20px; background-color: {color}; border-radius: 50%;'></div> ",
                        unsafe_allow_html=True
                    )
                    response = severity
                    query = output['full_query']
                else:
                    response = output['questions'][-1].content
                    st.markdown(response, unsafe_allow_html=True)    
                st.session_state.chat_history.extend([AIMessage(content=str(response))])

        if severity:
            with st.spinner(
                ("The emergency agent is thinking to find a solution..." if severity > 2 else
                "The agent for common situations is thinking to find a solution...") if language != "it" else
                ("L'agente per le emergenze sta pensando per trovare una soluzione..." if severity > 2 else
                "L'agente per le situazioni comuni sta pensando per trovare una soluzione...")
            ):
                # Call the LLM with the Jinja prompt and DataFrame context
                with st.chat_message("assistant"):
                    input = {
                        "full_query": query,
                        "prompt": prompt_emergency if severity>2 else prompt_everyday,
                        "severity" : severity,
                        "history" : st.session_state.chat_history[:-1],
                        "retry_count_youtube": 0,
                        "retry_count_web_search": 0, 
                        "user_location" : user_location,
                        "ensemble_retriever" : ensemble_retriever_emergency,
                        "youtube_api_key": YOUTUBE_API_KEY,
                        "google_maps_api_key": GOOGLE_MAPS_API_KEY
                    }
                    start_time = time.time()
                    response, google_maps_link, hospital_name, youtube_link, video_title= emergency_agent.invoke(input)['final_result']
                    end_time = time.time()

                    # Initialize an empty string to store the full response as it is built
                    st.markdown(response, unsafe_allow_html=True)

                    if severity >2:
                        # Mostra il link di Google Maps
                        st.markdown(f"### Nearest hospital: **{hospital_name}**" if language != "it" else f"### Ospedale più vicino: **{hospital_name}**")
                        st.markdown(f"[Google Maps]({google_maps_link})")

                    if video_title:
                        # Mostra il video di YouTube
                        st.markdown(f"## YouTube Video:" if language != "it" else f"## Video YouTube:")
                        st.markdown(f"### {video_title}:")
                    st.session_state.chat_history.extend([{"role": "assistant", "content": response}])

                    if 'https' in youtube_link:
                        video_url = youtube_link.replace("watch?v=", "embed/")
                        youtube_embed = f'<iframe width="560" height="315" src="{video_url}" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'
                        st.markdown(f"<br>{youtube_embed}", unsafe_allow_html=True)
                st.session_state.chat_history.extend([AIMessage(content=str(response))])
        
        # # Save session data to GCS
        # response_time = end_time - start_time
        # bucket_name = st.secrets["GCP"]["BUCKET_NAME"]
        # session_filename = create_session_filename(session_id)
        # write_session_to_gcs(session_id, app_version, user_location, severity, query, response, response_time, bucket_name, session_filename, gcs_client)


if __name__ == "__main__":
    main()