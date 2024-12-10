import streamlit as st
from utils import *
import tempfile
import hashlib
#from crewai_utils import init_crew
from langgraph_utils import get_user_location, process_pdf, create_retriever, create_langgraph_agent

st.set_page_config(page_title="llama-first-aid", page_icon="🦙")
# Hash session ID using hashlib
if 'session_id' not in st.session_state:
    session_id = hashlib.sha256(str(datetime.now()).encode()).hexdigest()
    st.session_state.session_id = session_id
else:
    session_id = st.session_state.session_id

user_location = get_user_location()

# Initialize the LLM with the Google API key from secrets
llm = init_LLM(API_KEY=st.secrets["GROQ"]["GROQ_API_KEY"])
YOUTUBE_API_KEY = st.secrets["YOUTUBE"]["YOUTUBE_API_KEY"]
GOOGLE_MAPS_API_KEY = st.secrets["GOOGLE_MAPS"]["GOOGLE_MAPS_API_KEY"]
llm_text_model_name = "llama3-70b-8192"
llm_audio_model_name = "whisper-large-v3"
file_path = "../data/doc_emergency/pdf/manuale_primo_soccorso.pdf"
documents=None
ensemble_retriever = None
agent=None

@st.cache_resource
def load_documents(_file_path):
    return process_pdf(_file_path)

# Funzione per creare il retriever
@st.cache_resource
def load_retriever(_documents):
    return create_retriever(_documents)

# Funzione per creare l'agente
@st.cache_resource
def load_agent():
    return create_langgraph_agent()

documents = load_documents(_file_path=file_path)
ensemble_retriever = load_retriever(documents)
#crew = init_crew()
agent = load_agent()
# llm_vision_model_name = "llama-3.2-11b-vision-preview"

# GCS client to store session data
gcs_client = initialize_gcs_client(SERVICE_ACCOUNT_KEY=st.secrets["GCP"]["SERVICE_ACCOUNT_KEY"])


# Main function
def main():

    # Additional toggles for fine-grained control of image upload
    # st.sidebar.header("Modalità")
    # allow_images = st.sidebar.checkbox("Enable image upload (EU-NON COMPLIANT)")

    # Sidebar for project details
    st.sidebar.header("**Dettagli**")
    st.sidebar.write(""" 
        Sei pronto a intervenire in un'emergenza sanitaria?
         
        Con l'app **LLAMA** (Life-saving Live Assistant for Medical Assistance) **FIRST AID**, 
        avrai un operatore sanitario esperto sempre al tuo fianco. Che tu sia un neofita o abbia già esperienza nel primo soccorso, 
        l'app ti guiderà passo dopo passo nella gestione di situazioni critiche, offrendoti consigli rapidi e precisi. 
        Grazie a un'interfaccia intuitiva, potrai ricevere risposte in tempo reale alle domande cruciali e ottenere le istruzioni giuste per 
        intervenire al meglio. Inoltre, avrai accesso a video tutorial utili per apprendere e perfezionare le manovre di soccorso. Non lasciare
        nulla al caso, con **LLAMA** ogni emergenza diventa più gestibile!
    """)

    st.title("LLAMA FIRST AID 🦙")

    # User query input
    query = ""
    image_base64 = ""
    audio_value = ""

    query = st.chat_input("Descrivi il problema o la situazione di emergenza")
    audio_value = st.audio_input("Parla col tuo assistente (opzionale)")

    #if allow_images:
    #    captured_image = st.file_uploader("Carica un'immagine (opzionale)", type=["jpg", "jpeg", "png"])
    #    if captured_image:
    #        image_base64 = convert_image_to_base64(captured_image, resize=50)

    if (query or (query and image_base64)) or (audio_value or (audio_value and image_base64)):
        sys_message_template = load_template("templates/sys_message_template.jinja")
        #sys_message = sys_message_template.render()
        trscb_message_template = load_template("templates/trscb_message_template.jinja")
        trscb_message = trscb_message_template.render()
        
        if audio_value:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                temp_audio_path = save_uploaded_audio(audio_value.getvalue(), temp_audio_file.name)
            query = transcribe_audio(llm, llm_audio_model_name, temp_audio_path, trscb_message)
        
        # # Display user message in chat message container
        # with st.chat_message("user"):
        #     if query:
        #         st.markdown(f"**Testo:** {query}")
        #     if image_base64:
        #         st.markdown("**Immagine catturata**")
            
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [{"role": "user", "content": query}]
    
            if image_base64:
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": f"data:image/jpeg;base64,{image_base64}"
                })
        else:
            st.session_state.chat_history.append({"role": "user", "content": query})

        # Mostra la cronologia della conversazione
        for message in st.session_state.chat_history:
            if message["role"] != 'system':
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
        with st.spinner("Sto pensando..."):
            # Call the LLM with the Jinja prompt and DataFrame context
            with st.chat_message("assistant"):
                input = {
                    "query": query,
                    "history" : st.session_state.chat_history[:-1],
                    "retry_count_youtube": 0,  # Inizializzazione del conteggio
                    "retry_count_web_search": 0, 
                    "user_location" : user_location,
                    "ensemble_retriever" : ensemble_retriever,
                    "youtube_api_key": YOUTUBE_API_KEY,
                    "google_maps_api_key": GOOGLE_MAPS_API_KEY
                }
                response, google_maps_link, hospital_name, youtube_link, video_title= agent.invoke(input)['final_result']
                #response = testo_to_utf8(response.raw)

                # Initialize an empty string to store the full response as it is built
                st.markdown(response, unsafe_allow_html=True)

                # Mostra il link di Google Maps
                st.markdown(f"## Link all'ospedale più vicino:")
                st.markdown(f"[{hospital_name}]({google_maps_link})")

                # Mostra il video di YouTube
                st.markdown(f"## Video YouTube:")
                st.markdown(f"### {video_title}:")
                st.session_state.chat_history.extend([{"role": "assistant", "content": response}])
        
                # Extract YouTube link from the response and embed it
                #youtube_link = extract_youtube_link(response)

                if 'https' in youtube_link:
                    video_url = youtube_link.replace("watch?v=", "embed/")
                    youtube_embed = f'<iframe width="560" height="315" src="{video_url}" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'
                    st.markdown(f"<br>{youtube_embed}", unsafe_allow_html=True)

            #if response.strip():
            #    audio_path = text_to_speech(response, language="it")
            #    st.audio(audio_path, format="audio/mp3", start_time=0)
            
        # Save session data to GCS
        bucket_name = st.secrets["GCP"]["BUCKET_NAME"]
        session_filename = create_session_filename(session_id)
        write_session_to_gcs(session_id, user_location, query, response, bucket_name, session_filename, gcs_client)


if __name__ == "__main__":
    main()