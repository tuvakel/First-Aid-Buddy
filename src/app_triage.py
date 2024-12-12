import streamlit as st
from utils import *
import tempfile
import hashlib
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
#from crewai_utils import init_crew
from triage_utils import create_retriever, create_triage_agent
from streamlit_js_eval import get_geolocation
import time

st.set_page_config(page_title="llama-first-aid", page_icon="🦙")
# Hash session ID using hashlib
if 'session_id' not in st.session_state:
    session_id = hashlib.sha256(str(datetime.now()).encode()).hexdigest()
    st.session_state.session_id = session_id
else:
    session_id = st.session_state.session_id

if st.checkbox("Check my location", value=True):
    with st.spinner("Retrieving location..."):
        user_location_info = None
        while user_location_info is None:
            user_location_info = get_geolocation()
            if user_location_info is None:
                time.sleep(0.5)
    user_location_info = user_location_info.get('coords', None)
    user_location = (user_location_info['latitude'], user_location_info['longitude'])
else:
    user_location = (3,4)

# Initialize the LLM with the Google API key from secrets
llm = init_LLM(API_KEY=st.secrets["GROQ"]["GROQ_API_KEY"])
YOUTUBE_API_KEY = st.secrets["YOUTUBE"]["YOUTUBE_API_KEY"]
GOOGLE_MAPS_API_KEY = st.secrets["GOOGLE_MAPS"]["GOOGLE_MAPS_API_KEY"]
llm_text_model_name = "llama3-70b-8192"
llm_audio_model_name = "whisper-large-v3"
file_path = "../data/doc_triage/pdf/Manuale-Triage.pdf"
documents=None
ensemble_retriever = None
agent=None

# Funzione per creare il retriever
@st.cache_resource
def load_retriever(_file_path):
    return create_retriever(_file_path)

# Funzione per creare l'agente
@st.cache_resource
def load_agent():
    return create_triage_agent()

#documents = load_documents(_file_path=file_path)
ensemble_retriever = load_retriever(file_path)
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
            
        with st.spinner("Sto pensando..."):
            # Call the LLM with the Jinja prompt and DataFrame context
            with st.chat_message("assistant"):
                input = {
                    "messages":st.session_state.chat_history,
                    "ensemble_retriever" : ensemble_retriever,
                    "questions" : []
                }
                #config = {"configurable": {"thread_id": "1"}}
                output = agent.invoke(input)
                severity = output.get('severity', None)
                if severity:
                    response = severity
                else:
                    response = output['questions'][-1].content
                #response = testo_to_utf8(response.raw)

                # Initialize an empty string to store the full response as it is built
                st.markdown(response, unsafe_allow_html=True)

                st.session_state.chat_history.extend([AIMessage(content=str(response))])
            
        # Save session data to GCS
        bucket_name = st.secrets["GCP"]["BUCKET_NAME"]
        session_filename = create_session_filename(session_id)
        write_session_to_gcs(session_id, user_location, query, response, bucket_name, session_filename, gcs_client)


if __name__ == "__main__":
    main()