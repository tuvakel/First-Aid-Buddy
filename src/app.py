import streamlit as st
from utils import *
import tempfile
import geocoder
import hashlib


# Hash session ID using hashlib
if 'session_id' not in st.session_state:
    session_id = hashlib.sha256(str(datetime.now()).encode()).hexdigest()
    st.session_state.session_id = session_id
else:
    session_id = st.session_state.session_id

# Get location
location = geocoder.ip('me')
# Get geographical location of the user
user_location = location.latlng if location.latlng else None

# Initialize the LLM with the Google API key from secrets
llm = init_LLM(API_KEY=st.secrets["GROQ"]["GROQ_API_KEY"])
llm_text_model_name = "llama3-70b-8192"
llm_audio_model_name = "whisper-large-v3"
# llm_vision_model_name = "llama-3.2-11b-vision-preview"

# GCS client to store session data
gcs_client = initialize_gcs_client(SERVICE_ACCOUNT_KEY=st.secrets["GCP"]["SERVICE_ACCOUNT_KEY"])


# Main function
def main():
    st.set_page_config(page_title="llama-first-aid", page_icon="🦙")

    # Additional toggles for fine-grained control of image upload
    # st.sidebar.header("Modalità")
    # allow_images = st.sidebar.checkbox("Enable image upload (EU-NON COMPLIANT)")

    # Sidebar for project details
    st.sidebar.header("Dettagli")
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
        sys_message = sys_message_template.render()
        trscb_message_template = load_template("templates/trscb_message_template.jinja")
        trscb_message = trscb_message_template.render()
        
        if audio_value:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                temp_audio_path = save_uploaded_audio(audio_value.getvalue(), temp_audio_file.name)
            query = transcribe_audio(llm, llm_audio_model_name, temp_audio_path, trscb_message)

        ctx_message_template = load_template("templates/ctx_message_template.jinja")
        ctx_message = ctx_message_template.render(user_request=query)
        
        # Display user message in chat message container
        with st.chat_message("user"):
            if query:
                st.markdown(f"**Testo:** {query}")
            if image_base64:
                st.markdown("**Immagine catturata**")
            
        # Call the LLM with the Jinja prompt and DataFrame context
        with st.chat_message("assistant"):        
            if image_base64 == "":
                stream = call_llm(llm=llm, llm_model_name=llm_text_model_name, sys_message=sys_message, context_message=ctx_message)
            #else: 
            #    stream = call_llm(llm=llm, llm_model_name=llm_vision_model_name, sys_message=sys_message, context_message=ctx_message, base64_image=image_base64)

            # Initialize an empty string to store the full response as it is built
            response = ""
            line_placeholder = st.empty()
            for chunk in stream:
                chunk_text = chunk.choices[0].delta.content
                clean_chunk = testo_to_utf8(chunk_text)
                response += clean_chunk
                line_placeholder.markdown(response, unsafe_allow_html=True)
        
        # Save session data to GCS
        bucket_name = st.secrets["GCP"]["BUCKET_NAME"]
        session_filename = create_session_filename(session_id)
        write_session_to_gcs(session_id, user_location, query, response, bucket_name, session_filename, gcs_client)


if __name__ == "__main__":
    main()

