import streamlit as st
from groq import Groq
from jinja2 import Environment, FileSystemLoader, Template
from PIL import Image
from io import BytesIO
import base64
import json
import os
import re
from google.cloud import storage
from google.auth import credentials
from datetime import datetime
from gtts import gTTS
import requests


def get_latest_commit_hash(github_repo: str):
    try:
        response = requests.get(f"https://api.github.com/repos/{github_repo}/commits?per_page=1")
        response.raise_for_status()
        commits = response.json()
        return commits[0]['sha']
    except:
        return None


def get_previous_commit_hash(last_commit_file: str):
    if os.path.exists(last_commit_file):
        try:
            with open(last_commit_file, 'r') as file:
                return file.read().strip()
        except:
            return None
    return None


def save_commit_hash(commit_hash: str, last_commit_file: str):
    commit_dir = os.path.dirname(last_commit_file)
    if not os.path.exists(commit_dir):
        os.makedirs(commit_dir)
    try:
        with open(last_commit_file, 'w') as file:
            file.write(commit_hash)
    except:
        pass


def get_current_version(version_file: str):
    if os.path.exists(version_file):
        try:
            with open(version_file, 'r') as file:
                return file.read().strip()
        except:
            return "v0.1.0"
    return "v0.1.0"


def increment_version(version):
    try:
        version_parts = version.lstrip('v').split('.')
        major, minor, patch = map(int, version_parts)
        patch += 1
        new_version = f"v{major}.{minor}.{patch}"
        return new_version
    except:
        return version


def generate_app_id(github_repo: str, last_commit_file: str, version_file: str):
    previous_commit_hash = get_previous_commit_hash(last_commit_file)
    current_commit_hash = get_latest_commit_hash(github_repo)

    current_version = get_current_version(version_file)

    if current_commit_hash:
        if previous_commit_hash != current_commit_hash:
            new_version = increment_version(current_version)
            save_commit_hash(current_commit_hash, last_commit_file)
            with open(version_file, 'w') as file:
                file.write(new_version)
            return new_version
        else:
            return current_version if current_version != "Unknown" else "Unknown"
    return "Unknown"


def get_language(location):
    url = f"https://nominatim.openstreetmap.org/reverse?lat={location[0]}&lon={location[1]}&format=json&addressdetails=1"
    headers = {
        'User-Agent': 'LLamaFirstAid/1.0'
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        country = data.get('address', {}).get('country', None)
        detailed_location = data.get('address', {}).get('county', None) \
            + ', ' + data.get('address', {}).get('state', None)  \
            + ', ' + data.get('address', {}).get('country', None)
        if country.lower() == 'italia':
            return 'it', detailed_location
        else: return 'en', detailed_location
    else:
        print(f"Error getting language from location") 
        return 'en', 'Disabled'


def get_sidebar(language):
    if (language == "it"):
        # st.sidebar.header("Modalità")
        # allow_images = st.sidebar.checkbox("Abilita il caricamento delle immagini (EU-NON COMPLIANT)")
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
    else:
        # st.sidebar.header("Modalità")
        # allow_images = st.sidebar.checkbox("Enable image upload (EU-NON COMPLIANT)")
        st.sidebar.header("**Details**")
        st.sidebar.write(""" 
            Are you ready to respond in a medical emergency?
            
            With the **LLAMA** app (Life-saving Live Assistant for Medical Assistance) **FIRST AID**, 
            you'll have an experienced healthcare operator by your side at all times. Whether you're a beginner or already have experience in first aid, 
            the app will guide you step by step in managing critical situations, providing you with quick and accurate advice. 
            Thanks to an intuitive interface, you’ll be able to receive real-time answers to crucial questions and get the right instructions to 
            respond effectively. Additionally, you'll have access to useful video tutorials to learn and perfect lifesaving techniques. Don’t leave 
            anything to chance, with **LLAMA** every emergency becomes more manageable!
        """)


def resize_image(image_file, new_width):
    with Image.open(image_file) as img:
        aspect_ratio = img.height / img.width
        new_height = int(new_width * aspect_ratio)
        resized_img = img.resize((new_width, new_height))
        img_byte_arr = BytesIO()
        resized_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return img_byte_arr


def convert_image_to_base64(image_file, resize: None):
    if resize: 
        resized_image = resize_image(image_file, new_width=resize)
    else: resized_image = image_file
    img_bytes = resized_image.read()
    base64_image = base64.b64encode(img_bytes).decode('utf-8')
    return base64_image


def load_template(template_path: str) -> Template:
    env = Environment(loader=FileSystemLoader(os.path.dirname(template_path)))
    template = env.get_template(os.path.basename(template_path))
    return template


def init_LLM(API_KEY=None):
    client = Groq(
        api_key= API_KEY,
    )
    return client


def call_llm(llm:Groq, llm_model_name, temperature: float = 0.5, max_tokens: int = None, top_p: float = 0.8, stop: str = None, chat_history = []) -> str:
    
    response_stream = llm.chat.completions.create(
        model=llm_model_name,
        messages=chat_history,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=None,
        stream=True
    )

    return response_stream



mapping = {
    'Ã¨': 'è',
    'Ã ': 'à',
    'Ã ': 'à',
    'Ã©': 'é',
    'Ã¹': 'ù',
    'Ã²': 'ò',
    'Ã¬': 'ì',
    'Ã§': 'ç',
    'Ã³': 'ó',
    'Ã¤': 'ä',
    'Ã¼': 'ü',
    'Ã«': 'ë',
    'Ã¢': 'â',
    'Ãª': 'ê',
    'Ã®': 'î',
    'Ã´': 'ô',
    'Ã¶': 'ö',
    'ÃŸ': 'ß',
    'Ã¸': 'ø',
    'Ã…': 'Å',
    'Ã†': 'Æ',
    'Â©': '©',
    'Â®': '®',
    'â‚¬': '€'
}

def testo_to_utf8(testo, mapping = mapping):
    if testo:
        for errato, corretto in mapping.items():
            testo = testo.replace(errato, corretto)
    else:
        testo = ""
    return testo


def transcribe_audio(llm, llm_audio_model_name, audio_file_path, trscb_message, language):
    try:
        with open(audio_file_path, "rb") as file:
            transcription = llm.audio.transcriptions.create(
                file=(os.path.basename(audio_file_path), file.read()),
                model=llm_audio_model_name,
                prompt=trscb_message,
                response_format="text",
                language=language,
            )
        return transcription  # This is now directly the transcription text
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None
    

def save_uploaded_audio(audio_bytes, output_filename):
    with open(output_filename, "wb") as f:
        f.write(audio_bytes)

    return output_filename


def text_to_speech(text: str, language = 'it', audio_file = "output.mp3"):
    # Converte il testo in audio
    tts = gTTS(text=text, lang=language, slow=False)
    # Salva l'audio in un file
    tts.save(audio_file)
    return audio_file


def extract_youtube_link(response_text):
    youtube_url_pattern = r'(https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+)'
    youtube_urls = re.findall(youtube_url_pattern, response_text)
    
    # Return the first YouTube URL found or None if no link is found
    return youtube_urls[0] if youtube_urls else None


# 1. Initialize GCS Client
def initialize_gcs_client(SERVICE_ACCOUNT_KEY):
    # Load the service account JSON
    service_account_info = json.loads(SERVICE_ACCOUNT_KEY)
    
    # Initialize the storage client with the service account credentials
    client = storage.Client.from_service_account_info(service_account_info)
    return client

# 2. Create a unique session file name using session_id
def create_session_filename(session_id: str):
    # Create a filename based on session_id
    return f"session_{session_id}.json"

# 3. Write a new session data file to Google Cloud Storage (GCS)
def write_session_to_gcs(session_id: str, app_version: str, user_location: list, severity: int, query: str, response: str, response_time: float, bucket_name: str, session_filename: str, client: storage.Client):
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(session_filename)

    try:
        # Try to download the existing content (if any)
        try:
            content_str = blob.download_as_text()
            existing_data = json.loads(content_str)
        except Exception as e:
            # If the file doesn't exist, start with an empty list
            existing_data = []

        # Look for the session with the same session_id
        session_found = False
        for session in existing_data:
            if session['session_id'] == session_id:
                # Append the new user_query and response, response times and updating severity for the existing session
                session['severity'] = severity
                session['queries'].append(query)
                session['responses'].append(response)
                session['response_times'].append(response_time)
                session_found = True
                break
        
        if not session_found:
            # If the session doesn't exist, create a new session entry
            new_session = {
                "session_id": session_id,
                "app_version": app_version,
                "location": user_location,
                "timestamp": datetime.now().isoformat(),
                "severity": severity,
                "queries": [query], 
                "responses": [response],
                "response_times": [response_time] 
            }
            existing_data.append(new_session)

        # Convert the updated data back to JSON string
        updated_content_str = json.dumps(existing_data, indent=4)

        # Upload the updated content back to GCS
        blob.upload_from_string(updated_content_str, content_type='application/json')
        print(f"Session file {session_filename} updated successfully.")

    except Exception as e:
        print(f"Error writing session to GCS: {e}")
    