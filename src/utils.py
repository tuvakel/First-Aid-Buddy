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
    if location == (None, None):
        return 'en', 'Disabled'
    else:
        url = f"https://nominatim.openstreetmap.org/reverse?lat={location[0]}&lon={location[1]}&format=json&addressdetails=1"
        headers = {
            'User-Agent': 'LLamaFirstAid/0.1'
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


def translate(llm: Groq, llm_model_name, temperature: float = 0.0, message: str = "", target_language: str = "") -> str:
    translate_command = f"""
        You are a language model capable of translating text between languages.
        Your task is to detect the source language from the given message and translate it into the target language. 
        Keep the original format intact (including Markdown elements like headers, lists, and code blocks) while translating the text.

        Input:
        - Message: {message}
        - Target Language: {target_language}

        If target_language and source_language are the same, return the original message without changes.
        You must provide a response in the following JSON format:
        {{
            "translated_query": "the translated query in the target language",
            "source_language": "the detected source language"
        }}

        Do exactly the required task and return a JSON in the required format.
        Do not add any additional information in the response.
    """

    response = llm.chat.completions.create(
        model=llm_model_name,
        messages=[{"role": "user", "content": translate_command}],
        temperature=temperature,
        stop=None
    )

    response_content = response.choices[0].message.content

    try:
        translated_query_match = re.search(r'"translated_query"\s*:\s*"([^"]+)', response_content)
        source_language_match = re.search(r'"source_language"\s*:\s*"([^"]+)', response_content)
        
        if translated_query_match:
            translated_query = translated_query_match.group(1)
        else:
            raise ValueError(f"Unable to extract translated query from response: {response_content}")

        if source_language_match:
            source_language = source_language_match.group(1)
        else:
            raise ValueError(f"Unable to extract source language from response: {response_content}")

    except Exception as e:
        raise ValueError(f"Error extracting data: {e}")

    return translated_query, source_language


def get_medical_class(llm: Groq, llm_model_name, temperature: float = 0.0, chat_history: list = []) -> str:
    if not chat_history or len(chat_history) < 1:
        raise ValueError("Chat history is insufficient for classification.")

    medical_specialties = [
        "cardiology", "psychiatry", "dermatology", "pulmonology", "gastroenterology", 
        "neurology", "orthopedics", "endocrinology", "hematology", "oncology", 
        "ophthalmology", "gynecology", "urology", "rheumatology", "infectious disease", 
        "anesthesiology", "pediatrics", "general surgery", "plastic surgery", "geriatrics", 
        "family medicine", "radiology", "nephrology", "trauma surgery", "vascular surgery", 
        "internal medicine"
    ]

    classify_command = f"""
        You are a medical expert capable of classifying medical issues based on conversations. 
        Based on the following conversation, identify the medical specialty most relevant to the issue discussed.
        
        Please choose one of the following specialties:
        {', '.join(medical_specialties)}.
        
        If there is insufficient information to classify, or if you cannot infer the specialty, return "None".

        Here is the conversation:
        {chat_history}

        Your task is to return the medical specialty as a JSON object:
        {{
            "medical_class": "the identified medical specialty (e.g., 'cardiology') or None if undetermined"
        }}

        Please return only the JSON object, nothing else. The response must be **always** in **English**.
    """

    response = llm.chat.completions.create(
        model=llm_model_name,
        messages=[{"role": "user", "content": classify_command}],
        temperature=temperature,
        stop=None
    )

    response_content = response.choices[0].message.content.strip()

    try:
        classification = json.loads(response_content)
        medical_class = classification.get("medical_class", None)
        if medical_class == "None" or medical_class not in medical_specialties:
            return None
    except json.JSONDecodeError:
        raise ValueError(f"Error parsing the response: {response_content}")

    return medical_class


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
    return f"session_{session_id}.json"

# 3. Write a new session data file either locally or within Google Cloud Storage (GCS)
def store_session_data(session_id: str, app_version: str, user_location: list,
                        medical_class: str, severity: int,
                        hospital_details: list, youtube_video_details: list, query: str, response: str,
                        response_time: float, session_filename: str, local_path_name: str = None,
                        bucket_name: str = None, client: storage.Client = None):
    def process_session_data(existing_data, session_found=False):
        """ Helper function to process and update the session data. """
        for session in existing_data:
            if session['session_id'] == session_id:
                session['medical_class'] = medical_class
                session['severity'] = severity
                session['hospital'] = {"name": hospital_details[0], "gmaps_link": hospital_details[1]}
                session['youtube_video'] = {"title": youtube_video_details[0], "link": youtube_video_details[1]}
                session['queries'].append(query)
                session['responses'].append(response)
                session['response_times'].append(response_time)
                session_found = True
                break

        if not session_found:
            new_session = {
                "session_id": session_id,
                "app_version": app_version,
                "location": user_location,
                "timestamp": datetime.now().isoformat(),
                "medical_class": medical_class,
                "severity": severity,
                "hospital": {"name": hospital_details[0], "gmaps_link": hospital_details[1]},
                "youtube_video": {"title": youtube_video_details[0], "link": youtube_video_details[1]},
                "queries": [query],
                "responses": [response],
                "response_times": [response_time]
            }
            existing_data.append(new_session)
        return existing_data

    # If data should be saved locally
    if local_path_name:
        local_file_path = f"{local_path_name}/{session_filename}"
        os.makedirs(local_path_name, exist_ok=True) 

        try:
            existing_data = []
            if os.path.exists(local_file_path):
                with open(local_file_path, 'r') as f:
                    existing_data = json.load(f)

            existing_data = process_session_data(existing_data)

            with open(local_file_path, 'w') as f:
                json.dump(existing_data, f, indent=4)

            print(f"Session file {session_filename} saved locally.")

        except Exception as e:
            print(f"Error writing session data locally: {e}")
    
    # If data should be saved to GCS
    if client and bucket_name:
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(session_filename)

        try:
            try:
                content_str = blob.download_as_text()
                existing_data = json.loads(content_str)
            except Exception:
                existing_data = []

            existing_data = process_session_data(existing_data)

            updated_content_str = json.dumps(existing_data, indent=4)
            blob.upload_from_string(updated_content_str, content_type='application/json')

            print(f"Session file {session_filename} updated successfully in GCS.")

        except Exception as e:
            print(f"Error writing session to GCS: {e}")

    