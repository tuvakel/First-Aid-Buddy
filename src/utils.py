from groq import Groq
from jinja2 import Environment, FileSystemLoader
from PIL import Image
from io import BytesIO
import base64
import json
import os
from google.cloud import storage
from google.auth import credentials
from datetime import datetime
from gtts import gTTS


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


def load_template(template_path: str) -> str:
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


def transcribe_audio(llm, llm_audio_model_name, audio_file_path, trscb_message):
    try:
        with open(audio_file_path, "rb") as file:
            transcription = llm.audio.transcriptions.create(
                file=(os.path.basename(audio_file_path), file.read()),
                model=llm_audio_model_name,
                prompt=trscb_message,
                response_format="text",
                language="it",
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
def write_session_to_gcs(session_id: str, user_location: list, query: str, response: str, bucket_name: str, session_filename: str, client: storage.Client):
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
                # Append the new user_query and response to the existing session
                session['queries'].append(query)
                session['responses'].append(response)
                session_found = True
                break
        
        if not session_found:
            # If the session doesn't exist, create a new session entry
            new_session = {
                "session_id": session_id,
                "location": user_location,
                "timestamp": datetime.now().isoformat(),
                "queries": [query], 
                "responses": [response] 
            }
            existing_data.append(new_session)

        # Convert the updated data back to JSON string
        updated_content_str = json.dumps(existing_data, indent=4)

        # Upload the updated content back to GCS
        blob.upload_from_string(updated_content_str, content_type='application/json')
        print(f"Session file {session_filename} updated successfully.")

    except Exception as e:
        print(f"Error writing session to GCS: {e}")
    