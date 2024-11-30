from groq import Groq
from jinja2 import Environment, FileSystemLoader
from PIL import Image
from io import BytesIO
import base64
import json
import os
from datetime import datetime


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


def call_llm(llm, llm_model_name, sys_message: str, context_message: str, base64_image: str = None,
            temperature: float = 0.5, max_tokens: int = None, top_p: float = 0.8, stop: str = None) -> str:
    
    messages = [{"role": "system", "content": sys_message}, {"role": "user", "content": context_message}]
    
    if base64_image:
        messages.append({
            "role": "user",
            "content": f"data:image/jpeg;base64,{base64_image}"
        })

    response_stream = llm.chat.completions.create(
        model=llm_model_name,
        messages=messages,
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


# Function to save the session data to a JSON file
def save_session_data(file_path, session_id, location, query, response):
    # Check if the file exists and load existing data
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            all_sessions = json.load(file)
    else:
        all_sessions = []
    
    # Check if the session ID already exists in the data
    session_exists = False
    for session in all_sessions:
        if session["session_id"] == session_id:
            # Append query and response to the existing session's queries_responses list
            session["queries_responses"].append({"query": query, "response": response})
            session_exists = True
            break
    
    # If session does not exist, create a new session entry
    if not session_exists:
        new_session = {
            "session_id": session_id,
            "location": location,
            "queries_responses": [{"query": query, "response": testo_to_utf8(response)}],
            "timestamp": datetime.now().isoformat()  # Timestamp when the session is first created
        }
        all_sessions.append(new_session)
    
    # Save the updated data back to the file with UTF-8 encoding
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(all_sessions, file, indent=4, ensure_ascii=False)