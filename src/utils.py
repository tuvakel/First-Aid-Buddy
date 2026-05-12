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

from datetime import datetime

import requests
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage




def get_language(location):
    """
    Detect country and location details from GPS coordinates via Nominatim.
    Always returns English as the UI language.
    Falls back safely if location is unavailable or the API call fails.
    """
    if location == (None, None):
        return 'en', None, None

    url = f"https://nominatim.openstreetmap.org/reverse?lat={location[0]}&lon={location[1]}&format=json&addressdetails=1"
    headers = {'User-Agent': 'FirstAidBuddy/1.0'}

    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            country = data.get('address', {}).get('country', None)
            address = data.get('address', {})
            detailed_location = ', '.join(filter(None, [
                address.get('county'),
                address.get('state'),
                address.get('country')
            ]))
            return 'en', detailed_location, country
        else:
            print(f"Nominatim returned status {response.status_code}")
            return 'en', None, None

    except Exception as e:
        print(f"Error getting language from location: {e}")
        return 'en', None, None
        


def get_sidebar():
    
    st.sidebar.header("**Details**")
    st.sidebar.write(""" 
            Are you ready to respond in a medical emergency?
            
            With the First-Aid Buddy app, 
            you'll have an experienced healthcare operator by your side at all times. Whether you're a beginner or already have experience in first aid, 
            the app will guide you step by step in managing critical situations, providing you with quick and accurate advice. 
            Thanks to an intuitive interface, you’ll be able to receive real-time answers to crucial questions and get the right instructions to 
            respond effectively. Additionally, you'll have access to useful video tutorials to learn and perfect lifesaving techniques. Don’t leave 
            anything to chance, with First-Aid Buddy every emergency becomes more managableeable!
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
        api_key=API_KEY,
    )
    return client

MODELS = [
    "llama-3.3-70b-versatile",  # best quality, try first
    "gemma2-9b-it",              # fallback 1
    "llama3-8b-8192",            # fallback 2
]

def init_chat_LLM(api_key):
    return ChatGroq(model=MODELS[0], api_key=api_key)



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

    models_to_try = [llm_model_name] + [m for m in MODELS if m != llm_model_name]

    response_content = None

    for model in models_to_try:
        try:
            response = llm.chat.completions.create(
                model=model,        # ✅ tries each model in order
                messages=[{"role": "user", "content": translate_command}],
                temperature=temperature,
                stop=None
            )
            response_content = response.choices[0].message.content
            print(f"✅ translate() using model: {model}")
            break   # ✅ stop trying once one works

        except Exception as e:
            if "rate_limit_exceeded" in str(e):
                print(f"⚠️ {model} rate limited in translate(), trying next...")
                continue    # try next model
            raise e         # non-rate-limit error — crash immediately

    if response_content is None:
        raise Exception("❌ All models rate limited during translation.")

    

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

# ✅ Add this dictionary
EMERGENCY_NUMBERS = {
    "Kenya": "999",
    "United States": "911",
    "United Kingdom": "999",
    "Australia": "000",
    "Canada": "911",
    "Germany": "112",
    "France": "15",
    "Italy": "118",
    "Spain": "112",
    "India": "108",
    "South Africa": "10177",
    "Nigeria": "199",
    "Ghana": "193",
    "Uganda": "999",
    "Tanzania": "114",
    "Ethiopia": "907",
    "default": "112"  # ✅ international standard fallback
}

def get_emergency_number(country: str) -> str:
    if not country:
        return EMERGENCY_NUMBERS["default"]
    # ✅ case-insensitive match
    for key, number in EMERGENCY_NUMBERS.items():
        if key.lower() == country.lower():
            return number
    return EMERGENCY_NUMBERS["default"]


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
def store_session_data(session_id: str, user_location: list, country:str,
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
                
                "location": user_location,
                "country": country,
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

    