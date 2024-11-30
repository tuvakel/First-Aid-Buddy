from groq import Groq
from jinja2 import Environment, FileSystemLoader
from PIL import Image
from io import BytesIO
import base64, os


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


def transcribe_audio(llm, audio_file_path):
    """
    Transcribe audio using Groq's Whisper implementation.
    """
    try:
        with open(audio_file_path, "rb") as file:
            transcription = llm.audio.transcriptions.create(
                file=(os.path.basename(audio_file_path), file.read()),
                model="whisper-large-v3",
                prompt="""L'audio proviene da una persona che descrive un'emergenza medica o una situazione di primo soccorso. La persona potrebbe menzionare sintomi, lesioni o manovre di primo soccorso che richiedono assistenza (punture, ustioni, ferite, svenimenti, soffocamenti, etc.). L'obiettivo è fornire una trascrizione chiara e accurata per aiutare nell'analisi della situazione e nella fornitura di istruzioni di primo soccorso.""",
                response_format="text",
                language="it",
            )
        return transcription  # This is now directly the transcription text
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None
    

def save_uploaded_audio(audio_bytes, output_filename):
    """
    Salva un file audio fornito come bytes in formato WAV.
    """
    with open(output_filename, "wb") as f:
        f.write(audio_bytes)

    return output_filename