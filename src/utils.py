from groq import Groq


def init_LLM(API_KEY=None):
    client = Groq(
        api_key= API_KEY,
    )
    return client


def call_llm(llm, llm_model_name, sys_message: str, context_message: str, base64_image: str = "", 
            temperature: float = 0.5, max_tokens: int = None, top_p: float = 0.8, stop: str = None) -> str:
    if base64_image == "":
        messages = [
            {"role": "system", "content": sys_message},
            {"role": "user", "content": context_message}
        ]
    else:
        messages = [
            {"role": "system", "content": sys_message},
            {"role": "user", "content": context_message},
            {"role": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        ]

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
