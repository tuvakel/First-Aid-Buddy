from groq import Groq


def init_LLM(API_KEY=None):
    client = Groq(
        api_key= API_KEY,
    )
    return client


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
