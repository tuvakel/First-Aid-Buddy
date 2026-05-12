from langchain_groq import ChatGroq
from langgraph.graph import StateGraph
import pickle
from typing import TypedDict, Annotated, List, Any
from langgraph.graph.message import add_messages

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_community.utilities import GoogleSerperAPIWrapper
import requests
from bs4 import BeautifulSoup
import re
import json
from dotenv import load_dotenv
load_dotenv()
import os
import streamlit as st
from jinja2 import Template

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
#from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema import Document

# from pdf2image import convert_from_path, convert_from_bytes
from pdf2image import convert_from_path
import pytesseract


# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from src.utils import init_chat_LLM
llm_70b = init_chat_LLM(api_key=st.secrets["GROQ"]["GROQ_API_KEY"])
os.environ["SERPER_API_KEY"] = st.secrets["SERPER"]["SERPER_API_KEY"]




def process_pdf_emergency(file_path):
    """
    Load and process the St John Ambulance First Aid Manual using OCR.
    Converts each PDF page to an image and extracts text with pytesseract.
    Skips the first 12 pages (cover, foreword, contents).

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        list[Document]: LangChain-compatible Document objects, one per page.
    """
    print('process_pdf_emergency — OCR mode (St John Ambulance)')

    # Convert all pages to images at once — cross-platform, no poppler_path needed
    images = convert_from_path(file_path, dpi=200)

    documents = []
    for i, image in enumerate(images[12:], start=12):  # skip first 12 pages
        text = pytesseract.image_to_string(image)

        # Clean common OCR artefacts
        text = text.replace("-\n", "")             # fix hyphenated line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)     # collapse excessive blank lines
        text = re.sub(r'[^\x00-\x7F]+', ' ', text) # remove non-ASCII noise

        # Skip near-empty pages (full-page photos with no text)
        if len(text.strip()) < 50:
            continue

        # Use first non-empty line as section title
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        title = lines[0] if lines else f"Page {i}"

        documents.append(Document(
            page_content=text,
            metadata={"title": title, "page_nr": i}
        ))

        if i % 20 == 0:
            print(f"  OCR progress: page {i}/{len(images) + 12}")

    print(f"process_pdf_emergency — extracted {len(documents)} documents")
    return documents


def create_bm25_retriever_emergency(pdf_file_path: str, bm25_index_path):

    try:
        os.makedirs(os.path.dirname(bm25_index_path), exist_ok=True)

        if os.path.exists(bm25_index_path):
            with open(bm25_index_path, "rb") as f:
                bm25_retriever = pickle.load(f)
                bm25_retriever.k = 3
                documents = []
        else:
            documents = process_pdf_emergency(pdf_file_path)
            bm25_retriever = BM25Retriever.from_documents(documents)
            bm25_retriever.k = 3

            with open(bm25_index_path, "wb") as f:
                pickle.dump(bm25_retriever, f)

        return bm25_retriever, documents

    except Exception as e:
        print("BM25 emergency creation failed:", e)

        # fallback (VERY IMPORTANT)
        documents = process_pdf_emergency(pdf_file_path)
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 3

        return bm25_retriever, documents


def create_emergency_retriever(pdf_file_path,  bm25_index_path, faiss_path):
    # Step 1: Configure the BM25 index for titles.
    bm25_retriever, documents = create_bm25_retriever_emergency(pdf_file_path, bm25_index_path)
    # Step 2: Configure FAISS for the content.
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    faiss_index_file = os.path.join(faiss_path, "index.faiss")
    if os.path.exists(faiss_index_file):
        vectorstore = FAISS.load_local(faiss_path, embeddings=embedding, allow_dangerous_deserialization=True)
        print('load emergency retriever')
    else:
        if documents:
            vectorstore = FAISS.from_documents(documents, embedding=embedding)
            vectorstore.save_local(faiss_path)
        else:
            documents = process_pdf_emergency(pdf_file_path)
            vectorstore = FAISS.from_documents(documents, embedding=embedding)
            vectorstore.save_local(faiss_path)
    similarity_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4})

    # Step 3: Configure a MultiRetriever.
    ensemble_retriever = EnsembleRetriever(retrievers=[
        bm25_retriever,
        similarity_retriever
    ], weights=[0.3, 0.7])
    class SafeRetriever:
        def __init__(self, retriever):
            self.retriever = retriever

        def invoke(self, query):
            if not isinstance(query, str):
                print("⚠️ FIXING QUERY TYPE:", type(query))
                query = str(query)
            return self.retriever.invoke(query)

    return SafeRetriever(ensemble_retriever)



class AgentState(TypedDict):
    query: str
    full_query:str
    severity: int
    messages: Annotated[list, add_messages]
    prompt: str

    rag_answer : str
    ensemble_retriever : Any

    keywords_youtube: str
    search_results: str
    video_title:str
    youtube_api_key : str
    retry_count_youtube: int

    
    google_maps_url: str
    user_location : List[str]
    hospital_name : str
    emergency_number: str 
    
    web_search_keywords : str
    retry_count_web_search : int
    web_answer : str
    web_info: Any 
    final_result: List[str]


def answer_from_rag(state: AgentState):
    log_state("answer_from_rag", state)
    full_query = state['full_query']
    ensemble_retriever = state['ensemble_retriever']
    emergency_number = state.get('emergency_number', '112')  # ✅ get it

    retrieved_info = None
    if ensemble_retriever:
        try:
            retrieved_docs = ensemble_retriever.invoke(full_query)
            retrieved_info = [doc.page_content for doc in retrieved_docs[:2]]
        except Exception as e:
            print(f"⚠️ Retriever failed: {e}")

    prompt_path = state['prompt']
    from jinja2 import Environment, FileSystemLoader
    import os
    env = Environment(loader=FileSystemLoader(os.path.dirname(prompt_path)))
    template = env.get_template(os.path.basename(prompt_path))
    prompt = template.render(
        full_query=full_query,
        retrieved_info=retrieved_info,
        emergency_number=emergency_number  # ✅ pass to template
    )

    response = llm_70b.invoke([HumanMessage(content=prompt)]).content.strip()
    return {"rag_answer": response, "full_query": full_query}


def log_state(node_name, state:AgentState):
    print(f"Node '{node_name}' State: {state}")


def web_search(state: AgentState) -> str:
    """
    Searches the Internet to retrieve reliable and certified information related to a specific medical query.

    Args:
        query (str): A simplified string, optimized for an effective Google search based on the user's query.

    Returns:
        str: A string containing useful and relevant information retrieved from certified websites related to the user's query. 
             If no pertinent information is found, it returns a message indicating the absence of results.
    """
    # Phase 1: Internet search.
    log_state("web_search", state)
    query = state['web_search_keywords']
    if not isinstance(query, str):
        return {"web_info": "NO Info"}
    serper = GoogleSerperAPIWrapper()

    compliant_links = ['webmd', 'mayoclinic']
    try:
        search_results = serper.results(query)['organic']
        # Filter and select one link for each compliant domain.
        selected_links = []
        for domain in compliant_links:
            for result in search_results:
                if domain in result['link']:
                    selected_links.append(result['link'])
                    break  # Exit the loop to move on to the next domain.

        general_content = []
        if not selected_links:
            return {"web_info": "NO Info"}
        selected_links = [selected_links[0]]
        for url in selected_links:
            try:
                # Make a request to the website
                response = requests.get(url)
                response.raise_for_status()  # Check if the request was successful
                
                # Parse the page content with BeautifulSoup."
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract the main content of the page (you may need to adapt the selector).
                page_content = soup.get_text(separator=' ', strip=True)
                
                general_content.append(page_content)

            except requests.exceptions.RequestException as e:
                print(f"NO Info")
        return {"web_info" : general_content}
    except:
        return {"web_info" : "NO Info"}
    

def extract_keywords_web_search(state:AgentState):
   log_state("extract_keywords_web_search", state)
   query = state['full_query']
   previous_keywords = state.get('web_search_keywords', '')
    # Build the prompt
   prompt = f"""You are a highly skilled virtual assistant with expertise in first aid. Your task is to extract the most relevant medical keywords from the user's query. These keywords will help optimize searches for first aid guidance on various websites. Follow these instructions carefully:
    
    1. **Understand User Needs:** Analyze the user query to understand the specific medical needs or issues.
    2. **Focus on Medical Relevance:** Extract only essential information about the medical issue or injury, including:
    - Type of injury or symptom (e.g., "cut," "burn," "panic attack").
    - Cause of the issue, if specified (e.g., "knife," "hot water," "bee sting").
    3. **Omit redundant or irrelevant details:** Ignore unnecessary context, such as who the injury happened to or extraneous background information.
    4. **Output format:** Return the result strictly as a JSON object with the key 'keywords' containing the extracted keywords. **Do not include any other text outside the JSON object.**""" + \
    """
    1. Query: "I am feeling anxious, I think I am having a panic attack. What should I do?"  
   Output : {"keywords": "panic attack, first aid"}

    2. Query: "What should I do if I get stung by a bee?"  
    Output : {"keywords": "bee sting, first aid"}

    3. Query: "How do I treat a deep cut from a knife?"  
    Output : {"keywords": "deep cut knife, first aid"}

    4. Query: "How do I treat a burn from boiling water?"  
    Output : {"keywords": "boiling water burn, first aid"}

    5. Query: "What should I do in case of a sudden allergic reaction?"  
    Output : {"keywords": "allergic reaction, first aid"}

    6. Query: "A friend of mine is having a panic attack"  
    Output : {"keywords": "panic attack, first aid"}
    """

   if previous_keywords:
        prompt += f" Previous search with keywords '{previous_keywords}' returned no results. Try a different search query."
    
   prompt += f"""    ### Input:
   Query: '{query}'

    Return the data strictly as a JSON object, with the following structure:
    {{
        "keywords": "allergic reaction help, first aid"
    }}
    """
        
   # Call the LLM model
   response = llm_70b.invoke([HumanMessage(content=prompt)])
   try:
       keywords = json.loads(response.content)["keywords"]
   except (json.JSONDecodeError, KeyError):
       keywords = query[:50]  # use truncated query as fallback
   return {"web_search_keywords": keywords, "retry_count_web_search": state["retry_count_web_search"] + 1}

# Function to check whether to continue
def should_continue_web_search(state: AgentState):
    web_info = state.get('web_info', '')               # correct field
    retry_count_web_search = state.get('retry_count_web_search', 0)
    if (not web_info or web_info == "NO Info") and retry_count_web_search < 2:
        return "retry"
    return "end"


# Function to check whether to continue
def should_web_search(state:AgentState):
    rag_answer = state.get('rag_answer', '')
    if not rag_answer or "no info available" in rag_answer.lower():
        return "web_search"
    return "end"


def extract_keywords_youtube(state:AgentState):
   log_state("extract_keywords_youtube", state)
   query = state['full_query']
   previous_keywords = state.get('keywords_youtube', '')
    # Build the prompt
   prompt = f"""From the following user medical situation: '{query}', extract the most relevant keywords to optimize the search for a video on YouTube. 
    Return just a Json object with the key: 'keywords'
    Here are examples of user queries and the corresponding optimized output:""" + \
    """
    1. Query: "I am feeling anxious, I think I am having a panic attack. What should I do?" 
       Output : {"keywords": "panic attack, first aid"}
    2. Query: "What should I do if I get stung by a bee?"
       Output : {"keywords": "bee sting treatment, first aid"}
    3. Query: "What happens if I was stung by a bee?"
       Output : {"keywords": "bee sting treatment, first aid"}
    3. Query: "How to treat a deep cut made with a knife?"
       Output : {"keywords": "knife deep cut treatment, first aid"}
    4. Query: "How to treat a burn from boiling water?"
       Output : {"keywords": "boiling water burn, first aid"}
    5. Query: "What to do in case of a sudden allergic reaction?"
       Output : {"keywords": "allergic reaction help, first aid"} 
    
    ### Output:
    Return strictly as a JSON object in form of, in form:
    {"keywords": "allergic reaction help, first aid"}

    """

   if previous_keywords:
        prompt += f" Previous search with keywords '{previous_keywords}' returned no results. Try a different search query."
   
    # Call the LLM model.
   response = llm_70b.invoke([HumanMessage(content=prompt)])
   try:
       # Strip markdown fences if present before parsing
       clean = response.content.strip().replace("```json", "").replace("```", "").strip()
       keywords = json.loads(clean)["keywords"]
   except (json.JSONDecodeError, KeyError):
    # Fall back to a truncated version of the raw query
       keywords = query[:50]

   return {
       "keywords_youtube": keywords,
       "retry_count_youtube": state["retry_count_youtube"] + 1
   }


# Function to check whether to continue
def should_continue_youtube(state:AgentState):
    search_results = state.get('search_results', '')
    retry_count_youtube = state.get('retry_count_youtube', 0)
    if (not search_results or "No videos found" in search_results) and retry_count_youtube < 2:
        return "retry"
    return "end" 


# Function to check whether to continue
def should_find_hospital(state:AgentState):
    severity = state.get('severity')
    if severity>2:
        return "high_severity"
    return "low_severity"


def create_response_from_web_search(state:AgentState):
    web_info = state.get('web_info', '')
    query = state.get('full_query', '')
    prompt = f"""Using the following context: {web_info}, provide a detailed and comprehensive response to the user query: "{query}". Focus on offering practical and actionable support for someone already facing the issue. Avoid mentioning precautions unless explicitly relevant to resolving the problem. Ensure your answer is clear, accurate, and concise, and limit it in a range of 400-1000 words."""
    response = llm_70b.invoke([HumanMessage(content=prompt)])
    return {"web_answer" : response.content}


def search_youtube_videos(state:AgentState) -> str:
    """
    Search for videos on YouTube from a certified list of trusted channels.

    Args:
        query (str): A simplified English version of the user's query, 
        optimised for a YouTube search.
    Returns:
        str:  A useful video link related to the query, or a message 
        indicating that no videos were found.
    """
    #log_state("search_youtube_videos", state)
    keywords = state['keywords_youtube']
    print(f"keywords: {keywords}")
    if not isinstance(keywords, str):
        fallback = f"https://www.youtube.com/results?search_query=first+aid"
        return {"search_results": fallback, "video_title": "First Aid Videos"}
    
    YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
    allowed_channels=[
        'UCwywRelPfy7U8jAI312J_Xw',  # First Aid
        'UCQK834Q3xqlo85LJqrEd7fw',  # ChatterDocs
        'UCVVXqSUGEr7oYBR3PGNqhAg',  # American Red Cross
        'UCqDFgQMSplDoKJFsxh6MUtA',  # St John Ambulance
        'UC6107grRI4m0o2-emgoDnAA',  # Skill Share (medical)
    ]  #'UCTVZkcCKSqFD0TTJ8BjYLDQ' Croce Rossa, 
    max_results = 5
    relevance_prompt = """
    You are tasked with determining if a YouTube video is relevant to a described medical situation. The situation provides details about a **medical problem affecting a person**. Analyze the situation and the video title, and decide if the video could be useful. Respond strictly with "YES" or "NO". Do not provide explanations or additional information.

    ### Guidelines:
    1. Assume the described situation pertains to a medical issue involving a person unless explicitly stated otherwise.
    2. Focus only on the **relevance** of the video to the medical situation described.
    3. Base your decision solely on the details provided in the medical situation and the video title.
    4. Respond with **"YES"** or **"NO"** only. Do not provide any explanations.

    ### Input Format:
    - Medical Situation: [Description of the patient's medical situation]
    - Video Title: [Title of the YouTube video]

    ### Output Format:
    - "YES" or "NO"

    ### Examples:
    - Medical Situation: "The patient was stung by a bee and has never had allergic reactions or symptoms such as swelling, itching, or difficulty breathing after being stung by an insect in the past."
      Video Title: "First Aid for Bee Stings"
      Output: "YES"

    - Medical Situation: "The patient was stung by a bee, but suffers from severe seasonal allergies."
      Video Title: "How to Treat Seasonal Allergies"
      Output: "NO"

    - Medical Situation: "The patient accidentally cut their hand with a knife and is experiencing minor bleeding."
      Video Title: "Emergency Care for Cuts"
      Output: "YES"

    - Medical Situation: "A person is having a heart attack."
      Video Title: "First Aid - Heart Attack"
      Output: "YES"

    ### Now process the following input:
    Medical Situation: {query}  
    Video Title: {video_title}
    """
    def try_search(channel_id=None):
        params = {
            "part": "snippet",
            "q": keywords,
            "maxResults": max_results,
            "type": "video",
            "key": state['youtube_api_key'],
        }
        if channel_id:
            params["channelId"] = channel_id

        try:
            response = requests.get(YOUTUBE_SEARCH_URL, params=params)
            data = response.json()
            if "items" not in data or len(data["items"]) == 0:
                return None, None

            for item in data["items"]:
                video_id = item["id"]["videoId"]
                video_title = item["snippet"]["title"]
                check = llm_70b.invoke([HumanMessage(content=relevance_prompt.format(
                    query=state['full_query'],
                    video_title=video_title
                ))]).content
                if 'yes' in check.strip().lower():
                    return f"https://www.youtube.com/watch?v={video_id}", video_title
        except Exception as e:
            print(f"YouTube search error: {e}")
        return None, None

    # 1️⃣ Try allowed channels first
    for channel_id in allowed_channels:
        url, title = try_search(channel_id)
        if url:
            return {"search_results": url, "video_title": title}

    # 2️⃣ Fallback — unrestricted search
    print("⚠️ No video in allowed channels, trying general search...")
    url, title = try_search(channel_id=None)
    if url:
        return {"search_results": url, "video_title": title}

    fallback_search_url = f"https://www.youtube.com/results?search_query={keywords.replace(' ', '+')}"
    return {
        "search_results": fallback_search_url,
          "video_title": f"First Aid: {keywords}"
    }

def get_google_maps_url(state: AgentState):
    lat, lng = state['user_location']
    print(f"🏥 Hospital search - lat: {lat}, lng: {lng}")

    if lat is None or lng is None:
        print("❌ No location provided")
        return {"hospital_name": None, "google_maps_url": None}

    # ✅ Multiple Overpass endpoints to try if one is rate-limited
    overpass_endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ]

    for radius in [7000, 15000, 25000]:
        for endpoint in overpass_endpoints:
            overpass_query = f"""
            [out:json][timeout:25];
            (
              node["amenity"="hospital"](around:{radius},{lat},{lng});
              way["amenity"="hospital"](around:{radius},{lat},{lng});
              relation["amenity"="hospital"](around:{radius},{lat},{lng});
            );
            out center 1;
            """
            try:
                response = requests.post(  # ✅ POST is more reliable than GET for Overpass
                    endpoint,
                    data={"data": overpass_query},
                    timeout=25
                )
                print(f"🌐 {endpoint} — status: {response.status_code}, length: {len(response.text)}")

                if response.status_code != 200 or not response.text.strip():
                    continue  # try next endpoint

                data = response.json()
                elements = data.get("elements", [])
                print(f"🏥 Radius {radius}m — {len(elements)} results")

                if elements:
                    hospital = elements[0]
                    hospital_name = hospital.get("tags", {}).get("name", "Nearest Hospital")

                    if hospital.get("type") == "node":
                        h_lat = hospital.get("lat")
                        h_lng = hospital.get("lon")
                    else:
                        center = hospital.get("center", {})
                        h_lat = center.get("lat")
                        h_lng = center.get("lon")

                    if h_lat and h_lng:
                        osm_url = f"https://www.openstreetmap.org/directions?from={lat},{lng}&to={h_lat},{h_lng}"
                        print(f"✅ Found: {hospital_name}")
                        return {"hospital_name": hospital_name, "google_maps_url": osm_url}

            except Exception as e:
                print(f"⚠️ Endpoint {endpoint} failed: {e}")
                continue

    # ✅ Guaranteed fallback — direct OSM search link, always works
    print("⚠️ All Overpass endpoints failed — using OSM search fallback")
    fallback_url = f"https://www.openstreetmap.org/search?query=hospital#map=13/{lat}/{lng}"
    return {
        "hospital_name": "Nearest hospitals (tap to search)",
        "google_maps_url": fallback_url
    }
    

def start_emergency_bot(state:AgentState):
    # Initial coordination node, returns the state unchanged.
    return {}


def combine_results(state:AgentState):
    video_result = state.get("search_results", "No video found.")
    video_title = state.get("video_title", "No video found.")
    google_maps_url = state.get("google_maps_url", "")
    hospital_name = state.get("hospital_name", "No hospital information found.")
    if state.get("web_answer", ""):
        doc_answer = state["web_answer"]
    else:
        doc_answer = state.get("rag_answer", "")
    
    return {"final_result": [doc_answer, google_maps_url, hospital_name, video_result, video_title]}


def create_emergency_agent():
    graph = StateGraph(AgentState)

    # ✅ 1. ADD ALL NODES FIRST
    graph.add_node("start_emergency_bot", start_emergency_bot)
    graph.add_node("extract_keywords_youtube", extract_keywords_youtube)
    graph.add_node("search_youtube_videos", search_youtube_videos)
    graph.add_node("answer_from_rag", answer_from_rag)
    graph.add_node("web_search", web_search)
    graph.add_node("create_response_from_web_search", create_response_from_web_search)
    graph.add_node("extract_keywords_web_search", extract_keywords_web_search)
    graph.add_node("get_google_maps_url", get_google_maps_url)
    graph.add_node("combine_results", combine_results)

    # ✅ 2. SET ENTRY POINT
    graph.set_entry_point("start_emergency_bot")

    # ✅ 3. ADD EDGES

    # YouTube flow
    graph.add_edge("extract_keywords_youtube", "search_youtube_videos")

    graph.add_conditional_edges(
        "search_youtube_videos",
        should_continue_youtube,
        {
            "retry": "extract_keywords_youtube",
            "end": "combine_results",
        }
    )

    # RAG + Web fallback
    graph.add_conditional_edges(
        "answer_from_rag",
        should_web_search,
        {
            "web_search": "extract_keywords_web_search",
            "end": "combine_results",
        }
    )

    graph.add_edge("extract_keywords_web_search", "web_search")

    graph.add_conditional_edges(
        "web_search",
        should_continue_web_search,
        {
            "retry": "extract_keywords_web_search",
            "end": "create_response_from_web_search",
        }
    )

    graph.add_edge("create_response_from_web_search", "combine_results")

    # Location flow
    graph.add_edge("get_google_maps_url", "combine_results")

    # Start node branching
    graph.add_edge("start_emergency_bot", "extract_keywords_youtube")
    graph.add_edge("start_emergency_bot", "answer_from_rag")

    graph.add_conditional_edges(
        "start_emergency_bot",
        should_find_hospital,
       {
            "high_severity": "get_google_maps_url",
            "low_severity": "combine_results",
        }
    )

    # ✅ 4. FINISH
    graph.set_finish_point("combine_results")

    app = graph.compile()

    # Optional visualization
    #img_bytes = app.get_graph().draw_mermaid_png()
    #with open('presentation/agents/specialized.png', 'wb') as f:
        #f.write(img_bytes)

    return app