import streamlit as st
import tempfile
import hashlib
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from src.triage_utils import create_triage_retriever, create_triage_agent, severity_to_color
from src.emergency_utils import create_emergency_retriever, create_emergency_agent
from streamlit_js_eval import streamlit_js_eval
import time
import base64
from datetime import datetime
from src.utils import (
    init_LLM,
    translate,
    get_language,
    get_sidebar,
    get_emergency_number,
    get_medical_class,
    load_template,
    create_session_filename,
    store_session_data,
    initialize_gcs_client,
    convert_image_to_base64,
)



st.set_page_config(page_title="First-Aid Buddy", page_icon="presentation/logo/logo.png", layout="wide", initial_sidebar_state="expanded")

STORE_SESSIONS_DATA_LOCALLY = False
STORE_SESSIONS_DATA_GCS = False




# Hash session ID using hashlib
if 'session_id' not in st.session_state:
    session_id = hashlib.sha256(str(datetime.now()).encode()).hexdigest()
    st.session_state.session_id = session_id
else:
    session_id = st.session_state.session_id

if 'chats' not in st.session_state:
    st.session_state.chats = {}

if 'active_chat_id' not in st.session_state:
    st.session_state.active_chat_id = None

if st.sidebar.checkbox("Use my current location", value=False):
    if 'user_location' not in st.session_state:
        st.sidebar.info("📍 Fetching location...")
        user_location_info = streamlit_js_eval(
            js_expressions="""
                new Promise((resolve) => {
                    if (!navigator.geolocation) { resolve(null); return; }
                    navigator.geolocation.getCurrentPosition(
                        (pos) => resolve({ lat: pos.coords.latitude, lng: pos.coords.longitude }),
                        ()    => resolve(null),
                        { enableHighAccuracy: true, timeout: 10000 }
                    );
                })
            """,
            key="geo_fetch"
        )
        if user_location_info is not None:
            lat = user_location_info.get('lat')
            lng = user_location_info.get('lng')
            if lat and lng:
                st.session_state['user_location'] = (lat, lng)
                st.rerun()
            else:
                st.sidebar.warning("⚠️ Could not get location. Check browser permissions.")

    user_location = st.session_state.get('user_location', (None, None))
    if user_location != (None, None):
        st.sidebar.success("📍 Location captured")
else:
    st.session_state.pop('user_location', None)
    st.session_state.pop('geo_fetch', None)  # clear cached JS result too
    user_location = (None, None)
    
language, detailed_location, country = get_language(user_location)

# Initialize the LLM with the Google API key from secrets
llm = init_LLM(API_KEY=st.secrets["GROQ"]["GROQ_API_KEY"])
YOUTUBE_API_KEY = st.secrets["YOUTUBE"]["YOUTUBE_API_KEY"]
llm_text_model_name ="llama-3.3-70b-versatile"
llm_audio_model_name = "whisper-large-v3"
file_path_triage = "data/doc_triage/pdf/esi_triage_handbook.pdf"
file_path_emergency = "data/doc_emergency/pdf/sja_first_aid_manual.pdf"
prompt_emergency_file_path = "src/templates/emergency_prompt.jinja"
prompt_everyday_file_path = "src/templates/everyday_prompt.jinja"
prompt_emergency = load_template(prompt_emergency_file_path)
prompt_everyday = load_template(prompt_everyday_file_path)
ensemble_retriever_emergency = None
ensemble_retriever_triage = None
triage_agent=None

# Function to create the retriever
@st.cache_resource
def load_triage_retriever(file_path, bm25_path, faiss_path):
    return create_triage_retriever(file_path, bm25_path, faiss_path)

# Function to create the retriever
@st.cache_resource
def load_emergency_retriever(file_path, bm25_path, faiss_path):
    return create_emergency_retriever(file_path, bm25_path, faiss_path)

# Function to create the triage agent
@st.cache_resource
def load_triage_agent():
    return create_triage_agent()

@st.cache_resource
def load_emergency_agent():
    return create_emergency_agent()


ensemble_retriever_triage = load_triage_retriever(file_path_triage, bm25_path="data/bm_25/bm25_triage_index.pkl", faiss_path="data/faiss/faiss_triage_index")
ensemble_retriever_emergency = load_emergency_retriever(file_path_emergency, bm25_path="data/bm_25/bm25_emergency_index.pkl", faiss_path="data/faiss/faiss_emergency_index")
triage_agent = load_triage_agent()
emergency_agent = load_emergency_agent()

def safe_translate(message):
    if not message or message.strip() == "":
        return message, "unknown"

    try:
        translated, lang = translate(
            llm=llm,
            llm_model_name=llm_text_model_name,
            message=message,
            target_language="English"
        )
        return translated, lang
    except Exception as e:
        print("Translation failed:", e)
        return message, "unknown"
    

# Main function
def main():
    detailed_location_new = "Unknown" if detailed_location is None else detailed_location
    st.sidebar.markdown(f"**Location details:** {detailed_location_new}") 
    st.sidebar.markdown("---")
    st.sidebar.subheader("💬 Chats")

    if st.sidebar.button("➕ New Chat"):
        new_id = hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:8]
        st.session_state.chats[new_id] = {
            "title": "New Chat",
            "history": [],
            "history_translated": []
        }
        st.session_state.active_chat_id = new_id
        st.rerun()
    else:
        new_id = None  

    for chat_id, chat_data in list(st.session_state.chats.items()):
        col1, col2 = st.sidebar.columns([4, 1])
        is_active = chat_id == st.session_state.active_chat_id
        label = f"**{chat_data['title']}**" if is_active else chat_data['title']
        if col1.button(label, key=f"chat_{chat_id}"):
            st.session_state.active_chat_id = chat_id
            st.rerun()
        if col2.button("🗑️", key=f"del_{chat_id}"):
            del st.session_state.chats[chat_id]
            if st.session_state.active_chat_id == chat_id:
                st.session_state.active_chat_id = None
            st.rerun()
            
    get_sidebar()
    if st.session_state.active_chat_id is None or \
        st.session_state.active_chat_id not in st.session_state.chats:
    
    # Logo
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with open("presentation/logo/logo.png", "rb") as f:
                encoded = base64.b64encode(f.read()).decode()
            st.markdown(f"""
            <div style='text-align:center;'>
            <img src='data:image/png;base64,{encoded}' width='120'>
            </div>
            <h1 style='text-align:center;'>First-Aid Buddy</h1>
            <p style='text-align:center; font-size:18px; color:gray;'>
            Your personal AI-powered first aid assistant
            </p>
            <hr>
            <p style='text-align:center; font-size:15px;'>
                🚑 Get instant guidance for medical emergencies<br><br>
                🏥 Find the nearest hospital based on your location<br><br>
                📹 Watch relevant first aid video tutorials<br><br>
            </p>
            <p style='text-align:center; color:gray; font-size:13px;'>
            Click <strong>➕ New Chat</strong> in the sidebar to get started.
            </p>
            """, unsafe_allow_html=True)
        return

    active_chat = st.session_state.chats[st.session_state.active_chat_id]

    st.title("First-Aid Buddy")

    # User query input
    query = ""
    image_base64 = ""
   

    query = st.chat_input("How can I help ?")
    

    

    # Default parameters values
    severity, hospital_name, google_maps_link, video_title, youtube_link = None, None, None, None, None
    
    if active_chat["history"]:
        for message in active_chat["history"]:
            if not isinstance(message, SystemMessage):
                if isinstance(message, HumanMessage):
                    role = "user"
                elif isinstance(message, AIMessage):
                    role = "assistant"
                else:
                    continue
                with st.chat_message(role):
                    st.markdown(message.content)
                
    if query :
        start_time = time.time()   # ✅ safe default
        end_time = time.time()     # ✅ safe default  
        response = ""              # ✅ safe default
        
        translated_query, source_language = safe_translate(query)

        
        
            
        if len(active_chat["history"]) == 0:
            active_chat["history"] = [HumanMessage(content=query)]
            active_chat["history_translated"] = [HumanMessage(content=translated_query)]
             # Auto-title from first message
            title = query[:30] + "..." if len(query) > 30 else query
            st.session_state.chats[st.session_state.active_chat_id]["title"] = title
        else:
            active_chat["history"].append(HumanMessage(content=query))
            active_chat["history_translated"].append(HumanMessage(content=translated_query))

        # Show the conversation history
        
        triage_agent_output = translated_query    
        
        with st.spinner("Assessing emergency severity..."):
            # Call the LLM with the Jinja prompt and DataFrame context
            with st.chat_message("assistant"):
                triage_input = {
                    "messages": active_chat["history_translated"],
                    "ensemble_retriever_triage": ensemble_retriever_triage,
                    "questions": []
                }

                print("INPUT TYPE:", type(triage_input))
                print("INPUT:", triage_input)
                start_time = time.time()
                output = triage_agent.invoke(triage_input)
                end_time = time.time()
                severity = output.get('severity', None)
                try:
                    severity = int(severity) if severity is not None else None
                except:
                    severity = None
                if severity:
                    color = severity_to_color[severity]
                    st.markdown(
                        f"<span style='font-size: 16px;'>"
                        f"<span style='display:inline-block;width:12px;height:12px;border-radius:50%;background:{color};margin-right:6px;vertical-align:middle;'></span>"
                        f"Emergency has <strong>severity {severity}</strong></span>",
                        unsafe_allow_html=True
                    )
                    response = severity
                    triage_agent_output = output.get('full_query', translated_query)
                else:
                    response = output.get('questions', ["No question"])[-1].content
                    response, _ = safe_translate(response)
                    st.markdown(response, unsafe_allow_html=True)    
                active_chat["history"].extend([AIMessage(content=str(response))])

        if severity:
            with st.spinner(
                ("The emergency agent is thinking to find a solution..." if severity > 2 else
                "The agent for common situations is thinking to find a solution...") 
            ):
                emergency_number = get_emergency_number(country)
                print(f"📞 Emergency number for {country}: {emergency_number}")
                # Call the LLM with the Jinja prompt and DataFrame context
                with st.chat_message("assistant"):
                    agent_input = {
                        "full_query": triage_agent_output,
                        "prompt": prompt_emergency_file_path if severity > 2 else prompt_everyday_file_path,
                        "severity": severity,
                        "messages": active_chat["history"][:-1], # ✅ USE THIS INSTEAD
                        "retry_count_youtube": 0,
                        "retry_count_web_search": 0,
                        "user_location": user_location,
                        "ensemble_retriever": ensemble_retriever_emergency,
                        "youtube_api_key": YOUTUBE_API_KEY,
                        "emergency_number": emergency_number,
                    
                    }
                    start_time = time.time()
                    result = emergency_agent.invoke(agent_input)
                    final_result = result.get('final_result', None)
                    if not final_result or len(final_result) < 5:
                        print("⚠️ final_result missing or incomplete:", final_result)
                        response, google_maps_link, hospital_name, youtube_link, video_title = "No response available.", None, None, None, None
                    else:
                        response, google_maps_link, hospital_name, youtube_link, video_title = final_result
                    end_time = time.time()

                    # Initialize an empty string to store the full response as it is built
                    response, _ = safe_translate(response)
                    response = response or "No response available."
                    st.markdown(response.replace("\\n", "\n"), unsafe_allow_html=True)

                    if severity > 2:
                        if hospital_name and google_maps_link and 'https' in str(google_maps_link):
                            st.markdown(
                                f"### Nearest hospital: **{hospital_name}**" 
                            )
                            st.markdown(
                                f"[🗺️ Get Directions (OpenStreetMap)]({google_maps_link})" 
                            )
                        else:
                            st.info(
                                "⚠️ Enable your location in the sidebar to find the nearest hospital."
                                
                            )

                    st.markdown("---")
                    st.markdown("## 📹 Related First Aid Video")

                    if youtube_link and 'https' in youtube_link:
                        if "watch?v=" in youtube_link:
                            # ✅ Direct video found — embed it
                            video_url = youtube_link.replace("watch?v=", "embed/")
                            st.markdown(f"### {video_title}")
                            youtube_embed = f'''
                              <iframe 
                                  width="560" height="315" 
                                  src="{video_url}" 
                                  frameborder="0" 
                                  allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" 
                                  allowfullscreen>
                                </iframe>
                            '''
                            st.markdown(youtube_embed, unsafe_allow_html=True)

                        elif "results?search_query=" in youtube_link:
                            # ✅ Fallback search link — no embed possible, show clickable link
                            st.markdown(f"No specific video found, but here are relevant results:")
                            st.markdown(f"🔗 [{video_title}]({youtube_link})")
                    else:
                        # ✅ Absolute last resort — build a search from full_query
                        fallback = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}+first+aid"
                        st.markdown(f"🔗 [Search YouTube for related first aid videos]({fallback})")
                active_chat["history"].extend([AIMessage(content=str(response))])
        

        # Save session data either locally or to GCS, if enabled
        if STORE_SESSIONS_DATA_LOCALLY or STORE_SESSIONS_DATA_GCS:
            response_time = end_time - start_time
            session_filename = create_session_filename(session_id)
            local_path_name = "data/sessions_history" if STORE_SESSIONS_DATA_LOCALLY else None
            bucket_name = st.secrets["GCP"]["BUCKET_NAME"] if STORE_SESSIONS_DATA_GCS else None
            gcs_client = initialize_gcs_client(SERVICE_ACCOUNT_KEY=st.secrets["GCP"]["SERVICE_ACCOUNT_KEY"]) if STORE_SESSIONS_DATA_GCS else None
            store_session_data(
                session_id=session_id, 
                user_location=user_location,
                country=country, 
                medical_class=get_medical_class(llm=llm, llm_model_name=llm_text_model_name,chat_history=active_chat["history"]), 
                severity=severity,
                hospital_details=[hospital_name, google_maps_link],
                youtube_video_details=[video_title, youtube_link],
                query=query, response=response, response_time=response_time,
                session_filename=session_filename,
                local_path_name=local_path_name,
                bucket_name=bucket_name, client=gcs_client
            )


if __name__ == "__main__":
    main()