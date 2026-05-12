🩹 First-Aid Buddy

Your personal AI-powered first aid assistant — providing instant, location-aware, and multilingual emergency guidance.


📋 Overview
First-Aid Buddy is an intelligent first aid assistant built with Python and Streamlit. It combines large language models, retrieval-augmented generation (RAG), and real-time external APIs to guide users through medical emergencies step by step.
The system uses a two-agent architecture:

A Triage Agent that assesses the severity of the situation (scale of 1–5) based on the Emergency Severity Index (ESI)
An Emergency Agent that generates structured first aid instructions, finds the nearest hospital, and retrieves a relevant first aid video tutorial


✨ Features

🚑 Severity triage — assesses how serious a situation is before responding
📋 First aid guidance — grounded in the St John Ambulance First Aid Manual via RAG
🏥 Nearest hospital finder — uses GPS location and OpenStreetMap (Overpass API)
📹 First aid video tutorials — retrieves relevant videos from verified YouTube channels
🌍 Multilingual support — automatically detects and translates any input language
📞 Local emergency numbers — displays the correct emergency number for the user's country
💬 Multi-session chat — manage multiple independent conversation threads


🏗️ Architecture
User Input
    ↓
Translation Layer (safe_translate)
    ↓
Triage Agent (LangGraph)
  └── ESI Handbook RAG → Severity Score (1–5) or Clarifying Question
    ↓
Emergency Agent (LangGraph) — three parallel branches
  ├── RAG Answer       → St John Ambulance Manual
  ├── YouTube Search   → Verified first aid channels
  └── Hospital Finder  → Overpass API (severity > 2 only)
    ↓
Combined Response → Streamlit UI

🛠️ Tech Stack
CategoryTechnologyUI FrameworkStreamlitLLM InferenceGroq Cloud (llama-3.3-70b-versatile)Agent OrchestrationLangGraphVector SearchFAISS + HuggingFace (all-MiniLM-L6-v2)Keyword SearchBM25RetrieverPDF ProcessingPyPDFLoader, pdf2image, pytesseract (OCR)Prompt TemplatingJinja2Web Search FallbackGoogle Serper API + BeautifulSoupHospital LookupOverpass API (OpenStreetMap)Reverse GeocodingNominatim API (OpenStreetMap)Session LoggingGoogle Cloud Storage (optional)Browser Geolocationstreamlit-js-eval

📁 Project Structure
first-aid-buddy/
│
├── app.py                          # Main Streamlit application
│
├── src/
│   ├── triage_utils.py             # Triage agent (LangGraph)
│   ├── emergency_utils.py          # Emergency agent (LangGraph)
│   ├── utils.py                    # Shared utilities (LLM, translation, geocoding)
│   └── templates/
│       ├── emergency_prompt.jinja  # Prompt template for emergency agent
│       └── everyday_prompt.jinja   # Prompt template for low-severity cases
│
├── data/
│   ├── doc_triage/pdf/             # ESI Triage Handbook PDF
│   ├── doc_emergency/pdf/          # St John Ambulance First Aid Manual PDF
│   ├── bm_25/                      # Persisted BM25 indexes (.pkl)
│   ├── faiss/                      # Persisted FAISS indexes
│   └── sessions_history/           # Local session logs (if enabled)
│
├── presentation/
│   └── logo/                       # App logo assets
│
├── requirements.txt
└── README.md

⚙️ Installation
1. Clone the repository
bashgit clone https://github.com/your-username/first-aid-buddy.git
cd first-aid-buddy
2. Create and activate a virtual environment
bashpython -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
3. Install dependencies
bashpip install -r requirements.txt
4. Install Tesseract OCR (required for emergency PDF processing)

macOS: brew install tesseract
Ubuntu/Debian: sudo apt install tesseract-ocr
Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki

5. Configure API keys
Create a .streamlit/secrets.toml file in the project root:
toml[GROQ]
GROQ_API_KEY = "your_groq_api_key"

[YOUTUBE]
YOUTUBE_API_KEY = "your_youtube_data_api_key"

[SERPER]
SERPER_API_KEY = "your_serper_api_key"

[GCP]
BUCKET_NAME = "your_gcs_bucket_name"           # optional
SERVICE_ACCOUNT_KEY = "your_service_account_json"  # optional
6. Add knowledge base documents
Place the following PDFs in their respective directories:
data/doc_triage/pdf/esi_triage_handbook.pdf
data/doc_emergency/pdf/sja_first_aid_manual.pdf

Note: On first startup the system will automatically build and persist the BM25 and FAISS indexes from these documents. This may take several minutes, but only happens once.


🚀 Running the App
bashstreamlit run app.py
The app will open in your browser at http://localhost:8501.

🔑 API Keys Required
APIPurposeGet it atGroq CloudLLM inferencehttps://console.groq.comYouTube Data API v3Video searchhttps://console.cloud.google.comGoogle Serper APIWeb search fallbackhttps://serper.devGoogle Cloud StorageSession logging (optional)https://console.cloud.google.com

🌍 Supported Emergency Numbers
The system automatically detects the user's country via GPS and displays the correct local emergency number. Currently supported countries include Kenya, United States, United Kingdom, Australia, Canada, Germany, France, Italy, Spain, India, South Africa, Nigeria, Ghana, Uganda, Tanzania, and Ethiopia. All other countries fall back to the international standard 112.

⚠️ Disclaimer
First-Aid Buddy is an academic project and is intended as a supplementary tool only. It is not a substitute for professional medical advice, diagnosis, or treatment. In any life-threatening emergency, always call your local emergency services immediately.



🙏 Acknowledgements

St John Ambulance for the First Aid Manual
AHRQ for the ESI Triage Handbook
LangChain and LangGraph for the agent framework
Groq for fast LLM inference
OpenStreetMap for geolocation and hospital data
Streamlit for the UI framework