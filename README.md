🤖 Smart AI Agent: RAG & Web Search

This project is an AI Agent that answers user queries using internal knowledge (RAG) and real-time web search, with intelligent routing to choose the best information source.

✨ Key Features

Hybrid AI: Combines internal knowledge base (RAG) with live web search.

Web Search Control: Users can enable or disable web search.

Transparent Workflow: Logs routing decisions and info retrieval steps.

PDF Knowledge Upload: Users can upload PDFs to expand the agent’s knowledge.

Persistent Conversation Memory: Maintains context across turns.

📦 Project Structure
agentBot/
├── frontend/
│   ├── app.py                  # Streamlit app
│   ├── ui_components.py        # UI components
│   └── backend_api.py          # API communication
│
├── backend/
│   ├── main.py                 # FastAPI backend
│   ├── agent.py                # AI agent workflow
│   └── vectorstore.py          # RAG logic
│
├── requirements.txt            # Python dependencies
└── .env                        # API keys (not committed)

⚙️ Tech Stack

Python 3.9+

Frontend: Streamlit

Backend: FastAPI

AI Orchestration: LangGraph

LLMs & Tools: LangChain, Groq (Llama 3)

Embeddings: sentence-transformers/all-MiniLM-L6-v2

Vector Store: Pinecone

PDF Loader: PyPDFLoader

Web Search: Tavily API

🛠️ Setup & Installation
1. Clone the repo
git clone https://github.com/your-username/agentBot.git
cd agentBot

2. Create virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Create .env file
GROQ_API_KEY="your_groq_api_key_here"
PINECONE_API_KEY="your_pinecone_api_key_here"
PINECONE_ENVIRONMENT="your_pinecone_environment"
TAVILY_API_KEY="your_tavily_api_key"
FASTAPI_BASE_URL="http://localhost:8000"

🏃 Running the App
Backend (FastAPI)
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

Frontend (Streamlit)
cd ..
streamlit run frontend/app.py