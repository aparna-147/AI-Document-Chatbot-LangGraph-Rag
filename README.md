# 🤖 AI Document Assistant (LangGraph + RAG)

An AI-powered document chatbot built using **LangChain, LangGraph, Google Gemini, and Retrieval-Augmented Generation (RAG)**.
The application allows users to upload PDF documents and screenshots, then ask questions to receive accurate, context-aware answers from the uploaded content.

---

## ✨ Features

- 📄 Upload one or multiple PDF documents
- 🖼 Upload screenshots/images for analysis
- 💬 Interactive AI chat interface
- 🔍 Context-aware question answering using RAG
- 🧠 LangGraph workflow with conversation memory
- 📚 Semantic search using a vector database
- ⚡ Fast document retrieval with embeddings
- 💾 SQLite checkpointing for chat persistence
- 🔐 Secure API key management using `.env`
- 🎨 Clean and user-friendly Streamlit interface

---

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- LangGraph
- Google Gemini 2.5 Flash
- FAISS (Vector Database)
- HuggingFace Embeddings
- PyMuPDF (PDF Processing)
- Recursive Character Text Splitter
- SQLite
- Python Dotenv

---

## 📂 Project Structure

```text
AI-Document-Assistant/
│
├── app.py                  # Streamlit frontend
├── langgraph_backend.py    # LangGraph + RAG backend
├── requirements.txt        # Project dependencies
├── .env.example            # API key template
├── README.md
└── chatbot.db              # SQLite memory database
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone <repository-link>
cd AI-Document-Assistant
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Configure API Key

Create a `.env` file and add your Google Gemini API key.

```env
GOOGLE_API_KEY=your_api_key_here
```

### 4️⃣ Run the application

```bash
streamlit run app.py
```

## 🔄 Workflow

```text
User Uploads PDF / Screenshot
            │
            ▼
      Extract Text
            │
            ▼
     Create Embeddings
            │
            ▼
 Store in FAISS Vector DB
            │
            ▼
   User Asks Question
            │
            ▼
   LangGraph Workflow
            │
            ▼
 Retrieve Relevant Context
            │
            ▼
    Context Available?
      │           │
     Yes          No
      │           │
      ▼           ▼
 Generate      General
 Response      Response
      │           │
      └─────┬─────┘
            ▼
     Display Answer
```

## 🧠 Architecture

This project follows the **Retrieval-Augmented Generation (RAG)** architecture.

Instead of relying only on the LLM's built-in knowledge, the chatbot:

1. Extracts text from uploaded PDFs and screenshots.
2. Splits the text into smaller chunks.
3. Converts chunks into embeddings.
4. Stores embeddings in a FAISS vector database.
5. Retrieves the most relevant chunks for each user query.
6. Uses LangGraph to manage the conversation flow.
7. Generates accurate answers with Google Gemini.

This approach significantly improves response accuracy and reduces hallucinations.

---

## 📸 Application Preview

- Home Page
- PDF Upload
- Screenshot Upload
- Chat Interface
- AI Responses

## 🚀 Future Improvements

- Support DOCX, TXT, and Excel files
- Voice-based interaction
- Multi-user authentication
- Chat history export
- Cloud deployment (Render/AWS)
- Citation and source highlighting
- Streaming AI responses

