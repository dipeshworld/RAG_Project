# 📄 Contextual RAG System for PDF Documents

A Retrieval-Augmented Generation (RAG) system that enhances document search by generating **context-aware chunks** using LLMs.

This project improves traditional semantic search by adding **LLM-generated contextual summaries** to each chunk, leading to better retrieval accuracy.

---

## 🚀 Overview

This system processes PDF documents and enables intelligent querying by:

1. Splitting documents into chunks  
2. Generating contextual summaries using an LLM  
3. Creating embeddings for semantic search  
4. Storing data in a vector database (ChromaDB)  
5. Retrieving the most relevant chunks for a query  

---

## 🧠 Architecture
PDFs → Chunking → Context Generation (LLM)
→ Embeddings → ChromaDB → Retriever → Query Results


---

## ✨ Features

- 📚 PDF ingestion using PyMuPDF  
- ✂️ Smart chunking with LangChain  
- 🧠 Context generation using OpenAI LLM  
- 🔍 Semantic similarity search  
- 🗂️ Persistent vector database (ChromaDB)  
- ⚡ Improved retrieval using contextual enrichment  

---

## 📂 Project Structure
.
├── main.py
├── sample_data/ # Input PDF files
├── RAG_System_db/ # Vector database storage
├── requirements.txt
└── README.md


---

## ⚙️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-username/contextual-rag-system.git
cd contextual-rag-system
2. Install Dependencies
pip install -r requirements.txt
3. Set OpenAI API Key
export OPENAI_API_KEY="your_api_key"

Or enter it when prompted.

▶️ Usage
Add your PDF files to:
./sample_data/
Run the script:
python main.py
The system will:
Process PDFs
Generate contextual chunks
Store embeddings
Run sample queries

🔑 Core Components
1. Context Generation

Each chunk is enriched using an LLM:

generate_chunk_context(org_doc, chunk)
Produces a short (3–4 line) contextual summary
Helps improve retrieval relevance
2. Document Processing
create_pdf_contextual_chunks(pdf)
Loads PDF
Splits into chunks
Adds contextual summaries
3. Vector Database
Uses ChromaDB
Stores embeddings for efficient similarity search
4. Retriever
sim_retriever = chroma_db.as_retriever(...)
Retrieves top-k relevant chunks based on query

🧪 Example Queries
What are the main components of a RAG model?
Explain Transformer encoder layers
How does positional encoding work?

🧩 Tech Stack
Component	Tool/Library
LLM	OpenAI (GPT-4o-mini)
Embeddings	text-embedding-3-small
Framework	LangChain
Vector Store	ChromaDB
PDF Loader	PyMuPDF

📌 Use Cases
Research paper analysis
Knowledge retrieval systems
AI-powered document assistants
Enterprise search solutions

⚠️ Limitations
Requires OpenAI API key
Processing large PDFs can be slow
Works best with text-based PDFs (not scanned images)
