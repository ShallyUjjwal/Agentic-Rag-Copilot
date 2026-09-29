# HR Policy Copilot

### Agentic RAG-based Employee Support Platform

An AI-powered employee support assistant that answers HR policy questions using a private company knowledge base. The system evaluates retrieved evidence and dynamically decides whether to answer, search the web, or retry retrieval.

## What Is It?

HR Policy Copilot allows employees to ask questions about:

* Leave and attendance
* Benefits
* Remote work
* Expenses
* Company policies
* Employee support

Instead of relying on a simple chatbot, it uses **Agentic RAG** to retrieve and evaluate information before generating an answer.

## How It Works

```text
Employee Question
       ↓
     Router
       ↓
HR / Policy Question?
   ↓           ↓
  Yes          No
   ↓           ↓
Pinecone     Direct
Retrieval    Answer
   ↓
Evidence Grading
   ↓
 ┌───────┴────────┐
Good              Weak
 ↓                  ↓
Answer          Tavily Search
                    ↓
               Grade Evidence
                    ↓
             Retry / Answer
```

The system follows a **private knowledge-first** approach. Company documents are searched before external web sources are considered.

## Technologies

| Technology                  | Purpose                                |
| --------------------------- | -------------------------------------- |
| **Python**                  | Application development                |
| **FastAPI**                 | Backend API                            |
| **LangGraph**               | Agent workflow and orchestration       |
| **Groq**                    | LLM inference                          |
| **Hugging Face**            | Text embeddings                        |
| **Pinecone**                | Vector database and semantic retrieval |
| **Tavily**                  | Web-search fallback                    |
| **LangChain**               | RAG and document processing            |
| **HTML / CSS / JavaScript** | Frontend                               |
| **SQLite**                  | Audit logging                          |
| **LangSmith**               | AI tracing and observability           |

## Core AI Concepts

* **RAG** — retrieves relevant documents before generating answers.
* **Agentic RAG** — dynamically decides the next action based on evidence quality.
* **Evidence Grading** — evaluates whether retrieved information is sufficient.
* **Query Rewriting** — improves queries when retrieval is unsuccessful.
* **Web Fallback** — searches external information when the private knowledge base is insufficient.
* **Grounded Generation** — generates answers using retrieved evidence.

## Architecture

```text
Frontend
   ↓
FastAPI
   ↓
LangGraph Agent
   ├── Groq → Reasoning & Generation
   ├── Pinecone → Private Knowledge Retrieval
   ├── Hugging Face → Embeddings
   └── Tavily → Web Search Fallback
```

## Project

**HR Policy Copilot**
Product × Data × AI
