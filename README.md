# HR RAG Chatbot

## Project Overview
This project is a Retrieval-Augmented Generation (RAG) chatbot that answers employee handbook questions.

## Technologies Used
- Python
- LangChain
- FAISS
- HuggingFace Embeddings
- OpenRouter (Llama 3.1 8B)

## Project Structure

```
Week2_RAG_Project/
│
├── data/
│   └── employee_handbook.pdf
├── src/
│   ├── ingest.py
│   └── chatbot.py
├── vectorstore/
├── screenshots/
└── .gitignore
```

## Features
- Loads an employee handbook PDF
- Splits the document into chunks
- Stores embeddings in FAISS
- Retrieves relevant information
- Answers employee questions using an LLM

## Example Questions
- What is the dress code?
- What are the working hours?
- Can employees work remotely?
- What are the employee benefits?