RAG Project Structure:
 
medical-rag/

│

├── app.py              # Streamlit frontend

├── ingest.py           # Data ingestion + FAISS index creation

├── rag_pipeline.py     # RAG pipeline definition

├── main.py             # FastAPI backend

├── requirements.txt    # Python dependencies

├── Dockerfile          # Docker build file

├── README.md

└── .env

├── .github/

│   └── workflows/

│       └── ci-cd.yml   # GitHub Actions pipeline

└── data/

    ├── chatdoctor5k.json

    └── format_dataset.csv
  
