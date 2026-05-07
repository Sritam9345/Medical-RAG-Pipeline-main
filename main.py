from fastapi import FastAPI, Query

# import from local python files

from rag_pipeline import load_pipeline, ask_question
 
app = FastAPI(title = "Medical RAG API")
 
chain = load_pipeline()
 
@app.post("/chat")

def chat(query: str = Query(..., description = "User query")):

    result = ask_question(chain, query)

    return result
 
