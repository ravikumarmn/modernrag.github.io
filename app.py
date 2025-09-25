import asyncio
import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import time

from modernrag import retrieve_augment_generate

app = FastAPI(title="ModernRAG Demo")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query", response_class=HTMLResponse)
async def query(request: Request, query: str = Form(...)):
    start_time = time.time()
    
    # Retrieval phase
    retrieval_start = time.time()
    # In a real implementation, this would call the vector store
    retrieval_time = time.time() - retrieval_start
    
    # Generation phase
    generation_start = time.time()
    try:
        response = await retrieve_augment_generate(query)
    except Exception as e:
        response = f"Error: {str(e)}"
    generation_time = time.time() - generation_start
    
    total_time = time.time() - start_time
    
    return templates.TemplateResponse(
        "result.html", 
        {
            "request": request,
            "query": query,
            "response": response,
            "retrieval_time": f"{retrieval_time:.2f}",
            "generation_time": f"{generation_time:.2f}",
            "total_time": f"{total_time:.2f}"
        }
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
