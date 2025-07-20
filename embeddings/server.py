from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import StringIO
import logging
import os
from pathlib import Path
import sys
from typing import Annotated

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

import embedding

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stderr)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("[%(levelname)s]: %(message)s"))
logger.addHandler(handler)

@dataclass
class Config:
    embeddings_folder: Path = Path(__file__).parent / "embeddings"

@asynccontextmanager
async def lifespan(app: FastAPI):
    config.embeddings_folder.mkdir(parents=True,exist_ok=True)
    yield

app = FastAPI(lifespan=lifespan)
config = Config()

@app.get("/health")
def health():
    return "embedding-server: OK"

@app.post("/process_documents")
async def process_documents(documents_files:list[UploadFile]):
    documents_contents = []
    for f in documents_files:
        documents_contents.append( (await f.read()).decode("utf-8") )

    processed_documents = [
            pd.model_dump() for pd in embedding.process_documents(documents_contents)
    ]
    logger.info(processed_documents)
    return JSONResponse(processed_documents)

@app.post("/create")
async def create_embedding(chunks_files:list[UploadFile], embedding_name:Annotated[str, Form()]):
    if len(embedding_name.split(".")) == 1:
        embedding_name += ".npy"
    
    documents_contents = []
    for f in chunks_files:
        documents_contents.append( (await f.read()).decode("utf-8") )

    embedding_full_path = config.embeddings_folder / embedding_name
    embedding_vectors = embedding.create_embedding(documents_contents)
    embedding.write_embedding(embedding_vectors,embedding_full_path)


    sstream = StringIO()
    embedding.write_embedding(embedding_vectors,sstream,"text")
    sstream.seek(0)
    embedding_vectors_csv_string = sstream.read()
    return JSONResponse({"embedding_name":embedding_name,"embedding":embedding_vectors_csv_string})

@app.post("/get")
async def get_embedding(text:Annotated[str,Form()]):
    embedding_vector = embedding.create_embedding([text])

    sstream = StringIO()
    embedding.write_embedding(embedding_vector,sstream,"text")
    sstream.seek(0)
    embedding_vectors_csv_string = sstream.read()
    return JSONResponse(embedding_vectors_csv_string)

@app.get("/list")
def list_embeddings():
    return JSONResponse( [f.name for f in config.embeddings_folder.iterdir() ] )

@app.post("/remove")
def remove_embedding(embedding_name: Annotated[str, Form()]):
    if len(embedding_name.split(".")) == 1:
        embedding_name += ".npy"

    embedding_full_path = config.embeddings_folder / embedding_name
    logger.info(embedding_full_path)
    if embedding_full_path.exists():
        os.remove(embedding_full_path)
        return JSONResponse({"embedding_name":embedding_name})
    else:
        raise HTTPException(status_code=404, detail="Embedding does not exist")

