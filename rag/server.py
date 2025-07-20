from contextlib import asynccontextmanager
from dataclasses import dataclass
import io
import logging
import numpy as np
import os
from pathlib import Path
import sys
from typing import Any, List, Optional
from typing_extensions import Annotated

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import httpx
from pydantic import BaseModel

import rag

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stderr)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("[%(levelname)s]: %(message)s"))
logger.addHandler(handler)

@dataclass
class Config:
    index_folder: Path = Path(__file__).parent / "index"
    embedding_endpoint: str = "http://localhost:8001"

config = Config()

@asynccontextmanager
async def lifespan(app: FastAPI):
    config.index_folder.mkdir(parents=True,exist_ok=True)
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    pass

class DocumentProperties(BaseModel):
    index: int
    name: str
    total_chunks: int

class IndexProperties(BaseModel):
    index_name: str
    documents: List[DocumentProperties]

@app.post("/add")
async def add_embedding(index_name: Annotated[str,Form()],embedding_file: UploadFile, document_name:Annotated[str,Form()]):
    index_main_folder = config.index_folder / index_name

    if not index_main_folder.exists():
        raise HTTPException(status_code=500,detail=f"The index {index_name} does not exist")

    with open(index_main_folder / "index_properties.json") as f:
        ip = IndexProperties.model_validate_json(f.read())

    if document_name in [d.name for d in ip.documents]:
        return JSONResponse(ip.model_dump_json())

    content = io.BytesIO(await embedding_file.read())
    embedding_vectors = rag.load_embedding_from_binary_file(content)
    shape = rag.add_to_index(index_main_folder / index_name,embedding_vectors)

    logger.info(f"Shape: {shape}")
    
    dp = DocumentProperties(
        index=len(ip.documents),
        name=document_name,
        total_chunks=shape[0]
    )
    ip.documents.append(dp)

    with open(index_main_folder / "index_properties.json", "w") as f:
        f.write(ip.model_dump_json(indent=2))

    return JSONResponse(ip.model_dump_json())

@app.post("/create")
async def create(index_name: Annotated[str,Form()]):
    new_index_folder = config.index_folder / index_name
    if new_index_folder.exists():
        return f"Index {index_name} exist already"
    
    new_index_folder.mkdir(parents=True,exist_ok=True)

    ip = IndexProperties(
        index_name=index_name,
        documents=[]
    )

    with open(new_index_folder / "index_properties.json", "w") as f:
        f.write(ip.model_dump_json(indent=2))

    return JSONResponse({"index_name":index_name})

@app.post("/remove")
async def remove(index_name: Annotated[str,Form()]):
    index_folder = config.index_folder / index_name

    if not index_folder.exists():
        return f"Index {index_name} does not exist"

    for f in index_folder.iterdir():
        os.remove(f)

    index_folder.rmdir()

    return f"Index {index_name} removed"

@app.get("/list")
def list_indexes():
    return JSONResponse([f.name for f in config.index_folder.iterdir()])


def _to_csv_string(X:np.ndarray) -> str:
    sstream = io.StringIO()
    np.savetxt(sstream,X)
    sstream.seek(0)
    return sstream.read()


@app.post("/query")
async def query(query_string:Annotated[str,Form()], index_name:Annotated[str,Form()]):
    index_folder = config.index_folder / index_name

    with open(index_folder / "index_properties.json") as f:
        ip = IndexProperties.model_validate_json(f.read())

    get_embedding_url = f"{config.embedding_endpoint}/get"
    data = {"text":query_string}
    async with httpx.AsyncClient() as client:
        response: httpx.Response  = await client.post(get_embedding_url,data=data)

        if response.status_code == 200:
            csv_string:str = response.json()
            sstream = io.StringIO(csv_string)
            embedding_vector = rag.load_embedding_from_text_file(sstream)
            logger.info(embedding_vector.shape)
            D,I = rag.query_index(index_folder / index_name,embedding_vector)

            query_response:dict[str,Any] = {
                "index_properties": ip.model_dump(),
                "distances":_to_csv_string(D),
                "neighbors":_to_csv_string(I)
            }
            logger.info(query_response)
            return JSONResponse(query_response)
        else:
            raise HTTPException(status_code=response.status_code, detail=response.content)

