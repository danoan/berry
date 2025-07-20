from contextlib import asynccontextmanager
from dataclasses import dataclass
import datetime
import io
import json
import logging
import numpy as np
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

class IndexProperties(BaseModel):
    index_name: str
    embeddings: List[str]

@app.post("/from_embeddings/create")
async def create_from_embeddings(embedding_files: List[UploadFile]):
    index_name = datetime.datetime.now().strftime("%Y-%m-%d")
    new_index_folder = config.index_folder / index_name
    new_index_folder.mkdir(parents=True,exist_ok=True)
    for embedding in embedding_files:
        content = io.BytesIO(await embedding.read())
        rag.write_to_index(new_index_folder / index_name,rag.load_embedding_from_binary_file(content))

    filenames = []
    for e in embedding_files:
        if e.filename:
            filenames.append(Path(e.filename).stem)

    ip = IndexProperties(
        index_name=index_name,
        embeddings=filenames
    )

    with open(new_index_folder / "index_properties.json", "w") as f:
        f.write(ip.model_dump_json(indent=2))

    return JSONResponse({"index_name":index_name})

@app.get("/list")
def list_indexes():
    return JSONResponse([f.name for f in config.index_folder.iterdir()])


def _to_csv_string(X:np.ndarray) -> str:
    sstream = io.StringIO()
    np.savetxt(sstream,X)
    sstream.seek(0)
    return sstream.read()


@app.post("/query")
async def query(query_string:Annotated[str,Form()], index:Optional[str]=Form(None)):
    if index is None:
        index_folder = sorted(list(config.index_folder.iterdir()))[-1]
        index_name = index_folder.name
    else:
        index_folder = config.index_folder / index
        index_name = index

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
            embedding_matrix = embedding_vector.reshape(1,embedding_vector.shape[0])
            D,I = rag.query_index(index_folder / index_name,embedding_matrix)

            query_response:dict[str,Any] = {
                "embedding_names": ip.embeddings,
                "distances":_to_csv_string(D),
                "neighbors":_to_csv_string(I)
            }
            return JSONResponse(query_response)
        else:
            raise HTTPException(status_code=response.status_code, detail=response.content)

