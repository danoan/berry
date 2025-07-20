import bisect
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
import io
import json
import logging
import numpy as np
import os
from pathlib import Path
import queue
import sys
import tempfile
import threading
import time
from typing import Annotated, Any, Callable, List, Optional

from pydantic import BaseModel

import sounddevice as sd
from vosk import Model, KaldiRecognizer, SetLogLevel

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import httpx

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stderr)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("[%(levelname)s]: %(message)s"))
logger.addHandler(handler)

###########################
# Model
###########################

class Language(StrEnum):
    BrazilianPortuguese="pt-br"
    AmericanEnglish="en-us"
    FrenchOfFrance="fr-fr"
    Italian="it"

class ASRModelVariation(StrEnum):
    Small="small"
    Large="large"

class ASRModel(BaseModel):
    language: Language
    variation: ASRModelVariation
    path: Path


class StoppableThread:
    def __init__(self):
        self._stop_event = threading.Event()
        self._thread : threading.Thread | None = None

    def _worker(self):
        raise NotImplementedError()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        self._thread = None

    def start(self):
        if self._thread:
            raise RuntimeError("Thread is running already")

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def is_alive(self) -> bool:
        if not self._thread:
            return False
        return self._thread.is_alive()


class ThreadManager:
    def __init__(self):
        self._threads: dict[str,StoppableThread] = {}

    def register(self, id:str, thread:StoppableThread):
        if id in self._threads:
            logger.warning(f"Thread with id {id} exists already.")
            return
        self._threads[id] = thread

    def unregister(self, id:str):
        t = self.get(id)
        if t:
            if t.is_alive():
                logger.warning(f"Not possible to unregister thread {id} because it is still running")
                return
            del self._threads[id]

    def get(self, id:str) -> StoppableThread | None:
        if id not in self._threads:
            logger.warning(f"No thread with id {id} is registered.")
            return None
        return self._threads[id]


    def start(self, id:str):
        t = self.get(id)
        if t is None:
            return
        if t.is_alive():
            logger.warning(f"Thread {id} is current running.")
            return

        logger.info(f"Start thread {id}")
        t.start()

    def stop(self,id:str):
        t = self.get(id)
        if t is None:
            return
        if not t.is_alive():
            logger.warning(f"Thread {id} is not running.")
            return
        logger.info(f"Stop thread {id}")
        t.stop()

class StreamListener:
    def start(self):
        raise NotImplementedError()

    def stop(self):
        raise NotImplementedError()

    def transcript(self, data: dict[str,Any]):
        raise NotImplementedError()
    
    def transcript_partial(self, data: dict[str,Any]):
        raise NotImplementedError()

class ModelLoadListener:
    def start(self, asr_model: ASRModel):
        raise NotImplementedError()

    def finish(self, asr_model: ASRModel):
        raise NotImplementedError()

AudioConsumer = Callable[[queue.Queue],None]

class MicrophoneStream(StoppableThread):
    ID: str = "MicrophoneStream"
    def __init__(self, asr_model:ASRModel, audio_consumer:AudioConsumer, listener: StreamListener):
        super().__init__()
        self._asr_model = asr_model
        self._audio_consumer = audio_consumer
        self._listener = listener

    def _worker(self):
        q = queue.Queue()
        def stream_callback(indata, frames, time, status):
            q.put(bytes(indata))
        
        with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                               channels=1, callback=stream_callback):
            self._listener.start()
            while not self._stop_event.is_set():
                self._audio_consumer(q)
        self._listener.stop()

    def get_model(self) -> ASRModel:
        return self._asr_model

###########################
# Helpers
###########################

def get_microphone_listener_thread(asr_model: ASRModel, model_load_listener: ModelLoadListener, stream_listener: StreamListener) -> MicrophoneStream:
    model_load_listener.start(asr_model) 
    model = Model(str(asr_model.path))
    rec = KaldiRecognizer(model, 16000)
    model_load_listener.finish(asr_model) 

    def audio_consumer(q: queue.Queue):
        data = q.get()
        if rec.AcceptWaveform(data):
            stream_listener.transcript(json.loads(rec.Result()))
        else:
            stream_listener.transcript_partial(json.loads(rec.PartialResult()))

    return MicrophoneStream(asr_model,audio_consumer, stream_listener)

def get_model(language: Language, variation: ASRModelVariation) -> ASRModel:
    asr_model_repository=config.asr_models_folder
    model_path = Path(asr_model_repository) / language / variation
    return ASRModel(language=language, variation=variation, path=model_path) 

def create_file_data(file_path:Path, field_name:str):
    with open(file_path, "rb") as f:
        return (field_name, (f.name,f.read(), "application/octet-stream"))

def create_list_file_data(file_paths:list[Path], field_name:str):
    file_data = []
    for file_path in file_paths:
        file_data.append(create_file_data(file_path,field_name))
    return file_data


###########################
# Core
###########################

class MicrophoneStreamListener(StreamListener):
    def start(self):
        logger.info("Listening... Press Ctrl+C to stop")

    def stop(self):
        logger.info("Stop listening")

    def transcript(self, data: dict[str,Any]):
        logger.info(f"[Transcript]: {data.get('text','')}")

    def transcript_partial(self, data: dict[str,Any]):
        logger.info(f"[Partial]: {data.get('partial','')}")

class ASRModelLoadListener(ModelLoadListener):
    def start(self, asr_model: ASRModel):
        logger.info(f"Loading model {asr_model.language}-{asr_model.variation}")

    def finish(self, asr_model: ASRModel):
        logger.info(f"Finished loading model {asr_model.language}-{asr_model.variation}")

class Query(BaseModel):
    content: str

###########################
# Configuration
###########################

@dataclass
class Config:
    documents_folder: Path = Path(__file__).parent / "documents"
    asr_models_folder: Path = Path(__file__).parent / "asr-models" 
    embedding_endpoint: str = "http://localhost:8001"
    index_endpoint: str = "http://localhost:8002"
    selected_embedding: Optional[str] = None


###########################
# Main
###########################

@asynccontextmanager
async def lifespan(app: FastAPI):
    # select_model(str(Language.BrazilianPortuguese),str(ASRModelVariation.Small))
    config.documents_folder.mkdir(parents=True, exist_ok=True)
    yield
    # stop()

SetLogLevel(-1)
app = FastAPI(lifespan=lifespan)
TM = ThreadManager()
config = Config()

@app.get("/health")
def health():
    t = TM.get(MicrophoneStream.ID)
    if t is None:
        return "No model running"
    else:
        match t:
            case MicrophoneStream():
                return t.get_model()

@app.get("/select/model/{language_str}/{variation_str}")
def select_model(language_str: str, variation_str: str):
    try:
        language = Language(language_str)
    except ValueError:
        return f"Language not recognized: {language_str}"

    try:
        variation = ASRModelVariation(variation_str)
    except ValueError:
        return f"Variation not recognized: {variation_str}"
   
    asr_model = get_model(language,variation)
    if TM.get(MicrophoneStream.ID) is not None:
        TM.stop(MicrophoneStream.ID)
        TM.unregister(MicrophoneStream.ID)

    TM.register(MicrophoneStream.ID,get_microphone_listener_thread(asr_model, ASRModelLoadListener(), MicrophoneStreamListener()))
    TM.start(MicrophoneStream.ID)


@app.get("/stop")
def stop():
    TM.stop(MicrophoneStream.ID)
    TM.unregister(MicrophoneStream.ID)

#--------------------------
# Documents 
#--------------------------
async def _process_document(filepath: Path) -> list[str]:
    """
    Process document into ProcessedDocument objects.

    Returns a map indexed by the position of the first chunk of text belonging to 
    the document.
    """
    documents_data = [create_file_data(filepath,"document_file")]
    process_documents_url = f"{config.embedding_endpoint}/process_documents"

    async with httpx.AsyncClient() as client:
        response: httpx.Response  = await client.post(process_documents_url, files=documents_data)

        if response.status_code == 200:
            return response.json()  
        else:
            raise HTTPException(status_code=response.status_code, detail=response.content)


def _create_chunks(chunk_contents: list[str], chunks_folder: Path) -> None:
    chunks_folder.mkdir(parents=True,exist_ok=True)

    chunk_number = 0
    for chunk in chunk_contents:
        chunk_filename = f"{chunk_number}.txt"
        chunk_path = chunks_folder / chunk_filename
        with open(chunk_path,"w") as f:
            f.write(chunk)
        chunk_number+=1


@app.post("/documents/upload")
async def upload_document(file: UploadFile):
    if file and file.filename:
        document_filename = file.filename
        document_name = Path(file.filename).stem
        document_folder = config.documents_folder / document_name
        document_folder.mkdir(parents=True,exist_ok=True)

        logger.info("Uploading document")
        destination_path = document_folder / document_filename
        with open(destination_path, "wb") as f:
            f.write(await file.read())


        logger.info("Processing document")
        try:
            chunks_contents = await _process_document(destination_path)
        except HTTPException as ex:
            return JSONResponse({"origin":"process_document", "status_code": ex.status_code, "detail": ex.detail})

        logger.info(f"Storing chunks")
        chunks_folder = document_folder / "chunks"
        _create_chunks(chunks_contents,chunks_folder)

        return JSONResponse({"filename": file.filename})
    else:
        return JSONResponse({"origin":"documents/upload", "status_code": 500, "detail": "Error while uploading the file"})


@app.get("/documents/list")
def list_documents():
    return json.dumps( [f.name for f in config.documents_folder.iterdir() ] ,ensure_ascii=False)

@app.post("/documents/remove")
def remove_document(filename: Annotated[str, Form()]):
    document_full_path = config.documents_folder / filename
    if document_full_path.exists():
        os.remove(document_full_path)
        return JSONResponse({"filename":filename})
    else:
        return JSONResponse({"origin":"documents/remove", "status_code": 500, "detail": "Document does not exist"})

#--------------------------
# Vector Database 
#--------------------------
async def _create_embedding(chunks_data) -> dict[str,str]:
    """
    Create an embedding for a list of ProcessedDocument.

    The embedding is supposed to be stored in a vector database and the text chunks
    are recoverable using the embedding chunk index.
    """

    create_embedding_url = f"{config.embedding_endpoint}/create"
    async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
        response: httpx.Response  = await client.post(create_embedding_url, files=chunks_data)

        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.content)

@app.post("/index/create")
async def create_index(index_name:Annotated[str,Form()]):
    create_index_url = f"{config.index_endpoint}/create"

    request_data = {"index_name":index_name} 
    async with httpx.AsyncClient() as client:
        response: httpx.Response  = await client.post(create_index_url, data=request_data)

        if response.status_code == 200:
            return JSONResponse(response.json())
        else:
            return JSONResponse({"origin":"index/create", "status_code":500, "detail":"Error while creating index"})

@app.post("/index/remove")
async def remove_index(index_name:Annotated[str,Form()]):
    remove_index_url = f"{config.index_endpoint}/remove"

    request_data = {"index_name":index_name} 
    async with httpx.AsyncClient() as client:
        response: httpx.Response  = await client.post(remove_index_url, data=request_data)

        if response.status_code == 200:
            return JSONResponse(response.json())
        else:
            return JSONResponse({"origin":"index/remove", "status_code":500, "detail":"Error while removing index"})

@app.post("/index/add")
async def add_document(index_name:Annotated[str,Form()], document_name: Annotated[str,Form()]):
    """
    Add a previously uploaded document to the index.

    It first creates an embedding for the document content and then send the embedding
    to the index endpoint.
    """
    add_embedding_url = f"{config.index_endpoint}/add"

    document_folder = config.documents_folder / document_name
    chunks_folder = document_folder / "chunks"

    logger.info("Creating embedding")
    chunks_path = list(chunks_folder.iterdir())
    chunks_data = create_list_file_data(chunks_path,"chunks_files")
   
    logger.info("Saving embedding vectors")
    try:
        embedding_data: dict[str,str] = await _create_embedding(chunks_data)
    except HTTPException as ex:
        return JSONResponse({"origin":"_create_embedding", "status_code":ex.status_code, "detail":ex.detail})

    with tempfile.TemporaryDirectory() as temp_dir:
        sstream = io.StringIO(embedding_data["embedding"])
        embedding_vectors: np.ndarray = np.loadtxt(sstream)
        logger.info(embedding_vectors.shape)
        embedding_temp_filepath = Path(temp_dir) / f"{index_name}-{document_name}.npy"
        np.save(embedding_temp_filepath, embedding_vectors)

        embedding_file_data = [create_file_data(embedding_temp_filepath,"embedding_file")]
        request_data = {"index_name":index_name, "document_name": document_name} 
        async with httpx.AsyncClient() as client:
            response: httpx.Response  = await client.post(add_embedding_url, files=embedding_file_data, data=request_data)

            if response.status_code == 200:
                return JSONResponse(response.json())
            else:
                raise HTTPException(status_code=response.status_code, detail=response.content)

# TODO: Make LLM request (use llm-assistant)
# TODO: Output TTS
# TODO: Setup proper packages and add tests
# TODO: Use websockets (realtime api)
# TODO: Make possible to add documents to an index without recreating from scratch

#### Query ####
class DocumentProperties(BaseModel):
    index: int
    name: str
    total_chunks: int

class IndexProperties(BaseModel):
    index_name: str
    documents: List[DocumentProperties]

class QueryResponse(BaseModel):
    index_properties: IndexProperties
    distances: str
    neighbors: str

def _load_ndarray_from_string(ndarray_string:str) -> np.ndarray:
    sstream = io.StringIO(ndarray_string)
    return np.loadtxt(sstream)

def _build_document_vector_searcher(ip: IndexProperties) -> list[int]:
    """
    Build an auxiliary data structure to search the document that contains the given chunk.

    The data structure is a vector of chunk indexes representing ranges.
        v = [a,b,c]
        v[n] is the last chunk index of the nth document

    If chunk index k is:
        < a -> chunk belongs to first document
        < b -> chunk belongs to second document
    """
    total_chunks =0
    v = []
    for d in ip.documents:
        v.append( total_chunks + d.total_chunks )
        total_chunks+=d.total_chunks
    return v


def _get_document_index_from_chunk(chunk_index:int, ip: IndexProperties) -> tuple[int,int]:
    searcher = _build_document_vector_searcher(ip)
    document_index = bisect.bisect_left(searcher,chunk_index)
    if document_index>0:        
        relative_chunk_index = chunk_index - searcher[document_index-1]
    else:
        relative_chunk_index = chunk_index
    logger.info(chunk_index)
    logger.info(searcher)
    logger.info(f"Document Index: {document_index}, Relative Document Index: {relative_chunk_index}")
    return document_index, relative_chunk_index


def _get_chunk(chunk_index:int, ip:IndexProperties) -> str:
    document_index, relative_chunk_index = _get_document_index_from_chunk(chunk_index,ip)
    document = ip.documents[document_index]

    chunks_folder = config.documents_folder / document.name / "chunks"
    chunk_path = chunks_folder / f"{relative_chunk_index}.txt"
    with open(chunk_path) as f:
        return f.read()


@app.post("/query")
async def query(index_name:Annotated[str,Form()],query_string: Annotated[str,Form()]):
    query_url = f"{config.index_endpoint}/query"
    data = {"query_string": query_string, "index_name":index_name}
    async with httpx.AsyncClient() as client:
        response: httpx.Response  = await client.post(query_url, data=data)

        if response.status_code == 200:
            query_response = QueryResponse.model_validate(response.json())
            distances = _load_ndarray_from_string(query_response.distances)
            neighbors = _load_ndarray_from_string(query_response.neighbors)

            logger.info(query_response)

            chunk_index = int(neighbors[0])
            # TODO: Use this chunk to make a llm request
            chunk = _get_chunk(chunk_index,query_response.index_properties)
            return JSONResponse({"chunk":chunk,"distances":[str(d) for d in distances],"neighbors":[str(n) for n in neighbors]})
        else:
            return JSONResponse({"origin":"query", "status_code":response.status_code, "detail":response.content})

if __name__ == "__main__":
    default_asr_model = get_model(Language.BrazilianPortuguese,ASRModelVariation.Large)
    select_model("pt-br","large")
    time.sleep(5)
    select_model("en-us","large")
    time.sleep(5)
    TM.stop(MicrophoneStream.ID)

