from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum
import io
import json
import logging
import numpy as np
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Annotated, Any, Callable, List, Optional

from pydantic import BaseModel

import sounddevice as sd
from vosk import Model, KaldiRecognizer, SetLogLevel

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import httpx

from embeddings import embedding

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

def create_file_data(file_paths:list[Path], field_name:str):
    file_data = []
    for embedding_path in file_paths:
        with open(embedding_path, "rb") as f:
            file_data.append(
                (field_name, (f.name,f.read(), "application/octet-stream"))
            )
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
    embeddings_folder: Path = Path(__file__).parent / "embeddings"
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
    config.embeddings_folder.mkdir(parents=True, exist_ok=True)
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
@app.post("/documents/upload")
async def upload_document(file: UploadFile):
    if file and file.filename:
        destination_path = config.documents_folder / file.filename
        with open(destination_path, "wb") as f:
            f.write(await file.read())
        return JSONResponse({"filename": file.filename})
    else:
        return HTTPException(status_code=500, detail="Error while uploading the file")


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
        raise HTTPException(status_code=404, detail="File does not exist")

#--------------------------
# Embeddings
#--------------------------
class ProcessedDocument(BaseModel):
    filename: str
    index: int
    chunks: list[str]

class DocumentChunkIndex(BaseModel):
    document_filename: str
    chunk_index_range: tuple[int,int] 

class EmbeddingChunkIndex(BaseModel):
    indexes: list[DocumentChunkIndex] = []

async def _process_documents(filenames: list[str]) -> list[ProcessedDocument]:
    """
    Process documents into ProcessedDocument objects.

    Returns a map indexed by the position of the first chunk of text belonging to 
    the document.
    """
    full_documents_path = [ config.documents_folder / filename for filename in filenames ]
    
    documents_data = create_file_data(full_documents_path,"documents_files")
    process_documents_url = f"{config.embedding_endpoint}/process_documents"

    async with httpx.AsyncClient() as client:
        response: httpx.Response  = await client.post(process_documents_url, files=documents_data)

        if response.status_code == 200:
            return [ ProcessedDocument(filename=filenames[i], index=i, chunks=e["chunks"]) for i,e in enumerate(response.json()) ]
        else:
            raise HTTPException(status_code=response.status_code, detail=response.content)


def _to_embedding_chunk_index(processed_documents: list[ProcessedDocument]) -> EmbeddingChunkIndex:
    """
    Map documents to chunk indexes.

    This index is later used to recover the chunks of data to send to a LLM.
    """
    eci = EmbeddingChunkIndex()
    total_chunks = 0
    for pd in processed_documents:
        processed_document = ProcessedDocument.model_validate(pd)
        first_chunk_index = total_chunks
        last_chunk_index = total_chunks + len(processed_document.chunks) - 1
        total_chunks += len(processed_document.chunks)
        eci.indexes.append(DocumentChunkIndex(document_filename=processed_document.filename,
                                              chunk_index_range=(first_chunk_index,last_chunk_index)))
    return eci

def _create_chunks(processed_documents: list[ProcessedDocument], chunks_folder: Path) -> None:
    chunks_folder.mkdir(parents=True,exist_ok=True)

    chunk_number = 0
    for pd in processed_documents:
        processed_document = ProcessedDocument.model_validate(pd)
        logger.info(f"Creating chunk files for {processed_document.filename}")
        for chunk in processed_document.chunks:
            chunk_filename = f"{chunk_number}.txt"
            chunk_path = chunks_folder / chunk_filename
            with open(chunk_path,"w") as f:
                f.write(chunk)
            chunk_number+=1


async def _create_embedding(chunks_data, embedding_name: str) -> dict[str,str]:
    """
    Create an embedding for a list of ProcessedDocument.

    The embedding is supposed to be stored in a vector database and the text chunks
    are recoverable using the embedding chunk index.
    """

    create_embedding_url = f"{config.embedding_endpoint}/create"
    request_data = {"embedding_name": embedding_name}
    async with httpx.AsyncClient() as client:
        response: httpx.Response  = await client.post(create_embedding_url, files=chunks_data, data=request_data)

        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.content)


@app.post("/embeddings/create")
async def create_embedding(comma_separated_list_filenames:Annotated[str,Form()], embedding_name:Annotated[str,Form()]):
    filenames: list[str] = comma_separated_list_filenames.split(";")

    new_embedding_folder = config.embeddings_folder / embedding_name
    new_embedding_folder.mkdir(parents=True,exist_ok=True)

    logger.info("Processing documents")
    processed_documents = await _process_documents(filenames)
    eci = _to_embedding_chunk_index(processed_documents)
    
    logger.info(f"Creating chunk index {embedding_name}.index.json")
    embedding_index_path = new_embedding_folder / f"{embedding_name}.index.json"
    with open(embedding_index_path,"w") as f:
        f.write(eci.model_dump_json())
    
    logger.info(f"Storing chunks")
    chunks_folder = new_embedding_folder / "chunks"
    _create_chunks(processed_documents,chunks_folder)

    logger.info("Creating embedding")
    chunks_path = list(chunks_folder.iterdir())
    chunks_data = create_file_data(chunks_path,"chunks_files")
   
    logger.info("Saving embedding vectors")
    embedding_data: dict[str,str] = await _create_embedding(chunks_data,embedding_name)
    sstream = io.StringIO(embedding_data["embedding"])
    embedding_vectors: np.ndarray = np.loadtxt(sstream)
    np.save(new_embedding_folder / f"{embedding_name}.npy", embedding_vectors)

    return JSONResponse({"embedding_name": embedding_name})



@app.get("/embeddings/list")
async def list_embeddings():
    list_embeddings_url = f"{config.embedding_endpoint}/list"
    async with httpx.AsyncClient() as client:
        response = await client.get(list_embeddings_url)
        
        if response.status_code == 200:
            return JSONResponse(response.json())
        else:
            raise HTTPException(status_code=response.status_code, detail=response.content)

@app.post("/embeddings/remove")
async def remove_embedding(embedding_name: Annotated[str, Form()]):
    # TODO: Currently I am storing the embedding both on the embedding server and in bentinho.
    # do not store in the embedding server.
    remove_embeddings_url = f"{config.embedding_endpoint}/remove"
    request_data = {"embedding_name": embedding_name}
    async with httpx.AsyncClient() as client:
        response = await client.post(remove_embeddings_url,data=request_data)

        to_remove_folder = config.embeddings_folder / embedding_name 
        if to_remove_folder not in config.embeddings_folder.iterdir():
            raise HTTPException(status_code=500, detail="The embedding does not exist")

        for c in (to_remove_folder / "chunks").iterdir():
            os.remove(c)
        (to_remove_folder / "chunks").rmdir()
        
        for f in to_remove_folder.iterdir():
            os.remove(f)
        to_remove_folder.rmdir()
        
        if response.status_code == 200:
            return JSONResponse(response.json())
        elif response.status_code == 404:
            return f"{response.status_code}:{response.content}"
        else:
            raise HTTPException(status_code=response.status_code, detail=response.content)

#--------------------------
# Vector Database 
#--------------------------
@app.post("/index/from_embeddings/create")
async def create_index_from_embeddings(comma_separated_list_embedding_names:Annotated[str,Form()]):
    create_embedding_url = f"{config.index_endpoint}/from_embeddings/create"
    embedding_full_path = []

    for name in comma_separated_list_embedding_names.split(";"):
        embedding_full_path.append( config.embeddings_folder / name / f"{name}.npy" )

    embedding_data = create_file_data(embedding_full_path,"embedding_files")
    async with httpx.AsyncClient() as client:
        response: httpx.Response  = await client.post(create_embedding_url, files=embedding_data)

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
class QueryResponse(BaseModel):
    embedding_names: List[str]
    distances: str
    neighbors: str

def _load_ndarray_from_string(ndarray_string:str) -> np.ndarray:
    sstream = io.StringIO(ndarray_string)
    return np.loadtxt(sstream)

def _get_chunk(chunk_pos:int, embedding_name:str) -> str:
    chunks_folder = config.embeddings_folder / embedding_name / "chunks"
    chunk_path = chunks_folder / f"{chunk_pos}.txt"
    with open(chunk_path) as f:
        return f.read()


@app.post("/query")
async def query(query_string: Annotated[str,Form()]):
    query_url = f"{config.index_endpoint}/query"
    data = {"query_string": query_string}
    async with httpx.AsyncClient() as client:
        response: httpx.Response  = await client.post(query_url, data=data)

        if response.status_code == 200:
            query_response = QueryResponse.model_validate(response.json())
            distances = _load_ndarray_from_string(query_response.distances)
            neighbors = _load_ndarray_from_string(query_response.neighbors)

            chunk_index = int(neighbors[0])
            # TODO: Index built from more than one embedding is not supported yet
            embedding_name = query_response.embedding_names[0]
            
            # TODO: Use this chunk to make a llm request
            chunk = _get_chunk(chunk_index,embedding_name)
            return JSONResponse({"chunk":chunk,"distances":[str(d) for d in distances],"neighbors":[str(n) for n in neighbors]})
        else:
            raise HTTPException(status_code=response.status_code, detail=response.content)

if __name__ == "__main__":
    default_asr_model = get_model(Language.BrazilianPortuguese,ASRModelVariation.Large)
    select_model("pt-br","large")
    time.sleep(5)
    select_model("en-us","large")
    time.sleep(5)
    TM.stop(MicrophoneStream.ID)

