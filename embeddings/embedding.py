from argparse import ArgumentParser
from io import FileIO, TextIOBase
from pathlib import Path
import sys
from typing import Optional, Literal, TextIO, Union

import numpy as np
from pydantic import BaseModel
from torch.serialization import save
from transformers import AutoTokenizer, AutoModel
import torch

###########################
# Helpers
###########################

def _split_text(text: str, max_length: int=512, overlap:int =50) -> list[str]:
    """
    Split a text in overlapping chunks suited for embedding creation.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_length - overlap):
        chunk = " ".join(words[i:i+max_length])
        chunks.append(chunk)
    return chunks

def _mean_pool(last_hidden_state, mask):
    mask_exp = mask.unsqueeze(-1).expand(last_hidden_state.size())
    return (last_hidden_state * mask_exp).sum(1) / mask_exp.sum(1)

def _embed_texts(texts: list[str]):
    """
    For each string, creates an embedding in a 768-dimensional space.
    """
    # Load model and tokenizer
    model_name = "BAAI/bge-base-en-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_embeddings = []
    for i in range(0, len(texts), 16):  # batch size = 16
        batch = texts[i:i+16]
        # For BGE, prepend "passage: "
        batch = ["passage: " + t for t in batch]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        pooled = _mean_pool(outputs.last_hidden_state, inputs['attention_mask'])
        batch_embeddings.append(pooled.cpu())
    return torch.cat(batch_embeddings, dim=0)

def _l2_normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

###########################
# API
###########################

def process_documents(document_content: str) -> list[str]:
    """
    Split documents into a series of chunks.

    Returns a list of ProcessedDocuments, which consists of a list 
    of chunks of text. Each chunk is supposed to be later embedded 
    into a vector space.
    """
    return _split_text(document_content)

def write_embedding(embedding: np.ndarray, output_path:Union[Path,FileIO,TextIO], mode:Literal["text","binary"]="binary"):
    """
    Write embedding into a file or output it on stdout.
    """
    if mode=="binary":
        if not isinstance(output_path,TextIO):
            np.save(output_path,embedding)
    else:
        np.savetxt(output_path,embedding)

def create_embedding(documents: list[str]) -> np.ndarray:
    """
    Create an embedding from a list of documents.
    """
    embeddings = _embed_texts(documents)
    return _l2_normalize(embeddings.numpy())  # shape: (num_chunks, 768)

###########################
# Parser
###########################

def _write_embedding(documents_paths: list[Path], text_mode:bool=False, save_as_numpy_binary:Optional[Path]=None):
    documents_contents = []
    
    if text_mode:
        documents_contents.append(documents_paths[0])
    else:
        for p in documents_paths:
            with open(p) as f:
                documents_contents.append(f.read())

    embedding = create_embedding(documents_contents)
    if save_as_numpy_binary:
        write_embedding(embedding,save_as_numpy_binary,mode="binary")
    else:
        write_embedding(embedding,sys.stdout,mode="text")

if __name__ == "__main__":
    description = create_embedding.__doc__
    parser = ArgumentParser("ce",description=description)

    parser.add_argument("documents_paths",nargs="+",type=Path, help="Path to one or more text documents")
    parser.add_argument("--text-mode", action="store_true", help="If passed, only the first document is processed and it is interpreted as the document content instead of its path")
    parser.add_argument("--save-as-numpy-binary", type=Path, help="Path to the numpy binary file")


    _write_embedding(**vars(parser.parse_args()))

