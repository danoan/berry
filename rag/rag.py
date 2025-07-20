from argparse import ArgumentParser
from io import BytesIO
from pathlib import Path
import sys
from typing import Optional, TextIO, Tuple, Union

import faiss
import numpy as np

###################
# API
###################

def write_to_index(index_path: Path, embedding: np.ndarray):
    """
    Write embeddings.
    """
    # embeddings = np.load("../embeddings/embedding.npy")

    index = faiss.IndexFlatIP(768)  # Use IndexFlatL2 if no normalization
    index.add(embedding.astype(np.float32))  # embeddings must be float32

    # Save to disk
    faiss.write_index(index, str(index_path))


def load_embedding_from_binary_file(from_file: Union[Path,BytesIO]) -> np.ndarray:
    return np.load(from_file)

def load_embedding_from_text_file(from_file: Union[Path,TextIO]) -> np.ndarray:
    return np.loadtxt(from_file)

def query_index(index_path: Path, query_embedding: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    index = faiss.read_index(str(index_path))
    D,I= index.search(query_embedding.astype(np.float32), k=5)
    return (D,I)

###################
# Parser
###################

def write_parser(subparser):
    command_name = "write"
    description = write_to_index.__doc__
    help = description.split(".")[0] if description else ""

    parser : ArgumentParser = subparser.add_parser(command_name,description=description,help=help)
    parser.add_argument("index_path", type=Path, help="Path to FAISS index")

    input_format = parser.add_mutually_exclusive_group(required=True)
    input_format.add_argument("--from-file", type=Path, help="Write from numpy binary file")
    input_format.add_argument("--from-stdin",action="store_true", help="Write from numpy text format")

    def pre_call(index_path: Path, from_file: Optional[Path]=None,from_stdin: Optional[bool]=None,**kwargs):
        if from_stdin:
            data = np.loadtxt(sys.stdin.read())
        elif from_file:
            data = load_embedding_from_binary_file(from_file)

        else:
            raise RuntimeError("Data could not be read")

        return write_to_index(index_path, data)

    parser.set_defaults(func=pre_call,subcommand_help=parser.print_help)
    return parser

def query_parser(subparser):
    command_name = "query"
    description = write_to_index.__doc__
    help = description.split(".")[0] if description else ""

    parser : ArgumentParser = subparser.add_parser(command_name,description=description,help=help)
    parser.add_argument("index_path", type=Path, help="Path to FAISS index")

    input_format = parser.add_mutually_exclusive_group(required=True)
    input_format.add_argument("--from-file", type=Path, help="Query using embedding from numpy binary file")
    input_format.add_argument("--from-stdin",action="store_true", help="Query using embedding from numpy text format")

    def pre_call(index_path: Path, from_file: Optional[Path]=None,from_stdin: Optional[bool]=None,**kwargs):
        if from_stdin:
            data = np.loadtxt(sys.stdin.read())
        elif from_file:
            data = np.load(from_file)

        else:
            raise RuntimeError("Data could not be read")

        return query_index(index_path, data)

    parser.set_defaults(func=pre_call,subcommand_help=parser.print_help)
    return parser

###################
# Main 
###################

if __name__ == "__main__":
    description = "Query and write embeddings"
    parser = ArgumentParser("qe",description=description)

    subparser = parser.add_subparsers()
    query_parser(subparser)
    write_parser(subparser)

    args = parser.parse_args()
    if "func" in args:
        args.func(**vars(args))
    elif "subcommand_help" in args:
        args.subcommand_help()
    else:
        parser.print_help()

