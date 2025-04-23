from tqdm import tqdm
import numpy as np
import safetensors.torch
import torch
from pathlib import Path
from typing import List, Dict, Tuple
import faiss
import json

def load_all_image_embeddings(embedding_dir: str):
    # Collect all embedding files
    embeddings_path = Path(embedding_dir)
    files = sorted(embeddings_path.glob("*.safetensors"))
    if not files:
        print(f"No .safetensors files found in {embeddings_path}")
        return

    # Load embeddings into array
    vectors = []
    keys = []
    for f in tqdm(files, desc="Loading embeddings"):
        data = safetensors.torch.load_file(str(f))
        emb = data.get("embedding")
        # ensure CPU numpy float32
        emb_np = emb.to(torch.float32).cpu().numpy()
        # flatten if necessary
        if emb_np.ndim > 1:
            emb_np = emb_np.reshape(-1, emb_np.shape[-1])
        # if multiple vectors per file, handle each row separately
        for vec in emb_np:
            vectors.append(vec)
            keys.append(f.stem)

    all_vectors = np.vstack(vectors).astype(np.float32)
    d = all_vectors.shape[1]
    print(f"Loaded {all_vectors.shape[0]} vectors of dimension {d}")
    return all_vectors


def load_query_embedding(query_embedding_dir: str, question_id: str) -> torch.Tensor:
    """
    Load the embedding for a single question.
    """
    path = Path(query_embedding_dir) / f"{question_id}.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"No embedding for '{question_id}' in {query_embedding_dir}")
    
    data = safetensors.torch.load_file(str(path))
    emb = data.get("embeddings")
    if emb is None:
        raise KeyError(f"'embeddings' key not found in {path.name}")
    
    return emb.to(torch.float32).cpu()


def load_multiple_query_embeddings(
    query_embedding_dir: str,
    question_ids: List[str],
    skip_missing: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Load embeddings for a list of question_ids from `query_dir`.
    
    Args:
        query_dir:     Path to your `.../{split}/queries_embedding` folder.
        question_ids:  List of question_id stems (e.g. ["q1","q42","q99"]).
        skip_missing:  If True, silently skip IDs whose file isn’t found.
    
    Returns:
        A dict mapping each question_id → Tensor of shape [seq_len, emb_dim].
    """
    embeddings: Dict[str, torch.Tensor] = {}
    for qid in question_ids:
        try:
            embeddings[qid] = load_query_embedding(query_embedding_dir, qid)
        except FileNotFoundError:
            if skip_missing:
                continue
            else:
                raise
    return embeddings


def load_index(index_dir: str) -> Tuple[faiss.Index, List[str]]:
    """
    Load FAISS index and keys from the given directory.

    Args:
        index_dir: Path to directory containing 'image_embeddings.index' and 'keys.json'
    Returns:
        index: FAISS index instance
        keys: List of image ID strings
    """
    index_path = index_dir + "/image_embeddings.index"
    keys_path = index_dir + "/keys.json"

    print(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(str(index_path))

    print(f"Loading keys from {keys_path}")
    with open(keys_path, "r") as f:
        keys = json.load(f)

    return index, keys

