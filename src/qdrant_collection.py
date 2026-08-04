# =================================================
# Codes are from MIRA
# =================================================

import json
from typing import Any, Dict, List
from uuid import uuid4

from dotenv import load_dotenv
from openai import OpenAI

# qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    InitFrom,
    MatchValue,
    PointStruct,
    VectorParams,
)
from tqdm import tqdm


class Qdrant_Collection:
    r"""A Qdrant vector database collection."""

    _EMBEDDING_PARAMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "jinaai/jina-embeddings-v3": 1024,
    }

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedding_client: Any,
        collection_name: str,
        embedding_model: str,
        distance: Distance = Distance.COSINE,
    ):
        self.qdrant_client = qdrant_client
        self.embedding_client = embedding_client
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.distance = distance

        vector_dim = self._EMBEDDING_PARAMS[embedding_model]

        if not self.qdrant_client.collection_exists(collection_name):
            print(
                f"Collection {collection_name} does not exist. Creating with shape {vector_dim} per vector..."
            )
            self.init_new(collection_name, vector_dim, distance)

    def init_new(
        self,
        collection_name: str,
        vector_dim: int,
        distance: Distance = Distance.COSINE,
    ):
        """
        Initialize a new collection.
        Args:
            collection_name (str): Name of the new collection to be created.
            vector_dim (int): Dimension of the vectors.
            distance (Distance): Distance metric to be used.
        """
        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_dim, distance=distance, on_disk=True  # use memmap
            ),
            hnsw_config=HnswConfigDiff(on_disk=True),
            on_disk_payload=True,  # use memmap for metadata as well
        )

    def add(self, points: List[PointStruct]):
        """Add points to the collection. Generic method - points need to be created with the correct method externally."""
        self.qdrant_client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query, query_filter: Filter, top_k: int = 300):
        """Search the collection with a query and filter. Generic method - query_filter needs to be created with the correct method externally. Returns a list of points."""
        query_embedding = self.embedding_client.encode(query)
        # query_embedding = to_numpy_list(self.embedding_client.encode(query))
        # print('query_embedding after:', type(query_embedding), query_embedding.shape if hasattr(query_embedding, "shape") else len(query_embedding))

        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=top_k,
        )
        return results


def create_procedures_points(data: List[Dict[str, Any]], collection: Qdrant_Collection):
    """Creates procedures points from data.

    Args:
        data (List[Dict[str, Any]]): Data to create procedures points from.

    Returns:
        List[PointStruct]: List of procedures points.
    """
    clean_texts = [code["long_title"] for code in data]
    embeddings = collection.embedding_client.encode(clean_texts)

    points = []
    for embedding, payload in zip(embeddings, data):  # preserved order
        point = PointStruct(
            id=str(uuid4()),
            vector=embedding,
            payload=payload,
        )
        points.append(point)

    return points


def batch_upsert_procedures(
    collection: Qdrant_Collection,
    data: List[Dict[str, Any]],
    max_batch_size: int = 100,
):
    """Upserts data in batches of max_batch_size.
    Args:
        collection (Qdrant_Collection): Qdrant collection to upsert data to.
        data (List[Dict[str, Any]]): Data to upsert.
        max_batch_size (int): Maximum batch size.

    Returns:
        None
    """
    for i in tqdm(range(0, len(data), max_batch_size)):
        batch = data[i : i + max_batch_size]
        procedures_points = create_procedures_points(batch, collection)
        collection.add(procedures_points)


def to_numpy_list(x):
    import numpy as np
    if 'torch' in str(type(x)):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, list):
        return x
    raise ValueError(f"Cannot convert embedding of type {type(x)} to list")
