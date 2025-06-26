
import numpy as np
import pytest

from enhanced_rag_csd.retrieval.faiss_vectordb import FaissVectorDB


@pytest.fixture
def db():
    return FaissVectorDB(dimension=4)


def test_add_documents(db):
    embeddings = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
    documents = ["doc1", "doc2"]
    metadata = [{"id": 1}, {"id": 2}]

    db.add_documents(embeddings, documents, metadata)

    assert db.get_statistics()["total_vectors"] == 2


def test_search(db):
    embeddings = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
    documents = ["doc1", "doc2"]
    metadata = [{"id": 1}, {"id": 2}]
    db.add_documents(embeddings, documents, metadata)

    query = np.array([1, 2, 3, 5], dtype=np.float32)
    results = db.search(query, top_k=1)

    assert len(results) == 1
    assert results[0]["chunk"] == "doc1"
