
import json
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import uuid
from datetime import datetime

class EpisodicMemory:
    """
    Stores past successful analysis runs (Episodes) in Qdrant.
    Allows retrieving similar past market situations.
    """
    def __init__(self, host="qdrant", port=6333, collection_name="market_episodes"):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            # Create collection if not exists
            # We assume a 768 dim vector (e.g. from Ollama/Nomic)
            # For simplicity in this Demo, we might simulate embeddings or use a lightweight model
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )

    def save_episode(self, ticker: str, summary: str, embedding: list):
        """Save a new episode."""
        point_id = str(uuid.uuid4())
        payload = {
            "ticker": ticker,
            "summary": summary,
            "date": datetime.now().isoformat()
        }
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=point_id, vector=embedding, payload=payload)]
        )

    def recall(self, query_embedding: list, limit: int = 3):
        """Retrieve relevant past episodes."""
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )
