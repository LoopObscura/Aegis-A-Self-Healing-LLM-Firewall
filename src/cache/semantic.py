from typing import Optional, List, Dict, Any
from loguru import logger
from openai import AsyncOpenAI
from qdrant_client.async_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import json
import time
from config.settings import settings


class SemanticCache:
    """
    SemanticCache leverages vector embeddings for semantic similarity matching.
    Cached responses are retrieved based on prompt vector similarity to previously seen prompts.
    """

    def __init__(self) -> None:
        """Initialize semantic cache with Qdrant client and OpenAI embedding service."""
        self.openai_client: AsyncOpenAI = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.qdrant_client: AsyncQdrantClient = AsyncQdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None,
        )
        self.collection_name: str = settings.VECTOR_COLLECTION_NAME
        self.embedding_model: str = settings.EMBEDDING_MODEL
        self.vector_dimension: int = settings.VECTOR_DIMENSION
        self.point_counter: int = 0

        logger.info(
            "SemanticCache initialized: collection={}, model={}, dimension={}",
            self.collection_name,
            self.embedding_model,
            self.vector_dimension
        )

    async def _initialize_collection(self) -> None:
        """Ensure the Qdrant collection exists with proper configuration."""
        try:
            collections = await self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                logger.info("Creating new Qdrant collection: {}", self.collection_name)
                await self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_dimension, distance=Distance.COSINE),
                )
                logger.info("Collection {} created successfully", self.collection_name)
            else:
                logger.debug("Collection {} already exists", self.collection_name)

        except Exception as e:
            logger.error("Error initializing Qdrant collection: {}", e)
            raise

    async def _get_embedding(self, text: str) -> List[float]:
        """
        Generate a vector embedding for the given text using OpenAI.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        try:
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text,
                timeout=settings.REQUEST_TIMEOUT,
            )
            embedding: List[float] = response.data[0].embedding
            logger.debug("Generated embedding for text (length: {})", len(text))
            return embedding

        except Exception as e:
            logger.error("Failed to generate embedding: {}", e)
            raise

    async def get_cached_response(self, prompt: str) -> Optional[str]:
        """
        Retrieve a cached response if semantically similar prompt exists.

        Args:
            prompt: The user prompt to search for in cache.

        Returns:
            The cached response text if a match is found, None otherwise.
        """
        try:
            await self._initialize_collection()

            # Generate embedding for the incoming prompt
            prompt_embedding: List[float] = await self._get_embedding(prompt)

            # Search Qdrant for similar vectors
            search_results = await self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=prompt_embedding,
                limit=1,
                score_threshold=settings.CACHE_SIMILARITY_THRESHOLD,
                timeout=settings.REQUEST_TIMEOUT,
            )

            if search_results and len(search_results) > 0:
                result = search_results[0]
                logger.info(
                    "Cache HIT: similarity_score={:.4f}",
                    result.score
                )

                # Extract cached response from payload
                if result.payload and "response" in result.payload:
                    cached_response: str = result.payload["response"]
                    return cached_response
                else:
                    logger.warning("Cache entry found but response payload missing")
                    return None
            else:
                logger.debug("Cache MISS: no similar prompts found (threshold={})", settings.CACHE_SIMILARITY_THRESHOLD)
                return None

        except Exception as e:
            logger.error("Error retrieving from cache: {}. Proceeding without cache.", e)
            return None

    async def set_cached_response(self, prompt: str, response: str) -> None:
        """
        Store a prompt-response pair in the semantic cache.

        Args:
            prompt: The user prompt.
            response: The LLM response to cache.
        """
        try:
            await self._initialize_collection()

            # Generate embedding for the prompt
            prompt_embedding: List[float] = await self._get_embedding(prompt)

            # Create point with embedding and metadata
            point_id: int = int(time.time() * 1_000_000) + self.point_counter
            self.point_counter += 1

            point: PointStruct = PointStruct(
                id=point_id,
                vector=prompt_embedding,
                payload={
                    "prompt": prompt[:500],  # Store truncated prompt for debugging
                    "response": response,
                    "timestamp": int(time.time()),
                    "embedding_model": self.embedding_model,
                },
            )

            # Upsert point to collection
            await self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point],
                timeout=settings.REQUEST_TIMEOUT,
            )

            logger.info("Cached response stored: point_id={}", point_id)

        except Exception as e:
            logger.error("Error storing response in cache: {}. Continuing without cache update.", e)

    async def clear_cache(self) -> None:
        """Delete the entire collection (useful for testing/reset)."""
        try:
            await self.qdrant_client.delete_collection(
                collection_name=self.collection_name,
                timeout=settings.REQUEST_TIMEOUT,
            )
            logger.info("Cache collection {} cleared", self.collection_name)
        except Exception as e:
            logger.error("Error clearing cache: {}", e)

    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Retrieve cache statistics.

        Returns:
            Dictionary containing collection stats.
        """
        try:
            collection_info = await self.qdrant_client.get_collection(
                collection_name=self.collection_name,
                timeout=settings.REQUEST_TIMEOUT,
            )

            stats: Dict[str, Any] = {
                "collection_name": self.collection_name,
                "points_count": collection_info.points_count,
                "vector_size": self.vector_dimension,
                "status": str(collection_info.status),
            }

            logger.debug("Cache stats: {}", stats)
            return stats

        except Exception as e:
            logger.warning("Could not retrieve cache stats: {}", e)
            return {
                "collection_name": self.collection_name,
                "points_count": 0,
                "vector_size": self.vector_dimension,
                "status": "unavailable",
            }
