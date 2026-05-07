from dataclasses import dataclass
from typing import Any

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

@dataclass
class EmbeddingsResponse:
    model_name: str
    dimensions: int
    embeddings: list[float]
    tokens: int


class HuggingFaceEmbeddings:
    def __init__(
        self,
        model: str,
        run_inference_on: str = "cpu",
        batch_size: int = 32,
        parallel_processes: int = 1,
        chunk_size: int | None = None,
    ):
    
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if parallel_processes < 1:
            raise ValueError("parallel_processes must be at least 1.")

        self.device = run_inference_on
        self.batch_size = batch_size
        self.parallel_processes = parallel_processes
        self.chunk_size = chunk_size
        self.model_name = model
        self.model = SentenceTransformer(model, device=run_inference_on)
        self._pool: dict[str, Any] | None = None

    def __enter__(self) -> "HuggingFaceEmbeddings":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        pool = getattr(self, "_pool", None)
        if pool is not None:
            self.model.stop_multi_process_pool(pool)
            self._pool = None

    def _get_pool(self) -> dict[str, Any] | None:
        if self.parallel_processes == 1:
            return None

        if self._pool is None:
            target_devices = [self.device] * self.parallel_processes
            self._pool = self.model.start_multi_process_pool(target_devices)

        return self._pool

    def __count_tokens(self, text: str, tokenizer_model: str = "GPT2") -> int:
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        text = "Hugging Face makes token counting easy!"
        tokens = tokenizer.encode(text)
        token_count = len(tokens)
        return token_count

    def embed_documents(
        self,
        documents: list[str],
        batch_size: int | None = None,
        show_progress_bar: bool = False,
    ) -> list[EmbeddingsResponse]:
        
        if not documents:
            return []

        embeddings = self.model.encode(
            documents,
            batch_size=batch_size or self.batch_size,
            pool=self._get_pool(),
            chunk_size=self.chunk_size,
            show_progress_bar=show_progress_bar,
        )
        responses = [
            EmbeddingsResponse(
                model_name=self.model_name,
                dimensions=len(embedding),
                embeddings=embedding.tolist(),
                tokens=self.__count_tokens(doc),
            )
            for embedding, doc in zip(embeddings, documents)
        ]    
        return responses
