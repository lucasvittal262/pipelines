from typing import Any

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from models.embeddings import EmbeddingsResponse


class HuggingFaceEmbeddings:
    def __init__(
        self,
        model: str,
        run_inference_on: str = "cpu",
        parallel_processes: int = 1,
        chunk_size: int | None = None,
        tokenizer_model: str = "GPT2",
    ):
    
        
        if parallel_processes < 1:
            raise ValueError("parallel_processes must be at least 1.")

        self.device = run_inference_on
        self.parallel_processes = parallel_processes
        self.chunk_size = chunk_size
        self.model_name = model
        self.model = SentenceTransformer(model, device=run_inference_on)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
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

    def __count_tokens(self, documents: list[str]) -> list[int]:
        encoded_documents = self.tokenizer(
            documents,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return [len(tokens) for tokens in encoded_documents["input_ids"]]

    def embed_documents(
        self,
        documents: list[str],
        batch_size: int = 10,
        show_progress_bar: bool = False,
    ) -> list[EmbeddingsResponse]:
        
        if not documents:
            return []

        effective_batch_size = batch_size or self.batch_size
        pool = self._get_pool()
        effective_chunk_size = self.chunk_size
        if pool is not None and effective_chunk_size is None:
            effective_chunk_size = effective_batch_size
        
        embeddings = self.model.encode(
            documents,
            batch_size=effective_batch_size,
            pool=pool,
            chunk_size=effective_chunk_size,
            show_progress_bar=show_progress_bar,
        )
        token_counts = self.__count_tokens(documents)
        responses = [
            EmbeddingsResponse(
                model_name=self.model_name,
                dimensions=len(embedding),
                embeddings=embedding,
                tokens=token_count,
            )
            for embedding, token_count in zip(embeddings, token_counts)
        ]    
        return responses
