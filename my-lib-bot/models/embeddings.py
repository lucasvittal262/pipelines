from dataclasses import asdict, dataclass

@dataclass
class EmbeddingsResponse:
    model_name: str
    dimensions: int
    embeddings: list[float]
    tokens: int

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "EmbeddingsResponse":
        return EmbeddingsResponse(
            model_name=data["model_name"],
            dimensions=data["dimensions"],
            embeddings=data["embeddings"],
            tokens=data["tokens"]
        )