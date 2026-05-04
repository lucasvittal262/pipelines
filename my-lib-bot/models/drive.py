from dataclasses import asdict, dataclass


@dataclass
class DriveFileMetadata:
    id: str
    name: str
    mimeType: str
    webViewLink: str
    size_mb: float | None = None

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "DriveFileMetadata":
        return DriveFileMetadata(
            id=data["id"],
            name=data["name"],
            mimeType=data["mimeType"],
            webViewLink=data["webViewLink"],
            size_mb=data.get("size_mb"),
        )
