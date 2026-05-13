from dataclasses import asdict, dataclass

@dataclass
class BookSection:
    level: int
    title: str
    start_page: int

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "BookSection":
        return BookSection(
            level=data["level"],
            title=data["title"],
            start_page=data["start_page"]
        )
        
@dataclass
class TextChunk:
    book_title: str
    title: str
    page: int
    text: str
    tokens: int
    
    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "TextChunk":
        return TextChunk(
            book_title=data["book_title"],
            title=data["title"],
            text=data["text"],
            page=data["page"],
            tokens=data["tokens"]
        )