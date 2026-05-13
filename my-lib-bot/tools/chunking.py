from pathlib import Path
import re
from typing import Any
import unicodedata

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
import yaml

from models.chunking import BookSection, TextChunk


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file and return an empty dict when the file is empty."""
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


class PDFSectionalChunker:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    METADATA_DIR = PROJECT_ROOT / "metadata"

    def __init__(self, tokenizer_model: str = "cl100k_base"):
        self.books_metadata: list = []
        self.books_processed: int = 0
        self.total_number_tokens = 0
        self.tokenizer = self.__get_tokenizer(tokenizer_model)

    def __get_tokenizer(self, tokenizer_model: str) -> tiktoken.Encoding:
        try:
            return tiktoken.encoding_for_model(tokenizer_model)
        except KeyError:
            try:
                return tiktoken.get_encoding(tokenizer_model.lower())
            except Exception as error:
                raise ValueError(
                    f"Could not load tokenizer '{tokenizer_model}'. Use an OpenAI "
                    "model name like 'gpt-4o-mini' or a tiktoken encoding name "
                    "available locally, like 'cl100k_base'."
                ) from error

    def __normalize_key(self, value: str) -> str:
        value = unicodedata.normalize("NFKC", value)
        value = re.sub(r"\s+", " ", value)
        return value.strip()

    def __get_yaml_value_by_key(self, data: dict[str, Any], key: str) -> Any:
        normalized_key = self.__normalize_key(key)
        for yaml_key, value in data.items():
            if self.__normalize_key(str(yaml_key)) == normalized_key:
                return value

        return None

    def __get_book_key(self, file_name: str) -> str:
        book_keys = read_yaml(self.METADATA_DIR / "books_keys.yaml")
        return self.__get_yaml_value_by_key(book_keys, file_name) or file_name

    def __get_excluded_sections(self, file_name: str) -> set[str]:
        section_map = read_yaml(self.METADATA_DIR / "exclude_section_map.yaml")
        book_key = self.__get_book_key(file_name)
        excluded_by_file_name = self.__get_yaml_value_by_key(section_map, file_name) or {}
        excluded_by_book_key = self.__get_yaml_value_by_key(section_map, book_key) or {}
        return set(
            excluded_by_file_name.get("excluded_sections")
            or excluded_by_book_key.get("excluded_sections")
            or []
        )

    def __get_book_sections(self, doc: fitz.Document, book_key: str) -> list[BookSection]:
        exclude_sections = self.__get_excluded_sections(book_key)
        sections = [
            BookSection(*section)
            for section in doc.get_toc()
            if section[1] not in exclude_sections
        ]
        return sections

    def __get_tokens_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def __split_text_by_tokens(
        self,
        text: str,
        max_chunk_size: int,
    ) -> list[str]:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=self.tokenizer.name,
            chunk_size=max_chunk_size,
            chunk_overlap=0,
        )
        text_chunks = [
            self.__normalize_text(text_chunk)
            for text_chunk in splitter.split_text(text)
            if text_chunk.strip()
        ]
        oversized_chunks = [
            self.__get_tokens_count(text_chunk)
            for text_chunk in text_chunks
            if self.__get_tokens_count(text_chunk) > max_chunk_size
        ]
        if oversized_chunks:
            raise ValueError(
                "LangChain fallback produced chunks above the max token size. "
                f"Largest chunk has {max(oversized_chunks)} tokens; max_chunk_size "
                f"is {max_chunk_size}. Use a higher max_chunk_size."
            )

        return text_chunks

    def __get_parent_index(
        self,
        sections: list[BookSection],
        section_index: int,
    ) -> int | None:
        section = sections[section_index]
        for index in range(section_index - 1, -1, -1):
            if sections[index].level < section.level:
                return index

        return None

    def __get_section_end_index(
        self,
        sections: list[BookSection],
        section_index: int,
    ) -> int:
        section = sections[section_index]
        for index in range(section_index + 1, len(sections)):
            if sections[index].level <= section.level:
                return index

        return len(sections)

    def __get_child_indexes(
        self,
        sections: list[BookSection],
        section_index: int,
    ) -> list[int]:
        section = sections[section_index]
        section_end_index = self.__get_section_end_index(sections, section_index)
        child_levels = [
            sections[index].level
            for index in range(section_index + 1, section_end_index)
            if sections[index].level > section.level
        ]
        if not child_levels:
            return []

        child_level = min(child_levels)
        return [
            index
            for index in range(section_index + 1, section_end_index)
            if sections[index].level == child_level
        ]

    def __get_section_text(
        self,
        doc: fitz.Document,
        sections: list[BookSection],
        section_index: int,
    ) -> str:
        section = sections[section_index]
        section_end_index = self.__get_section_end_index(sections, section_index)
        start_page = section.start_page - 1
        end_page = (
            sections[section_end_index].start_page - 1
            if section_end_index < len(sections)
            else len(doc)
        )

        text = "\n".join(
            doc[page_num].get_text("text")
            for page_num in range(start_page, end_page)
        )
        return self.__normalize_text(text)


    def __normalize_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\x0c", "\n")
        text = text.replace("\u00a0", " ")
        text = text.replace("\t", " ")
        text = re.sub(r"[^\S\n]+", " ", text)
        text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
        text = re.sub(r"[^\S\n]*\n[^\S\n]*", "\n", text)
        text = "\n".join(line.strip() for line in text.splitlines())
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[\x00-\x08\x0b\x0e-\x1f\x7f]", "", text)
        return text.strip()
    
    def __chunk_section(
        self,
        file_name: str,
        doc: fitz.Document,
        sections: list[BookSection],
        section_index: int,
        max_chunk_size: int,
    ) -> list[TextChunk]:
        section = sections[section_index]
        text = self.__get_section_text(doc, sections, section_index)
        token_count = self.__get_tokens_count(text)

        if token_count <= max_chunk_size:
            return [TextChunk(
                book_title=file_name,
                title=section.title,
                page=section.start_page,
                text=self.__normalize_text(text),
                tokens=token_count
            )]

        child_indexes = self.__get_child_indexes(sections, section_index)
        if not child_indexes:
            return [
                TextChunk(
                    book_title=file_name,
                    title=f"{section.title} ({index + 1})",
                    page=section.start_page,
                    text=self.__normalize_text(text_chunk),
                    tokens=self.__get_tokens_count(text_chunk),
                )
                for index, text_chunk in enumerate(
                    self.__split_text_by_tokens(text, max_chunk_size)
                )
            ]

        chunks = []
        for child_index in child_indexes:
            chunks.extend(
                self.__chunk_section(file_name, doc, sections, child_index, max_chunk_size)
            )

        return chunks
    
    

    def get_chunks(self, doc_path: str, max_chunk_size: int = 1000) -> list[TextChunk]:
        if max_chunk_size < 1:
            raise ValueError("max_chunk_size must be at least 1.")

        file_name = Path(doc_path).stem
        doc = fitz.open(doc_path)
        try:
            sections = self.__get_book_sections(doc, file_name)
            chunks = []

            for section_index in range(len(sections)):
                if self.__get_parent_index(sections, section_index) is not None:
                    continue
                chunks.extend(
                    self.__chunk_section(file_name, doc, sections, section_index, max_chunk_size)
                )

            return chunks
        finally:
            doc.close()


if __name__ == "__main__":
    import pprint
    chunker = PDFSectionalChunker(tokenizer_model="cl100k_base")
    chunks = chunker.get_chunks("/home/lucas-vital/Downloads/AI Engineering_ Building Applications with Foundation Models .pdf", max_chunk_size=800)
    for chunk in chunks:
        pprint.pprint(chunk)
        print("-" * 80)
