from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from rag.parser import parse_policy_markdown


class ChromaPolicyStore:
    """Student scaffold for the real Chroma-backed policy index."""

    def __init__(
        self,
        persist_directory: Path,
        embedding_model: Any,
        collection_name: str = "policy_chunks",
    ) -> None:
        self.persist_directory = persist_directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def ensure_index(self, markdown_path: Path) -> None:
        if self.collection.count() == 0:
            self.rebuild(markdown_path)

    def rebuild(self, markdown_path: Path) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        markdown_text = markdown_path.read_text(encoding="utf-8")
        chunks = parse_policy_markdown(markdown_text)
        if not chunks:
            raise ValueError(f"No policy chunks parsed from {markdown_path}")

        documents = [chunk["rendered_text"] for chunk in chunks]
        embeddings = self.embedding_model.embed_documents(documents)
        metadatas = [
            {
                "section_h2": chunk["section_h2"],
                "section_h3": chunk["section_h3"],
                "citation": chunk["citation"],
                "source": markdown_path.name,
            }
            for chunk in chunks
        ]
        ids = [f"policy-{index:03d}" for index in range(len(chunks))]
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        if self.collection.count() == 0:
            return []
        query_embedding = self.embedding_model.embed_query(query)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max(1, top_k),
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]
        hits: list[dict[str, Any]] = []
        for doc_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            hits.append(
                {
                    "id": doc_id,
                    "citation": metadata.get("citation", ""),
                    "section_h2": metadata.get("section_h2", ""),
                    "section_h3": metadata.get("section_h3", ""),
                    "content": document,
                    "distance": float(distance),
                }
            )
        return hits

    def get_sections(self, section_h3_values: list[str]) -> list[dict[str, Any]]:
        if self.collection.count() == 0:
            return []
        result = self.collection.get(include=["documents", "metadatas"])
        wanted = set(section_h3_values)
        hits: list[dict[str, Any]] = []
        for doc_id, document, metadata in zip(
            result.get("ids", []),
            result.get("documents", []),
            result.get("metadatas", []),
        ):
            if metadata.get("section_h3") not in wanted:
                continue
            hits.append(
                {
                    "id": doc_id,
                    "citation": metadata.get("citation", ""),
                    "section_h2": metadata.get("section_h2", ""),
                    "section_h3": metadata.get("section_h3", ""),
                    "content": document,
                    "distance": 0.0,
                }
            )
        order = {section: index for index, section in enumerate(section_h3_values)}
        hits.sort(key=lambda hit: order.get(hit["section_h3"], 999))
        return hits
