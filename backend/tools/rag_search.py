"""
RAG Search Tool for Mini-OpenClaw.

Compatible with both old and new LlamaIndex package layouts. If LlamaIndex
is unavailable or mismatched, the tool degrades gracefully instead of blocking
backend startup.
"""

import os
from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import PrivateAttr

try:
    from llama_index import (
        SimpleDirectoryReader,
        StorageContext,
        VectorStoreIndex,
        load_index_from_storage,
    )
except Exception:
    try:
        from llama_index.core import (  # type: ignore
            SimpleDirectoryReader,
            StorageContext,
            VectorStoreIndex,
            load_index_from_storage,
        )
    except Exception:
        SimpleDirectoryReader = None  # type: ignore
        StorageContext = None  # type: ignore
        VectorStoreIndex = None  # type: ignore
        load_index_from_storage = None  # type: ignore


class RAGSearchTool(BaseTool):
    """
    Hybrid search tool using LlamaIndex.
    """

    name: str = "search_knowledge_base"
    description: str = (
        "Search the knowledge base for relevant information using hybrid search. "
        "Use this when the user asks about specific topics that might be in local documents."
    )

    _knowledge_dir: str = PrivateAttr()
    _storage_dir: str = PrivateAttr()
    _index = PrivateAttr(default=None)
    _available: bool = PrivateAttr(default=False)
    _init_error: Optional[str] = PrivateAttr(default=None)

    def __init__(self, knowledge_dir: str, storage_dir: str):
        super().__init__()
        self._knowledge_dir = os.path.abspath(knowledge_dir)
        self._storage_dir = os.path.abspath(storage_dir)
        self._initialize_index()

    def _initialize_index(self) -> None:
        """Initialize or load the vector index."""
        if not all(
            [SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage]
        ):
            self._available = False
            self._init_error = (
                "LlamaIndex imports are unavailable in the current environment. "
                "Install a compatible llama-index version to enable knowledge search."
            )
            return

        try:
            os.makedirs(self._knowledge_dir, exist_ok=True)
            os.makedirs(self._storage_dir, exist_ok=True)

            docstore_path = os.path.join(self._storage_dir, "docstore.json")
            if os.path.exists(docstore_path):
                storage_context = StorageContext.from_defaults(persist_dir=self._storage_dir)
                self._index = load_index_from_storage(storage_context)
                self._available = True
                return

            if not os.listdir(self._knowledge_dir):
                self._available = False
                self._init_error = (
                    f"No documents found in {self._knowledge_dir}. "
                    "Add PDF/MD/TXT files to enable knowledge search."
                )
                return

            documents = SimpleDirectoryReader(self._knowledge_dir).load_data()
            if not documents:
                self._available = False
                self._init_error = "Knowledge directory exists, but no readable documents were loaded."
                return

            self._index = VectorStoreIndex.from_documents(documents)
            self._index.storage_context.persist(persist_dir=self._storage_dir)
            self._available = True
        except Exception as e:
            self._available = False
            self._index = None
            self._init_error = f"Failed to initialize knowledge base: {e}"

    def _run(self, query: str) -> str:
        """Execute knowledge-base search."""
        if not self._available or self._index is None:
            return self._init_error or "Knowledge base is not available."

        try:
            query_engine = self._index.as_query_engine(
                similarity_top_k=5,
                response_mode="compact",
            )
            response = query_engine.query(query)
            return str(response)
        except Exception as e:
            return f"Error searching knowledge base: {e}"

    async def _arun(self, query: str) -> str:
        """Async version."""
        return self._run(query)


def create_rag_search_tool(
    knowledge_dir: Optional[str] = None, storage_dir: Optional[str] = None
) -> RAGSearchTool:
    """Create the RAG search tool."""
    if knowledge_dir is None:
        knowledge_dir = os.path.join(os.getcwd(), "knowledge")

    if storage_dir is None:
        storage_dir = os.path.join(os.getcwd(), "storage")

    return RAGSearchTool(knowledge_dir=knowledge_dir, storage_dir=storage_dir)
