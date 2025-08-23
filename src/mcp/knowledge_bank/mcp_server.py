from typing import List
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from langchain.schema import Document
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession

from stores.faiss_vector_store import FAISSIngestionHandler
from loaders.doc_loader import load_document

# --- Lifecycle ---

@dataclass
class AppContext:
    """Application context with typed dependencies."""

    handler: FAISSIngestionHandler

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with type-safe context."""
    # Initialize on startup
    handler = FAISSIngestionHandler.load_local("./stores/_vectorstore/faiss_db")
    try:
        yield AppContext(handler=handler)
    finally:
        # Cleanup on shutdown
        handler.save_local("./stores/_vectorstore/faiss_db")

mcp = FastMCP("rag_faiss_server", lifespan=app_lifespan)

# --- Tools ---

@mcp.tool()
async def load_files(ctx: Context[ServerSession, AppContext], path: str) -> List[dict]:
    """Load file into Documents, return structured metadata + preview text."""
    docs: List[Document] = load_document(path)
    return [{"content": d.page_content[:200], "metadata": d.metadata} for d in docs]

@mcp.tool()
async def ingest(ctx: Context[ServerSession, AppContext], path: str) -> dict:
    """Load + ingest file into FAISS vectorstore."""
    handler = ctx.request_context.lifespan_context.handler

    docs: List[Document] = load_document(path)
    n = handler.add(docs)
    return {"chunks_ingested": n}

@mcp.tool()
async def query(ctx: Context[ServerSession, AppContext], query: str, k: int = 5) -> List[dict]:
    """Query the FAISS vectorstore for semantically similar chunks."""
    handler = ctx.request_context.lifespan_context.handler
    
    results = handler.similarity_search(query, k=k)
    return [
        {"content": r.page_content, "metadata": r.metadata}
        for r in results
    ]

if __name__ == "__main__":
    mcp.run(transport="streamable-http")