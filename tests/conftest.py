import pytest
from src.providers.interfaces import ILLMProvider, IVectorStore
from src.dependencies import AppDependencies
from src.config import Settings
from src.core.graph_builder import build_graph

class MockLLMProvider(ILLMProvider):
    def __init__(self):
        self.responses = []

    def invoke_json(self, prompt: str, system: str) -> dict:
        import json
        resp = self.responses.pop(0) if self.responses else "{}"
        try:
            return json.loads(resp)
        except json.JSONDecodeError:
            return {"error": "invalid json in mock"}

    def invoke_text(self, prompt: str, system: str, *, use_power: bool = False) -> str:
        return self.responses.pop(0) if self.responses else "Default mock response"

    async def astream_text(self, prompt: str, system: str):
        resp = self.responses.pop(0) if self.responses else "Default mock response"
        yield resp

class MockVectorStore(IVectorStore):
    def __init__(self):
        self.next_docs = []

    def similarity_search(self, query: str, k: int = 4) -> list[dict]:
        return self.next_docs

    def get_document_count(self) -> int:
        return len(self.next_docs)

    def add_documents(self, documents: list) -> None:
        pass

@pytest.fixture
def mock_llm():
    return MockLLMProvider()

@pytest.fixture
def mock_vectorstore():
    return MockVectorStore()

from src.config import load_settings

@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test_key")
    monkeypatch.setenv("MAX_QUERY_REWRITES", "2")
    return load_settings()

@pytest.fixture
def mock_deps(mock_llm, mock_vectorstore, settings):
    deps = AppDependencies(
        settings=settings,
        vectorstore=mock_vectorstore,
        graph=None
    )
    deps.graph = build_graph(llm=mock_llm, vectorstore=mock_vectorstore, settings=settings)
    return deps

@pytest.fixture
def test_client(mock_deps):
    from fastapi.testclient import TestClient
    from src.api.app import create_app
    from contextlib import asynccontextmanager

    app = create_app()
    
    @asynccontextmanager
    async def mock_lifespan(app):
        yield
    app.router.lifespan_context = mock_lifespan
    
    app.state.deps = mock_deps
    
    with TestClient(app) as client:
        yield client
