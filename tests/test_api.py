from langchain_core.documents import Document

def test_health_healthy(test_client, mock_vectorstore):
    mock_vectorstore.next_docs = [Document(page_content="", metadata={}), Document(page_content="", metadata={})] # Make doc count > 0
    resp = test_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["document_count"] == 2

def test_health_empty(test_client, mock_vectorstore):
    mock_vectorstore.next_docs = [] # Empty vector store
    resp = test_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "degraded"
    assert data["document_count"] == 0

def test_chat_sync(test_client, mock_llm, mock_vectorstore):
    mock_vectorstore.next_docs = [Document(page_content="Return in 30 days.", metadata={"source": "policy.md", "distance": 0.1})]
    
    mock_llm.responses = [
        '{"route": "rag"}',          # router
        '{"relevant": "yes"}',       # doc_grader
        'You have 30 days to return', # generator
        '{"grounded": "yes"}'        # hallucination_grader
    ]
    
    resp = test_client.post("/chat", json={"question": "How to return?"})
    assert resp.status_code == 200
    data = resp.json()
    
    assert data["answer"] == "You have 30 days to return"
    assert data["route"] == "rag"
    assert data["is_escalated"] is False
    assert data["documents_used"] == 1
    assert len(data["thought_trace"]) == 5

def test_chat_stream_format(test_client, mock_llm, mock_vectorstore):
    # Test that the SSE endpoint returns 200 and the correct content-type
    # Testing the actual stream contents with TestClient is complex, 
    # but we can verify it doesn't crash on connection.
    mock_vectorstore.next_docs = [Document(page_content="docs", metadata={})]
    mock_llm.responses = ['{"route": "chitchat"}', 'Hello!']
    
    resp = test_client.post("/chat/stream", json={"question": "Hi"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"
    
    # Check that it streamed "done" at the end
    content = resp.text
    assert "event: done" in content
