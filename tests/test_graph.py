import pytest
from langchain_core.documents import Document

def _get_clean_state(question: str):
    return {
        "question": question,
        "generation": "",
        "documents": [],
        "route_decision": "",
        "docs_are_relevant": False,
        "generation_is_grounded": False,
        "query_rewrite_count": 0,
        "generation_retry_count": 0,
        "thought_trace": [],
        "is_escalated": False,
    }

@pytest.mark.asyncio
async def test_graph_happy_path_rag(mock_deps, mock_llm, mock_vectorstore):
    mock_vectorstore.next_docs = [Document(page_content="Return in 30 days.", metadata={"source": "policy.md", "distance": 0.1})]
    
    mock_llm.responses = [
        '{"route": "rag"}',          # router
        '{"relevant": "yes"}',       # doc_grader
        'You have 30 days to return', # generator
        '{"grounded": "yes"}'        # hallucination_grader
    ]
    
    result = await mock_deps.graph.ainvoke(_get_clean_state("How to return?"))
    
    assert result["generation"] == 'You have 30 days to return'
    assert result["is_escalated"] is False
    assert result["route_decision"] == "rag"
    assert result["generation_is_grounded"] is True

@pytest.mark.asyncio
async def test_graph_chitchat(mock_deps, mock_llm):
    mock_llm.responses = [
        '{"route": "chitchat"}',     # router
        'Hello there!'               # direct_responder
    ]
    
    result = await mock_deps.graph.ainvoke(_get_clean_state("Hi"))
    
    assert result["generation"] == 'Hello there!'
    assert result["route_decision"] == "chitchat"

@pytest.mark.asyncio
async def test_graph_escalation_on_missing_docs(mock_deps, mock_llm, mock_vectorstore):
    mock_vectorstore.next_docs = [Document(page_content="irrelevant", metadata={})]
    
    mock_llm.responses = [
        '{"route": "rag"}',          # router
        '{"relevant": "no"}',        # doc_grader (attempt 1)
        'Rewritten 1',               # query_rewriter (attempt 1)
        '{"relevant": "no"}',        # doc_grader (attempt 2)
        'Rewritten 2',               # query_rewriter (attempt 2)
        '{"relevant": "no"}',        # doc_grader (attempt 3)
        # -> Escalate!
    ]
    
    result = await mock_deps.graph.ainvoke(_get_clean_state("Weird specific thing"))
    
    assert result["is_escalated"] is True
    assert "human support agent" in result["generation"]
    assert result["query_rewrite_count"] == 2

@pytest.mark.asyncio
async def test_graph_escalation_on_hallucination(mock_deps, mock_llm, mock_vectorstore):
    mock_vectorstore.next_docs = [Document(page_content="Relevant but doesn't mention free shipping", metadata={})]
    
    mock_llm.responses = [
        '{"route": "rag"}',          # router
        '{"relevant": "yes"}',       # doc_grader
        'You get free shipping with code VIP!', # generator (attempt 1)
        '{"grounded": "no"}',        # hallucination_grader (attempt 1)
        'Free shipping is not available.', # generator (attempt 2)
        '{"grounded": "no"}',        # hallucination_grader (attempt 2)
        # -> Escalate!
    ]
    
    result = await mock_deps.graph.ainvoke(_get_clean_state("How to get free shipping?"))
    
    assert result["is_escalated"] is True
    assert "human support agent" in result["generation"]
    assert result["generation_retry_count"] == 2

@pytest.mark.asyncio
async def test_graph_out_of_domain(mock_deps, mock_llm):
    mock_llm.responses = [
        '{"route": "out_of_domain"}', # router
        # -> Escalate directly!
    ]
    
    result = await mock_deps.graph.ainvoke(_get_clean_state("Do you sell video games?"))
    
    assert result["route_decision"] == "out_of_domain"
    assert result["is_escalated"] is True
    assert "human support agent" in result["generation"]
