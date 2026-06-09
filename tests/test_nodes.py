import pytest
from src.core.nodes.router import RouterNode
from src.core.nodes.doc_grader import DocGraderNode
from src.core.nodes.generator import GeneratorNode
from src.core.nodes.hallucination_grader import HallucinationGraderNode
from src.core.nodes.query_rewriter import QueryRewriterNode
from src.core.nodes.direct_responder import DirectResponderNode
from langchain_core.documents import Document

def test_router_node_rag(mock_llm):
    mock_llm.responses = ['{"route": "rag"}']
    state = {"question": "How to return?", "thought_trace": []}
    
    node = RouterNode(mock_llm)
    result = node(state)
    
    assert result["route_decision"] == "rag"
    assert result["thought_trace"][0]["step"] == "router"
    assert "rag" in result["thought_trace"][0]["detail"]

def test_router_node_chitchat(mock_llm):
    mock_llm.responses = ['{"route": "chitchat"}']
    state = {"question": "Hello", "thought_trace": []}
    
    node = RouterNode(mock_llm)
    result = node(state)
    
    assert result["route_decision"] == "chitchat"

def test_router_node_out_of_domain(mock_llm):
    mock_llm.responses = ['{"route": "out_of_domain"}']
    state = {"question": "Do you sell video games?", "thought_trace": []}
    
    node = RouterNode(mock_llm)
    result = node(state)
    
    assert result["route_decision"] == "out_of_domain"

def test_doc_grader_node(mock_llm):
    mock_llm.responses = ['{"relevant": "yes"}', '{"relevant": "no"}']
    state = {
        "question": "test",
        "documents": [
            Document(page_content="doc1", metadata={"source": "s1"}),
            Document(page_content="doc2", metadata={"source": "s2"})
        ],
        "thought_trace": []
    }
    
    node = DocGraderNode(mock_llm)
    result = node(state)
    
    assert result["docs_are_relevant"] is True
    assert len(result["documents"]) == 1
    assert result["documents"][0].page_content == "doc1"
    assert result["thought_trace"][0]["detail"]["passed"] == 1

def test_doc_grader_node_all_fail(mock_llm):
    mock_llm.responses = ['{"relevant": "no"}']
    state = {
        "question": "test",
        "documents": [Document(page_content="doc1", metadata={})],
        "thought_trace": []
    }
    
    node = DocGraderNode(mock_llm)
    result = node(state)
    
    assert result["docs_are_relevant"] is False
    assert len(result["documents"]) == 0

def test_generator_node(mock_llm):
    mock_llm.responses = ["Here is the answer."]
    state = {
        "question": "test",
        "documents": [Document(page_content="context", metadata={})],
        "generation_retry_count": 0,
        "thought_trace": []
    }
    
    node = GeneratorNode(mock_llm)
    result = node(state)
    
    assert result["generation"] == "Here is the answer."
    assert result["generation_retry_count"] == 1

def test_hallucination_grader_node(mock_llm):
    mock_llm.responses = ['{"grounded": "yes"}']
    state = {
        "generation": "test answer",
        "documents": [],
        "thought_trace": []
    }
    
    node = HallucinationGraderNode(mock_llm)
    result = node(state)
    
    assert result["generation_is_grounded"] is True

def test_query_rewriter_node(mock_llm):
    mock_llm.responses = ["rewritten query"]
    state = {
        "question": "original query",
        "query_rewrite_count": 0,
        "thought_trace": []
    }
    
    node = QueryRewriterNode(mock_llm)
    result = node(state)
    
    assert result["question"] == "rewritten query"
    assert result["query_rewrite_count"] == 1

def test_direct_responder_node(mock_llm):
    mock_llm.responses = ["Direct chitchat response"]
    state = {
        "question": "hello",
        "thought_trace": []
    }
    
    node = DirectResponderNode(mock_llm)
    result = node(state)
    
    assert result["generation"] == "Direct chitchat response"
