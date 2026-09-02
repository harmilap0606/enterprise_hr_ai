"""
app/rag/__init__.py
RAG module for enterprise HR AI grounded strictly on real project documentation
and O*NET occupational descriptions.
"""
from app.rag.retriever import retrieve
from app.rag.qa_chain import answer_question
from app.rag.pipeline import GroundedRAGPipeline, answer_grounded_question

__all__ = ["retrieve", "answer_question", "GroundedRAGPipeline", "answer_grounded_question"]
