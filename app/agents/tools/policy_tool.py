"""
app/agents/tools/policy_tool.py
===============================
Dedicated Policy Retrieval Tool for the Enterprise HR AI Agentic Layer.
Encapsulates access to the isolated Synthetic HR Policy RAG capability:
- ChromaDB collection: 'enterprise_hr_policies_bge'
- BM25 sparse index: 'data/rag/policy_sparse_index/'

Enforces strict namespace boundaries:
- Throws an assertion if configured against the general knowledge collection.
- Encapsulates tool definition schema and isolated execution boundary.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional

from app.rag.embeddings import BGEEmbedder
from app.rag.retrieval.hybrid_retriever import HybridRetriever
from app.rag.retrieval.reranker import CrossEncoderReranker
from app.rag.retrieval.schemas import RetrievalConfig
from app.rag.pipeline import GroundedRAGPipeline, REFUSAL_MESSAGE
from app.utils.config import BASE_DIR
from app.utils.logger import logger

POLICY_COLLECTION_NAME = "enterprise_hr_policies_bge"
POLICY_SPARSE_DIR = BASE_DIR / "data" / "rag" / "policy_sparse_index"
FORBIDDEN_COLLECTION_NAME = "enterprise_hr_knowledge_bge"


class PolicyRetrievalTool:
    """
    Dedicated tool interface for querying the Enterprise HR Policy repository.
    
    Provides two distinct aspects:
    1. Tool Specification: Metadata describing capabilities, arguments, and name for agent planning.
    2. Execution Boundary: Safe, authorized execution method running the Policy RAG pipeline.
    """

    name: str = "search_hr_policies"
    description: str = (
        "Authoritative retrieval tool for enterprise HR policies, AI decision governance, "
        "model usage thresholds, skill gap severity rules, employee review requirements, "
        "and data monitoring limitations. Returns grounded policy excerpts and provenance."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Specific question regarding HR policies, compliance, governance, or thresholds."
            }
        },
        "required": ["query"]
    }

    _instance: Optional["PolicyRetrievalTool"] = None

    def __init__(
        self,
        collection_name: str = POLICY_COLLECTION_NAME,
        sparse_dir: Path = POLICY_SPARSE_DIR,
        pipeline: Optional[GroundedRAGPipeline] = None
    ):
        # Strict isolation guardrails:
        assert collection_name == POLICY_COLLECTION_NAME, (
            f"PolicyRetrievalTool must use '{POLICY_COLLECTION_NAME}', got '{collection_name}'"
        )
        assert collection_name != FORBIDDEN_COLLECTION_NAME, (
            f"PolicyRetrievalTool must NOT use '{FORBIDDEN_COLLECTION_NAME}'"
        )

        self.collection_name = collection_name
        self.sparse_dir = Path(sparse_dir)
        
        if pipeline is not None:
            self._pipeline = pipeline
        else:
            self._pipeline = self._build_policy_pipeline()

    def _build_policy_pipeline(self) -> GroundedRAGPipeline:
        """Constructs an isolated GroundedRAGPipeline targeting only the policy corpus."""
        logger.info(f"Initializing PolicyRetrievalTool pipeline on collection '{self.collection_name}'...")
        embedder = BGEEmbedder()
        retriever_config = RetrievalConfig(
            dense_top_k=20,
            sparse_top_k=20,
            final_top_k=15,
            dense_weight=0.8,
            sparse_weight=0.2
        )
        retriever = HybridRetriever(
            config=retriever_config,
            embedder=embedder,
            collection_name=self.collection_name,
            sparse_dir=self.sparse_dir
        )
        reranker = CrossEncoderReranker()
        return GroundedRAGPipeline(retriever=retriever, reranker=reranker)

    @classmethod
    def get_instance(cls) -> "PolicyRetrievalTool":
        """Singleton accessor to prevent redundant model/index reloads."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def execute(self, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool Execution Boundary:
        Validates arguments and executes the isolated policy RAG pipeline.
        
        Args:
            tool_args: Dictionary containing 'query'.
            
        Returns:
            Structured dictionary containing:
            - answer: Natural-language grounded response or refusal message.
            - provenance: List of audited source metadata dictionaries.
            - refusal_status: Boolean indicating if query was refused.
            - collection: Name of collection queried.
            - sources_count: Number of source chunks retrieved.
        """
        query = tool_args.get("query", "").strip()
        if not query:
            return {
                "answer": REFUSAL_MESSAGE,
                "provenance": [],
                "refusal_status": True,
                "collection": self.collection_name,
                "sources_count": 0
            }

        logger.info(f"PolicyRetrievalTool executing query: '{query}'")
        try:
            rag_response = self._pipeline.run(query)
        except Exception as e:
            if "does not exist" in str(e).lower() or "notfound" in str(e).lower():
                logger.info("Chroma collection was recreated; refreshing policy pipeline...")
                self._pipeline = self._build_policy_pipeline()
                rag_response = self._pipeline.run(query)
            else:
                raise e

        # Check refusal: explicit refusal message, negative score gate (score < 0.0), or degenerate output
        raw_answer = rag_response.answer.strip()
        top_score = rag_response.sources[0].score if rag_response.sources else -999.0
        confidence_gate_triggered = (top_score < 0.0)

        is_refusal = (
            confidence_gate_triggered
            or raw_answer == REFUSAL_MESSAGE
            or REFUSAL_MESSAGE.lower() in raw_answer.lower()
            or "does not contain" in raw_answer.lower()
            or "not contain sufficient" in raw_answer.lower()
            or len(raw_answer) <= 2
            or len(rag_response.sources) == 0
        )

        final_answer = REFUSAL_MESSAGE if is_refusal else raw_answer
        provenance_list = [s.model_dump() for s in rag_response.sources]

        return {
            "answer": final_answer,
            "provenance": provenance_list,
            "refusal_status": is_refusal,
            "collection": self.collection_name,
            "sources_count": len(provenance_list)
        }
