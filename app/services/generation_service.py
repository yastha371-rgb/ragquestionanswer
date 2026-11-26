import time
from typing import List, Tuple, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except ImportError:
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.output_parser import StrOutputParser

from app.config import settings


class GenerationService:
    """Service for synthesizing academic answers using Google Gemini and retrieved context."""

    def __init__(self):
        self.llm = None
        self._ensure_client()

        self.prompt_template = ChatPromptTemplate.from_template("""
You are an expert academic research assistant designed for enterprise academic inquiry.
Answer the user's research question strictly based on the provided context excerpts from indexed literature.

Context Excerpts:
---------------------
{context}
---------------------

Question: {question}

Rigorous Academic Guidelines:
1. Provide a clear, scholarly, and concise answer supported by the text.
2. If the context does not contain adequate evidence to answer the question, state clearly: "I don't have enough information in the provided academic documents to answer this question."
3. Cite the relevant source titles and page numbers inline (e.g. [Document Title, Page X]) wherever facts are asserted.
4. Avoid unsubstantiated speculation or external facts not verifiable within the provided context.

Answer:
""")

    def _ensure_client(self) -> None:
        """Verify Gemini API key and model before invoking model."""
        key = settings.api_key if settings.api_key else "dummy-key-for-init"
        model_name = settings.MODEL_NAME or "gemini-3.8-flash"

        if (
            self.llm is None
            or getattr(self.llm, "model", None) != model_name
            or getattr(self.llm, "google_api_key", None) != key
        ):
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=key,
                temperature=0.3
            )

    def generate_answer(self, question: str, context: str) -> Tuple[str, float]:
        """
        Generate a synthesized answer using LCEL chain with Google Gemini.
        """
        self._ensure_client()
        start_time = time.time()

        chain = self.prompt_template | self.llm | StrOutputParser()

        answer = chain.invoke({
            "question": question,
            "context": context
        })

        processing_time_ms = round((time.time() - start_time) * 1000, 2)
        return answer.strip(), processing_time_ms

    def calculate_confidence_score(self, retrieved_docs: List[Dict[str, Any]]) -> float:
        """
        Calculate an overall confidence score based on retrieval relevance scores.
        """
        if not retrieved_docs:
            return 0.0

        top_k = retrieved_docs[:3]
        scores = [float(doc.get("relevance_score", 0.0)) for doc in top_k]

        if not scores:
            return 0.0

        if len(scores) == 1:
            confidence = scores[0]
        else:
            weights = [0.5, 0.3, 0.2][:len(scores)]
            norm_weights = [w / sum(weights) for w in weights]
            confidence = sum(s * w for s, w in zip(scores, norm_weights))

        return max(0.0, min(1.0, round(confidence, 4)))
