from __future__ import annotations

import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from rag.config import (
        DEFAULT_EMBED_MODEL,
        DEFAULT_LLM_MODEL,
        DEFAULT_RERANK_MODEL,
        SYSTEM_PROMPT_TEMPLATE,
    )
except ImportError:
    from config import (
        DEFAULT_EMBED_MODEL,
        DEFAULT_LLM_MODEL,
        DEFAULT_RERANK_MODEL,
        SYSTEM_PROMPT_TEMPLATE,
    )


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    1. Loads parsed PDF chunks (from the pdf_parser output JSON)
    2. Creates embeddings using a sentence-transformer model
    3. Retrieves top-k relevant chunks via cosine similarity
    4. Generates answers via Groq LLM grounded in retrieved context
    """

    def __init__(
        self,
        groq_api_key: str,
        embed_model: str = DEFAULT_EMBED_MODEL,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        llm_model: str = DEFAULT_LLM_MODEL,
    ):
        from groq import Groq
        from sentence_transformers import CrossEncoder

        self.embed_model_name = embed_model
        self.embedder = SentenceTransformer(embed_model)
        
        print(f"  Loading reranker model: {rerank_model}")
        self.reranker = CrossEncoder(rerank_model, max_length=512)
        
        self.groq = Groq(api_key=groq_api_key)
        self.llm_model = llm_model

        self.chunks: list[dict] = []
        self.chunk_texts: list[str] = []
        self.embeddings: np.ndarray | None = None



    def load_chunks(self, json_path: str):
        """Load parsed chunks from JSON and create/load embeddings."""
        with open(json_path, encoding="utf-8") as f:
            raw_chunks = json.load(f)

        # Keep only chunks with embeddable text
        self.chunks = []
        self.chunk_texts = []
        for chunk in raw_chunks:
            text = self._chunk_to_text(chunk)
            if text.strip():
                self.chunks.append(chunk)
                self.chunk_texts.append(text)

        if not self.chunks:
            print("  No embeddable chunks found!")
            return

        # Use cached embeddings if available, keyed by content hash and model
        import hashlib
        text_combo = "".join(self.chunk_texts).encode("utf-8")
        text_hash = hashlib.md5(text_combo).hexdigest()[:8]
        safe_model = self.embed_model_name.replace("/", "_")
        cache_path = json_path.replace(".json", f"_embs_{safe_model}_{text_hash}.npy")

        if os.path.exists(cache_path):
            self.embeddings = np.load(cache_path)
            if len(self.embeddings) == len(self.chunks):
                print(f"  Loaded cached embeddings ({len(self.chunks)} chunks)")
                return
            print("  Cache mismatch, re-embedding...")

        self._create_embeddings(cache_path)

    def _create_embeddings(self, cache_path: str):
        """Create embeddings and save to disk."""
        print(f"  Creating embeddings for {len(self.chunk_texts)} chunks...")
        self.embeddings = self.embedder.encode(
            self.chunk_texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        np.save(cache_path, self.embeddings)
        print(f"  Embeddings cached -> {cache_path}")

    def _chunk_to_text(self, chunk: dict) -> str:
        """
        Convert a chunk dict to an embeddable string context rich in semantics.
        """
        parts = []

        # 1. Metadata labels help the embedding model understand structure
        meta = []
        if chunk.get("page_number"):
            meta.append(f"Page: {chunk['page_number']}")
        if chunk.get("type"):
            meta.append(f"Type: {chunk['type']}")
        if meta:
            parts.append(f"[{' | '.join(meta)}]")

        # 2. Section context
        if chunk.get("section_heading"):
            parts.append(f"Section Heading: {chunk['section_heading']}")

        # 3. Main content
        if chunk.get("text"):
            if chunk.get("type") == "table":
                parts.append("--- Table Data Start ---")
            parts.append(chunk["text"])
            if chunk.get("type") == "table":
                parts.append("--- Table Data End ---")

        # 4. Imagery context
        if chunk.get("type") == "figure" and chunk.get("images"):
            parts.append(f"Visual Figure attached: {', '.join(chunk['images'])}")
        
        # 5. OCR context
        if chunk.get("type") == "figure" and chunk.get("ocr_text"):
             parts.append(f"Extracted Text from Figure:\n{chunk['ocr_text']}")

        return "\n".join(parts)



    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[dict, float]]:
        """
        Stage 1: Retrieve top_k * 4 chunks using dense vector cosine similarity (High Recall).
        Stage 2: Cross-Encoder Reranking to find the absolute best top_k (Precision).
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        # -- STAGE 1: Sparse/Dense Vector Retrieval --
        query_emb = self.embedder.encode(
            [query], normalize_embeddings=True
        )

        n_retrieve = min(len(self.chunks), top_k * 4)

        # Cosine similarity (embeddings are already normalized)
        scores = np.dot(self.embeddings, query_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:n_retrieve]
        
        stage1_candidates = [self.chunks[i] for i in top_indices]
        candidate_texts = [self.chunk_texts[i] for i in top_indices]

        # -- STAGE 2: Cross-Encoder Reranking --
        # Formally scores the exact query against each candidate context
        pairs = [[query, text] for text in candidate_texts]
        rerank_scores = self.reranker.predict(pairs)
        
        # CrossEncoder returns a float if len(pairs)==1, ensure it's a list
        if isinstance(rerank_scores, float):
            rerank_scores = [rerank_scores]
            
        # Zip chunks with their new precision score and sort
        reranked_results = list(zip(stage1_candidates, rerank_scores))
        reranked_results.sort(key=lambda x: x[1], reverse=True)

        return reranked_results[:top_k]



    def ask(
        self,
        query: str,
        chat_history: list[dict] | None = None,
        top_k: int = 5,
    ) -> tuple[str, list[dict]]:
        """
        Ask a question with RAG context.

        Args:
            query: user question
            chat_history: list of {"role": "user"/"assistant", "content": "..."}
            top_k: number of chunks to retrieve

        Returns:
            (answer_text, list_of_source_references)
        """
        # If the user asks for a summary of the document, we need a lot more context 
        # because top-3 limits us to random fragments
        is_summary_request = any(word in query.lower() for word in ["summarize", "summary", "overview", "what is this document about"])
        effective_top_k = min(len(self.chunks), max(15, top_k * 3)) if is_summary_request else top_k

        results = self.retrieve(query, effective_top_k)
        context = self._build_context(results)

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_TEMPLATE.format(context=context),
            }
        ]

        if chat_history:
            messages.extend(chat_history)

        messages.append({"role": "user", "content": query})

        try:
            response = self.groq.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.3,
                max_tokens=2048,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Error calling Groq API: {e}"

        sources = []
        for chunk, score in results:
            sources.append({
                "page": chunk.get("page_number"),
                "section": chunk.get("section_heading"),
                "type": chunk.get("type"),
                "score": round(score, 3),
            })

        return answer, sources

    def _build_context(self, results: list[tuple[dict, float]]) -> str:
        """Format retrieved chunks into LLM-readable context."""
        parts = []
        for i, (chunk, score) in enumerate(results, 1):
            header = (
                f"[Chunk {i} | Page {chunk.get('page_number', '?')} | "
                f"Type: {chunk.get('type', '?')} | Relevance: {score:.2f}]"
            )
            body = []
            if chunk.get("section_heading"):
                body.append(f"Section: {chunk['section_heading']}")
            if chunk.get("type") == "table":
                body.append("[Table Data]")
            if chunk.get("text"):
                body.append(chunk["text"])
            if chunk.get("type") == "figure" and chunk.get("images"):
                body.append(f"[Figure: {', '.join(chunk['images'])}]")
            if chunk.get("type") == "figure" and chunk.get("ocr_text"):
                body.append(f"[Image Text]: {chunk['ocr_text']}")
            parts.append(f"{header}\n" + "\n".join(body))

        return "\n\n".join(parts)
