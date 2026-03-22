from nodes.base import BaseNode
from db import get_pool


# Lazy-load the embedding model so startup stays fast
_model = None

def get_embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("intfloat/multilingual-e5-large")
        print("✅ Embedding model loaded")
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    # multilingual-e5 expects "query: " prefix for queries
    prefixed = [f"query: {t}" for t in texts]
    embeddings = model.encode(prefixed, normalize_embeddings=True)
    return embeddings.tolist()


class RAGNode(BaseNode):
    """
    Queries the translation memory (pgvector) for each segment.
    Classifies matches as:
      exact  — cosine similarity >= exact_threshold  (auto-fill)
      fuzzy  — cosine similarity >= fuzzy_threshold  (suggestion)
      new    — below fuzzy threshold (send to LLM)
    """

    async def run(self, context: dict) -> dict:
        segments: list[str] = context.get("segments", [])
        if not segments:
            return {**context, "rag_matches": [], "rag_stats": {}}

        exact_threshold: float = self.config.get("exact_threshold", 1.0)
        fuzzy_threshold: float = self.config.get("fuzzy_threshold", 0.75)
        top_k: int = self.config.get("top_k", 5)
        target_language: str = context.get("target_language", "hi")

        embeddings = embed(segments)
        pool = get_pool()

        rag_matches = []
        stats = {"exact": 0, "fuzzy": 0, "new": 0}

        for segment, embedding in zip(segments, embeddings):
            # pgvector cosine similarity query
            # Requires a translation_memory table — see SQL below
            rows = await pool.fetch(
                """
                SELECT source_text, target_text, target_language,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM translation_memory
                WHERE target_language = $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                str(embedding),
                target_language,
                top_k,
            )

            if not rows:
                match_type = "new"
                stats["new"] += 1
                rag_matches.append({
                    "segment": segment,
                    "match_type": "new",
                    "matches": [],
                })
                continue

            best_score = float(rows[0]["similarity"])

            if best_score >= exact_threshold:
                match_type = "exact"
                stats["exact"] += 1
            elif best_score >= fuzzy_threshold:
                match_type = "fuzzy"
                stats["fuzzy"] += 1
            else:
                match_type = "new"
                stats["new"] += 1

            rag_matches.append({
                "segment": segment,
                "match_type": match_type,
                "best_score": round(best_score, 4),
                "matches": [
                    {
                        "source": r["source_text"],
                        "translation": r["target_text"],
                        "score": round(float(r["similarity"]), 4),
                    }
                    for r in rows
                ],
            })

        return {
            **context,
            "rag_matches": rag_matches,
            "rag_stats": stats,
        }


# ─── SQL to create the translation_memory table ──────────────────────────────
#
# Run this once in your PostgreSQL instance:
#
# CREATE EXTENSION IF NOT EXISTS vector;
#
# CREATE TABLE translation_memory (
#   id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
#   source_text TEXT NOT NULL,
#   target_text TEXT NOT NULL,
#   source_language TEXT NOT NULL DEFAULT 'en',
#   target_language TEXT NOT NULL,
#   embedding vector(1024),   -- multilingual-e5-large produces 1024-dim vectors
#   created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
# );
#
# CREATE INDEX ON translation_memory
#   USING ivfflat (embedding vector_cosine_ops)
#   WITH (lists = 100);