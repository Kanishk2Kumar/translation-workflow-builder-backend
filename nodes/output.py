import json
from datetime import datetime, timezone
from nodes.base import BaseNode
from db import get_pool


class OutputNode(BaseNode):
    """
    Packages the final result and writes it to the executions table.
    Seeds the translation memory with approved translations.
    """

    async def run(self, context: dict) -> dict:
        output_format: str = self.config.get("format", "json")
        include_audit: bool = self.config.get("include_audit", True)

        translated_text: str = context.get("translated_text", "")
        rag_stats: dict = context.get("rag_stats", {})
        logs: list = context.get("_logs", [])
        execution_id: str = context.get("execution_id", "")
        workflow_id: str = context.get("workflow_id", "")
        raw_text: str = context.get("raw_text", "")           # ← pulled from context
        segments: list = context.get("segments", [raw_text])  # ← safe fallback to full text

        output_payload = {
            "translated_text": translated_text,
            "source_language": "en",
            "target_language": context.get("target_language", "hi"),
            "segment_count": context.get("segment_count", 0),
            "rag_stats": rag_stats,
            "model": context.get("llm_model", ""),
            "tm_hit": context.get("tm_hit", False),
            "token_usage": {
                "input": context.get("input_tokens", 0),
                "output": context.get("output_tokens", 0),
            },
        }

        if include_audit:
            output_payload["audit"] = {
                "execution_id": execution_id,
                "workflow_id": workflow_id,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "node_logs": logs,
            }

        # ── Get pool once, reuse for all DB writes ────────────────────────────
        pool = get_pool()

        # Write final status + output to executions table
        if execution_id:
            await pool.execute(
                """
                UPDATE executions
                SET status = 'success',
                    output = $1,
                    logs   = $2,
                    completed_at = CURRENT_TIMESTAMP
                WHERE id = $3
                """,
                json.dumps(output_payload),
                json.dumps(logs),
                execution_id,
            )

        # ── Seed translation memory (only when LLM was used, not a TM hit) ───
        if translated_text and not context.get("tm_hit") and segments:
            try:
                from nodes.rag_tm import embed
                embeddings = embed(segments)
                for segment, embedding in zip(segments, embeddings):
                    await pool.execute(
                        """
                        INSERT INTO translation_memory
                          (source_text, target_text, source_language, target_language, embedding)
                        VALUES ($1, $2, 'en', $3, $4::vector)
                        ON CONFLICT DO NOTHING
                        """,
                        segment,
                        translated_text,
                        context.get("target_language", "hi"),
                        str(embedding),
                    )
            except Exception as e:
                # TM seeding failure should never crash the response
                print(f"⚠️  TM seeding failed: {e}")

        return {
            **context,
            "final_output": output_payload,
        }