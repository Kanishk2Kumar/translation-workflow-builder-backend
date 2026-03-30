import base64
import json
import os
import tempfile
from datetime import datetime, timezone
from nodes.base import BaseNode
from db import get_pool

# Persistent temp dir for translated documents
TRANSLATED_DOCS_DIR = "/tmp/translatio_outputs"
os.makedirs(TRANSLATED_DOCS_DIR, exist_ok=True)


class OutputNode(BaseNode):

    async def run(self, context: dict) -> dict:
        include_audit: bool = self.config.get("include_audit", True)

        translated_text: str = context.get("translated_text", "")
        rag_stats: dict = context.get("rag_stats", {})
        logs: list = context.get("_logs", [])
        execution_id: str = context.get("execution_id", "")
        workflow_id: str = context.get("workflow_id", "")
        raw_text: str = context.get("raw_text", "")

        output_doc_bytes: bytes | None = context.get("output_document_bytes")
        output_doc_format: str = context.get("output_document_format", "text")

        # ── Save document to disk, store path in DB ───────────────────────────
        doc_file_path: str | None = None
        if output_doc_bytes and execution_id:
            ext = output_doc_format if output_doc_format in ("docx", "pdf") else "bin"
            doc_file_path = os.path.join(TRANSLATED_DOCS_DIR, f"{execution_id}.{ext}")
            with open(doc_file_path, "wb") as f:
                f.write(output_doc_bytes)

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
            # Store path + format in DB — not the bytes themselves
            "document_path": doc_file_path,
            "document_format": output_doc_format if output_doc_bytes else None,
        }

        if include_audit:
            output_payload["audit"] = {
                "execution_id": execution_id,
                "workflow_id": workflow_id,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "node_logs": logs,
            }

        pool = get_pool()

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

        # ── Seed TM ───────────────────────────────────────────────────────────
        if translated_text and not context.get("tm_hit"):
            segment_translations: dict = context.get("segment_translations", {})

            if segment_translations:
                try:
                    from nodes.rag_tm import embed
                    segs = list(segment_translations.keys())
                    embeddings = embed(segs)

                    for segment, embedding in zip(segs, embeddings):
                        target_text = segment_translations[segment]

                        existing = await pool.fetchval(
                            """
                            SELECT id FROM translation_memory
                            WHERE source_text = $1 AND target_language = $2
                            LIMIT 1
                            """,
                            segment,
                            context.get("target_language", "hi"),
                        )
                        if not existing:
                            await pool.execute(
                                """
                                INSERT INTO translation_memory
                                  (source_text, target_text, source_language, target_language, embedding)
                                VALUES ($1, $2, 'en', $3, $4::vector)
                                """,
                                segment,
                                target_text,
                                context.get("target_language", "hi"),
                                str(embedding),
                            )
                except Exception as e:
                    print(f"⚠️  TM seeding failed: {e}")

        # Return bytes in context so the route handler can stream them directly
        # without a second DB/disk read if it wants to
        return {
            **context,
            "final_output": output_payload,
            "output_document_bytes": output_doc_bytes,  # keep in context for route
        }