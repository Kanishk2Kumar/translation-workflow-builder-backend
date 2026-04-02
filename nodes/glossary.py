import re
from nodes.base import BaseNode
from db import get_pool


class GlossaryNode(BaseNode):
    """
    Fetches user-specific glossary terms for the target language
    and injects them into context so LLMAgentNode can use them in the prompt.

    Also does pre-translation term locking: if a source term has an exact
    match in the glossary, it's replaced with a locked placeholder
    {{GLOSSARY_term}} that the LLM is instructed to keep as-is, then
    restored with the correct target term after translation.
    """

    async def run(self, context: dict) -> dict:
        user_id: str = context.get("user_id", "")
        target_language: str = context.get("target_language", "hi")
        source_lang: str = context.get("source_language", "en")
        segments: list[str] = context.get("segments", [])

        if not user_id:
            print("⚠️  GlossaryNode: no user_id in context — skipping")
            return {**context, "glossary_terms": [], "glossary_map": {}}

        # Fetch all glossary terms for this user + language pair
        pool = get_pool()
        rows = await pool.fetch(
            """
            SELECT source_term, target_term, case_sensitive, domain
            FROM glossary_terms
            WHERE user_id = $1
              AND source_lang = $2
              AND target_lang = $3
            ORDER BY length(source_term) DESC  -- longer terms first to avoid partial matches
            """,
            user_id,
            source_lang,
            target_language,
        )

        if not rows:
            return {**context, "glossary_terms": [], "glossary_map": {}}

        glossary_terms = [dict(r) for r in rows]
        glossary_map: dict[str, str] = {}        # placeholder → target_term
        locked_segments: list[str] = list(segments)

        for row in glossary_terms:
            source_term = row["source_term"]
            target_term = row["target_term"]
            case_sensitive = row["case_sensitive"]

            # Create a safe placeholder that survives translation
            # Use the source term slug so it's readable in the prompt
            slug = re.sub(r"[^a-zA-Z0-9]", "_", source_term)
            placeholder = f"{{{{GLOS_{slug}}}}}"
            glossary_map[placeholder] = target_term

            flags = 0 if case_sensitive else re.IGNORECASE
            pattern = r"\b" + re.escape(source_term) + r"\b"

            locked_segments = [
                re.sub(pattern, placeholder, seg, flags=flags)
                for seg in locked_segments
            ]

        print(f"✅ GlossaryNode: {len(glossary_terms)} terms, "
              f"{len(glossary_map)} placeholders injected")

        return {
            **context,
            "segments": locked_segments,
            "raw_text": "\n".join(locked_segments),
            "glossary_terms": glossary_terms,
            "glossary_map": glossary_map,           # passed to LLM prompt builder
        }