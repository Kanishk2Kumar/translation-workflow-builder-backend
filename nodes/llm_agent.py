from openai import OpenAI
from nodes.base import BaseNode
from config import settings

LANGUAGE_NAMES = {
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "mr": "Marathi",
    "ta": "Tamil",
    "te": "Telugu",
}

TONE_INSTRUCTIONS = {
    "clinical":         "Use formal clinical terminology. This text is for healthcare professionals.",
    "patient_friendly": "Use simple, clear language. This text is for patients with no medical background.",
    "formal":           "Use formal language appropriate for official documents.",
    "technical":        "Use precise technical language. Preserve all technical terms.",
}


def build_prompt(
    source_text: str,
    target_language: str,
    tone: str,
    rag_matches: list[dict],
    system_prompt: str | None,
) -> tuple[str, str]:
    lang_name = LANGUAGE_NAMES.get(target_language, target_language)
    tone_instruction = TONE_INSTRUCTIONS.get(tone, TONE_INSTRUCTIONS["formal"])

    examples_block = ""
    useful_matches = [
        m for m in rag_matches
        if m.get("match_type") in ("fuzzy",) and m.get("matches")
    ]
    if useful_matches:
        examples = []
        for match in useful_matches[:3]:
            best = match["matches"][0]
            examples.append(
                f"Source: {best['source']}\n"
                f"Translation ({lang_name}): {best['translation']}"
            )
        examples_block = (
            "\n\nReference translations from approved memory:\n"
            + "\n\n".join(examples)
        )

    system = system_prompt or (
        f"You are an expert medical translator specialising in English to {lang_name} translation. "
        f"{tone_instruction} "
        f"Never translate ICD codes, CPT codes, or MRN numbers — keep them exactly as-is. "
        f"Return only the translated text with no explanation or preamble."
    )

    user = (
        f"Translate the following medical text from English to {lang_name}."
        f"{examples_block}"
        f"\n\nText to translate:\n{source_text}"
    )

    return system, user


class LLMAgentNode(BaseNode):
    """
    Translates source text using OpenAI.
    - If ALL segments are exact TM matches → skip OpenAI entirely (instant)
    - If SOME segments are exact → use cached translations + OpenAI for new ones
    - If NO matches → full OpenAI translation
    """

    async def run(self, context: dict) -> dict:
        raw_text: str = context.get("raw_text", "")
        if not raw_text:
            raise ValueError("LLMAgentNode: no raw_text in context")

        target_language: str = context.get(
            "target_language", self.config.get("target_language", "hi")
        )
        tone: str = self.config.get("tone", "clinical")
        model: str = self.config.get("model", "gpt-4o")
        max_tokens: int = self.config.get("max_tokens", 2048)
        system_prompt: str | None = self.config.get("system_prompt")
        rag_matches: list = context.get("rag_matches", [])

        # ── Fast path: ALL segments are exact TM matches ──────────────────────
        if rag_matches and all(
            m.get("match_type") == "exact" and m.get("matches")
            for m in rag_matches
        ):
            translated_text = " ".join(
                m["matches"][0]["translation"] for m in rag_matches
            )
            return {
                **context,
                "translated_text": translated_text,
                "llm_model": "translation_memory",
                "input_tokens": 0,
                "output_tokens": 0,
                "tm_hit": True,
            }

        # ── Partial path: some exact, some new — build combined text ──────────
        # Exact segments reuse cached translation; new segments go to OpenAI
        new_segments = []
        cached_parts = {}

        for match in rag_matches:
            seg = match["segment"]
            if match.get("match_type") == "exact" and match.get("matches"):
                cached_parts[seg] = match["matches"][0]["translation"]
            else:
                new_segments.append(seg)

        # Text to translate = only the new segments (or full text if no rag_matches)
        text_to_translate = (
            " ".join(new_segments) if new_segments else raw_text
        )

        system, user = build_prompt(
            source_text=text_to_translate,
            target_language=target_language,
            tone=tone,
            rag_matches=rag_matches,
            system_prompt=system_prompt,
        )

        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )

        llm_translation = response.choices[0].message.content.strip()

        # Merge cached + LLM translations
        if cached_parts:
            cached_text = " ".join(cached_parts.values())
            translated_text = f"{cached_text} {llm_translation}".strip()
        else:
            translated_text = llm_translation

        return {
            **context,
            "translated_text": translated_text,
            "llm_model": model,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "tm_hit": False,
        }