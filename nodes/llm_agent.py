from openai import OpenAI
from nodes.base import BaseNode
from config import settings

LANGUAGE_NAMES = {
    "hi": "Hindi", "es": "Spanish", "fr": "French", "de": "German",
    "ja": "Japanese", "mr": "Marathi", "ta": "Tamil", "te": "Telugu",
}

TONE_INSTRUCTIONS = {
    "clinical":         "Use formal clinical terminology. This text is for healthcare professionals.",
    "patient_friendly": "Use simple, clear language. This text is for patients with no medical background.",
    "formal":           "Use formal language appropriate for official documents.",
    "technical":        "Use precise technical language. Preserve all technical terms.",
}


def build_prompt(source_text, target_language, tone, rag_matches, system_prompt):
    lang_name = LANGUAGE_NAMES.get(target_language, target_language)
    tone_instruction = TONE_INSTRUCTIONS.get(tone, TONE_INSTRUCTIONS["formal"])

    examples_block = ""
    useful_matches = [
        m for m in rag_matches
        if m.get("match_type") == "fuzzy" and m.get("matches")
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
        f"Translate the following text from English to {lang_name}."
        f"{examples_block}"
        f"\n\nText to translate:\n{source_text}"
    )

    return system, user


class LLMAgentNode(BaseNode):

    async def run(self, context: dict) -> dict:
        raw_text: str = context.get("raw_text", "")
        if not raw_text:
            raise ValueError("LLMAgentNode: no raw_text in context")

        target_language: str = context.get("target_language", self.config.get("target_language", "hi"))
        tone: str = self.config.get("tone", "formal")
        model: str = self.config.get("model", "gpt-4o")
        max_tokens: int = self.config.get("max_tokens", 2048)
        system_prompt: str | None = self.config.get("system_prompt")
        rag_matches: list = context.get("rag_matches", [])

        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        total_input_tokens = 0
        total_output_tokens = 0

        # ── Fast path: ALL segments exact TM hits ─────────────────────────────
        if rag_matches and all(
            m.get("match_type") == "exact" and m.get("matches")
            for m in rag_matches
        ):
            segment_translations = {
                m["segment"]: m["matches"][0]["translation"] for m in rag_matches
            }
            translated_text = " ".join(
                segment_translations[m["segment"]] for m in rag_matches
            )
            return {
                **context,
                "translated_text": translated_text,
                "segment_translations": segment_translations,
                "llm_model": "translation_memory",
                "input_tokens": 0,
                "output_tokens": 0,
                "tm_hit": True,
            }

        # ── Build per-segment translation map ────────────────────────────────
        # For every segment: exact → use cache, anything else → call OpenAI
        segment_translations: dict[str, str] = {}

        if rag_matches:
            for match in rag_matches:
                seg = match["segment"]

                if match.get("match_type") == "exact" and match.get("matches"):
                    # Reuse cached translation directly
                    segment_translations[seg] = match["matches"][0]["translation"]

                else:
                    # fuzzy or new → call OpenAI for this segment
                    system, user = build_prompt(
                        source_text=seg,
                        target_language=target_language,
                        tone=tone,
                        rag_matches=[match],   # pass this segment's own matches as context
                        system_prompt=system_prompt,
                    )
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            max_tokens=max_tokens,
                            messages=[
                                {"role": "system", "content": system},
                                {"role": "user",   "content": user},
                            ],
                        )
                        segment_translations[seg] = response.choices[0].message.content.strip()
                        total_input_tokens  += response.usage.prompt_tokens
                        total_output_tokens += response.usage.completion_tokens
                    except Exception as e:
                        print(f"⚠️  LLM failed for segment '{seg[:40]}...': {e}")
                        segment_translations[seg] = seg  # fallback: keep original

            # Assemble final text preserving original segment order
            translated_text = "\n".join(
                segment_translations.get(m["segment"], m["segment"])
                for m in rag_matches
            )

        else:
            # No RAG node in pipeline — translate full text as one call
            system, user = build_prompt(
                source_text=raw_text,
                target_language=target_language,
                tone=tone,
                rag_matches=[],
                system_prompt=system_prompt,
            )
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            translated_text = response.choices[0].message.content.strip()
            segment_translations = {}  # no per-segment map without RAG
            total_input_tokens  = response.usage.prompt_tokens
            total_output_tokens = response.usage.completion_tokens

        print(f"DEBUG llm: {len(segment_translations)} segments translated, "
              f"tokens in={total_input_tokens} out={total_output_tokens}")
        print(f"DEBUG llm: rag_matches={len(rag_matches)}, " f"segment_translations={len(segment_translations)}, "f"tm_hit={False}")
        return {
            **context,
            "translated_text": translated_text,
            "segment_translations": segment_translations,  # ← THIS is what rebuilder needs
            "llm_model": model,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "tm_hit": False,
        }