import re

from openai import OpenAI
from nodes.base import BaseNode
from config import settings

LANGUAGE_NAMES = {
    "hi": "Hindi", "es": "Spanish", "fr": "French", "de": "German",
    "ja": "Japanese", "mr": "Marathi", "ta": "Tamil", "te": "Telugu",
}

TONE_INSTRUCTIONS = {
    "clinical": "Use formal clinical terminology. This text is for healthcare professionals.",
    "patient_friendly": "Use simple, clear language. This text is for patients with no medical background.",
    "formal": "Use formal language appropriate for official documents.",
    "technical": "Use precise technical language. Preserve all technical terms.",
}


def build_prompt(source_text, target_language, tone, rag_matches, system_prompt, glossary_terms=None):
    lang_name = LANGUAGE_NAMES.get(target_language, target_language)
    tone_instruction = TONE_INSTRUCTIONS.get(tone, TONE_INSTRUCTIONS["formal"])

    # Build glossary instruction block
    glossary_block = ""
    if glossary_terms:
        term_lines = "\n".join(
            f'  - "{t["source_term"]}" must always be translated as "{t["target_term"]}"'
            for t in glossary_terms[:30]
        )
        glossary_block = (
            "\n\nMANDATORY GLOSSARY - these translations are fixed and must not be altered:\n"
            f"{term_lines}\n"
            "Do not paraphrase, synonymize, or skip any of the above terms."
        )

    # RAG fuzzy match examples block
    examples_block = ""
    useful_matches = [m for m in rag_matches if m.get("match_type") == "fuzzy" and m.get("matches")]
    if useful_matches:
        examples = [
            f"Source: {m['matches'][0]['source']}\nTranslation ({lang_name}): {m['matches'][0]['translation']}"
            for m in useful_matches[:3]
        ]
        examples_block = "\n\nReference translations from approved memory:\n" + "\n\n".join(examples)

    system = system_prompt or (
        f"You are an expert translator specialising in English to {lang_name} translation. "
        f"{tone_instruction} "
        "Never translate ICD codes, CPT codes, MRN numbers, or passport numbers - keep them exactly as-is. "
        "Return only the translated text with no explanation or preamble."
        f"{glossary_block}"
    )

    user = (
        f"Translate the following text from English to {lang_name}."
        f"{examples_block}"
        f"\n\nText to translate:\n{source_text}"
    )

    return system, user


def restore_glossary(text: str, glossary_map: dict[str, str]) -> str:
    """
    Post-translation correction pass.
    1. If the source term still appears in translated text -> replace with target term
    2. No placeholder logic needed - terms go via prompt instructions
    """
    if not glossary_map:
        return text

    for source_term, target_term in glossary_map.items():
        pattern = r"(?<!\w)" + re.escape(source_term) + r"(?!\w)"
        if re.search(pattern, text, flags=re.IGNORECASE):
            text = re.sub(pattern, target_term, text, flags=re.IGNORECASE)
            print(f"glossary correction: '{source_term}' -> '{target_term}'")

    return text


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
        glossary_terms = context.get("glossary_terms", [])
        glossary_map: dict[str, str] = context.get("glossary_map", {})

        print(f"DEBUG llm_agent: glossary_map has {len(glossary_map)} entries: {glossary_map}")

        # Fast path: ALL segments exact TM hits
        if rag_matches and all(
            m.get("match_type") == "exact" and m.get("matches")
            for m in rag_matches
        ):
            segment_translations = {
                m["segment"]: restore_glossary(m["matches"][0]["translation"], glossary_map)
                for m in rag_matches
            }
            translated_text = "\n".join(
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

        # Build per-segment translation map
        # For every segment: exact -> use cache, anything else -> call OpenAI
        segment_translations: dict[str, str] = {}

        if rag_matches:
            for match in rag_matches:
                seg = match["segment"]
                if match.get("match_type") == "exact" and match.get("matches"):
                    segment_translations[seg] = restore_glossary(
                        match["matches"][0]["translation"],
                        glossary_map,
                    )
                else:
                    system, user = build_prompt(
                        source_text=seg,
                        target_language=target_language,
                        tone=tone,
                        rag_matches=[match],
                        system_prompt=system_prompt,
                        glossary_terms=glossary_terms,
                    )
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            max_tokens=max_tokens,
                            messages=[
                                {"role": "system", "content": system},
                                {"role": "user", "content": user},
                            ],
                        )
                        raw_translation = response.choices[0].message.content.strip()
                        print(f"DEBUG raw_translation: '{raw_translation[:100]}'")
                        segment_translations[seg] = restore_glossary(raw_translation, glossary_map)
                        total_input_tokens += response.usage.prompt_tokens
                        total_output_tokens += response.usage.completion_tokens
                    except Exception as e:
                        print(f"LLM failed for segment: {e}")
                        segment_translations[seg] = restore_glossary(seg, glossary_map)

            translated_text = "\n".join(
                segment_translations.get(m["segment"], m["segment"])
                for m in rag_matches
            )

        else:
            # No RAG - full text single call
            system, user = build_prompt(
                source_text=raw_text,
                target_language=target_language,
                tone=tone,
                rag_matches=[],
                system_prompt=system_prompt,
                glossary_terms=glossary_terms,
            )
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            raw_translation = response.choices[0].message.content.strip()
            print(f"DEBUG raw_translation: '{raw_translation[:100]}'")
            translated_text = restore_glossary(raw_translation, glossary_map)
            segment_translations = {}
            total_input_tokens = response.usage.prompt_tokens
            total_output_tokens = response.usage.completion_tokens

        print(
            f"DEBUG llm: {len(segment_translations)} segments translated, "
            f"tokens in={total_input_tokens} out={total_output_tokens}"
        )
        print(
            f"DEBUG llm: rag_matches={len(rag_matches)}, "
            f"segment_translations={len(segment_translations)}, tm_hit={False}"
        )
        return {
            **context,
            "translated_text": translated_text,
            "segment_translations": segment_translations,
            "llm_model": model,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "tm_hit": False,
        }
