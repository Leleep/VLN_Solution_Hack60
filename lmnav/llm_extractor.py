"""
LLM Landmark Extractor
======================
Extracts ordered landmarks from natural language navigation instructions.

Backends (in priority order):
  1. Ollama (llama3) — primary, runs locally
  2. spaCy — offline fallback, noun-chunk extraction
  3. OpenAI — optional cloud fallback

Few-shot prompt adapted from lm_nav/landmark_extraction.py (lines 7-21)
for indoor navigation context.
"""

import json
import re
from typing import List, Optional

import requests


# ─── Few-Shot Prompt ─────────────────────────────────────────────────────────
# Adapted from lm_nav paper Appendix A / lm_nav/landmark_extraction.py
# Changed from outdoor (buildings, streets) to indoor (furniture, rooms)

INDOOR_FEW_SHOT_PROMPT = """You extract ordered landmarks from indoor navigation instructions.
Given a navigation instruction, return ONLY the ordered list of visual landmarks 
that the robot should look for, in the order they appear along the path.

Examples:

Instruction: Go past the sofa and find the TV in the living room.
Ordered landmarks:
1. a sofa
2. a TV

Instruction: Walk to the kitchen. You will pass a dining table before reaching the refrigerator.
Ordered landmarks:
1. a dining table
2. a refrigerator

Instruction: Go to the bedroom and look for the bed near the wardrobe.
Ordered landmarks:
1. a wardrobe
2. a bed

Instruction: Walk down the hallway past the shoe rack. Turn and find the coffee table near the sofa.
Ordered landmarks:
1. a shoe rack
2. a coffee table
3. a sofa

Instruction: {instruction}
Ordered landmarks:
1."""


def extract_landmarks(
    instruction: str,
    backend: str = "ollama",
    model: str = "llama3",
    ollama_host: str = "http://localhost:11434",
) -> List[str]:
    """
    Extract ordered landmarks from a navigation instruction.
    
    Args:
        instruction: Natural language navigation instruction
        backend: "ollama" | "spacy" | "openai"
        model: LLM model name (for ollama/openai backends)
        ollama_host: Ollama server URL
        
    Returns:
        Ordered list of landmark strings, e.g. ["a sofa", "a TV"]
    """
    if backend == "ollama":
        try:
            return _extract_ollama(instruction, model, ollama_host)
        except Exception as e:
            print(f"⚠️  Ollama failed ({e}), falling back to spaCy")
            return _extract_spacy(instruction)
    elif backend == "spacy":
        return _extract_spacy(instruction)
    elif backend == "openai":
        return _extract_openai(instruction, model)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'ollama', 'spacy', or 'openai'.")


def _extract_ollama(
    instruction: str,
    model: str = "llama3",
    host: str = "http://localhost:11434",
) -> List[str]:
    """Extract landmarks using Ollama (local LLM) via chat API."""

    system_msg = (
        "You extract ordered visual landmarks from indoor navigation instructions. "
        "Given an instruction, output ONLY a numbered list of landmarks in order. "
        "Each landmark should be a short noun phrase like 'a sofa' or 'a refrigerator'. "
        "Do NOT include any explanation, just the numbered list."
    )

    user_msg = f"""Examples:

Instruction: Go past the sofa and find the TV in the living room.
Ordered landmarks:
1. a sofa
2. a TV

Instruction: Walk to the kitchen. You will pass a dining table before reaching the refrigerator.
Ordered landmarks:
1. a dining table
2. a refrigerator

Instruction: Go to the bedroom and look for the bed near the wardrobe.
Ordered landmarks:
1. a wardrobe
2. a bed

Instruction: Walk down the hallway past the shoe rack. Turn and find the coffee table near the sofa.
Ordered landmarks:
1. a shoe rack
2. a coffee table
3. a sofa

Now extract landmarks for this instruction:

Instruction: {instruction}
Ordered landmarks:"""

    print(f"🧠 Querying Ollama ({model})...")
    response = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 256,
            },
        },
        timeout=60,
    )
    response.raise_for_status()

    result_text = response.json()["message"]["content"]
    print(f"   LLM raw response: {result_text[:200]}")
    landmarks = _parse_numbered_list(result_text)

    if not landmarks:
        print(f"⚠️  Could not parse landmarks from LLM response: {result_text[:200]}")
        print("    Falling back to spaCy...")
        return _extract_spacy(instruction)

    print(f"✅ Extracted landmarks: {landmarks}")
    return landmarks


def _extract_spacy(instruction: str) -> List[str]:
    """
    Extract landmarks using spaCy noun-chunk extraction.
    
    Adapted from lm_nav/landmark_extraction.py lines 42-46:
        nlp = spacy.load("en_core_web_lg")
        doc = nlp(text)
        black_list = ["you", "left", "right", "left turn", "right turn"]
        return [chunk.text for chunk in doc.noun_chunks if ...]
    """
    try:
        import spacy
    except ImportError:
        raise ImportError("spaCy required for fallback. Install: pip install spacy && python -m spacy download en_core_web_sm")

    # Try loading models in order of preference
    nlp = None
    for model_name in ["en_core_web_lg", "en_core_web_sm"]:
        try:
            nlp = spacy.load(model_name)
            break
        except OSError:
            continue

    if nlp is None:
        raise RuntimeError(
            "No spaCy model found. Install one:\n"
            "  python -m spacy download en_core_web_sm"
        )

    print(f"🔤 Extracting landmarks with spaCy ({nlp.meta['name']})...")

    doc = nlp(instruction)

    # Blacklist from lm_nav + additions for indoor context
    blacklist = {
        "you", "i", "it", "we", "they",
        "left", "right", "left turn", "right turn",
        "turn", "end", "way", "direction", "side",
        "room", "area", "the room", "the area",
    }

    def _remove_articles(text: str) -> str:
        """Remove leading articles (a, an, the)."""
        words = text.split()
        articles = {"a", "an", "the"}
        filtered = [w for w in words if w.lower() not in articles]
        return " ".join(filtered)

    landmarks = []
    for chunk in doc.noun_chunks:
        clean = _remove_articles(chunk.text.lower())
        if clean and clean not in blacklist and len(clean) > 1:
            landmarks.append(chunk.text.lower())

    # Deduplicate while preserving order
    seen = set()
    unique_landmarks = []
    for lm in landmarks:
        if lm not in seen:
            seen.add(lm)
            unique_landmarks.append(lm)

    print(f"✅ Extracted landmarks (spaCy): {unique_landmarks}")
    return unique_landmarks


def _extract_openai(instruction: str, model: str = "gpt-3.5-turbo") -> List[str]:
    """Extract landmarks using OpenAI API (optional fallback)."""
    try:
        import openai
    except ImportError:
        raise ImportError("openai package required. Install: pip install openai")

    import os
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    prompt = INDOOR_FEW_SHOT_PROMPT.format(instruction=instruction)

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You extract ordered landmarks from navigation instructions."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=256,
    )

    result_text = response.choices[0].message.content
    landmarks = _parse_numbered_list(result_text)
    print(f"✅ Extracted landmarks (OpenAI): {landmarks}")
    return landmarks


def _parse_numbered_list(text: str) -> List[str]:
    """
    Parse a numbered list from LLM output.
    
    Handles formats like:
      "1. a sofa\n2. a TV\n"
      " a sofa\n2. a TV"  (first item may lack number since prompt ends with "1.")
    """
    # Prepend "1." if the text doesn't start with a number
    # (because our prompt already prints "1." and the LLM continues)
    text = text.strip()
    if text and not re.match(r'^\d+\.', text):
        text = "1. " + text

    lines = text.strip().split("\n")
    landmarks = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Match "1. a sofa" or "2. the refrigerator"
        match = re.match(r'^\d+\.\s*(.+)$', line)
        if match:
            landmark = match.group(1).strip()
            if landmark:
                landmarks.append(landmark)
        else:
            # Stop at first non-matching line (end of list)
            if landmarks:
                break

    return landmarks
