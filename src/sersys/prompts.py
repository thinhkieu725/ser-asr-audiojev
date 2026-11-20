"""SER choice prompt builder, reproducing audiojev_inference's Choice prompt.

AudioJev (audiojev_inference.api / backend) builds a Choice prompt like this:

* Options are the criteria descriptions. If they are all distinct and none uses a
  relative phrase ("none of the above", "option B", "the latter", ...), the library
  sorts them by their UTF-8 bytes. That sorted order is the *reference order* here.
* Option j is labelled LABELS[j] ("0", "1", ..., "9", "A", ...), one token each.
* The user turn is [audio, text]; the text is `render_prompt(instructions, options)`.
  It is wrapped in the Qwen2.5-Omni chat template with the fixed system prompt and
  `add_generation_prompt=True`.
* The answer distribution is softmax over the logits of the label tokens at the last
  prompt position (see scoring.py). There is no post-hoc calibration step.

The prompt strings are imported from the library itself so they cannot drift.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from audiojev_inference.api import LABELS, _RELATIVE, render_prompt
from audiojev_inference.backend import SYSTEM_PROMPT

__all__ = ["LABELS", "SYSTEM_PROMPT", "LabelSpace", "load_label_space", "reference_order",
           "render_choice_text", "choice_conversation", "question_with_transcript", "asr_conversation"]


class LabelSpace:
    """Canonical classes plus candidate descriptions and question for one prompt language."""

    def __init__(self, classes: list[str], descriptions: list[str], question: str):
        if len(classes) != len(descriptions):
            raise ValueError("one description per class required")
        if not 2 <= len(classes) <= 36:
            raise ValueError("AudioJev Choice supports 2-36 options")
        self.classes = list(classes)
        self.descriptions = [d.strip() for d in descriptions]
        self.question = question.strip()
        self.reference_order = reference_order(self.descriptions)

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def criteria(self) -> dict[str, str]:
        """The equivalent `Choice(criteria=...)` mapping for audiojev_inference."""
        return dict(zip(self.classes, self.descriptions))


def load_label_space(labels_path: str | Path, lang: str, question: str | None = None) -> LabelSpace:
    spec = yaml.safe_load(Path(labels_path).read_text(encoding="utf-8"))
    classes = spec["classes"]
    desc = spec["descriptions"][lang]
    if set(desc) != set(classes):
        raise ValueError(f"descriptions[{lang}] keys {sorted(desc)} != classes {sorted(classes)}")
    return LabelSpace(classes, [desc[c] for c in classes], question or spec["questions"][lang])


def reference_order(descriptions: list[str]) -> list[int]:
    """Canonical class indices in the order audiojev_inference would display them."""
    idx = list(range(len(descriptions)))
    texts = [d.strip() for d in descriptions]
    if len(set(texts)) == len(texts) and not any(_RELATIVE.search(t) for t in texts):
        idx.sort(key=lambda i: texts[i].encode("utf-8"))
    return idx


def render_choice_text(question: str, shown_descriptions: list[str]) -> str:
    """The user-turn text with options listed in the given (already permuted) order."""
    return render_prompt(question, [(str(j), d) for j, d in enumerate(shown_descriptions)])


def choice_conversation(question: str, shown_descriptions: list[str]) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "audio", "audio": "audio.wav"},
                                     {"type": "text", "text": render_choice_text(question, shown_descriptions)}]},
    ]


def question_with_transcript(question: str, transcript: str, template: str) -> str:
    """E2.4 transcribe-then-classify: the transcript goes into the Choice instructions."""
    return question + "\n" + template.format(transcript=transcript.strip())


def asr_conversation(instruction: str) -> list[dict]:
    """ASR prompt (PLAN 2.1): same system prompt and audio slot, a fixed instruction."""
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "audio", "audio": "audio.wav"},
                                     {"type": "text", "text": instruction}]},
    ]
