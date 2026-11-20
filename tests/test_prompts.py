"""The SER prompt must equal what audiojev_inference builds for the same Choice."""
from audiojev_inference import Choice
from audiojev_inference.api import _prepare, render_prompt

from sersys.prompts import load_label_space, render_choice_text


def test_reference_order_and_text_match_library(cfg):
    for lang in ("en", "vi"):
        space = load_label_space(cfg["labels"], lang)
        kind, instructions, options = _prepare(Choice(instructions=space.question, criteria=space.criteria()))
        assert kind == "choice"
        assert [space.classes[i] for i in space.reference_order] == [key for key, _ in options]
        ours = render_choice_text(space.question, [space.descriptions[i] for i in space.reference_order])
        assert ours == render_prompt(instructions, options)
