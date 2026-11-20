"""Batch round trip (PLAN 1.2): shapes, masks, readout positions, candidate order, audio features."""
import torch

from sersys.prompts import LABELS
from sersys.scoring import ChoiceCollator, build_choice_batch, label_token_ids, random_perms, tta_perms


def _batch(processor, space, audios, seed):
    label_ids = label_token_ids(processor.tokenizer, space.num_classes)
    perms = random_perms(len(audios), space.num_classes, torch.Generator().manual_seed(seed))
    return build_choice_batch(processor, audios, space.question, space.descriptions, perms, label_ids), perms


def test_batch_shapes_and_readout(processor, space, synthetic_audios):
    b, _ = _batch(processor, space, synthetic_audios, 0)
    B = len(synthetic_audios)
    assert b["input_ids"].shape == b["attention_mask"].shape and b["input_ids"].shape[0] == B
    assert b["input_features"].shape[:2] == (B, 128)
    assert b["feature_attention_mask"].shape == (B, b["input_features"].shape[2])
    am = b["attention_mask"]
    for i in range(B):
        pos = int(b["readout_pos"][i])
        assert am[i, pos] == 1 and am[i, pos + 1:].sum() == 0
        # The prompt ends with the generation prompt, so the next token is the answer label.
        tail = processor.tokenizer.decode(b["input_ids"][i, : pos + 1][-5:])
        assert tail.endswith("<|im_start|>assistant\n"), tail
    assert b["label_ids"].tolist() == label_token_ids(processor.tokenizer, space.num_classes)


def test_candidate_order_in_text(processor, space, synthetic_audios):
    b, perms = _batch(processor, space, synthetic_audios, 1)
    for i in range(len(synthetic_audios)):
        text = processor.tokenizer.decode(b["input_ids"][i][b["attention_mask"][i].bool()])
        for j, k in enumerate(perms[i].tolist()):
            assert f"\n{LABELS[j]}. {space.descriptions[k]}\n" in text


def test_batched_equals_single(processor, space, synthetic_audios):
    """Each row of a padded batch holds exactly the tokens and valid features of its clip alone."""
    b, perms = _batch(processor, space, synthetic_audios, 2)
    label_ids = b["label_ids"].tolist()
    for i, audio in enumerate(synthetic_audios):
        s = build_choice_batch(processor, [audio], space.question, space.descriptions, perms[i:i + 1], label_ids)
        assert torch.equal(b["input_ids"][i][b["attention_mask"][i].bool()], s["input_ids"][0])
        frames = int(s["feature_attention_mask"].sum())
        assert int(b["feature_attention_mask"][i].sum()) == frames
        torch.testing.assert_close(b["input_features"][i, :, :frames], s["input_features"][0, :, :frames],
                                   atol=1e-5, rtol=0)


def test_collator_views(processor, space, synthetic_audios):
    label_ids = label_token_ids(processor.tokenizer, space.num_classes)
    items = [{"id": f"u{i}", "wave": a, "label": i % 7, "index": i} for i, a in enumerate(synthetic_audios)]
    out = ChoiceCollator(processor, space, label_ids, mode="fixed", perms=tta_perms(space.reference_order, 3))(items)
    assert len(out["views"]) == 3
    assert out["views"][0]["perm"][0].tolist() == space.reference_order
    assert out["labels"].tolist() == [it["label"] for it in items]
    out = ChoiceCollator(processor, space, label_ids, mode="random")(items)
    assert len(out["views"]) == 1 and out["views"][0]["perm"].shape == (len(items), space.num_classes)
