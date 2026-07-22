import argparse
import json
import os
from typing import Any, Dict

import torch

from data_utils import (
    build_fields_from_slot_lists,
    decode_bio_slots,
    encode_text,
    extract_slots_by_rules,
    render_template,
    infer_intent_by_rules,
)
from model import IntentClassifier


def load_runtime(artifact_dir: str, device):
    with open(os.path.join(artifact_dir, "metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    id2label = {int(k): v for k, v in meta["id2label"].items()}
    id2intent = {int(k): v for k, v in meta["id2intent"].items()}
    margs = meta["model_args"].copy()

    state_dict = torch.load(os.path.join(artifact_dir, "model.pt"), map_location=device)
    embedding_weight = state_dict.get("embedding.weight")
    if embedding_weight is not None:
        checkpoint_vocab_size = int(embedding_weight.shape[0])
        metadata_vocab_size = int(margs["vocab_size"])
        if checkpoint_vocab_size != metadata_vocab_size:
            margs["vocab_size"] = checkpoint_vocab_size

    model = IntentClassifier(**margs).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, meta, id2label, id2intent


INVALID_PROMPT_MESSAGE = (
    "Prompt is invalid or out of context. Please provide a supported "
    "web test prompt, such as navigating to a URL, filling fields, "
    "clicking an element, or comparing element text."
)


def validate_intent_requirements(intent: str, parsed: Dict[str, Any]) -> str | None:
    """Return a validation error when required prompt data is missing.

    The model can emit high-confidence slots for arbitrary input (for example,
    treating ``11111`` as a click target). Every current executable template
    starts with a navigate step, so accepting a plan without a URL creates an
    invalid Selenium script that fails at runtime instead of rejecting the
    prompt up front.
    """
    if intent.startswith("navigate_") and not parsed.get("url"):
        return "A supported web test prompt must include a target URL to navigate to."

    if intent == "navigate_click_by_id" and not parsed.get("click_id"):
        return "A click prompt must include a target element attribute, such as id='submit'."

    if intent in {"navigate_fill_fields", "navigate_fill_fields_click"} and not parsed.get("fields"):
        return "A fill prompt must include at least one field attribute and value."

    text_assertion = parsed.get("text_assertion") or {}
    if intent == "find_tag_text_compare" and not all(
        text_assertion.get(key) for key in ("tag_name", "attribute_name", "id", "value")
    ):
        return "A text comparison prompt must include the element locator and expected text."

    return None


def predict(prompt: str, artifact_dir: str = "artifacts_nfields", repair_with_rules: bool = True, repair_intent: bool = True) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta, id2label, id2intent = load_runtime(artifact_dir, device)

    max_len = meta["max_len"]
    input_ids, length = encode_text(prompt, meta["vocab"], max_len)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    lengths = torch.tensor([length], dtype=torch.long, device=device)

    with torch.no_grad():
        slot_logits, intent_logits = model(x, lengths)
        slot_ids = slot_logits.argmax(-1)[0].cpu().tolist()
        intent_id = int(intent_logits.argmax(-1)[0].cpu().item())

    intent = id2intent[intent_id]
    raw_slot_lists = decode_bio_slots(prompt[:max_len], slot_ids, id2label)
    parsed = {
        "url": raw_slot_lists.get("URL", [None])[0],
        "fields": build_fields_from_slot_lists(raw_slot_lists),
        "click_id": raw_slot_lists.get("CLICK_ID", [None])[0],
        "click_attribute_name": raw_slot_lists.get("CLICK_ATTRIB_NAME", ["id"])[0],
        "text_assertion": None,
    }

    if repair_with_rules:
        repaired = extract_slots_by_rules(prompt, intent=intent)
        if repaired.get("url"):
            parsed["url"] = repaired["url"]
        if repaired.get("fields"):
            parsed["fields"] = repaired["fields"]
        if repaired.get("click_id"):
            parsed["click_id"] = repaired["click_id"]
        if repaired.get("click_attribute_name"):
            parsed["click_attribute_name"] = repaired["click_attribute_name"]
        if repaired.get("text_assertion"):
            parsed["text_assertion"] = repaired["text_assertion"]

    model_intent = intent
    validation_error = None
    if repair_intent:
        repaired_intent = infer_intent_by_rules(prompt, parsed)
        if repaired_intent in meta["templates"]:
            intent = repaired_intent
        else:
            validation_error = INVALID_PROMPT_MESSAGE

    if not validation_error and intent is not None:
        validation_error = validate_intent_requirements(intent, parsed)

    if validation_error:
        return {
            "intent": None,
            "model_intent": model_intent,
            "raw_slot_lists": raw_slot_lists,
            "parsed": parsed,
            "json": None,
            "is_valid": False,
            "validation_error": validation_error,
        }

    if intent != "navigate_click_by_id":
        parsed["click_id"] = None

    final_json = render_template(meta["templates"][intent], parsed)
    return {
        "intent": intent,
        "model_intent": model_intent,
        "raw_slot_lists": raw_slot_lists,
        "parsed": parsed,
        "json": final_json,
        "is_valid": True,
        "validation_error": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts_attributes")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--no-repair", action="store_true")
    parser.add_argument("--no-intent-repair", action="store_true")
    args = parser.parse_args()

    result = predict(args.prompt, artifact_dir=args.artifacts, repair_with_rules=not args.no_repair, repair_intent=not args.no_intent_repair)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
