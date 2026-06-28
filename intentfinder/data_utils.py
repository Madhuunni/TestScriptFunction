import copy
import json
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Generic slot labels.
# ATTRIB_NAME + FIELD_ID can repeat N times:
#   input[id='email']
#          ^   ^^^^^
#          |   FIELD_ID
#          ATTRIB_NAME
# -----------------------------
SLOT_NAMES = [
    "URL",
    "ATTRIB_NAME",
    "FIELD_ID",
    "FIELD_VALUE",
    "ENV_VALUE",
    "CLICK_ATTRIB_NAME",
    "CLICK_ID",
    "ELEMENT_TAG",
]
LABELS = ["O"]
for slot in SLOT_NAMES:
    LABELS.extend([f"B-{slot}", f"I-{slot}"])
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)
# Attribute names supported by CSS: id, name, data-testid, aria-label, formControlName, etc.
ATTR_NAME = r"[A-Za-z_][A-Za-z0-9_:\-.]*"
ATTR_RE = re.compile(
    rf"(?P<attr>{ATTR_NAME})\s*=\s*['\"](?P<field_id>[^'\"]+)['\"]",
    re.IGNORECASE,
)
ID_RE = re.compile(r"id\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
QUOTED_RE = re.compile(r"'([^']*)'")
ENV_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")

TAG_RE = re.compile(r"\b(?:tag|element)\s+([A-Za-z][A-Za-z0-9_-]*)\b", re.IGNORECASE)
FIND_TAG_RE = re.compile(r"\b(?:find|locate)\s+(?P<tag>[A-Za-z][A-Za-z0-9_-]*)\b", re.IGNORECASE)
TEXT_COMPARE_RE = re.compile(
    r"(?:compare|verify|check|assert)[^.]{0,80}?(?:text\s+value|text|value)[^.]{0,40}?(?:is|equals|equal\s+to|to\s+be)\s*['\"](?P<text>[^'\"]+)['\"]",
    re.IGNORECASE,
)
GET_TEXT_RE = re.compile(r"\bget\s+the\s+text\s+value\b|\btext\s+value\b", re.IGNORECASE)

# Pair regexes. Each pattern must expose named groups:
#   attr, field_id and one of value/env.
PAIR_PATTERNS = [
    # Enter 'abc' in id='email' / Type 'abc' into input name='email'
    re.compile(
        rf"(?P<value>'[^']*')\s+(?:in|into|using|having|to|for|with)"
        rf"(?:\s+the\s+attribute)?(?:\s+input|\s+field)?[^.]{{0,80}}?"
        rf"(?P<attr>{ATTR_NAME})\s*=\s*['\"](?P<field_id>[^'\"]+)['\"]",
        re.IGNORECASE,
    ),
    # Enter 'abc' having the attribute data-testid='email-input'
    re.compile(
        rf"(?P<value>'[^']*')\s+having\s+the\s+attribute\s+"
        rf"(?P<attr>{ATTR_NAME})\s*=\s*['\"](?P<field_id>[^'\"]+)['\"]",
        re.IGNORECASE,
    ),
    # Fill id='email' with 'abc' / Set name='email' to 'abc'
    re.compile(
        rf"(?P<attr>{ATTR_NAME})\s*=\s*['\"](?P<field_id>[^'\"]+)['\"]"
        rf"\s*(?:with|to|value|=)\s*(?P<value>'[^']*')",
        re.IGNORECASE,
    ),
    # Type ENV_VAR into id='field'
    re.compile(
        rf"(?P<env>[A-Z][A-Z0-9_]{{2,}})\s+(?:in|into|using|to|for)"
        rf"(?:\s+input|\s+field)?[^.]{{0,80}}?"
        rf"(?P<attr>{ATTR_NAME})\s*=\s*['\"](?P<field_id>[^'\"]+)['\"]",
        re.IGNORECASE,
    ),
    # id='field' from environment variable ENV_VAR / name='field' from ENV_VAR
    re.compile(
        rf"(?P<attr>{ATTR_NAME})\s*=\s*['\"](?P<field_id>[^'\"]+)['\"]"
        rf"\s*(?:from\s+environment\s+variable|from)\s+(?P<env>[A-Z][A-Z0-9_]{{2,}})",
        re.IGNORECASE,
    ),
    # from environment variable ENV_VAR ... id='field'
    re.compile(
        rf"(?:from\s+environment\s+variable|from)\s+(?P<env>[A-Z][A-Z0-9_]{{2,}})"
        rf"[^.]{{0,120}}?(?P<attr>{ATTR_NAME})\s*=\s*['\"](?P<field_id>[^'\"]+)['\"]",
        re.IGNORECASE,
    ),
]

CLICK_ATTR_PATTERNS = [
    re.compile(
        rf"click[^.]{{0,80}}?(?P<attr>{ATTR_NAME})\s*=\s*['\"](?P<field_id>[^'\"]+)['\"]",
        re.IGNORECASE,
    ),
    re.compile(
        rf"button\s+(?P<attr>{ATTR_NAME})\s*=\s*['\"](?P<field_id>[^'\"]+)['\"]",
        re.IGNORECASE,
    ),
]

@dataclass
class Example:
    text: str
    intent: str
    labels: List[str]


def clean_url(raw: str) -> str:
    return raw.rstrip(".,;)\n\t ")


def extract_url(text: str) -> Optional[str]:
    match = URL_RE.search(text)
    return clean_url(match.group(0)) if match else None


def _span_overlaps(span: Tuple[int, int], spans: List[Tuple[int, int]]) -> bool:
    s, e = span
    for os, oe in spans:
        if s < oe and e > os:
            return True
    return False


def _strip_quotes(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == "'" and value[-1] == "'":
        return value[1:-1]
    return value


def extract_click_attr(text: str) -> Optional[Dict[str, str]]:
    for pattern in CLICK_ATTR_PATTERNS:
        match = pattern.search(text)
        if match:
            return {"attribute_name": match.group("attr"), "id": match.group("field_id")}

    # Fallback for click-only prompts: use the first attribute expression if no fill value exists.
    attrs = list(ATTR_RE.finditer(text))
    if attrs and not extract_field_pairs(text):
        m = attrs[0]
        return {"attribute_name": m.group("attr"), "id": m.group("field_id")}
    return None


def extract_field_pairs(text: str) -> List[Dict[str, Optional[str]]]:
    """Extract N input fields from the prompt.

    Each field now carries attribute_name so selectors can be generated as:
      input[attribute_name='field_id']

    Examples:
      id='email'              -> attribute_name=id, field_id=email
      name='email'            -> attribute_name=name, field_id=email
      data-testid='email-box' -> attribute_name=data-testid, field_id=email-box
    """
    pairs: List[Dict[str, Any]] = []
    seen = set()

    for pattern in PAIR_PATTERNS:
        for match in pattern.finditer(text):
            attr = match.groupdict().get("attr")
            field_id = match.groupdict().get("field_id")
            raw_value = match.groupdict().get("value")
            env_value = match.groupdict().get("env")
            if not attr or not field_id:
                continue

            key = (match.start(), match.end(), attr, field_id, raw_value, env_value)
            if key in seen:
                continue
            seen.add(key)

            pairs.append({
                "attribute_name": attr,
                "id": field_id,
                "value": _strip_quotes(raw_value),
                "value_from_env": env_value,
                "start": match.start(),
            })

    # Fallback pairing: collect attribute expressions and non-attribute quoted values/envs in order.
    if not pairs:
        attrs = [(m.start("field_id"), m.group("attr"), m.group("field_id")) for m in ATTR_RE.finditer(text)]
        attr_value_spans = [(m.start("field_id"), m.end("field_id")) for m in ATTR_RE.finditer(text)]
        attr_name_spans = [(m.start("attr"), m.end("attr")) for m in ATTR_RE.finditer(text)]
        skip_spans = attr_value_spans + attr_name_spans

        values = []
        for m in QUOTED_RE.finditer(text):
            if _span_overlaps((m.start(1), m.end(1)), skip_spans):
                continue
            values.append((m.start(1), m.group(1), "value"))

        url_match = URL_RE.search(text)
        url_span = (url_match.start(), url_match.end()) if url_match else None
        for m in ENV_RE.finditer(text):
            if url_span and m.start() >= url_span[0] and m.end() <= url_span[1]:
                continue
            if _span_overlaps((m.start(), m.end()), [(s, s + len(v)) for s, v, _ in values]):
                continue
            if _span_overlaps((m.start(), m.end()), skip_spans):
                continue
            values.append((m.start(), m.group(0), "env"))
        values.sort(key=lambda x: x[0])

        for idx, (_, attr, field_id) in enumerate(attrs):
            val = values[idx] if idx < len(values) else None
            pairs.append({
                "attribute_name": attr,
                "id": field_id,
                "value": val[1] if val and val[2] == "value" else None,
                "value_from_env": val[1] if val and val[2] == "env" else None,
                "start": attrs[idx][0],
            })

    # Remove duplicates by attr/id/value/env, preserve order.
    pairs.sort(key=lambda x: x["start"])
    out = []
    seen2 = set()
    for item in pairs:
        key = (item["attribute_name"], item["id"], item.get("value"), item.get("value_from_env"))
        if key in seen2:
            continue
        seen2.add(key)
        item.pop("start", None)
        out.append(item)
    return out



def extract_text_assertion(text: str) -> Optional[Dict[str, Optional[str]]]:
    """Extract a tag + attribute selector + expected text assertion.

    Example:
      find mat-card-title having the attribute class='mat-mdc-card-title'
      and get the text value. Compare the text value is 'Sales Overview'
    """
    if not GET_TEXT_RE.search(text):
        return None

    tag_match = FIND_TAG_RE.search(text) or TAG_RE.search(text)
    attr_match = ATTR_RE.search(text)
    text_match = TEXT_COMPARE_RE.search(text)

    if not tag_match or not attr_match or not text_match:
        return None

    tag = tag_match.group("tag") if "tag" in tag_match.groupdict() else tag_match.group(1)
    return {
        "tag_name": tag,
        "attribute_name": attr_match.group("attr"),
        "id": attr_match.group("field_id"),
        "value": text_match.group("text"),
    }

def extract_slots_by_rules(text: str, intent: Optional[str] = None) -> Dict[str, Any]:
    url = extract_url(text)
    click = extract_click_attr(text) if intent == "navigate_click_by_id" else None
    text_assertion = extract_text_assertion(text)
    fields = [] if intent in {"navigate_click_by_id", "find_tag_text_compare"} or text_assertion else extract_field_pairs(text)
    return {
        "url": url,
        "fields": fields,
        "click_id": click.get("id") if click else None,
        "click_attribute_name": click.get("attribute_name") if click else None,
        "text_assertion": text_assertion,
    }


def find_slot_spans(text: str, intent: Optional[str] = None) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []

    url_match = URL_RE.search(text)
    url_span = None
    if url_match:
        clean = clean_url(url_match.group(0))
        url_span = (url_match.start(), url_match.start() + len(clean))
        spans.append((url_span[0], url_span[1], "URL"))

    attr_name_label = "CLICK_ATTRIB_NAME" if intent == "navigate_click_by_id" else "ATTRIB_NAME"
    attr_value_label = "CLICK_ID" if intent == "navigate_click_by_id" else "FIELD_ID"
    attr_spans = []
    for m in ATTR_RE.finditer(text):
        spans.append((m.start("attr"), m.end("attr"), attr_name_label))
        spans.append((m.start("field_id"), m.end("field_id"), attr_value_label))
        attr_spans.append((m.start("attr"), m.end("attr")))
        attr_spans.append((m.start("field_id"), m.end("field_id")))

    tag_match = FIND_TAG_RE.search(text) if intent == "find_tag_text_compare" else None
    if tag_match:
        spans.append((tag_match.start("tag"), tag_match.end("tag"), "ELEMENT_TAG"))

    # Quoted strings that are not attribute='...' are literal field values.
    quoted_value_spans = []
    for m in QUOTED_RE.finditer(text):
        span = (m.start(1), m.end(1))
        if _span_overlaps(span, attr_spans):
            continue
        spans.append((span[0], span[1], "FIELD_VALUE"))
        quoted_value_spans.append(span)

    # Uppercase env vars outside URLs, quoted literals, and attribute expressions.
    for m in ENV_RE.finditer(text):
        span = (m.start(), m.end())
        if url_span and m.start() >= url_span[0] and m.end() <= url_span[1]:
            continue
        if _span_overlaps(span, quoted_value_spans):
            continue
        if _span_overlaps(span, attr_spans):
            continue
        spans.append((span[0], span[1], "ENV_VALUE"))

    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    # Drop overlapping smaller spans.
    filtered: List[Tuple[int, int, str]] = []
    occupied: List[Tuple[int, int]] = []
    for s, e, label in spans:
        if s < e and not _span_overlaps((s, e), occupied):
            filtered.append((s, e, label))
            occupied.append((s, e))
    return filtered


def make_bio_labels(text: str, intent: Optional[str] = None) -> List[str]:
    labels = ["O"] * len(text)
    for start, end, slot in find_slot_spans(text, intent):
        if start < 0 or end > len(text) or start >= end:
            continue
        labels[start] = f"B-{slot}"
        for pos in range(start + 1, end):
            labels[pos] = f"I-{slot}"
    return labels


def load_examples(json_path: str) -> Tuple[List[Example], Dict[str, Any], Dict[str, int], Dict[int, str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    templates: Dict[str, Any] = {}
    examples: List[Example] = []
    for intent in raw["intents"]:
        tag = intent["tag"]
        templates[tag] = intent["responses"][0]
        for pattern in intent.get("patterns", []):
            examples.append(Example(text=pattern, intent=tag, labels=make_bio_labels(pattern, tag)))

    intents = sorted(templates.keys())
    intent2id = {name: idx for idx, name in enumerate(intents)}
    id2intent = {idx: name for name, idx in intent2id.items()}
    return examples, templates, intent2id, id2intent


def build_char_vocab(texts: List[str]) -> Dict[str, int]:
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for ch in sorted(set("".join(texts))):
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_len: int) -> Tuple[List[int], int]:
    ids = [vocab.get(ch, vocab["<UNK>"]) for ch in text[:max_len]]
    length = len(ids)
    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids, length


def encode_labels(labels: List[str], max_len: int) -> List[int]:
    ids = [LABEL2ID[label] for label in labels[:max_len]]
    if len(ids) < max_len:
        ids += [-100] * (max_len - len(ids))
    return ids


def decode_bio_slots(text: str, pred_label_ids: List[int], id2label: Dict[int, str]) -> Dict[str, List[str]]:
    slots: Dict[str, List[str]] = {}
    current_slot: Optional[str] = None
    current_chars: List[str] = []

    def flush():
        nonlocal current_slot, current_chars
        if current_slot and current_chars:
            value = "".join(current_chars).strip()
            if value:
                slots.setdefault(current_slot, []).append(value)
        current_slot = None
        current_chars = []

    for ch, label_id in zip(text, pred_label_ids[: len(text)]):
        label = id2label.get(int(label_id), "O")
        if label == "O":
            flush()
            continue
        prefix, slot_name = label.split("-", 1)
        if prefix == "B" or slot_name != current_slot:
            flush()
            current_slot = slot_name
            current_chars = [ch]
        else:
            current_chars.append(ch)
    flush()
    return slots


def build_fields_from_slot_lists(slots: Dict[str, List[str]]) -> List[Dict[str, Optional[str]]]:
    attrs = slots.get("ATTRIB_NAME", [])
    ids = slots.get("FIELD_ID", [])
    values = slots.get("FIELD_VALUE", [])
    envs = slots.get("ENV_VALUE", [])
    fields = []
    for i, field_id in enumerate(ids):
        attr = attrs[i] if i < len(attrs) else "id"
        literal = values[i] if i < len(values) else None
        env = envs[i] if literal is None and i < len(envs) else None
        fields.append({
            "attribute_name": attr,
            "id": field_id,
            "value": literal,
            "value_from_env": env,
        })
    return fields


def infer_intent_by_rules(text: str, parsed: Dict[str, Any]) -> Optional[str]:
    """Optional production repair for obvious action structure."""
    lower = text.lower()
    has_click_word = any(word in lower for word in ["click", "submit", "press", "login", "sign in", "save"])
    has_fields = bool(parsed.get("fields"))
    has_click_id = bool(parsed.get("click_id"))
    has_text_assertion = bool(parsed.get("text_assertion")) or ("get the text" in lower and "compare" in lower)

    if has_text_assertion:
        return "find_tag_text_compare"
    if has_click_id and not has_fields:
        return "navigate_click_by_id"
    if has_fields and has_click_word:
        return "navigate_fill_fields_click"
    if has_fields:
        return "navigate_fill_fields"
    if parsed.get("url"):
        return "navigate_only"
    return None


def _replace_string(value: str, replacements: Dict[str, Any]) -> Any:
    # Preserve None for dynamic field values.
    if value == "__FIELD_VALUE__":
        return replacements.get("FIELD_VALUE")
    if value == "__ENV_VALUE__":
        return replacements.get("ENV_VALUE")
    if value == "__URL__":
        return replacements.get("URL")
    if value == "__CLICK_ID__":
        return replacements.get("CLICK_ID")
    if value == "__CLICK_ATTRIB_NAME__":
        return replacements.get("CLICK_ATTRIB_NAME")
    if value == "__ATTRIB_NAME__":
        return replacements.get("ATTRIB_NAME")
    if value == "__TEXT_TAG__":
        return replacements.get("TEXT_TAG")
    if value == "__TEXT_ATTRIB_NAME__":
        return replacements.get("TEXT_ATTRIB_NAME")
    if value == "__TEXT_FIELD_ID__":
        return replacements.get("TEXT_FIELD_ID")
    if value == "__TEXT_VALUE__":
        return replacements.get("TEXT_VALUE")
    if value == "__AUTO_STEP_NO__":
        return replacements.get("AUTO_STEP_NO")

    out = value
    for key, slot_value in replacements.items():
        if slot_value is None:
            continue
        out = out.replace(f"__{key}__", str(slot_value))
    return out


def _replace_any(obj: Any, replacements: Dict[str, Any]) -> Any:
    if isinstance(obj, str):
        return _replace_string(obj, replacements)
    if isinstance(obj, list):
        return [_replace_any(x, replacements) for x in obj]
    if isinstance(obj, dict):
        return {k: _replace_any(v, replacements) for k, v in obj.items()}
    return obj


def render_template(template: Any, parsed: Dict[str, Any]) -> Any:
    """Render updated dynamic response templates.

    Supports a step marker like:
      {"repeat_for": "__FIELDS__", "template": {...}}
    """
    result = copy.deepcopy(template)
    url = parsed.get("url")
    fields = parsed.get("fields") or []
    click_id = parsed.get("click_id")
    click_attr = parsed.get("click_attribute_name") or "id"
    text_assertion = parsed.get("text_assertion") or {}

    steps = []
    step_no = 1
    for step in result.get("steps", []):
        if isinstance(step, dict) and step.get("repeat_for") == "__FIELDS__":
            repeated_template = step["template"]
            for field in fields:
                replacements = {
                    "URL": url,
                    "ATTRIB_NAME": field.get("attribute_name") or "id",
                    "FIELD_ID": field.get("id"),
                    "FIELD_VALUE": field.get("value"),
                    "ENV_VALUE": field.get("value_from_env"),
                    "AUTO_STEP_NO": step_no,
                }
                rendered = _replace_any(repeated_template, replacements)
                steps.append(rendered)
                step_no += 1
            continue

        replacements = {
            "URL": url,
            "CLICK_ATTRIB_NAME": click_attr,
            "CLICK_ID": click_id,
            "AUTO_STEP_NO": step_no,
            "TEXT_TAG": text_assertion.get("tag_name"),
            "TEXT_ATTRIB_NAME": text_assertion.get("attribute_name"),
            "TEXT_FIELD_ID": text_assertion.get("id"),
            "TEXT_VALUE": text_assertion.get("value"),
        }
        rendered = _replace_any(step, replacements)
        steps.append(rendered)
        step_no += 1

    result["base_url"] = url
    result["steps"] = steps
    return result
