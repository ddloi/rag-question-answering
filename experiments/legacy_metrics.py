"""Alias-aware QA metrics and explicitly named evidence proxies; no ML imports."""
import re
import string
import unicodedata
from collections import Counter


def normalize_answer(value):
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    value = "".join(c for c in value if c not in string.punctuation and not unicodedata.category(c).startswith("P"))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", value).split())


def answers_for(record):
    return list(dict.fromkeys([record["answer"], *record.get("answer_aliases", [])]))


def select_answer_policy(record, policy):
    if policy == 'canonical':
        return {**record, 'answer_aliases': []}
    if policy == 'official-aliases':
        return dict(record)
    raise ValueError(f'Unknown answer policy: {policy}')


def exact_match_score(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction, ground_truth):
    pred, gold = normalize_answer(prediction).split(), normalize_answer(ground_truth).split()
    if not pred or not gold:
        return float(pred == gold)
    common = sum((Counter(pred) & Counter(gold)).values())
    return 2 * common / (len(pred) + len(gold))


def score_answer(prediction, record):
    aliases = answers_for(record)
    return {"em": max(exact_match_score(prediction, a) for a in aliases),
            "f1": max(f1_score(prediction, a) for a in aliases)}


def contains_answer(text, aliases):
    value = normalize_answer(text)
    return any(normalize_answer(a) and re.search(r"(?<!\w)" + re.escape(normalize_answer(a)) + r"(?!\w)", value)
               for a in aliases)


def token_support_fraction(prediction, context):
    """Lexical support only: cannot test factual entailment or negation."""
    pred = normalize_answer(prediction).split()
    context_tokens = set(normalize_answer(context).split())
    return sum(t in context_tokens for t in pred) / len(pred) if pred else 0.0


def classify_error(prediction, record, retrieved, used_context, corpus_has_answer):
    if score_answer(prediction, record)["em"]:
        return "correct"
    if not corpus_has_answer:
        return "corpus_evidence_missing"
    aliases = answers_for(record)
    if not any(contains_answer(p["text"], aliases) for p in retrieved):
        return "retrieval_evidence_missing"
    if not contains_answer(used_context, aliases):
        return "context_selection_or_truncation"
    return "generation_wrong_with_lexical_evidence"
