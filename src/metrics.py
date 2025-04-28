import re
import string
from typing import List, Set, Tuple, Union
from word2number import w2n  # pip install word2number


def precision_recall_at_k(retrieved_docs, ground_truth_docs, k):
    """
    Compute average precision@k and recall@k for a batch of queries.

    Args:
        retrieved_docs: List of lists, each list has retrieved doc IDs per query.
        ground_truth_docs: List of sets or lists, each has true relevant doc IDs.
        k: Integer cutoff for top-k.

    Returns:
        avg_precision: Float, average precision@k over all queries.
        avg_recall: Float, average recall@k over all queries.
    """
    precisions = []
    recalls = []
    if type(retrieved_docs[0]) != list:
        retrieved_docs = [retrieved_docs]
    if type(ground_truth_docs[0]) != list:
        ground_truth_docs= [ground_truth_docs]
        
    for retrieved, relevant in zip(retrieved_docs, ground_truth_docs):
        relevant_set = set(relevant)
        top_k = retrieved[:k]
        true_positives = relevant_set.intersection(top_k)
        
        precision = len(true_positives) / k
        recall = len(true_positives) / len(relevant_set) if relevant_set else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
    
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    
    return avg_precision, avg_recall



# ─── Unnormalized metrics ─────────────────────────────────────────────

def levenshtein_distance(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = [list(range(m + 1))] + [[i] + [0]*m for i in range(1, n+1)]
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    return dp[n][m]

def normalized_similarity(a: str, b: str) -> float:
    a, b = a.strip(), b.strip()
    max_len = max(len(a), len(b))
    if max_len == 0: return 1.0
    return 1.0 - (levenshtein_distance(a, b) / max_len)

def anls_score(pred: str, gold: str, threshold: float = 0.5) -> float:
    sim = normalized_similarity(pred, gold)
    return sim if sim >= threshold else 0.0

def exact_match(pred: str, gold: str) -> int:
    return int(pred.strip().lower() == gold.strip().lower())

def evaluate_unnormalized(
    preds: List[str],
    golds: List[str],
    threshold: float = 0.5
) -> dict:
    assert len(preds) == len(golds)
    anls = [anls_score(p, g, threshold) for p, g in zip(preds, golds)]
    em   = [exact_match(p, g)         for p, g in zip(preds, golds)]
    return {
        "ANLS": sum(anls) / len(anls),
        "EM":   sum(em)   / len(em)
    }

# ─── Normalization helpers ────────────────────────────────────────────

EXCLUDE = set(string.punctuation)

def _is_number(text: str) -> bool:
    try:
        float(text); return True
    except ValueError:
        return False

def _is_word_number(text: str) -> bool:
    try:
        w2n.word_to_num(text); return True
    except Exception:
        return False

def _remove_articles(text: str) -> str:
    return re.sub(r"\b(a|an|the)\b", " ", text)

def _white_space_fix(text: str) -> str:
    return " ".join(text.split())

def _remove_punc(text: str) -> str:
    if _is_number(text):
        return text
    return "".join(ch for ch in text if ch not in EXCLUDE)

def _lower(text: str) -> str:
    return text.lower()

def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(float(text))
    if _is_word_number(text):
        return str(float(w2n.word_to_num(text)))
    return text

def _tokenize(text: str) -> List[str]:
    return re.split(r"[ \-]", text)

def _normalize_answer(text: str) -> str:
    parts = []
    for tok in _tokenize(text):
        t = _lower(tok)
        t = _remove_punc(t)
        t = _normalize_number(t)
        t = _remove_articles(t)
        t = _white_space_fix(t)
        if t:
            parts.append(t)
    return " ".join(parts).strip()

def _answer_to_bags(
    answer: Union[str, List[str], Tuple[str, ...]]
) -> Tuple[List[str], List[Set[str]]]:
    spans = answer if isinstance(answer, (list, tuple)) else [answer]
    norm_spans, bags = [], []
    for raw in spans:
        norm = _normalize_answer(raw)
        norm_spans.append(norm)
        bags.append(set(norm.split()))
    return norm_spans, bags

# ─── Normalized metrics ───────────────────────────────────────────────

def normalized_exact_match(pred: str, gold: str) -> int:
    return int(_normalize_answer(pred) == _normalize_answer(gold))

def normalized_f1(pred: str, gold: str) -> float:
    _, pbag = _answer_to_bags(pred)
    _, gbag = _answer_to_bags(gold)
    common = pbag[0] & gbag[0]
    if not common:
        return 0.0
    prec = len(common) / len(pbag[0])
    rec  = len(common) / len(gbag[0])
    return 2 * prec * rec / (prec + rec)

def evaluate_normalized(
    preds: List[str],
    golds: List[str]
) -> dict:
    assert len(preds) == len(golds)
    nem = [normalized_exact_match(p, g) for p, g in zip(preds, golds)]
    nf1 = [normalized_f1(p, g)         for p, g in zip(preds, golds)]
    return {
        "NormalizedEM": sum(nem) / len(nem),
        "NormalizedF1": sum(nf1) / len(nf1)
    }



def evaluate_all(
    preds: List[str],
    golds: List[str],
    anls_threshold: float = 0.5
) -> dict:
    """
    Returns a dict with keys:
      ANLS, EM, NormalizedEM, NormalizedF1
    """
    res = {}
    res.update(evaluate_unnormalized(preds, golds, anls_threshold))
    res.update(evaluate_normalized(preds, golds))
    return res

