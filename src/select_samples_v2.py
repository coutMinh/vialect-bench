"""Select 100 samples/task with dialect-trigger coverage and failure likelihood.

Enhanced version that:
1. Uses weighted trigger scores (not raw counts) — Negation, Template, Connective weigh more
2. Prioritizes multi-category triggers (>= 2 categories or score >= 3)
3. Tracks trigger location for NLI (premise / hypothesis / both)
4. Scans MCQA question for triggers
5. Uses dialect-signature diversity (max per signature), not just lexical diversity
6. Enforces hard/medium/easy quota based on category difficulty
7. Task-specific category weights (QA prioritizes Interrogative, etc.)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset
import openpyxl


# ── Dialect lexicon parsing ──────────────────────────────────────────


def load_lexicon(xlsx_path: str) -> list[dict]:
    """Parse DIALECT_LEXICON_v2.xlsx into a list of entry dicts."""
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb[wb.sheetnames[0]]

    REGION_COLS = {
        6: "Bắc Bộ", 7: "Thanh Hoá", 8: "Nghệ An",
        9: "Hà Tĩnh", 10: "Thừa Thiên Huế", 11: "Quảng Nam",
        12: "Quảng Ngãi", 13: "Phú Yên", 14: "Nam Bộ",
    }

    category_map = {
        "1. Interrogatives": "Interrogative",
        "2. Demonstratives & Deixis": "Demonstrative",
        "3. Negation & Function Words": "Negation/Function",
        "4. Discourse Particles": "Particle",
        "5. Connectives & Aspect Markers": "Connective/Aspect",
        "6. Pronouns": "Pronoun",
        "7. Kinship & Honorifics": "Kinship",
        "8. Predicate Vocabulary": "Predicate",
        "9. Templates & Idioms": "Template/Idiom",
    }

    entries = []
    current_category = None

    for row in ws.iter_rows(min_row=3, values_only=True):
        col_b = row[1]
        if col_b is not None and isinstance(col_b, str):
            for prefix, cat_name in category_map.items():
                if col_b.strip().startswith(prefix):
                    current_category = cat_name
                    break
            continue

        col_c = row[2]
        if col_c is None:
            continue

        standard = row[3]
        if standard is None:
            continue

        regions = {}
        for col_idx, region_name in REGION_COLS.items():
            regions[region_name] = row[col_idx]

        task_impact = row[5] if row[5] and row[5] not in {
            "Interrogative", "Demonstrative", "Negation", "Function",
            "Particle", "Connective", "Aspect", "Pronoun", "Kinship",
            "Predicate", "Template", "Idiom",
        } else None

        entries.append({
            "standard": str(standard).strip(),
            "role": str(row[4]).strip() if row[4] else "",
            "category": current_category,
            "task_impact": str(task_impact).strip() if task_impact else "",
            "regions": regions,
        })

    wb.close()
    return entries


def build_trigger_index(entries: list[dict]) -> dict[str, list[dict]]:
    """Map lowercase standard forms → list of lexicon entries."""
    index: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        full = entry["standard"].lower()
        index[full].append(entry)
        for part in re.split(r"\s*/\s*", full):
            part = part.strip()
            if part and len(part) >= 2:
                index[part].append(entry)
    return dict(index)


def find_dialect_triggers(
    text: str,
    trigger_index: dict[str, list[dict]],
) -> list[dict]:
    """Find all dialect-triggerable entries in a text string."""
    text_lower = text.lower()
    found = []
    seen = set()

    sorted_keys = sorted(trigger_index.keys(), key=len, reverse=True)

    for key in sorted_keys:
        if len(key) < 2:
            continue
        if len(key) <= 4:
            pattern = r"(?<!\w)" + re.escape(key) + r"(?!\w)"
        else:
            pattern = re.escape(key)
        if re.search(pattern, text_lower):
            for entry in trigger_index[key]:
                dedup_key = (entry["standard"], entry["category"])
                if dedup_key not in seen:
                    seen.add(dedup_key)
                    found.append({
                        "standard": entry["standard"],
                        "category": entry["category"],
                        "matched_word": key,
                    })

    return found


# ── Task-specific weights and difficulty ─────────────────────────────

# How much each category's trigger contributes to failure likelihood.
# Higher weight = more likely to cause accuracy drop when dialect-transformed.
# Rationale:
#   Negation/Function: flips polarity ("không" → "nỏ") → can reverse meaning
#   Template/Idiom: multi-word, cannot be translated word-by-word
#   Connective/Aspect: changes clause logic ("chứ" → "chơ", "thì" → "bơ")
#   Interrogative: changes question target ("đâu" → "mô", "sao" → "răng")
#   Demonstrative: changes reference ("này" → "ni", "đó" → "nớ")
#   Pronoun: changes entity reference ("tôi" → "tui", "mày" → "mi")
#   Particle: tone/pragmatics only, rarely flips label
#   Predicate: lexical substitution, model may handle via context
#   Kinship: specific vocabulary, usually preserved in context
CATEGORY_WEIGHT: dict[str, dict[str, float]] = {
    "sentiment": {
        "Negation/Function": 3.0,
        "Template/Idiom": 3.0,
        "Connective/Aspect": 2.0,
        "Interrogative": 2.0,
        "Demonstrative": 1.5,
        "Pronoun": 1.5,
        "Kinship": 1.0,
        "Predicate": 1.0,
        "Particle": 0.5,
    },
    "nli": {
        "Negation/Function": 3.0,
        "Template/Idiom": 3.0,
        "Connective/Aspect": 3.0,
        "Interrogative": 2.0,
        "Demonstrative": 2.0,
        "Pronoun": 1.5,
        "Kinship": 1.0,
        "Particle": 1.0,
        "Predicate": 1.0,
    },
    "qa": {
        "Interrogative": 3.0,
        "Negation/Function": 3.0,
        "Template/Idiom": 2.5,
        "Demonstrative": 2.0,
        "Connective/Aspect": 2.0,
        "Pronoun": 1.5,
        "Predicate": 1.0,
        "Kinship": 1.0,
        "Particle": 0.5,
    },
    "mcqa": {
        "Interrogative": 3.0,
        "Negation/Function": 3.0,
        "Template/Idiom": 2.5,
        "Connective/Aspect": 2.0,
        "Demonstrative": 2.0,
        "Pronoun": 1.5,
        "Predicate": 1.0,
        "Kinship": 1.0,
        "Particle": 0.5,
    },
}

# Difficulty tiers based on which categories appear in the sample
HARD_CATEGORIES = {"Negation/Function", "Template/Idiom", "Connective/Aspect"}
MEDIUM_CATEGORIES = {"Interrogative", "Demonstrative"}
EASY_CATEGORIES = {"Pronoun", "Particle", "Predicate", "Kinship"}

# Target quota: hard 30%, medium 40%, easy 30%
DIFFICULTY_QUOTA = {"hard": 0.30, "medium": 0.40, "easy": 0.30}


def classify_difficulty(categories: list[str]) -> str:
    """Classify a sample's difficulty based on its trigger categories."""
    cats = set(categories)
    if cats & HARD_CATEGORIES:
        return "hard"
    if cats & MEDIUM_CATEGORIES:
        return "medium"
    return "easy"


def compute_trigger_score(triggers: list[dict], task: str) -> float:
    """Compute weighted trigger score for a sample."""
    weights = CATEGORY_WEIGHT.get(task, {})
    return sum(weights.get(t["category"], 1.0) for t in triggers)


def compute_dialect_score(
    triggers: list[dict],
    task: str,
) -> float:
    """Compute overall dialect failure-likelihood score.

    score = trigger_score + 0.5 * unique_categories
    """
    trigger_score = compute_trigger_score(triggers, task)
    unique_cats = len({t["category"] for t in triggers})
    return trigger_score + 0.5 * unique_cats


# ── Dataset configs ──────────────────────────────────────────────────

DATASETS = {
    "sentiment": {
        "dataset_id": "tridm/UIT-VSMEC",
        "split": "train",
        "label_field": "Emotion",
    },
    "nli": {
        "dataset_id": "uitnlp/ViANLI",
        "split": "train",
        "label_field": "label",
    },
    "qa": {
        "dataset_id": "taidng/UIT-ViQuAD2.0",
        "split": "train",
        "label_field": "is_impossible",
    },
    "mcqa": {
        "dataset_id": "uitnlp/vimmrc2.0",
        "split": "train",
        "label_field": "answers",
    },
}

MAX_SELECTION_POOL = 3000
MAX_MCQA_QUESTIONS_PER_TOPIC = 2
TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)

TASK_CATEGORIES = {
    "sentiment": {"Interrogative", "Negation/Function", "Particle", "Connective/Aspect", "Pronoun", "Kinship", "Predicate", "Template/Idiom"},
    "nli": {"Interrogative", "Demonstrative", "Negation/Function", "Particle", "Connective/Aspect", "Pronoun", "Kinship", "Template/Idiom"},
    "qa": {"Interrogative", "Demonstrative", "Negation/Function", "Connective/Aspect", "Pronoun", "Predicate", "Template/Idiom"},
    "mcqa": {"Interrogative", "Demonstrative", "Negation/Function", "Connective/Aspect", "Pronoun", "Predicate", "Template/Idiom"},
}

# Text fields per task for trigger scanning
TASK_TEXT_FIELDS = {
    "sentiment": {"full": ["Sentence"]},
    "nli": {"full": ["premise", "hypothesis"], "premise": ["premise"], "hypothesis": ["hypothesis"]},
    "qa": {"full": ["question"]},
    "mcqa": {"full": ["question"]},
}

# Dialect-signature diversity: max samples sharing the same sorted category set
MAX_PER_SIGNATURE = 15


# ── Filtering ────────────────────────────────────────────────────────

SENTIMENT_VULGAR_WORDS = [
    "đéo", "đụ", "lồn", "địt", "đm", "dm", "dmm", "cc", "fuck", "shit",
    "vãi", "vãi_chưởng", "chảnh chó", "con mẹ nó",
]

MCQA_TITLE_EXCLUDE_KEYWORDS = {
    "bac", "chu dong tu", "luom", "thanh giong", "son tinh",
    "thuy tinh", "y ec xanh", "mtao", "quan am", "ra ma", "xi ta",
    "ga li le", "nguyen", "tran dai nghia", "hai ba trung",
    "con rong chau tien", "bop nat qua cam", "ong trang", "trang",
}

MCQA_GENERAL_TITLE_KEYWORDS = {
    "bon mua", "cay", "chim", "ga", "gau", "ca", "cau chuyen", "con",
    "chiec", "chuyen", "mua", "hoa", "song", "bien", "rung", "dong",
    "dien thoai", "doi giay", "mau giay", "ban tay", "ban tin",
    "bai hat", "bai hoc", "bai tap", "bai kiem tra", "bai tho",
    "bai doc", "dong ho", "cai cau", "cua tung", "cuc nuoc da",
    "cuon so tay", "nang", "dat", "dem", "dua ghe", "huong",
    "mat troi", "bau troi", "canh dieu", "hat mam", "truong",
    "lop", "cho", "que", "kho bau", "phong nha", "thuoc la",
}


def is_vulgar(text: str) -> bool:
    lower = text.lower()
    for word in SENTIMENT_VULGAR_WORDS:
        i = lower.find(word)
        while i != -1:
            before_ok = i == 0 or not lower[i - 1].isalnum()
            after_ok = i + len(word) >= len(lower) or not lower[i + len(word)].isalnum()
            if before_ok and after_ok:
                return True
            i = lower.find(word, i + 1)
    return False


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    return value


def get_label(task: str, row: dict[str, Any]) -> str:
    if task == "qa":
        return "unanswerable" if row.get("is_impossible") else "answerable"
    if task == "mcqa":
        return str(row.get("answer", ""))
    return str(row[DATASETS[task]["label_field"]])


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def token_set(text: str) -> set[str]:
    return set(tokens(text))


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def selection_text(task: str, row: dict[str, Any]) -> str:
    if task == "sentiment":
        return str(row.get("Sentence", ""))
    if task == "nli":
        return f"{row.get('premise', '')}\n{row.get('hypothesis', '')}"
    if task == "qa":
        return str(row.get("question", ""))
    if task == "mcqa":
        return f"{row.get('title', '')}\n{row.get('question', '')}"
    raise ValueError(f"Unsupported task: {task}")


def passes_filter(task: str, row: dict[str, Any]) -> bool:
    if task == "sentiment":
        sentence = str(row.get("Sentence", ""))
        return len(tokens(sentence)) >= 5 and not is_vulgar(sentence)

    if task == "nli":
        premise = str(row.get("premise", ""))
        hypothesis = str(row.get("hypothesis", ""))
        return len(tokens(premise)) >= 5 and len(tokens(hypothesis)) >= 3

    if task == "qa":
        question = str(row.get("question", ""))
        answers = row.get("answers") or {}
        answer_texts = answers.get("text") or []
        question_len = len(tokens(question))
        return (
            not row.get("is_impossible")
            and 6 <= question_len <= 25
            and len(answer_texts) > 0
            and all(len(tokens(str(answer))) <= 12 for answer in answer_texts)
        )

    if task == "mcqa":
        return is_valid_mcqa_question(row)

    return True


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalized_title(title: str) -> str:
    text = str(title).replace("​", " ").replace("-", " ")
    text = strip_accents(text).lower()
    return re.sub(r"\s+", " ", text).strip()


def mcqa_grade_ok(grade: Any) -> bool:
    try:
        return int(str(grade).strip()) <= 3
    except ValueError:
        return False


def is_general_mcqa_topic(row: dict[str, Any]) -> bool:
    title = normalized_title(row.get("title", ""))
    topic_text = normalized_title(
        " ".join([
            str(row.get("title", "")),
            str(row.get("article", "")),
            " ".join(str(question) for question in row.get("questions", []) or []),
        ])
    )
    return (
        mcqa_grade_ok(row.get("grade"))
        and not any(keyword in topic_text for keyword in MCQA_TITLE_EXCLUDE_KEYWORDS)
        and any(keyword in title for keyword in MCQA_GENERAL_TITLE_KEYWORDS)
    )


def is_valid_mcqa_question(row: dict[str, Any]) -> bool:
    return (
        is_general_mcqa_topic(row)
        and len(tokens(str(row.get("article", "")))) >= 40
        and len(tokens(str(row.get("question", "")))) >= 5
        and isinstance(row.get("options"), list)
        and len(row.get("options") or []) == 4
        and str(row.get("answer", "")).strip() in {"A", "B", "C", "D"}
    )


def sample_key(row: dict[str, Any]) -> Any:
    return row.get("_sample_id", row["_source_index"])


def dialect_signature(categories: list[str]) -> str:
    """Create a hashable dialect signature from sorted categories."""
    return "|".join(sorted(set(categories)))


# ── Difficulty-aware selection ────────────────────────────────────────


def select_with_difficulty_quota(
    candidates: list[dict],
    task: str,
    sample_size: int,
    seed: int,
    max_per_topic: int | None = None,
) -> list[Any]:
    """Select samples with difficulty quota, category coverage, and dialect diversity.

    Strategy:
    1. Compute dialect_score and difficulty tier for each candidate.
    2. Fill difficulty quota (hard/medium/easy) — within each tier, pick
       highest dialect_score first.
    3. Enforce dialect-signature diversity (max MAX_PER_SIGNATURE per sig).
    4. Within each tier, use least-similar fill for lexical diversity.
    """
    if sample_size > len(candidates):
        raise SystemExit(
            f"Task '{task}' has only {len(candidates)} candidates, "
            f"but --sample-size is {sample_size}."
        )

    # Pre-compute features
    features = {}
    for i, row in enumerate(candidates):
        key = sample_key(row)
        features[key] = {
            "idx": i,
            "tokens": token_set(selection_text(task, row)),
            "categories": set(row.get("_dialect_categories", [])),
            "score": row.get("_dialect_score", 0),
            "difficulty": row.get("_difficulty", "easy"),
            "signature": dialect_signature(row.get("_dialect_categories", [])),
        }

    # Target counts per difficulty tier
    target_hard = max(1, round(sample_size * DIFFICULTY_QUOTA["hard"]))
    target_medium = max(1, round(sample_size * DIFFICULTY_QUOTA["medium"]))
    target_easy = sample_size - target_hard - target_medium

    tier_targets = {"hard": target_hard, "medium": target_medium, "easy": target_easy}
    tier_candidates: dict[str, list[Any]] = defaultdict(list)

    for row in candidates:
        tier = row.get("_difficulty", "easy")
        tier_candidates[tier].append(sample_key(row))

    # Sort each tier by dialect_score (descending) — prefer harder samples
    for tier in tier_candidates:
        tier_candidates[tier].sort(
            key=lambda k: features[k]["score"], reverse=True
        )

    selected_keys: list[Any] = []
    selected_set: set[Any] = set()
    signature_counts: Counter = Counter()
    topic_counts: Counter = Counter()
    tier_counts: Counter = Counter()

    def can_add(row_key: Any, row: dict) -> bool:
        if row_key in selected_set:
            return False
        topic_key = row.get("_topic_source_index", row["_source_index"])
        if max_per_topic is not None and topic_counts[topic_key] >= max_per_topic:
            return False
        sig = features[row_key]["signature"]
        if signature_counts[sig] >= MAX_PER_SIGNATURE:
            return False
        return True

    def add_key(key: Any, row: dict) -> None:
        selected_keys.append(key)
        selected_set.add(key)
        sig = features[key]["signature"]
        signature_counts[sig] += 1
        topic_key = row.get("_topic_source_index", row["_source_index"])
        topic_counts[topic_key] += 1
        tier_counts[row.get("_difficulty", "easy")] += 1

    # Build key→row map
    row_by_key = {sample_key(r): r for r in candidates}

    # Phase 1: Fill each difficulty tier by score
    for tier in ["hard", "medium", "easy"]:
        target = tier_targets[tier]
        added = 0
        for key in tier_candidates[tier]:
            if added >= target:
                break
            row = row_by_key[key]
            if can_add(key, row):
                add_key(key, row)
                added += 1

    # Phase 2: Fill remaining slots with least-similar diversity
    # Prioritize by dialect_score (descending), break ties by lexical diversity
    remaining = [r for r in candidates if sample_key(r) not in selected_set]
    remaining.sort(key=lambda r: features[sample_key(r)]["score"], reverse=True)

    selected_token_sets = [
        features[k]["tokens"] for k in selected_keys if k in features
    ]

    for row in remaining:
        if len(selected_keys) >= sample_size:
            break
        key = sample_key(row)
        if not can_add(key, row):
            continue
        cand_tokens = features[key]["tokens"]
        max_sim = max(
            (jaccard(cand_tokens, st) for st in selected_token_sets),
            default=0.0,
        )
        row["_diversity_score"] = features[key]["score"] - max_sim

    remaining.sort(key=lambda r: r.get("_diversity_score", 0), reverse=True)

    for row in remaining:
        if len(selected_keys) >= sample_size:
            break
        key = sample_key(row)
        if not can_add(key, row):
            continue
        add_key(key, row)
        if key in features:
            selected_token_sets.append(features[key]["tokens"])

    return selected_keys


# ── MCQA row builder ─────────────────────────────────────────────────


def build_mcqa_question_rows(dataset, dataset_id: str) -> list[dict[str, Any]]:
    rows = []
    for topic_index in range(len(dataset)):
        topic = to_builtin(dataset[topic_index])
        if not is_general_mcqa_topic(topic):
            continue
        questions = topic.get("questions") or []
        options = topic.get("options") or []
        answers = topic.get("answers") or []
        types = topic.get("types") or []
        for question_index, (question, option_set, answer) in enumerate(
            zip(questions, options, answers)
        ):
            question_row = {
                "_task": "mcqa",
                "_source_dataset": dataset_id,
                "_source_index": topic_index,
                "_sample_id": f"{topic_index}:{question_index}",
                "_topic_source_index": topic_index,
                "_question_index": question_index,
                "title": topic.get("title"),
                "article": topic.get("article"),
                "grade": topic.get("grade"),
                "author": topic.get("author"),
                "files": topic.get("files"),
                "isProse": topic.get("isProse"),
                "question": question,
                "options": option_set,
                "answer": answer,
                "type": types[question_index] if question_index < len(types) else None,
            }
            if is_valid_mcqa_question(question_row):
                rows.append(question_row)
    return rows


# ── I/O ──────────────────────────────────────────────────────────────


def normalize_row(task: str, dataset_id: str, source_index: int, row: dict[str, Any]) -> dict:
    row = to_builtin(row)
    return {"_task": task, "_source_dataset": dataset_id, "_source_index": source_index, **row}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_csv(path: Path, summary_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task", "dataset", "num_samples", "num_candidates",
                "num_with_triggers", "selection_pool", "seed", "labels",
                "category_coverage", "difficulty_dist", "avg_dialect_score",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


# ── Main selection pipeline ──────────────────────────────────────────


def select_for_task(
    task: str,
    sample_size: int,
    output_dir: Path,
    max_source_rows: int | None,
    split_override: str | None,
    seed: int,
    trigger_index: dict[str, list[dict]],
    min_score: float = 3.0,
    min_categories: int = 2,
) -> dict:
    if task not in DATASETS:
        known = ", ".join(sorted(DATASETS))
        raise SystemExit(f"Unknown task '{task}'. Known tasks: {known}")

    task_config = DATASETS[task]
    dataset_id = task_config["dataset_id"]
    split = split_override or task_config["split"]
    source_split = f"{split}[:{max_source_rows}]" if max_source_rows else split

    print(f"\n{'='*60}")
    print(f"Loading {task}: {dataset_id} (split={source_split})")

    # Try loading from HuggingFace; fall back to local JSON for gated datasets
    dataset = None
    loaded_from_fallback = False
    try:
        dataset = load_dataset(dataset_id, split=source_split)
    except Exception as e:
        print(f"  Cannot load from HuggingFace: {e}")

        local_dir = output_dir / dataset_id.split("/")[-1]
        local_file = local_dir / f"{split}_vimmrc.json" if task == "mcqa" else None
        if local_file and local_file.exists():
            print(f"  Loading from local file: {local_file}")
            with local_file.open(encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, list):
                candidates = []
                for topic_index, topic in enumerate(raw):
                    topic = to_builtin(topic)
                    if not is_general_mcqa_topic(topic):
                        continue
                    questions = topic.get("questions") or []
                    options = topic.get("options") or []
                    answers = topic.get("answers") or []
                    types = topic.get("types") or []
                    for question_index, (question, option_set, answer) in enumerate(
                        zip(questions, options, answers)
                    ):
                        question_row = {
                            "_task": "mcqa",
                            "_source_dataset": dataset_id,
                            "_source_index": topic_index,
                            "_sample_id": f"{topic_index}:{question_index}",
                            "_topic_source_index": topic_index,
                            "_question_index": question_index,
                            "title": topic.get("title"),
                            "article": topic.get("article"),
                            "grade": topic.get("grade"),
                            "author": topic.get("author"),
                            "files": topic.get("files"),
                            "isProse": topic.get("isProse"),
                            "question": question,
                            "options": option_set,
                            "answer": answer,
                            "type": types[question_index] if question_index < len(types) else None,
                        }
                        if is_valid_mcqa_question(question_row):
                            candidates.append(question_row)
            else:
                raise SystemExit(f"Unexpected format in {local_file}")
        else:
            existing = output_dir / f"{task}_{sample_size}.jsonl"
            if existing.exists():
                print(f"  Loading from existing JSONL: {existing}")
                candidates = []
                with existing.open(encoding="utf-8") as f:
                    for line in f:
                        candidates.append(json.loads(line))
                loaded_from_fallback = True
            else:
                raise SystemExit(f"No fallback data found for {task}")

    if dataset is not None:
        if task == "mcqa":
            candidates = build_mcqa_question_rows(dataset, dataset_id)
        else:
            candidates = [
                normalize_row(task, dataset_id, source_index, dataset[source_index])
                for source_index in range(len(dataset))
                if passes_filter(task, dataset[source_index])
            ]

    print(f"  After quality filter: {len(candidates)} candidates")

    # Find dialect triggers for each candidate using task-specific text fields
    text_fields_config = TASK_TEXT_FIELDS.get(task, {"full": []})

    for row in candidates:
        # Scan full text for all triggers
        full_text = " ".join(
            str(row.get(f, "")) for f in text_fields_config.get("full", [])
        )
        triggers = find_dialect_triggers(full_text, trigger_index)
        categories = list({t["category"] for t in triggers})

        row["_dialect_triggers"] = triggers
        row["_dialect_categories"] = categories
        row["_num_triggers"] = len(triggers)

        # Weighted scores
        row["_trigger_score"] = compute_trigger_score(triggers, task)
        row["_dialect_score"] = compute_dialect_score(triggers, task)
        row["_difficulty"] = classify_difficulty(categories)
        row["_num_categories"] = len(set(categories))

        # NLI: track trigger location (premise vs hypothesis vs both)
        if task == "nli":
            premise_text = str(row.get("premise", ""))
            hypothesis_text = str(row.get("hypothesis", ""))
            premise_triggers = find_dialect_triggers(premise_text, trigger_index)
            hypothesis_triggers = find_dialect_triggers(hypothesis_text, trigger_index)
            premise_cats = {t["category"] for t in premise_triggers}
            hypothesis_cats = {t["category"] for t in hypothesis_triggers}
            if premise_cats and hypothesis_cats:
                row["_trigger_location"] = "both"
            elif premise_cats:
                row["_trigger_location"] = "premise"
            elif hypothesis_cats:
                row["_trigger_location"] = "hypothesis"
            else:
                row["_trigger_location"] = "none"
        else:
            row["_trigger_location"] = "n/a"

    # Filter: require min_score >= threshold OR min_categories >= threshold
    # (sample must have enough failure likelihood)
    effective_min_score = 0 if loaded_from_fallback else min_score
    effective_min_categories = 0 if loaded_from_fallback else min_categories

    before = len(candidates)
    candidates = [
        r for r in candidates
        if r["_trigger_score"] >= effective_min_score
        or r["_num_categories"] >= effective_min_categories
    ]
    print(f"  After score filter (score>={effective_min_score} OR cats>={effective_min_categories}): "
          f"{len(candidates)} (removed {before - len(candidates)})")

    # Stats on filtered candidates
    triggered = sum(1 for r in candidates if r["_num_triggers"] > 0)
    avg_score = sum(r["_dialect_score"] for r in candidates) / max(len(candidates), 1)
    diff_dist = Counter(r["_difficulty"] for r in candidates)
    print(f"  With dialect triggers: {triggered}/{len(candidates)}")
    print(f"  Avg dialect score: {avg_score:.1f}")
    print(f"  Difficulty pool: {dict(diff_dist)}")

    if task == "nli":
        loc_dist = Counter(r["_trigger_location"] for r in candidates)
        print(f"  NLI trigger location: {dict(loc_dist)}")

    # Check we have enough
    if len(candidates) < sample_size:
        print(f"  WARNING: Only {len(candidates)} candidates for sample_size={sample_size}. Adjusting.")
        actual_sample_size = len(candidates)
    else:
        actual_sample_size = sample_size

    # Selection pool
    selection_pool = candidates
    if len(selection_pool) > MAX_SELECTION_POOL:
        rng = random.Random(seed)
        selection_pool = rng.sample(selection_pool, MAX_SELECTION_POOL)

    # Select with difficulty quota and dialect diversity
    source_indices = select_with_difficulty_quota(
        selection_pool,
        task,
        actual_sample_size,
        seed,
        max_per_topic=MAX_MCQA_QUESTIONS_PER_TOPIC if task == "mcqa" else None,
    )
    selected = [row for row in candidates if sample_key(row) in set(source_indices)]
    selected.sort(key=lambda row: source_indices.index(sample_key(row)))

    # Compute stats BEFORE cleaning internal fields
    all_cats_in_selected = set()
    trigger_counts_by_cat = Counter()
    total_triggers = 0
    total_score = 0
    diff_selected = Counter()
    for row in selected:
        cats = row.get("_dialect_categories", [])
        all_cats_in_selected.update(cats)
        for cat in cats:
            trigger_counts_by_cat[cat] += 1
        total_triggers += row.get("_num_triggers", 0)
        total_score += row.get("_dialect_score", 0)
        diff_selected[row.get("_difficulty", "easy")] += 1

    relevant_cats = TASK_CATEGORIES.get(task, set())
    coverage = len(all_cats_in_selected & relevant_cats) / len(relevant_cats) * 100 if relevant_cats else 0
    avg_dialect_score = total_score / max(len(selected), 1)

    labels = Counter(get_label(task, row) for row in selected)

    print(f"  Selected: {len(selected)} samples")
    print(f"  Label distribution: {dict(labels)}")
    print(f"  Category coverage: {coverage:.0f}% ({all_cats_in_selected & relevant_cats})")
    print(f"  Triggers by category: {dict(trigger_counts_by_cat)}")
    print(f"  Avg triggers/sample: {total_triggers/len(selected):.1f}")
    print(f"  Avg dialect score: {avg_dialect_score:.1f}")
    print(f"  Difficulty: {dict(diff_selected)}")

    if task == "nli":
        loc_selected = Counter(r.get("_trigger_location", "n/a") for r in selected)
        print(f"  NLI trigger location: {dict(loc_selected)}")

    # Clean up internal fields from output
    clean_keys = [
        "_dialect_triggers", "_dialect_categories", "_num_triggers",
        "_trigger_score", "_dialect_score", "_difficulty", "_num_categories",
        "_trigger_location", "_diversity_score",
    ]
    for row in selected:
        for key in clean_keys:
            row.pop(key, None)

    out_path = output_dir / f"{task}_{sample_size}.jsonl"
    write_jsonl(out_path, selected)

    return {
        "task": task,
        "dataset": dataset_id,
        "num_samples": len(selected),
        "num_candidates": len(candidates),
        "num_with_triggers": triggered,
        "selection_pool": len(selection_pool),
        "seed": seed,
        "labels": json.dumps(dict(labels), ensure_ascii=False),
        "category_coverage": f"{coverage:.0f}%",
        "difficulty_dist": json.dumps(dict(diff_selected)),
        "avg_dialect_score": f"{avg_dialect_score:.1f}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select 100 samples/task with dialect-trigger coverage for Vialect-Bench."
    )
    parser.add_argument(
        "--tasks", nargs="+",
        default=["sentiment", "nli", "qa", "mcqa"],
        help="Task names to sample.",
    )
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--max-source-rows", type=int, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument(
        "--lexicon", type=Path,
        default=Path("data/DIALECT_LEXICON_v2.xlsx"),
        help="Path to dialect lexicon Excel file.",
    )
    parser.add_argument(
        "--min-score", type=float, default=3.0,
        help="Minimum weighted trigger score for a sample to be eligible.",
    )
    parser.add_argument(
        "--min-categories", type=int, default=2,
        help="Minimum number of dialect categories a sample must trigger.",
    )
    args = parser.parse_args()

    print(f"Loading dialect lexicon from {args.lexicon}")
    entries = load_lexicon(str(args.lexicon))
    trigger_index = build_trigger_index(entries)
    print(f"  {len(entries)} lexicon entries, {len(trigger_index)} trigger keys")

    print(f"\nCategory weights by task:")
    for task, weights in CATEGORY_WEIGHT.items():
        print(f"  {task}: {dict(sorted(weights.items(), key=lambda x: -x[1]))}")

    print(f"\nFilters: min_score={args.min_score}, min_categories={args.min_categories}")
    print(f"Difficulty quota: {DIFFICULTY_QUOTA}")

    summaries = [
        select_for_task(
            task=task,
            sample_size=args.sample_size,
            output_dir=args.output_dir,
            max_source_rows=args.max_source_rows,
            split_override=args.split,
            seed=args.seed,
            trigger_index=trigger_index,
            min_score=args.min_score,
            min_categories=args.min_categories,
        )
        for task in args.tasks
    ]
    write_summary_csv(args.output_dir / "selection_summary.csv", summaries)
    print(f"\n{'='*60}")
    print("Selection complete!")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
