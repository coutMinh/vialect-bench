"""Compare full normalized predictions with the frozen direct run.

Main score matches EDA: QA token F1, and exact-match accuracy for other tasks.
Exact-match transitions are reported separately for all tasks.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import random
import re


MODEL_ORDER = (
    "qwen_2_5_0_5b_instruct", "qwen_2_5_3b_instruct", "qwen_2_5_7b_instruct",
    "llama_3_1_8b_instruct", "mistral_7b_instruct_v0_3", "gemma_2_9b_it",
    "gemma_3_4b_it", "vistral_7b_chat", "seallm_7b_chat", "gpt_4o",
)


def norm(value):
    if isinstance(value, list):
        value = value[0] if value else ""
    text = re.sub(r"\s+", " ", str(value or "").lower().strip())
    text = re.sub(r"""[\.,;:!?\(\)\[\]\{\}"']+""", "", text)
    return text.replace(chr(96), "").strip()


def exact(prediction, gold):
    return any(norm(prediction) == norm(item) for item in (gold if isinstance(gold, list) else [gold]))


def f1(prediction, gold):
    if isinstance(gold, list):
        return max((f1(prediction, item) for item in gold), default=0.0)
    p, g = norm(prediction).split(), norm(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    overlap = sum((Counter(p) & Counter(g)).values())
    return 2 * overlap / (len(p) + len(g)) if overlap else 0.0


def metric(row):
    return f1(row["prediction"], row["gold"]) if row["task"] == "qa" else float(exact(row["prediction"], row["gold"]))


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def key(row):
    return (row["task"], str(row["id"]), row["variant"])


def dialect_fields(task, variant):
    if task == "sentiment":
        return {"text": variant["text"]}
    if task == "nli":
        return {"hypothesis": variant["hypothesis"]}
    if task == "qa":
        return {"question": variant["question"]}
    if task == "mcqa":
        return {"question": variant["question"], "options": variant["options"]}
    raise ValueError(f"Unknown task: {task}")


def average(values):
    return sum(values) / len(values)


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cluster_ci(case_deltas, seed=20261028, replicates=2000):
    """Paired source-case bootstrap, stratified to preserve 100 cases/task."""
    if len(case_deltas) != 400:
        raise ValueError("Expected one delta for each of 400 source cases")
    by_task = defaultdict(list)
    for (task, _case_id), delta in case_deltas.items():
        by_task[task].append(delta)
    if set(by_task) != {"mcqa", "nli", "qa", "sentiment"} or any(len(values) != 100 for values in by_task.values()):
        raise ValueError("Expected 100 source cases per task")
    rng = random.Random(seed)
    means = sorted(
        sum(sum(rng.choices(by_task[task], k=100))
            for task in ("mcqa", "nli", "qa", "sentiment")) / 400
        for _ in range(replicates)
    )
    return means[int(.025 * replicates)], means[int(.975 * replicates)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-dir", type=Path, default=Path("outputs/direct_prompting"))
    parser.add_argument("--normalized-dir", type=Path, default=Path("outputs/normalized_direct_prompting_full"))
    parser.add_argument("--normalized-inputs", type=Path, default=Path("outputs/normalizer_full/normalized_case_dialects.jsonl"))
    parser.add_argument("--probe", type=Path, default=Path("data/Finalized/probe_dialects.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper_artifacts/direct_prompting"))
    args = parser.parse_args()

    input_rows = read(args.normalized_inputs)
    input_keys = {key(row) for row in input_rows}
    if len(input_rows) != 2400 or len(input_keys) != 2400:
        raise ValueError("Expected 2,400 unique normalized variants")
    if len({(row["normalizer_id"], row["normalizer_revision"]) for row in input_rows}) != 1:
        raise ValueError("Mixed normalizer checkpoints")
    probe = {str(row["id"]): row for row in read(args.probe)}
    if len(probe) != 400:
        raise ValueError("Expected 400 source cases")
    original_standard_changed = 0
    original_standard_changed_after_score_norm = 0
    exact_standard = 0
    source_exact_standard = 0
    changed_keys = set()
    field_counts = defaultdict(lambda: [0, 0, 0, 0])
    for item in input_rows:
        case = probe[item["id"]]
        if item["source"] != dialect_fields(item["task"], case["variants"][item["variant"]]):
            raise ValueError(f"Source input mismatch: {key(item)}")
        standard = dialect_fields(item["task"], case["variants"]["standard"])
        source_exact_standard += item["source"] == standard
        exact_standard += item["normalized"] == standard
        original_standard_changed += item["source"] == standard and item["normalized"] != standard
        if item["source"] == standard and item["normalized"] != standard:
            for field, source_value in item["source"].items():
                normalized_value = item["normalized"][field]
                source_values = source_value if isinstance(source_value, list) else [source_value]
                normalized_values = normalized_value if isinstance(normalized_value, list) else [normalized_value]
                if (len(source_values) != len(normalized_values)
                        or any(norm(source_text) != norm(normalized_text)
                               for source_text, normalized_text in zip(source_values, normalized_values))):
                    original_standard_changed_after_score_norm += 1
                    break
        if item["source"] != item["normalized"]:
            changed_keys.add(key(item))
        for field, source_value in item["source"].items():
            source_values = source_value if isinstance(source_value, list) else [source_value]
            normalized_value = item["normalized"][field]
            normalized_values = normalized_value if isinstance(normalized_value, list) else [normalized_value]
            standard_value = standard[field]
            standard_values = standard_value if isinstance(standard_value, list) else [standard_value]
            if not (len(source_values) == len(normalized_values) == len(standard_values)):
                raise ValueError(f"Field shape changed: {key(item)}, {field}")
            counter = field_counts[(item["task"], field)]
            for source_text, normalized_text, standard_text in zip(
                    source_values, normalized_values, standard_values, strict=True):
                counter[0] += 1
                counter[1] += source_text == standard_text
                counter[2] += normalized_text == standard_text
                counter[3] += source_text != normalized_text

    direct_files = {p.name: p for p in args.direct_dir.glob("*.jsonl")}
    normalized_files = {p.name: p for p in args.normalized_dir.glob("*.jsonl")}
    expected_filenames = {f"{slug}.jsonl" for slug in MODEL_ORDER}
    if direct_files.keys() != expected_filenames or normalized_files.keys() != expected_filenames:
        raise ValueError("Expected matching direct and normalized files for all 10 models")

    summary_rows, detail_rows, all_flips = [], [], Counter()
    case_outcomes = defaultdict(list)
    model_case_deltas = []
    all_standard_scores, all_direct_scores, all_normalized_scores = [], [], []
    control_results = []
    pooled_groups = defaultdict(lambda: [[], [], [], []])
    gold_probability_groups = defaultdict(list)
    for slug in MODEL_ORDER:
        filename = f"{slug}.jsonl"
        direct = read(direct_files[filename])
        normalized = read(normalized_files[filename])
        if len(direct) != 2800 or len(normalized) != 2400:
            raise ValueError(f"Wrong row count: {filename}")
        direct_map = {key(row): row for row in direct}
        normalized_map = {key(row): row for row in normalized}
        if len(direct_map) != 2800 or len(normalized_map) != 2400:
            raise ValueError(f"Duplicate result key: {filename}")
        if set(normalized_map) != {k for k in direct_map if k[2] != "standard"} or set(normalized_map) != input_keys:
            raise ValueError(f"Coverage mismatch: {filename}")
        model_name = normalized[0]["model_name"]

        standard_scores, direct_scores, normalized_scores = [], [], []
        groups = defaultdict(lambda: [[], [], [], []])
        source_deltas = defaultdict(list)
        changed_deltas, unchanged_deltas = [], []
        unchanged_flips = 0
        flips = Counter()
        for item_key, old in direct_map.items():
            if item_key[2] == "standard":
                standard_scores.append(metric(old))
                continue
            new = normalized_map[item_key]
            if new["gold"] != old["gold"] or new["prompt_strategy"] != "normalized_direct":
                raise ValueError(f"Metadata mismatch: {filename}, {item_key}")
            if not str(new.get("raw_output") or "").strip():
                raise ValueError(f"Empty LLM output: {filename}, {item_key}")
            old_score, new_score = metric(old), metric(new)
            old_prob, new_prob = old.get("gold_prob"), new.get("gold_prob")
            if (item_key[0] != "qa" and isinstance(old_prob, (int, float))
                    and isinstance(new_prob, (int, float))
                    and math.isfinite(old_prob) and math.isfinite(new_prob)):
                gold_probability_groups[(model_name, item_key[0])].append(
                    (old_prob, new_prob, item_key in changed_keys)
                )
            direct_scores.append(old_score)
            normalized_scores.append(new_score)
            groups[(item_key[0], item_key[2])][0].append(old_score)
            groups[(item_key[0], item_key[2])][1].append(new_score)
            source_deltas[(item_key[0], item_key[1])].append(new_score - old_score)
            transition = (exact(old["prediction"], old["gold"]), exact(new["prediction"], new["gold"]))
            groups[(item_key[0], item_key[2])][2].append(float(transition[0]))
            groups[(item_key[0], item_key[2])][3].append(float(transition[1]))
            for grouping in (("task", item_key[0]), ("dialect", item_key[2])):
                pooled_groups[grouping][0].append(old_score)
                pooled_groups[grouping][1].append(new_score)
                pooled_groups[grouping][2].append(float(transition[0]))
                pooled_groups[grouping][3].append(float(transition[1]))
            if item_key in changed_keys:
                changed_deltas.append(new_score - old_score)
            else:
                unchanged_deltas.append(new_score - old_score)
                unchanged_flips += transition[0] != transition[1]
            flips[transition] += 1
            all_flips[transition] += 1
            case_outcomes[item_key].append((old_score, new_score, *transition))

        s, d, n = average(standard_scores), average(direct_scores), average(normalized_scores)
        all_standard_scores.append(s)
        all_direct_scores.append(d)
        all_normalized_scores.append(n)
        source_means = {source: average(values) for source, values in source_deltas.items()}
        if len(source_means) != 400 or any(len(values) != 6 for values in source_deltas.values()):
            raise ValueError(f"Incomplete source case: {filename}")
        model_case_deltas.append(source_means)
        ci_low, ci_high = cluster_ci(source_means)
        gap, change = s - d, n - d
        control_results.append({
            "model": model_name,
            "changed_inputs": len(changed_deltas),
            "changed_input_gain_pp": 100 * average(changed_deltas),
            "unchanged_inputs": len(unchanged_deltas),
            "unchanged_input_gain_pp": 100 * average(unchanged_deltas),
            "unchanged_input_exact_match_flips": unchanged_flips,
        })
        summary_rows.append({
            "Model": model_name, "Standard score": f"{s*100:.1f}",
            "Direct dialect score": f"{d*100:.1f}",
            "Normalized dialect score": f"{n*100:.1f}",
            "Change (pp)": f"{change*100:+.1f}",
            "Change 95% CI low (pp)": f"{ci_low*100:+.1f}",
            "Change 95% CI high (pp)": f"{ci_high*100:+.1f}",
            "Standard gap closed": f"{change/gap*100:.1f}" if gap > 0 else "-",
            "Incorrect to correct": flips[(False, True)],
            "Correct to incorrect": flips[(True, False)],
            "Incorrect to incorrect": flips[(False, False)],
            "Correct to correct": flips[(True, True)],
        })
        for (task, dialect), (before, after, before_em, after_em) in sorted(groups.items()):
            detail_rows.append({
                "Model": model_name, "Task": task, "Dialect": dialect, "Rows": len(before),
                "Direct score": f"{average(before)*100:.1f}",
                "Normalized score": f"{average(after)*100:.1f}",
                "Change (pp)": f"{(average(after)-average(before))*100:+.1f}",
                "Direct EM": f"{average(before_em)*100:.1f}",
                "Normalized EM": f"{average(after_em)*100:.1f}",
                "EM change (pp)": f"{(average(after_em)-average(before_em))*100:+.1f}",
            })

    model_path = args.output_dir / "paper_table_normalizer_full.csv"
    detail_path = args.output_dir / "paper_table_normalizer_full_by_task_dialect.csv"
    write(model_path, summary_rows)
    write(detail_path, detail_rows)
    probability_rows = []
    for (model_name, task), values in sorted(gold_probability_groups.items()):
        before = average([pair[0] for pair in values])
        after = average([pair[1] for pair in values])
        changed = [pair[1] - pair[0] for pair in values if pair[2]]
        unchanged = [pair[1] - pair[0] for pair in values if not pair[2]]
        probability_rows.append({
            "Model": model_name, "Task": task, "Paired rows": len(values),
            "Direct mean gold probability": f"{before:.4f}",
            "Normalized mean gold probability": f"{after:.4f}",
            "Change": f"{after-before:+.4f}",
            "Changed-input change": f"{average(changed):+.4f}" if changed else "",
            "Unchanged-input change": f"{average(unchanged):+.4f}" if unchanged else "",
            "Rows with increased gold probability": sum(pair[1] > pair[0] for pair in values),
        })
    if probability_rows:
        write(args.output_dir / "paper_table_normalizer_full_gold_probability.csv", probability_rows)
        probability_task_rows = []
        for task in ("mcqa", "nli", "sentiment"):
            values = [pair for (model, group_task), pairs in gold_probability_groups.items()
                      if group_task == task for pair in pairs]
            if not values:
                continue
            before = average([pair[0] for pair in values])
            after = average([pair[1] for pair in values])
            probability_task_rows.append({
                "Task": task, "Paired model rows": len(values),
                "Direct mean gold probability": f"{before:.4f}",
                "Normalized mean gold probability": f"{after:.4f}",
                "Change": f"{after-before:+.4f}",
                "Rows with increased gold probability": sum(pair[1] > pair[0] for pair in values),
            })
        write(args.output_dir / "paper_table_normalizer_full_gold_probability_by_task.csv",
              probability_task_rows)
    tex_lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & Standard & Direct & Normalized & Gain (pp) & Gap closed (\%) \\",
        r"\midrule",
    ]
    for row in summary_rows:
        model = row["Model"].replace("_", r"\_").replace("&", r"\&")
        tex_lines.append(
            f"{model} & {row['Standard score']} & {row['Direct dialect score']} & "
            f"{row['Normalized dialect score']} & {row['Change (pp)']} & "
            f"{row['Standard gap closed']} " + r"\\"
        )
    mean_standard = average(all_standard_scores)
    mean_direct = average(all_direct_scores)
    mean_normalized = average(all_normalized_scores)
    mean_gap_closed = (100 * (mean_normalized - mean_direct) / (mean_standard - mean_direct)
                       if mean_standard > mean_direct else float("nan"))
    tex_lines.extend([
        r"\midrule",
        f"Mean & {mean_standard*100:.1f} & {mean_direct*100:.1f} & "
        f"{mean_normalized*100:.1f} & {(mean_normalized-mean_direct)*100:+.1f} & "
        f"{mean_gap_closed:.1f} " + r"\\",
    ])
    tex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (args.output_dir / "paper_table_normalizer_full.tex").write_text(
        "\n".join(tex_lines) + "\n", encoding="utf-8"
    )
    for grouping in ("task", "dialect"):
        pooled_rows = []
        for (kind, value), (before, after, before_em, after_em) in sorted(pooled_groups.items()):
            if kind != grouping:
                continue
            pooled_rows.append({
                grouping.title(): value, "Model rows": len(before),
                "Direct score": f"{average(before)*100:.1f}",
                "Normalized score": f"{average(after)*100:.1f}",
                "Change (pp)": f"{(average(after)-average(before))*100:+.1f}",
                "Direct EM": f"{average(before_em)*100:.1f}",
                "Normalized EM": f"{average(after_em)*100:.1f}",
                "EM change (pp)": f"{(average(after_em)-average(before_em))*100:+.1f}",
            })
        write(args.output_dir / f"paper_table_normalizer_full_by_{grouping}.csv", pooled_rows)
    case_rows = []
    for item in input_rows:
        item_key = key(item)
        outcomes = case_outcomes[item_key]
        if len(outcomes) != 10:
            raise ValueError(f"Missing model outcomes: {item_key}")
        case_rows.append({
            "task": item_key[0], "id": item_key[1], "variant": item_key[2],
            "source": json.dumps(item["source"], ensure_ascii=False),
            "normalized": json.dumps(item["normalized"], ensure_ascii=False),
            "standard": json.dumps(dialect_fields(item["task"], probe[item["id"]]["variants"]["standard"]), ensure_ascii=False),
            "text_changed": item["source"] != item["normalized"],
            "mean_direct_score": f"{average([x[0] for x in outcomes]):.4f}",
            "mean_normalized_score": f"{average([x[1] for x in outcomes]):.4f}",
            "mean_change": f"{average([x[1]-x[0] for x in outcomes]):+.4f}",
            "models_corrected_em": sum(not x[2] and x[3] for x in outcomes),
            "models_regressed_em": sum(x[2] and not x[3] for x in outcomes),
            "models_still_incorrect_em": sum(not x[2] and not x[3] for x in outcomes),
        })
    case_path = args.normalized_inputs.parent / "case_level_outcomes.csv"
    write(case_path, case_rows)
    field_rows = []
    for (task, field), (total, source_match, normalized_match, changed) in sorted(field_counts.items()):
        field_rows.append({
            "Task": task, "Field": field, "Text segments": total,
            "Source exact standard": source_match,
            "Normalized exact standard": normalized_match,
            "Changed segments": changed,
        })
    write(args.normalized_inputs.parent / "field_match_summary.csv", field_rows)
    summary = {
        "models": 10, "unique_variants": 2400, "model_rows": 24000,
        "changed_variants": sum(row["source"] != row["normalized"] for row in input_rows),
        "source_exact_standard_variants": source_exact_standard,
        "exact_standard_variants": exact_standard,
        "originally_standard_changed": original_standard_changed,
        "originally_standard_changed_after_score_norm": original_standard_changed_after_score_norm,
        "unchanged_prompt_controls": control_results,
        "exact_match_transitions": {
            "incorrect_to_correct": all_flips[(False, True)],
            "correct_to_incorrect": all_flips[(True, False)],
            "incorrect_to_incorrect": all_flips[(False, False)],
            "correct_to_correct": all_flips[(True, True)],
        },
    }
    sources = sorted(model_case_deltas[0])
    if any(set(model) != set(sources) for model in model_case_deltas):
        raise ValueError("Models cover different source cases")
    macro_case_deltas = {source: average([model[source] for model in model_case_deltas])
                         for source in sources}
    macro_ci_low, macro_ci_high = cluster_ci(macro_case_deltas)
    macro_standard = average(all_standard_scores)
    macro_direct = average(all_direct_scores)
    macro_normalized = average(all_normalized_scores)
    summary["macro_across_models"] = {
        "standard_score": macro_standard,
        "direct_dialect_score": macro_direct,
        "normalized_dialect_score": macro_normalized,
        "standard_reference_gap_pp": 100 * (macro_standard - macro_direct),
        "gain_pp": 100 * average(list(macro_case_deltas.values())),
        "standard_reference_gap_recovered_percent": (
            100 * (macro_normalized - macro_direct) / (macro_standard - macro_direct)
            if macro_standard > macro_direct else None
        ),
        "gain_case_cluster_bootstrap_95ci_pp": [100 * macro_ci_low, 100 * macro_ci_high],
        "bootstrap_unit": "source case (400), stratified by task (100 each), with six dialect variants and ten models per case",
        "bootstrap_replicates": 2000,
    }
    summary_path = args.normalized_inputs.parent / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {model_path}, {detail_path}, {case_path}, {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
