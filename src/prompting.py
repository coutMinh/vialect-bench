from __future__ import annotations

import json
import re
from typing import Any


def _variant_text(variant: Any) -> str:
    if isinstance(variant, dict):
        return str(variant.get("text") or variant.get("sentence") or "")
    return str(variant)


def _format_mcqa_options(options: list[str]) -> str:
    labels = ["A", "B", "C", "D"]
    return "\n".join(f"{label}. {text}" for label, text in zip(labels, options))


def build_prompt(case: dict, variant_name: str, variant: Any) -> str:
    task = case["task"]

    if task == "sentiment":
        text = _variant_text(variant)
        return (
            "Bạn là hệ thống phân loại cảm xúc tiếng Việt.\n"
            "Chọn đúng một nhãn trong danh sách sau: Anger, Disgust, Enjoyment, "
            "Fear, Sadness, Surprise, Other.\n"
            "Chỉ trả lời JSON hợp lệ, không giải thích.\n"
            'Định dạng: {"label":"<nhãn>"}\n'
            f"Câu: {text}\n"
            "JSON:"
        )

    if task == "nli":
        standard = case.get("variants", {}).get("standard", {})
        variant = variant if isinstance(variant, dict) else {"premise": str(variant)}
        premise = variant.get("premise") or standard.get("premise") or case.get("premise") or ""
        hypothesis = (
            variant.get("hypothesis")
            or standard.get("hypothesis")
            or case.get("hypothesis")
            or ""
        )
        return (
            "Xác định quan hệ NLI giữa tiền đề và giả thuyết.\n"
            "Chọn đúng một nhãn: entailment, neutral, contradiction.\n"
            "Chỉ trả lời JSON hợp lệ, không giải thích.\n"
            'Định dạng: {"label":"<nhãn>"}\n'
            f"Tiền đề: {premise}\nGiả thuyết: {hypothesis}\n"
            "JSON:"
        )

    if task == "qa":
        standard = case.get("variants", {}).get("standard", {})
        variant = variant if isinstance(variant, dict) else {"question": str(variant)}
        context = variant.get("context") or standard.get("context") or case.get("context") or ""
        question = (
            variant.get("question")
            or standard.get("question")
            or case.get("question")
            or ""
        )
        return (
            "Trả lời câu hỏi dựa trên ngữ cảnh.\n"
            "Câu trả lời phải là một cụm ngắn được tìm thấy trong ngữ cảnh. "
            "Nếu không có câu trả lời, dùng unanswerable.\n"
            "Chỉ trả lời JSON hợp lệ, không giải thích.\n"
            'Định dạng: {"answer":"<câu trả lời>"}\n'
            f"Ngữ cảnh: {context}\nCâu hỏi: {question}\n"
            "JSON:"
        )

    if task == "mcqa":
        standard = case.get("variants", {}).get("standard", {})
        variant = variant if isinstance(variant, dict) else {"question": str(variant)}
        context = variant.get("context") or standard.get("context") or case.get("context") or ""
        question = (
            variant.get("question")
            or standard.get("question")
            or case.get("question")
            or ""
        )
        options = variant.get("options") or standard.get("options") or case.get("options") or []
        return (
            "Đọc ngữ cảnh và chọn một đáp án đúng nhất trong bốn lựa chọn A, B, C, D.\n"
            "Chỉ trả lời JSON hợp lệ, không giải thích.\n"
            'Định dạng: {"answer":"A"}\n'
            f"Ngữ cảnh: {context}\n"
            f"Câu hỏi: {question}\n"
            f"Lựa chọn:\n{_format_mcqa_options(options)}\n"
            "JSON:"
        )

    if task == "mt":
        text = _variant_text(variant)
        return (
            "Chuyển câu tiếng Việt phương ngữ sau sang tiếng Việt phổ thông, "
            "giữ nguyên nghĩa.\n"
            "Chỉ trả lời JSON hợp lệ, không giải thích.\n"
            'Định dạng: {"standard":"<câu phổ thông>"}\n'
            f"Phương ngữ: {text}\n"
            "JSON:"
        )

    raise ValueError(f"Unsupported task: {task}")


def parse_prediction(task: str, output: str) -> str:
    text = output.strip()
    json_match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            if task in {"sentiment", "nli"} and data.get("label"):
                return normalize_label(task, str(data["label"]))
            if task == "qa" and data.get("answer"):
                return str(data["answer"]).strip()
            if task == "mcqa" and (data.get("answer") or data.get("label")):
                answer = str(data.get("answer") or data.get("label")).strip().upper()
                match = re.search(r"\b([ABCD])\b", answer)
                return match.group(1) if match else answer
            if task == "mt" and data.get("standard"):
                return str(data["standard"]).strip()
        except json.JSONDecodeError:
            pass

    first_line = text.splitlines()[0].strip() if text else ""
    normalized = first_line.lower()
    if task == "sentiment":
        label_map = {
            "anger": "Anger",
            "disgust": "Disgust",
            "enjoyment": "Enjoyment",
            "fear": "Fear",
            "sadness": "Sadness",
            "surprise": "Surprise",
            "other": "Other",
        }
        for key, label in label_map.items():
            if key in normalized:
                return label
    if task == "nli":
        for label in ["entailment", "contradiction", "neutral"]:
            if label in normalized:
                return label
    if task == "qa" and "unanswerable" in normalized:
        return "unanswerable"
    if task == "mcqa":
        match = re.search(r"\b([ABCD])\b", first_line.upper())
        if match:
            return match.group(1)
    return first_line or text


def normalize_label(task: str, label: str) -> str:
    normalized = label.strip().lower()
    if task == "sentiment":
        label_map = {
            "anger": "Anger",
            "disgust": "Disgust",
            "enjoyment": "Enjoyment",
            "fear": "Fear",
            "sadness": "Sadness",
            "surprise": "Surprise",
            "other": "Other",
        }
        return label_map.get(normalized, label.strip())
    if task == "nli":
        label_map = {
            "entailment": "entailment",
            "neutral": "neutral",
            "contradiction": "contradiction",
        }
        return label_map.get(normalized, label.strip())
    return label.strip()
