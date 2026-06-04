from __future__ import annotations


def build_prompt(case: dict, variant_key: str) -> str:
    text = case[variant_key]
    task = case["task"]

    if task == "sentiment":
        return (
            "Bạn là hệ thống phân loại cảm xúc tiếng Việt. "
            "Chỉ trả lời một nhãn ngắn: positive, negative, neutral, hoặc other.\n"
            f"Câu: {text}\nNhãn:"
        )

    if task == "mt":
        return (
            "Chuyển câu tiếng Việt phương ngữ sau sang tiếng Việt phổ thông, "
            "giữ nguyên nghĩa và không giải thích.\n"
            f"Phương ngữ: {text}\nPhổ thông:"
        )

    if task == "nli":
        premise = case.get(f"{variant_key}_premise") or case.get("premise") or text
        hypothesis = case.get(f"{variant_key}_hypothesis") or case.get("hypothesis") or ""
        return (
            "Xác định quan hệ NLI giữa tiền đề và giả thuyết. "
            "Chỉ trả lời một nhãn: entailment, neutral, contradiction.\n"
            f"Tiền đề: {premise}\nGiả thuyết: {hypothesis}\nNhãn:"
        )

    if task == "qa":
        context = case.get(f"{variant_key}_context") or case.get("context") or ""
        question = case.get(f"{variant_key}_question") or case.get("question") or text
        return (
            "Trả lời câu hỏi dựa trên ngữ cảnh. Nếu không có câu trả lời, ghi unanswerable.\n"
            f"Ngữ cảnh: {context}\nCâu hỏi: {question}\nTrả lời:"
        )

    if task == "ner":
        return (
            "Trích xuất thực thể trong câu tiếng Việt. "
            "Trả lời JSON list các object có text và type.\n"
            f"Câu: {text}\nJSON:"
        )

    raise ValueError(f"Unsupported task: {task}")


def parse_prediction(task: str, output: str) -> str:
    normalized = output.strip().lower()
    if task == "sentiment":
        for label in ["positive", "negative", "neutral", "other"]:
            if label in normalized:
                return label
    if task == "nli":
        for label in ["entailment", "contradiction", "neutral"]:
            if label in normalized:
                return label
    if task == "qa" and "unanswerable" in normalized:
        return "unanswerable"
    return output.strip()
