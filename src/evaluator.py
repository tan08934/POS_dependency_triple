"""
Module: evaluator.py (Đã sửa lỗi logic theo góp ý của nhóm)

Mục đích: Đánh giá chính xác Precision, Recall, F1, Accuracy cho đa bộ ba,
          sửa lỗi tích Đề-các khi tính S/R/O và sửa lỗi log cứng [0].
"""

import json
import csv
import os
from typing import List, Dict, Tuple, Any


def load_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Đọc dữ liệu từ file JSON kiểm thử."""
    if not os.path.exists(filepath):
        print(f"❌ Không tìm thấy file: {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_string(text: str) -> str:
    """Chuẩn hóa chuỗi ký tự."""
    if not text:
        return ""
    return " ".join(str(text).strip().lower().split())


def is_exact_match(gold_triple: List[str], pred_triple: List[str]) -> bool:
    """So sánh khớp chính xác 100% giữa 2 bộ ba."""
    if len(gold_triple) != 3 or len(pred_triple) != 3:
        return False
    return (normalize_string(gold_triple[0]) == normalize_string(pred_triple[0]) and
            normalize_string(gold_triple[1]) == normalize_string(pred_triple[1]) and
            normalize_string(gold_triple[2]) == normalize_string(pred_triple[2]))


def is_partial_match(gold_triple: List[str], pred_triple: List[str]) -> bool:
    """So sánh khớp một phần giữa 2 bộ ba."""
    if len(gold_triple) != 3 or len(pred_triple) != 3:
        return False
    
    g_s, g_r, g_o = [normalize_string(x) for x in gold_triple]
    p_s, p_r, p_o = [normalize_string(x) for x in pred_triple]

    r_match = (g_r in p_r) or (p_r in g_r)
    s_match = (g_s in p_s) or (p_s in g_s)
    o_match = (g_o in p_o) or (p_o in g_o)

    return r_match and s_match and o_match


def calculate_metrics(tp: int, fp: int, fn: int, total_sentences: int = 0, correct_sentences: int = 0) -> Dict[str, float]:
    """Tính các chỉ số đo đạc chuẩn."""
    total_triples = tp + fp + fn
    accuracy = (tp / total_triples) if total_triples > 0 else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    res = {
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2)
    }

    if total_sentences > 0:
        res["sentence_accuracy"] = round((correct_sentences / total_sentences) * 100, 2)

    return res


def evaluate_predictions(test_results: List[Dict[str, Any]], match_mode: str = 'exact') -> Dict[str, Any]:
    """
    Hàm đánh giá chính đã được sửa lỗi:
    1. Hỗ trợ đa bộ ba (multiple triples per sentence) cho cả câu đơn và câu phức.
    2. Sửa lỗi phóng đại TP/FP ở đánh giá phân rã S / R / O.
    """
    match_fn = is_exact_match if match_mode == 'exact' else is_partial_match

    overall_tp, overall_fp, overall_fn = 0, 0, 0
    total_sentences = len(test_results)
    correct_sentences = 0

    type_stats = {}
    comp_stats = {
        "subject": {"tp": 0, "fp": 0, "fn": 0},
        "relation": {"tp": 0, "fp": 0, "fn": 0},
        "object": {"tp": 0, "fp": 0, "fn": 0}
    }
    error_logs = []

    for item in test_results:
        sent_type = item.get("type", "simple")
        if sent_type not in type_stats:
            type_stats[sent_type] = {"tp": 0, "fp": 0, "fn": 0, "total_sent": 0, "correct_sent": 0}
        
        type_stats[sent_type]["total_sent"] += 1

        gold_list = item.get("gold_triples", [])
        pred_list = item.get("predicted_triples", [])

        # Chuyển đổi an toàn nếu pred_list/gold_list không đúng dạng list
        gold_list = [list(g) for g in gold_list] if gold_list else []
        pred_list = [list(p) for p in pred_list] if pred_list else []

        matched_gold_indices = set()
        matched_pred_indices = set()

        # Ghép cặp chính xác giữa Gold và Predicted
        for p_idx, pred in enumerate(pred_list):
            for g_idx, gold in enumerate(gold_list):
                if g_idx not in matched_gold_indices and match_fn(gold, pred):
                    matched_gold_indices.add(g_idx)
                    matched_pred_indices.add(p_idx)
                    break

        tp = len(matched_pred_indices)
        fp = len(pred_list) - tp
        fn = len(gold_list) - len(matched_gold_indices)

        overall_tp += tp
        overall_fp += fp
        overall_fn += fn

        type_stats[sent_type]["tp"] += tp
        type_stats[sent_type]["fp"] += fp
        type_stats[sent_type]["fn"] += fn

        # Đánh giá chính xác cấp câu (Sentence Accuracy)
        if fp == 0 and fn == 0 and tp == len(gold_list) and len(gold_list) > 0:
            correct_sentences += 1
            type_stats[sent_type]["correct_sent"] += 1
        else:
            # Sửa lỗi: Không dùng pred_list[0] nữa, ghi log lỗi toàn bộ danh sách bộ ba
            error_logs.append({
                "id": item.get("id"),
                "sentence": item.get("sentence", ""),
                "type": sent_type,
                "gold_triples": gold_list,
                "predicted_triples": pred_list,
                "unmatched_gold_count": fn,
                "unmatched_pred_count": fp
            })

        # Sửa lỗi tính S / R / O: Chỉ tính trên các cặp đã được ghép 1-1, không lặp Tích Đề-các
        for g_idx, p_idx in zip(sorted(matched_gold_indices), sorted(matched_pred_indices)):
            g_s, g_r, g_o = gold_list[g_idx]
            p_s, p_r, p_o = pred_list[p_idx]

            if normalize_string(p_s) == normalize_string(g_s):
                comp_stats["subject"]["tp"] += 1
            else:
                comp_stats["subject"]["fp"] += 1

            if normalize_string(p_r) == normalize_string(g_r):
                comp_stats["relation"]["tp"] += 1
            else:
                comp_stats["relation"]["fp"] += 1

            if normalize_string(p_o) == normalize_string(g_o):
                comp_stats["object"]["tp"] += 1
            else:
                comp_stats["object"]["fp"] += 1

        # Cập nhật FN/FP cho S/R/O dựa trên các bộ ba chưa matched
        unmatched_preds = [p for i, p in enumerate(pred_list) if i not in matched_pred_indices]
        unmatched_golds = [g for i, g in enumerate(gold_list) if i not in matched_gold_indices]

        for _ in unmatched_preds:
            comp_stats["subject"]["fp"] += 1
            comp_stats["relation"]["fp"] += 1
            comp_stats["object"]["fp"] += 1

        for _ in unmatched_golds:
            comp_stats["subject"]["fn"] += 1
            comp_stats["relation"]["fn"] += 1
            comp_stats["object"]["fn"] += 1

    by_type_results = {
        stype: calculate_metrics(
            stats["tp"], stats["fp"], stats["fn"],
            total_sentences=stats["total_sent"],
            correct_sentences=stats["correct_sent"]
        ) for stype, stats in type_stats.items()
    }

    by_comp_results = {
        comp: calculate_metrics(stats["tp"], stats["fp"], stats["fn"])
        for comp, stats in comp_stats.items()
    }

    return {
        "match_mode": match_mode,
        "overall": calculate_metrics(
            overall_tp, overall_fp, overall_fn,
            total_sentences=total_sentences,
            correct_sentences=correct_sentences
        ),
        "by_type": by_type_results,
        "by_component": by_comp_results,
        "error_logs": error_logs
    }


def save_reports(report: Dict[str, Any], json_output: str = 'data/evaluation.json', csv_output: str = 'data/dependency_errors.csv'):
    """Lưu kết quả báo cáo an toàn dạng JSON và CSV."""
    os.makedirs(os.path.dirname(json_output), exist_ok=True)
    
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    errors = report.get("error_logs", [])
    if errors:
        with open(csv_output, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Sentence", "Type", "Gold_Triples", "Predicted_Triples", "Unmatched_Gold", "Unmatched_Pred"])
            for err in errors:
                writer.writerow([
                    err.get("id"),
                    err.get("sentence"),
                    err.get("type"),
                    str(err.get("gold_triples")),
                    str(err.get("predicted_triples")),
                    err.get("unmatched_gold_count"),
                    err.get("unmatched_pred_count")
                ])
    print(f"✅ Đã lưu kết quả đánh giá vào {json_output} và log lỗi vào {csv_output}")


if __name__ == "__main__":
    dataset_path = 'data/dataset_test.json'
    data = load_dataset(dataset_path)
    if data:
        report = evaluate_predictions(data, match_mode='exact')
        save_reports(report)
