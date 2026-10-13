"""
Module: evaluator.py

Mục đích: Đánh giá hiệu năng trích xuất bộ ba (Subject, Relation, Object)
          Tính toán Accuracy, Precision, Recall, F1 theo Exact Match, Partial Match, 
          phân loại câu và phân rã các thành phần (Subject/Relation/Object).
"""

import json
from typing import List, Dict, Tuple, Any


def load_dataset(filepath: str) -> List[Dict[str, Any]]:
    """
    Hàm đọc file dataset kiểm thử định dạng JSON.

    Đầu vào:
        filepath (str): Đường dẫn đến file .json (ví dụ: 'data/dataset_test.json').

    Đầu ra:
        List[Dict[str, Any]]: Danh sách các câu kiểm thử cùng nhãn chuẩn (gold_triples).
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def normalize_string(text: str) -> str:
    """
    Hàm chuẩn hóa chuỗi ký tự trước khi so sánh.

    Đầu vào:
        text (str): Chuỗi đầu vào.

    Đầu ra:
        str: Chuỗi đã được viết thường và xóa khoảng trắng thừa.
    """
    if not text:
        return ""
    return " ".join(text.strip().lower().split())


def is_exact_match(gold_triple: List[str], pred_triple: List[str]) -> bool:
    """
    So sánh khớp chính xác (Exact Match) giữa bộ ba chuẩn và bộ ba dự đoán.

    Đầu vào:
        gold_triple (List[str]): Bộ ba chuẩn [Subject, Relation, Object].
        pred_triple (List[str]): Bộ ba dự đoán [Subject, Relation, Object].

    Đầu ra:
        bool: True nếu cả 3 thành phần trùng khớp 100%, ngược lại False.
    """
    if len(gold_triple) != 3 or len(pred_triple) != 3:
        return False
    return (normalize_string(gold_triple[0]) == normalize_string(pred_triple[0]) and
            normalize_string(gold_triple[1]) == normalize_string(pred_triple[1]) and
            normalize_string(gold_triple[2]) == normalize_string(pred_triple[2]))


def is_partial_match(gold_triple: List[str], pred_triple: List[str]) -> bool:
    """
    So sánh khớp một phần (Partial Match).

    Đầu vào:
        gold_triple (List[str]): Bộ ba chuẩn [Subject, Relation, Object].
        pred_triple (List[str]): Bộ ba dự đoán [Subject, Relation, Object].

    Đầu ra:
        bool: True nếu Quan hệ trùng/chứa nhau và Subject, Object có sự giao thoa từ vựng.
    """
    if len(gold_triple) != 3 or len(pred_triple) != 3:
        return False
    
    g_s, g_r, g_o = [normalize_string(x) for x in gold_triple]
    p_s, p_r, p_o = [normalize_string(x) for x in pred_triple]

    r_match = (g_r in p_r) or (p_r in g_r)
    s_match = (g_s in p_s) or (p_s in g_s)
    o_match = (g_o in p_o) or (p_o in g_o)

    return r_match and s_match and o_match


def calculate_metrics(tp: int, fp: int, fn: int, total_sentences: int = 0, correct_sentences: int = 0) -> Dict[str, float]:
    """
    Tính Accuracy, Precision, Recall, F1-score từ số lượng TP, FP, FN.

    Đầu vào:
        tp (int): Số lượng True Positive (Dự đoán đúng).
        fp (int): Số lượng False Positive (Dự đoán thừa/sai).
        fn (int): Số lượng False Negative (Bỏ sót).
        total_sentences (int, optional): Tổng số câu đánh giá.
        correct_sentences (int, optional): Số câu dự đoán chính xác toàn bộ.

    Đầu ra:
        Dict[str, float]: Dictionary chứa các chỉ số đã làm tròn 2 chữ số thập phân.

    Giải thích công thức:
        - Triple Accuracy = TP / (TP + FP + FN)
        - Precision = TP / (TP + FP)
        - Recall = TP / (TP + FN)
        - F1-score = 2 * P * R / (P + R)
        - Sentence Accuracy = Correct Sentences / Total Sentences (nếu có tham số)
    """
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
        sent_acc = (correct_sentences / total_sentences) * 100
        res["sentence_accuracy"] = round(sent_acc, 2)

    return res


def evaluate_predictions(test_results: List[Dict[str, Any]], match_mode: str = 'exact') -> Dict[str, Any]:
    """
    Hàm tổng hợp đánh giá toàn bộ hệ thống trên dataset kiểm thử.

    Đầu vào:
        test_results (List[Dict[str, Any]]): Danh sách chứa các câu kiểm thử và kết quả dự đoán.
        match_mode (str): 'exact' hoặc 'partial'.

    Đầu ra:
        Dict[str, Any]: Báo cáo tổng hợp Accuracy, Precision, Recall, F1 và danh sách câu lỗi.
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
        sent_type = item.get("type", "unknown")
        if sent_type not in type_stats:
            type_stats[sent_type] = {"tp": 0, "fp": 0, "fn": 0, "total_sent": 0, "correct_sent": 0}
        
        type_stats[sent_type]["total_sent"] += 1

        gold_list = item.get("gold_triples", [])
        pred_list = item.get("predicted_triples", [])

        matched_gold_indices = set()
        matched_pred_indices = set()

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

        # Kiểm tra câu hoàn toàn đúng (Sentence-level Accuracy)
        if fp == 0 and fn == 0 and tp == len(gold_list):
            correct_sentences += 1
            type_stats[sent_type]["correct_sent"] += 1
        else:
            # Ghi log lỗi nếu câu chưa hoàn hảo
            error_field = "multiple"
            if len(pred_list) > 0 and len(gold_list) > 0:
                p_s, p_r, p_o = pred_list[0]
                g_s, g_r, g_o = gold_list[0]
                if p_s != g_s and p_r == g_r and p_o == g_o:
                    error_field = "subject"
                elif p_s == g_s and p_r != g_r and p_o == g_o:
                    error_field = "relation"
                elif p_s == g_s and p_r == g_r and p_o != g_o:
                    error_field = "object"

            error_logs.append({
                "sentence": item.get("sentence", ""),
                "type": sent_type,
                "error_field": error_field,
                "gold": gold_list,
                "predicted": pred_list
            })

        # Đánh giá phân rã S / R / O
        for pred in pred_list:
            for gold in gold_list:
                if normalize_string(pred[0]) == normalize_string(gold[0]):
                    comp_stats["subject"]["tp"] += 1
                else:
                    comp_stats["subject"]["fp"] += 1

                if normalize_string(pred[1]) == normalize_string(gold[1]):
                    comp_stats["relation"]["tp"] += 1
                else:
                    comp_stats["relation"]["fp"] += 1

                if normalize_string(pred[2]) == normalize_string(gold[2]):
                    comp_stats["object"]["tp"] += 1
                else:
                    comp_stats["object"]["fp"] += 1

    # Tính toán kết quả cho từng nhóm
    by_type_results = {}
    for stype, stats in type_stats.items():
        by_type_results[stype] = calculate_metrics(
            stats["tp"], stats["fp"], stats["fn"],
            total_sentences=stats["total_sent"],
            correct_sentences=stats["correct_sent"]
        )

    by_comp_results = {}
    for comp, stats in comp_stats.items():
        by_comp_results[comp] = calculate_metrics(stats["tp"], stats["fp"], stats["fn"])

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


if __name__ == "__main__":
    print("--- CHẠY THỬ MODULE EVALUATOR CÓ ACCURACY ---")
    mock_pipeline_output = [
        {
            "sentence": "Trường Đại học A triển khai chương trình AI.",
            "type": "simple",
            "gold_triples": [["Trường Đại học A", "triển khai", "chương trình AI"]],
            "predicted_triples": [["Trường Đại học A", "triển khai", "chương trình AI"]]
        },
        {
            "sentence": "Công ty A không triển khai dự án B.",
            "type": "negative",
            "gold_triples": [["Công ty A", "không triển khai", "dự án B"]],
            "predicted_triples": [["Công ty A", "triển khai", "dự án B"]]
        }
    ]

    report = evaluate_predictions(mock_pipeline_output, match_mode='exact')
    print(json.dumps(report, ensure_ascii=False, indent=2))
