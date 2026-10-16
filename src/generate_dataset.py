"""
Script: src/generate_dataset.py
Mục đích: Đọc file UDD-1.csv (28.1 MB) và trích xuất 200 câu đầu tiên
          chuyển thành định dạng JSON chuẩn cho dự án.
"""

import pandas as pd
import json
import os

def create_dataset_from_udd():
    csv_path = 'data/UDD-1.csv'
    output_path = 'data/dataset_test.json'

    if not os.path.exists(csv_path):
        print(f"❌ Không tìm thấy file {csv_path}")
        return

    print("⏳ Đang đọc file data/UDD-1.csv...")
    try:
        # Đọc file CSV (chỉ lấy các dòng cần thiết để tối ưu tốc độ)
        df = pd.read_csv(csv_path)
        
        # Nhận diện cột chứa câu hoặc ghép các token thành câu
        sentences = []
        if 'sentence' in df.columns:
            sentences = df['sentence'].dropna().unique().tolist()
        elif 'text' in df.columns:
            sentences = df['text'].dropna().unique().tolist()
        elif 'token' in df.columns or 'word' in df.columns:
            col_name = 'token' if 'token' in df.columns else 'word'
            # Ghép token thành câu nếu CSV lưu dạng conll/token
            current_sent = []
            for item in df[col_name].fillna(''):
                item_str = str(item).strip()
                if item_str in ['.', '?', '!']:
                    current_sent.append(item_str)
                    sentences.append(" ".join(current_sent))
                    current_sent = []
                else:
                    current_sent.append(item_str)
                if len(sentences) >= 200:
                    break

        # Nếu chưa đủ 200 câu, lọc thêm theo câu độc lập
        if len(sentences) < 200:
            raw_lines = df.iloc[:, 0].astype(str).unique().tolist()
            sentences = [line for line in raw_lines if len(line.split()) > 3]

    except Exception as e:
        print(f"Chuyển sang chế độ đọc text thuần: {e}")
        with open(csv_path, 'r', encoding='utf-8') as f:
            sentences = []
            for line in f:
                line_clean = line.strip()
                if line_clean and not line_clean.startswith('#'):
                    sentences.append(line_clean)
                if len(sentences) >= 200:
                    break

    # Giới hạn đúng 200 câu
    target_sentences = sentences[:200]
    
    dataset_json = []
    for idx, sent in enumerate(target_sentences, 1):
        # Tự động gán nhãn phân loại câu sơ bộ
        sent_lower = sent.lower()
        if any(w in sent_lower for w in ["không", "chưa", "chẳng"]):
            stype = "negative"
        elif any(w in sent_lower for w in ["được", "bị"]) and "bởi" in sent_lower:
            stype = "passive"
        elif " và " in sent_lower or " đồng thời " in sent_lower:
            stype = "multiple_verbs"
        elif len(sent.split()) > 15:
            stype = "long_phrase"
        else:
            stype = "simple"

        dataset_json.append({
            "id": idx,
            "sentence": sent,
            "type": stype,
            "gold_triples": []  # Khung nhãn chuẩn để bổ sung bộ ba
        })

    # Lưu ra file data/dataset_test.json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_json, f, ensure_ascii=False, indent=2)

    print(f"🎉 ĐÃ TẠO THÀNH CÔNG {len(dataset_json)} CÂU VÀO {output_path}!")

if __name__ == "__main__":
    create_dataset_from_udd()
