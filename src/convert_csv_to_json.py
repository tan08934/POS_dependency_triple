import csv
import json
import os

def convert_csv_to_json():
    # 1. Khai báo đường dẫn file trong thư mục data/
    csv_path = 'data/gold_manual.csv'
    json_path = 'data/gold_annotated.json'

    if not os.path.exists(csv_path):
        print(f"Không tìm thấy file: {csv_path}")
        return

    dataset = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentence = row.get('sentence', row.get('text', '')).strip()
            sent_type = row.get('type', 'simple').strip()
            raw_triples = row.get('gold_triples_text', '').strip()

            # Bỏ qua những dòng chưa gán nhãn hoặc trống
            if not sentence or not raw_triples:
                continue

            # Tách chuỗi | và ; thành dạng mảng JSON
            gold_triples = []
            for t_str in raw_triples.split(';'):
                parts = [p.strip() for p in t_str.split('|')]
                if len(parts) == 3:
                    gold_triples.append(parts)

            dataset.append({
                "id": int(row.get('id', len(dataset) + 1)),
                "sent_id": row.get('sent_id', ''),
                "sentence": sentence,
                "type": sent_type if sent_type else "simple",
                "gold_triples": gold_triples
            })

    # 2. Đảm bảo thư mục data/ tồn tại và ghi file JSON
    os.makedirs('data', exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f" Đã chuyển đổi thành công {len(dataset)} câu!")
    print(f" File JSON đã lưu tại: {json_path}")

if __name__ == "__main__":
    convert_csv_to_json()
