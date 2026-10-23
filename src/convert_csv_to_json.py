import csv
import json
import os

def convert_csv_to_json():
    # Đường dẫn file CSV chính theo cấu trúc An thông báo
    csv_path = 'data/UDD-1.csv'
    json_path = 'data/gold_annotated.json'

    # Tìm kiếm linh hoạt các vị trí file dự phòng nếu chưa di chuyển
    if not os.path.exists(csv_path):
        fallback_paths = ['UDD-1.csv', 'data/UDD.csv', 'UDD.csv', 'data/gold_manual.csv']
        for p in fallback_paths:
            if os.path.exists(p):
                csv_path = p
                break

    if not os.path.exists(csv_path):
        print(f"❌ Không tìm thấy file CSV tại: {csv_path}")
        return

    print(f"📖 Đang đọc dữ liệu từ: {csv_path}")
    dataset = []

    with open(csv_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
        lines = f.readlines()
        if not lines:
            print("❌ File CSV rỗng!")
            return

        # Tự động phát hiện dấu phân cách (chủ yếu là ';')
        first_content = "".join(lines[:10])
        delimiter = ';' if ';' in first_content else ','

        reader = csv.reader(lines, delimiter=delimiter)
        header = None

        for row in reader:
            if not row or not any(row):
                continue

            row_lower = [str(c).strip().lower() for c in row]
            if 'sent_id' in row_lower and 'text' in row_lower:
                header = row_lower
                continue

            if header and len(row) >= len(header):
                row_dict = dict(zip(header, [str(c).strip() for c in row]))
                sent_id = row_dict.get('sent_id', '')
                sentence = row_dict.get('text', '') or row_dict.get('sentence', '')
                sent_type = row_dict.get('type', 'simple')
                raw_triples = row_dict.get('gold_triples_text', '') or row_dict.get('gold_triples', '')

                if not sentence or not raw_triples:
                    continue

                gold_triples = []
                for t_str in raw_triples.split(';'):
                    t_str = t_str.strip()
                    if not t_str:
                        continue
                    parts = [p.strip() for p in t_str.split('|')]
                    if len(parts) == 3:
                        gold_triples.append(parts)

                if gold_triples:
                    dataset.append({
                        "id": len(dataset) + 1,
                        "sent_id": sent_id,
                        "sentence": sentence,
                        "type": sent_type if sent_type else "simple",
                        "gold_triples": gold_triples
                    })

    os.makedirs('data', exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"🎉 Đã chuyển đổi thành công {len(dataset)} câu sang JSON!")
    print(f"📁 File JSON đã lưu tại: {json_path}")

if __name__ == "__main__":
    convert_csv_to_json()
