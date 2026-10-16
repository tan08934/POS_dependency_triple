POS_dependency_triple
Đồ án xây dựng quy trình xử lý ngôn ngữ tự nhiên cho văn bản tiếng Việt, bao gồm chuẩn hóa văn bản, tách từ, gán nhãn từ loại (POS tagging), phân tích phụ thuộc (dependency parsing) và trích xuất bộ ba (subject, relation, object).
Môi trường thực thi yêu cầu Python 3.11.9. Dự án sử dụng thư viện Underthesea để tách từ tiếng Việt.

1. Cài đặt Python 3.11.9
Tải Python 3.11.9 từ trang phát hành chính thức:
https://www.python.org/downloads/release/python-3119/
Sau khi cài đặt, mở PowerShell trên Windows hoặc Terminal trên macOS và kiểm tra phiên bản:
python --version
Trên Windows, nếu lệnh trên không hoạt động, sử dụng lệnh này trên powershell:
py -3.11 --version
Phiên bản hiển thị cần là Python 3.11.9.

2. Cài đặt thư viện Python
Mở PowerShell hoặc Terminal tại thư mục gốc của dự án và thực hiện lệnh:
python -m pip install --upgrade pip
python -m pip install underthesea pandas numpy
Nếu hệ thống Windows sử dụng py -3.11 để gọi Python, sử dụng:
py -3.11 -m pip install --upgrade pip
py -3.11 -m pip install underthesea 
Underthesea: cung cấp chức năng tách từ tiếng Việt.

3. Sử dụng Underthesea để tách từ
Sau khi cài đặt, kiểm tra thư viện bằng lệnh:
python -c "from underthesea import word_tokenize; print(word_tokenize('Chủ tọa phiên tòa phải hỏi riêng từng bị cáo.'))"
Nếu hệ thống Windows sử dụng py -3.11, chạy:
py -3.11 -c "from underthesea import word_tokenize; print(word_tokenize('Chủ tọa phiên tòa phải hỏi riêng từng bị cáo.'))"
Kết quả trả về là danh sách các từ sau khi tách. Một số từ tiếng Việt gồm nhiều âm tiết có thể được biểu diễn thành một phần tử trong danh sách, chẳng hạn Việt Nam.
Trong mã nguồn Python, có thể sử dụng chức năng tách từ như sau:
from underthesea import word_tokenize
def segment_text(text: str) -> list[str]:
    return word_tokenize(text)
Hàm segment_text nhận một câu tiếng Việt và trả về danh sách các đơn vị từ sau khi tách.
Tài liệu tham khảo: https://github.com/undertheseanlp/underthesea/blob/main/docs/docs/api/word-tokenize.md

4. Tải và chạy dự án
Tải mã nguồn bằng Git:
git clone https://github.com/tan08934/POS_dependency_triple.git
Di chuyển vào thư mục dự án:
cd POS_dependency_triple
Sau khi cài đặt Python, các thư viện cần thiết và bảo đảm mã nguồn đã được cập nhật để sử dụng Underthesea, khởi chạy chương trình bằng lệnh:
python src/main.py
Trên Windows, nếu cần sử dụng Python 3.11 thông qua py, chạy:
py -3.11 src/main.py

5. Lưu ý về môi trường thực thi
Trước khi thực hiện các lệnh, cần mở PowerShell hoặc Terminal tại thư mục POS_dependency_triple.
Nếu xảy ra lỗi khi import underthesea, kiểm tra xem thư viện đã được cài đặt trong đúng môi trường Python đang chạy chương trình hay chưa.
Nếu chương trình không khởi chạy, kiểm tra phiên bản Python, các thư viện đã cài đặt, đường dẫn mã nguồn và thông báo lỗi do chương trình cung cấp.


# Người 2 — Phân tích cú pháp phụ thuộc tiếng Việt bằng MaltParser

## 1. Mục tiêu

Xây dựng bước **Dependency Parsing** trong pipeline xử lý ngôn ngữ tự nhiên của nhóm.

| Hạng mục | Nội dung |
| :-- | :-- |
| Đầu vào | Câu tiếng Việt đã tách từ và gán nhãn từ loại (POS) |
| Đầu ra | Cây phụ thuộc: mỗi từ có **head** (từ chủ) và **deprel** (nhãn quan hệ) |
| Định dạng xử lý và đánh giá | CoNLL-X |
| Định dạng bàn giao cho Người 3 | JSONL (`data/dependency_results.jsonl`) |

## 2. Phương pháp

| Thành phần | Mô tả |
| :-- | :-- |
| Parser | MaltParser 1.9.2 (Java), transition-based |
| Thuật toán | `nivreeager` (đổi bằng tham số `--algorithm` của `train_maltparser.py`) |
| Bộ dữ liệu | Vietnamese Dependency Treebank (`UDD-1.csv`) |
| Chia dữ liệu | 90% train, 10% test, `seed=42` |
| Mô hình đã huấn luyện | `models/vietnamese.mco` (**không xóa, không đưa vào `.gitignore`**) |
| Độ đo đánh giá | UAS (đúng head), LAS (đúng head và nhãn quan hệ) |

## 3. Yêu cầu môi trường

Python 3.11, Java JDK 8 trở lên, MaltParser 1.9.2 (có sẵn `tools/maltparser-1.9.2.zip`, `main.py` tự giải nén).

```
py -3.11 --version
java -version
py -3.11 -m pip install underthesea pandas numpy matplotlib scikit-learn sklearn-crfsuite joblib ipython
```

## 4. Cách chạy trong terminal

Terminal phải đang ở thư mục gốc project (thư mục chứa `src/`, `data/`, `models/`). Nếu chưa ở đó thì `cd` vào (tên thư mục tùy máy, ví dụ):

```
cd POS_dependency_triple
py -3.11 -m pip install underthesea pandas numpy matplotlib scikit-learn sklearn-crfsuite joblib ipython
py -3.11 src\main.py --skip-pos-train
```


Bước `train_maltparser.py` có thể bỏ qua nếu đã có `models/vietnamese.mco`.

## 5. Đầu vào / đầu ra của từng bước

| Bước | File chạy | Chức năng | Đầu vào | Đầu ra |
| :-- | :-- | :-- | :-- | :-- |
| 1 | `csv_to_conll.py` | Đọc treebank, loại câu lỗi, chia 90/10 | `UDD-1.csv` | `data/train.conll`, `data/test.conll` (nhãn chuẩn); `data/input_pos.csv` (câu test chỉ có từ + POS) |
| 2 | `train_maltparser.py` | Huấn luyện MaltParser | `data/train.conll`, `maltparser-1.9.2.jar` | `models/vietnamese.mco` |
| 3 | `prepare_malt_input.py` | Đổi CSV sang CoNLL-X cho parser (HEAD/DEPREL điền giá trị giả) | `data/input_pos.csv` | `data/malt_input.conll` |
| 4 | `run_maltparser.py` | MaltParser dự đoán head và deprel | `data/malt_input.conll`, `models/vietnamese.mco`, file jar | `data/dependency_predictions.conll` |
| 5 | `validate_dependency.py` | Kiểm tra cấu trúc cây, xuất JSONL bàn giao | `data/dependency_predictions.conll` | `data/dependency_results.jsonl`, `data/dependency_errors.csv` |
| 6 | `evaluate_dependency.py` | Tính UAS, LAS so với đáp án | `data/test.conll` (gold), `data/dependency_predictions.conll` | `data/evaluation.json` |
```
## Module Trích xuất Bộ ba (Rule Extractor)
- File module: `src/rule_extractor.py`
- Cách sử dụng:
    Đã thêm thư viện sau:
  from src.rule_extractor import TokenNode, extract_triples_openie
  
- Đưa danh sách TokenNode từ dependency parser vào để lấy kết quả
