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


# Người 2 — Vietnamese Dependency Parsing with MaltParser

## 1. Mục tiêu

Xây dựng bước Dependency Parsing (phân tích cú pháp phụ thuộc) tiếng Việt trong pipeline xử lý ngôn ngữ tự nhiên của nhóm.

* **Đầu vào:** Dữ liệu tiếng Việt đã được tách từ (*Word Segmentation*) và gán nhãn từ loại (*POS Tagging*) từ file UDD-1.csv.
* **Đầu ra:** Cây cú pháp phụ thuộc, trong đó mỗi từ được xác định từ chủ (*head*) và nhãn quan hệ phụ thuộc (*dependency label*).
* **Định dạng dữ liệu:** CoNLL phục vụ xử lý và đánh giá; JSONL phục vụ bàn giao kết quả cho Người 3.

## 2. Phương pháp

| Thành phần              | Mô tả                                        |
| :---------------------- | :------------------------------------------- |
| **Parser**              | MaltParser 1.9.2                             |
| **Ngôn ngữ triển khai** | Java                                         |
| **Thuật toán**          | `nivreeager`                                 |
| **Bộ dữ liệu**          | Vietnamese Dependency Treebank (`UDD-1.csv`) |
| **Mô hình huấn luyện**  | `vietnamese.mco`                             |

MaltParser sử dụng phương pháp *transition-based dependency parsing*, phân tích cú pháp dựa trên các thao tác chuyển trạng thái. Thuật toán được sử dụng có thể cấu hình thông qua tham số `--algorithm` trong script huấn luyện.

## 3. Yêu cầu môi trường

Trước khi chạy chương trình, hãy đảm bảo các công cụ sau đã được cài đặt:

| Công cụ    | Phiên bản yêu cầu | Mục đích                                          |
| :--------- | :---------------- | :------------------------------------------------ |
| Python     | 3.8 trở lên       | Chạy các script xử lý dữ liệu và điều phối parser |
| Java JDK   | 8 trở lên         | Chạy MaltParser                                   |
| MaltParser | 1.9.2             | Huấn luyện và dự đoán cú pháp phụ thuộc           |

Ngoài ra, cần chuẩn bị bộ dữ liệu `UDD-1.csv` để huấn luyện mô hình.

### 3.1. Kiểm tra Python

Mở terminal tại thư mục dự án và chạy:

```bash
python --version
```

### 3.2. Kiểm tra Java

```bash
java -version
```

Nếu chưa cài Java trên Windows, có thể cài **Eclipse Temurin JDK 21** bằng Windows Package Manager:

```powershell
winget install EclipseAdoptium.Temurin.21.JDK
```

Sau khi cài đặt, hãy đóng và mở lại terminal hoặc VS Code để cập nhật môi trường. Tiếp theo, chạy lại `java -version` để xác nhận Java đã được cài đặt thành công.
### 3.3. Cài Maltparser
```bash
curl -L -o tools/maltparser-1.9.2.zip http://maltparser.org/dist/maltparser-1.9.2.zip
tar -xf tools/maltparser-1.9.2.zip -C tools
```

## 4. Chạy dự đoán, kiểm tra và đánh giá

### Bước 1. Tách dữ liệu

```bash
python src/csv_to_conll.py --input src/UDD-1.csv --outdir data
```

### Bước 2. Chạy huấn luyện

```bash
python src/train_maltparser.py --jar tools/maltparser-1.9.2/maltparser-1.9.2.jar --train data/train.conll
```

### Bước 3. Kiểm tra kết quả dự đoán

```bash
python src/prepare_malt_input.py --input data/input_pos.csv --output data/malt_input.conll
python src/run_maltparser.py --jar tools/maltparser-1.9.2/maltparser-1.9.2.jar --model models/vietnamese.mco --input data/malt_input.conll --output data/dependency_predictions.conll
python src/validate_dependency.py --input data/dependency_predictions.conll
python src/evaluate_dependency.py --gold data/test.conll --pred data/dependency_predictions.conll
```
## Module Trích xuất Bộ ba (Rule Extractor)
- File module: `src/rule_extractor.py`
- Cách sử dụng:
    Đã thêm thư viện sau:
  from src.rule_extractor import TokenNode, extract_triples_openie
  
- Đưa danh sách TokenNode từ dependency parser vào để lấy kết quả
