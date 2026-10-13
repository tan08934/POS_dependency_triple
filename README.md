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

Người 2 — Dependency Parsing tiếng Việt bằng MaltParser
1. Mục tiêu

Xây dựng bước phân tích cú pháp phụ thuộc (dependency parsing) trong pipeline của nhóm:

Đầu vào: câu tiếng Việt đã tách từ và gán POS (kết quả của Người 1).
Đầu ra: với mỗi từ, một quan hệ phụ thuộc gồm từ chủ (head) và nhãn quan hệ (dependency label). Kết quả được xuất ra JSONL để bàn giao cho Người 3.
2. Dữ liệu
Mục	Giá trị
Treebank	data/UDD-1.csv (có sẵn token, POS, head, nhãn dependency kiểu Universal Dependencies)
Số câu	20.000 (không câu nào bị loại do lỗi định dạng)
Số token	453.551
Chia dữ liệu	Ngẫu nhiên, seed = 42: 18.000 câu train / 2.000 câu test (45.375 token test)
Nhãn	Nhãn UD, gồm cả nhãn phụ như nsubj:pass, obl:comp, compound:svc

Tập test được tách ra trước khi huấn luyện. File input_pos.csv chỉ chứa từ và POS của tập test, không có head/nhãn. Vì vậy kết quả dự đoán là kết quả thật của parser, không phải đáp án chép lại.

3. Phương pháp
Parser: MaltParser 1.9.2 (transition-based, Java).
Thuật toán: nivreeager (mặc định của script, đổi được bằng --algorithm).
Định dạng: CoNLL-X 10 cột (ID, FORM, LEMMA, CPOSTAG, POSTAG, FEATS, HEAD, DEPREL, PHEAD, PDEPREL). Cột POS dùng trong huấn luyện: UPOS ở cột 4, XPOS ở cột 5.
Quy trình: chuyển CSV sang CoNLL → huấn luyện ra models/vietnamese.mco → dự đoán trên tập test → kiểm tra cấu trúc cây → tính UAS/LAS.
4. Hướng dẫn huấn luyện ra vietnamese.mco
Yêu cầu
Python 3.8 trở lên. Chỉ dùng thư viện chuẩn, không cần pip install. Kiểm tra: python --version.
Java 8 trở lên. Đã chạy với Temurin 21. Kiểm tra: java -version: Windows: winget install EclipseAdoptium.Temurin.21.JDK, sau đó đóng và mở lại VS Code/terminal.
Bước 1. Giải nén MaltParser

Windows (CMD/PowerShell) và macOS/Linux đều có tar:

tar -xf tools/maltparser-1.9.2.zip -C tools
Bước 2. Tách dữ liệu thành train/test
python src/csv_to_conll.py --input data/UDD-1.csv --outdir data
Bước 3. Huấn luyện MaltParser
python src/train_maltparser.py --jar tools/maltparser-1.9.2/maltparser-1.9.2.jar --train data/train.conll
5. Chạy dự đoán, kiểm tra và đánh giá
python src/prepare_malt_input.py --input data/input_pos.csv --output data/malt_input.conll
python src/run_maltparser.py --jar tools/maltparser-1.9.2/maltparser-1.9.2.jar --model models/vietnamese.mco --input data/malt_input.conll --output data/dependency_predictions.conll
python src/validate_dependency.py --input data/dependency_predictions.conll
python src/evaluate_dependency.py --gold data/test.conll --pred data/dependency_predictions.conll
