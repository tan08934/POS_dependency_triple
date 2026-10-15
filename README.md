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