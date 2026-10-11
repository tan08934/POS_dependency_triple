POS_dependency_triple:
Đồ án xây dựng quy trình xử lý ngôn ngữ tự nhiên cho văn bản tiếng Việt, bao gồm chuẩn hóa văn bản, tách từ, gán nhãn từ loại (POS tagging), phân tích phụ thuộc (dependency parsing) và trích xuất bộ ba (subject, relation, object).
Môi trường thực thi yêu cầu Python 3.11.9 và Java JDK 21. Ngoài ra, cần cài đặt các thư viện Python và chuẩn bị đầy đủ các tệp mô hình phục vụ quá trình xử lý ngôn ngữ.

1. Cài đặt Python 3.11.9
Tải Python 3.11.9 từ trang phát hành chính thức: https://www.python.org/downloads/release/python-3119/
Trên Windows, thực hiện cài đặt theo hướng dẫn của trình cài đặt. 
Trên macOS, tải bộ cài tương ứng với hệ điều hành và kiến trúc máy tính. 
Sau khi cài đặt, mở PowerShell trên Windows hoặc Terminal trên macOS và kiểm tra phiên bản:
python --version
Trên Windows, nếu lệnh trên không hoạt động, sử dụng:
py -3.11 --version
Phiên bản hiển thị cần là Python 3.11.9.

2. Cài đặt Java JDK 21
VnCoreNLP sử dụng Java để khởi chạy mô hình tách từ tiếng Việt. Cần cài đặt JDK 21 trước khi thực thi chức năng này.
Trên Windows, có thể cài Microsoft OpenJDK 21 bằng lệnh:
winget install --id Microsoft.OpenJDK.21 -e
Trên macOS đã cài Homebrew, sử dụng:
brew install --cask microsoft-openjdk@21
Kiểm tra quá trình cài đặt bằng các lệnh:
java -version
javac -version
Nếu hệ thống không nhận diện được Java, cần cấu hình biến môi trường JAVA_HOME trỏ đến thư mục JDK đã cài đặt và bổ sung thư mục bin của JDK vào PATH.

3. Cài đặt thư viện Python
Mở PowerShell hoặc Terminal tại thư mục gốc của dự án và thực thi lệnh:
python -m pip install --upgrade pip
python -m pip install py_vncorenlp pandas numpy
Nếu hệ thống Windows sử dụng py -3.11 để gọi Python, thay python bằng py -3.11 trong các lệnh trên.
Các thư viện được liệt kê phục vụ những chức năng xử lý dữ liệu và tách từ. 

4. Chuẩn bị mô hình tách từ VnCoreNLP
Dự án sử dụng thư viện py_vncorenlp để gọi VnCoreNLP phục vụ việc tách từ tiếng Việt. Chức năng này cần file JAR của VnCoreNLP và hai tệp dữ liệu của mô hình tách từ.
Các tệp của mô hinh cần nằm tại những vị trí sau:
models/vncorenlp/VnCoreNLP-1.2.jar
models/vncorenlp/models/wordsegmenter/vi-vocab
models/vncorenlp/models/wordsegmenter/wordsegmenter.rdr
Repository hiện đã chứa đầy đủ các tệp trên, không cần tải lại. Nếu thiếu hai tệp dữ liệu vi-vocab hoặc wordsegmenter.rdr, chạy lệnh sau tại thư mục gốc dự án:
python src/download_model.py
Lệnh này tải hai tệp dữ liệu vào thư mục models/vncorenlp/models/wordsegmenter/. Lệnh không tải file JAR và không cài đặt Java.
Nếu thiếu file VnCoreNLP-1.2.jar, cần tải file phù hợp từ nguồn chính thức của VnCoreNLP tại https://github.com/vncorenlp/VnCoreNLP và đặt file vào thư mục models/vncorenlp/ với tên VnCoreNLP-1.2.jar.
Đặt ba tệp mô hình vào đúng các thư mục đã liệt kê ở trên trước khi chạy chương trình.

5. Tải và chạy dự án
Tải mã nguồn bằng Git:
git clone https://github.com/tan08934/POS_dependency_triple.git
Di chuyển vào thư mục dự án:
cd POS_dependency_triple
Sau khi hoàn tất các bước cài đặt môi trường và chuẩn bị mô hình, khởi chạy chương trình bằng lệnh:
python src/main.py

6. Lưu ý về môi trường thực thi
Trước khi thực hiện các lệnh chạy chương trình, mở Terminal hoặc PowerShell tại thư mục POS_dependency_triple.
Việc một tệp tồn tại trên máy cục bộ không đồng nghĩa với việc tệp đó đã được lưu trong repository. Chỉ những tệp đã được đưa vào Git và đẩy lên GitHub mới có sẵn khi tải mã nguồn từ repository. Do đó, cần xác nhận các tệp mô hình đã được đưa lên repository trước khi giả định rằng bước chuẩn bị mô hình có thể được bỏ qua.
Nếu chương trình không khởi chạy, cần kiểm tra lần lượt phiên bản Python, cấu hình Java, các thư viện đã cài đặt, đường dẫn mô hình và thông báo lỗi do chương trình cung cấp.