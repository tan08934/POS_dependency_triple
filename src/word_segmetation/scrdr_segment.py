from pathlib import Path
import py_vncorenlp

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "vncorenlp"

print("Thư mục gốc:", ROOT)
print("Thư mục mô hình:", MODEL_DIR)
print("Thư mục models tồn tại:", (MODEL_DIR / "models").is_dir())
print("File JAR tồn tại:", (MODEL_DIR / "VnCoreNLP-1.2.jar").is_file())

#khởi tạo đối tượng của lớp vncorenlp
segmenter = py_vncorenlp.VnCoreNLP(
    annotators=["wseg"], #chỉ định dùng tách từ
    save_dir=str(MODEL_DIR.resolve())#cung cấp đường dẫn đến thư mục chứa file đã tải về
)

#nhận vào từng câu để tách
def segment_text(text: str):
    result =  segmenter.word_segment(text) 
    return result[0].split() if result else []#list không có split
# trả về danh sách các từ đã được tách 