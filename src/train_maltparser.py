"""
Train a MaltParser model from a gold CoNLL-X file (e.g. data/train.conll).
Output: models/<name>.mco  (used afterwards by run_maltparser.py)
"""
import argparse
import subprocess
from pathlib import Path

#đầu vào là train.conll
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jar", required=True)
    ap.add_argument("--train", default="data/train.conll")
    #help là để hướng dẫn
    ap.add_argument("--name", default="vietnamese", help="tên model (không có đuôi .mco)")
    ap.add_argument("--workdir", default="models")
    ap.add_argument("--algorithm", default="nivreeager", help="vd: nivreeager, stackproj, stacklazy")
    #các loại thuật toán tạo cây dependency của maltparser
    ap.add_argument("--java", default="java")
    ap.add_argument("--xmx", default="4g")
    #giới hạn bộ nhớ heap là 4 GB
    args = ap.parse_args()

    #kiểm tra các file có tồn tại không
    jar, train = Path(args.jar), Path(args.train)
    if not jar.exists():
        raise FileNotFoundError(f"Không tìm thấy MaltParser JAR: {jar}")
    if not train.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu train: {train} (chạy csv_to_conll.py trước)")
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    #-jar là chạy file MaltParser trên jar
    #-c là đặc tên mô hình
    #-i là chỉ định file dữ liệu đầu vào
    #-m learn là chọn chế độ huấn luyện
    #-a chọn thuật toán phân tích
    #-w là chỉ định thư mục làm việc
    cmd = [args.java, f"-Xmx{args.xmx}", "-jar", str(jar.resolve()),
           "-c", args.name, "-i", str(train.resolve()), "-m", "learn",
           "-a", args.algorithm, "-w", str(workdir.resolve()), "-grl", "root"]
    print("Running:", " ".join(cmd))
    #chạy chương trình bên ngoài chạy MaltParser
    res = subprocess.run(cmd, text=True, capture_output=True, cwd=workdir)
    print(res.stdout)
    #kiểm tra mã chương trình có khác 0 không, nếu khác 0 thì là thất bại
    if res.returncode != 0:
        print(res.stderr)
        raise SystemExit("Huấn luyện MaltParser thất bại.")
    print(f"Model: {(workdir / (args.name + '.mco')).resolve()}")
#xử lý chạy maltparser bằng java và trả về vietnamese.mco là mô hình maltparser đã được huấn luyện dùng để dự đoán

if __name__ == "__main__":
    main()
