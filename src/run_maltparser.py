#nguoi 2
import argparse
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jar", required=True, help="Đường dẫn maltparser-1.9.2.jar")
    ap.add_argument("--model", required=True, help="Đường dẫn model .mco đã huấn luyện")
    ap.add_argument("--input", default="data/malt_input.conll")
    ap.add_argument("--output", default="data/dependency_predictions.conll")
    ap.add_argument("--java", default="java")
    args = ap.parse_args()

    jar = Path(args.jar)
    model = Path(args.model)
    source = Path(args.input)
    target = Path(args.output)
    if not jar.exists():
        raise FileNotFoundError(f"Không tìm thấy MaltParser JAR: {jar}")
    if not model.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model: {model}. Cần model MaltParser đã huấn luyện cho dữ liệu tiếng Việt."
        )
    if not source.exists():
        raise FileNotFoundError(f"Không tìm thấy input: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.java, "-jar", str(jar),
        "-c", str(model),
        "-i", str(source),
        "-o", str(target),
        "-m", "parse",
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit(
            "MaltParser chạy thất bại. Kiểm tra model, phiên bản Java và định dạng CoNLL."
        )
    print(f"Đã lưu dependency dự đoán: {target.resolve()}")


if __name__ == "__main__":
    main()
