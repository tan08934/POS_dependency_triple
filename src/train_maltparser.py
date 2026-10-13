"""
Train a MaltParser model from a gold CoNLL-X file (e.g. data/train.conll).
Output: models/<name>.mco  (used afterwards by run_maltparser.py)
"""
import argparse
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jar", required=True)
    ap.add_argument("--train", default="data/train.conll")
    ap.add_argument("--name", default="vietnamese", help="tên model (không có đuôi .mco)")
    ap.add_argument("--workdir", default="models")
    ap.add_argument("--algorithm", default="nivreeager", help="vd: nivreeager, stackproj, stacklazy")
    ap.add_argument("--java", default="java")
    ap.add_argument("--xmx", default="4g")
    args = ap.parse_args()

    jar, train = Path(args.jar), Path(args.train)
    if not jar.exists():
        raise FileNotFoundError(f"Không tìm thấy MaltParser JAR: {jar}")
    if not train.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu train: {train} (chạy csv_to_conll.py trước)")
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    cmd = [args.java, f"-Xmx{args.xmx}", "-jar", str(jar.resolve()),
           "-c", args.name, "-i", str(train.resolve()), "-m", "learn",
           "-a", args.algorithm, "-w", str(workdir.resolve())]
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, text=True, capture_output=True, cwd=workdir)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        raise SystemExit("Huấn luyện MaltParser thất bại.")
    print(f"Model: {(workdir / (args.name + '.mco')).resolve()}")


if __name__ == "__main__":
    main()
