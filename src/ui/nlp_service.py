
import re
import threading
import traceback
from tkinter import messagebox

from .ui_widgets import (
    root,
    input_entry,
    analyze_button,
    normalized_label,
    note_label,
)
from .visualization import display_results


# Trạng thái xử lý phân tích
analysis_running = False


def show_analysis_error(error_message):
    """Hiển thị lỗi và khôi phục trạng thái nút phân tích."""
    global analysis_running

    analysis_running = False

    analyze_button.configure(
        state="normal",
        text="➤  Phân tích"
    )

    note_label.configure(
        text="Phân tích thất bại. Vui lòng kiểm tra thông tin lỗi."
    )

    messagebox.showerror(
        "Lỗi phân tích",
        error_message
    )


def run_pipeline(sentence):
    """Chạy pipeline NLP trong luồng nền."""

    try:
        # Gọi pipeline chính của project.
        from main import main as pipeline_main

        result = pipeline_main(sentence)

        if not isinstance(result, dict):
            raise TypeError(
                "main() phải trả về dict chứa kết quả phân tích."
            )

        # Chuẩn hóa câu để hiển thị.
        try:
            from preprocessing import preprocess_data
            normalized_text = preprocess_data(sentence)

        except ImportError:
            normalized_text = re.sub(r"\s+", " ", sentence)
            normalized_text = re.sub(
                r"\s+([,.!?;:])",
                r"\1",
                normalized_text
            ).strip()

        # Chuyển kết quả về luồng giao diện Tkinter.
        root.after(
            0,
            lambda r=result, n=normalized_text: finish_analysis(r, n)
        )

    except Exception:
        error_detail = traceback.format_exc()

        print("\n===== LỖI PIPELINE NLP =====")
        print(error_detail)

        root.after(
            0,
            lambda msg=error_detail: show_analysis_error(msg)
        )


def finish_analysis(result, normalized_text):
    """Hiển thị kết quả phân tích trên giao diện."""
    global analysis_running

    try:
        display_results(result, normalized_text)

    except Exception:
        error_detail = traceback.format_exc()

        print("\n===== LỖI HIỂN THỊ KẾT QUẢ =====")
        print(error_detail)

        show_analysis_error(error_detail)
        return

    analysis_running = False

    analyze_button.configure(
        state="normal",
        text="➤  Phân tích"
    )


def analyze():
    """Xử lý sự kiện nhấn nút Phân tích."""
    global analysis_running

    if analysis_running:
        return

    sentence = input_entry.get().strip()

    if not sentence:
        messagebox.showwarning(
            "Thông báo",
            "Vui lòng nhập câu tiếng Việt."
        )
        return

    # Chuẩn hóa khoảng trắng và dấu câu.
    sentence = re.sub(r"\s+", " ", sentence)
    sentence = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        sentence
    ).strip()

    analysis_running = True

    analyze_button.configure(
        state="disabled",
        text="Đang phân tích..."
    )

    normalized_label.configure(
        text="Đang xử lý văn bản..."
    )

    note_label.configure(
        text=(
            "Pipeline đang thực hiện POS Tagging, "
            "Dependency Parsing và Triple Extraction..."
        )
    )

    threading.Thread(
        target=run_pipeline,
        args=(sentence,),
        daemon=True
    ).start()
