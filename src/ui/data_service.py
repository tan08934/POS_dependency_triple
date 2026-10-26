import json
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

from . import app_state
from .ui_widgets import (
    triple_table, subject_label, relation_label,
    object_label, note_label
)

# XÓA BỘ BA ĐANG ĐƯỢC CHỌN
def delete_selected_triple():

    selected_items = triple_table.selection()

    if not selected_items:
        messagebox.showwarning(
            "Thông báo",
            "Vui lòng chọn bộ ba cần xóa."
        )
        return

    confirm = messagebox.askyesno(
        "Xác nhận xóa",
        "Bạn có chắc muốn xóa bộ ba đã chọn không?"
    )

    if not confirm:
        return

    # Lấy chỉ số các dòng được chọn
    row_indices = sorted(
        (triple_table.index(item) for item in selected_items),
        reverse=True
    )

    # Xóa khỏi dữ liệu và bảng
    for index in row_indices:
        if 0 <= index < len(app_state.current_triples):
            del app_state.current_triples[index]

    for item in selected_items:
        triple_table.delete(item)

    # Cập nhật khu vực hiển thị bộ ba đầu tiên
    if app_state.current_triples:
        subject_label.configure(
            text=app_state.current_triples[0]["subject"]
        )
        relation_label.configure(
            text=app_state.current_triples[0]["relation"]
        )
        object_label.configure(
            text=app_state.current_triples[0]["object"]
        )
    else:
        subject_label.configure(text="—")
        relation_label.configure(text="—")
        object_label.configure(text="—")

    note_label.configure(
        text=f"Còn lại {len(app_state.current_triples)} bộ ba."
    )
    
def tolist_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {
            key: tolist_json(value)
            for key, value in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [tolist_json(item) for item in obj]

    return obj

def export_json():
    if app_state.current_result is None:
        messagebox.showwarning(
            "Thông báo",
            "Vui lòng phân tích câu trước khi xuất JSON."
        )
        return

    file_path = filedialog.asksaveasfilename(
        title="Xuất kết quả phân tích",
        defaultextension=".json",
        filetypes=[("JSON files", "*.json")]
    )

    if not file_path:
        return

    try:
        # Sao chép kết quả phân tích ban đầu
        export_data = dict(app_state.current_result)

        # Thay bộ ba ban đầu bằng bộ ba đã chỉnh sửa trên giao diện
        export_data["triples"] = [
            {
                "subject": triple["subject"],
                "relation": triple["relation"],
                "object": triple["object"]
            }
            for triple in app_state.current_triples
        ]

        # Chuyển các kiểu dữ liệu NumPy thành kiểu Python
        export_data = tolist_json(export_data)

        # Ghi dữ liệu ra file JSON
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                export_data,
                f,
                ensure_ascii=False,
                indent=4
            )

        messagebox.showinfo(
            "Thành công",
            f"Đã lưu kết quả tại:\n{file_path}"
        )

    except (OSError, TypeError, ValueError, KeyError) as exc:
        messagebox.showerror(
            "Lỗi xuất JSON",
            str(exc)
        )

def edit_triple_cell(event):
    """Cho phép chỉnh sửa trực tiếp từng ô trong bảng bộ ba."""


    # Xác định ô được nhấp đúp
    region = triple_table.identify("region", event.x, event.y)

    if region != "cell":
        return

    row_id = triple_table.identify_row(event.y)
    column_id = triple_table.identify_column(event.x)

    if not row_id or not column_id:
        return

    # Xác định chỉ số dòng và cột
    row_index = triple_table.index(row_id)
    column_index = int(column_id.replace("#", "")) - 1

    if row_index >= len(app_state.current_triples):
        return

    if column_index not in (0, 1, 2):
        return

    # Lấy vị trí và kích thước ô
    bbox = triple_table.bbox(row_id, column_id)

    if not bbox:
        return

    x, y, width, height = bbox

    # Lấy nội dung hiện tại
    values = triple_table.item(row_id, "values")
    old_value = values[column_index]

    # Tạo ô nhập ngay trên ô được chọn
    editor = tk.Entry(triple_table)

    editor.place(
        x=x,
        y=y,
        width=width,
        height=height
    )

    editor.insert(0, old_value)
    editor.select_range(0, tk.END)
    editor.focus_set()

    def save_edit(event=None):
        new_value = editor.get().strip()

        if not new_value:
            messagebox.showwarning(
                "Thông báo",
                "Nội dung bộ ba không được để trống."
            )
            editor.focus_set()
            return

        # Cập nhật dữ liệu bộ ba
        keys = ["subject", "relation", "object"]

        app_state.current_triples[row_index][keys[column_index]] = new_value

        # Cập nhật bảng
        new_values = list(
            triple_table.item(row_id, "values")
        )
        new_values[column_index] = new_value

        triple_table.item(
            row_id,
            values=new_values
        )

        # Cập nhật khu vực hiển thị bộ ba đầu tiên
        if row_index == 0:
            subject_label.configure(
                text=app_state.current_triples[0]["subject"]
            )
            relation_label.configure(
                text=app_state.current_triples[0]["relation"]
            )
            object_label.configure(
                text=app_state.current_triples[0]["object"]
            )

        editor.destroy()

    def cancel_edit(event=None):
        editor.destroy()

    editor.bind("<Return>", save_edit)
    editor.bind("<Escape>", cancel_edit)
