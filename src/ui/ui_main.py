# File chạy chính: python ui_main.py
from .ui_widgets import (
    root, normalized_label, triple_table,
    analyze_button, export_button, delete_triple_button
)
from .visualization import draw_dependency_tree
from .nlp_service import analyze
from .data_service import (
    delete_selected_triple, export_json, edit_triple_cell
)

# Gắn lệnh cho các nút
analyze_button.configure(command=analyze)
export_button.configure(command=export_json)
delete_triple_button.configure(command=delete_selected_triple)

#KHỞI CHẠY
def initial_render():
    # Không tự hiển thị kết quả giả khi mở ứng dụng.
    normalized_label.configure(
        text="Nhập câu tiếng Việt rồi nhấn Phân tích."
    )

    draw_dependency_tree([])

root.title("POS - Dependency - Triple")

root.after(200, initial_render)

triple_table.bind(
    "<Double-1>",
    edit_triple_cell
)
root.mainloop()
