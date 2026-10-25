import tkinter as tk
from tkinter import ttk

# Màu sắc, font chữ và cấu hình giao diện
#BẢNG MÀU SỬ DỤNG
BLUE = "#2a3b90"
NAVY = "#2c345c"
DEEP_NAVY = "#0c0d49"
FOREST = "#183a23"
DARK = "#0f0f18"

BG = "#F5F6FA"
WHITE = "#FFFFFF"
LIGHT_BLUE = "#EEF0FA"
LIGHT_NAVY = "#E9EAF2"
LIGHT_GREEN = "#EDF3EE"
LIGHT_GRAY = "#F1F2F6"
BORDER = "#D9DCE8"

FONT = "Arial"

def make_panel(parent, title, accent, header_bg):
    panel = tk.Frame(
        parent,
        bg=WHITE,
        highlightbackground=BORDER, #màu đường viền
        highlightthickness=1 #độ dày đường viền
    )

    header = tk.Frame(
        panel,
        bg=header_bg,
        padx=12,
        pady=10
    )
    header.pack(fill="x") #hiển thị header lên đầu panel và tự động co giãn hết chiều ngang của panel

    tk.Label(
        header,
        text=title,
        font=(FONT, 10, "bold"),
        bg=header_bg,
        fg=accent,
        anchor="w"
    ).pack(fill="x")
    return panel

def make_table(parent, columns, headings, widths): #khởi tạo bảng dữ liệu
    table = ttk.Treeview(
        parent,
        columns=columns,
        show="headings",
        height=5
    )
    for col, heading, width in zip(columns, headings, widths):
        table.heading(col, text=heading)
        table.column(
            col,
            width=width,
            minwidth=60,
            anchor="w",
            stretch=True
        )
    return table

def replace_rows(table, rows): #xóa và làm mới dữ liệu bảng
    table.delete(*table.get_children())

    for row in rows:
        table.insert("", "end", values=row)