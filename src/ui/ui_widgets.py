import tkinter as tk
from tkinter import ttk

from .ui_components import (
    BLUE, NAVY, DEEP_NAVY, FOREST, DARK,
    BG, WHITE, LIGHT_BLUE, LIGHT_NAVY,
    LIGHT_GREEN, LIGHT_GRAY, BORDER, FONT,
    make_panel, make_table
)

#GIAO DIỆN CHÍNH
root = tk.Tk()
root.title()
root.geometry("1450x950")
root.minsize(1050, 750)
root.configure(bg=BG)

# Cấu hình bảng ttk, cần có để tạo bảng, cây
style = ttk.Style()
style.theme_use("clam") #cho phép đổi màu

style.configure(
    "Treeview", #bảng
    background=WHITE,
    foreground=DARK,
    fieldbackground=WHITE,
    rowheight=30,
    font=(FONT, 9),
    borderwidth=0
)

style.configure(
    "Treeview.Heading", #bảng
    background=LIGHT_NAVY,
    foreground=DEEP_NAVY,
    font=(FONT, 9, "bold"),
    padding=(8, 9),
    relief="flat"
)

style.map(
    "Treeview",
    background=[("selected", LIGHT_BLUE)], #nút bấm
    foreground=[("selected", DEEP_NAVY)]
)

style.configure(
    "Vertical.TScrollbar", #thanh cuộn
    background=LIGHT_NAVY,
    troughcolor=BG,
    arrowcolor=NAVY,
    bordercolor=BG
)

#HEADING
header = tk.Frame(
    root,
    bg=WHITE,
    height=76,
    highlightbackground=BORDER,
    highlightthickness=1
)
header.pack(fill="x")
header.pack_propagate(False)

brand = tk.Frame(header, bg=WHITE) #dùng để gom nhóm
brand.pack(side="left", padx=25, pady=10)

brand_text = tk.Frame(brand, bg=WHITE)
brand_text.pack(side="left")

tk.Label(
    brand_text,
    text="POS - Dependency - Triple",
    font=(FONT, 14, "bold"),
    bg=WHITE,
    fg=DEEP_NAVY
).pack(anchor="w")

tk.Label(
    brand_text,
    text="Hệ thống xử lý ngôn ngữ tự nhiên tiếng Việt",
    font=(FONT, 8),
    bg=WHITE,
    fg=NAVY
).pack(anchor="w", pady=(3, 0))

tk.Label(
    header,
    text="NLP PROJECT",
    font=(FONT, 10, "bold"),
    bg=WHITE,
    fg=BLUE
).pack(side="right", padx=25)

#THANH CUỘN
outer = tk.Frame(root, bg=BG) #tạo một khung chứa bên ngoài bọc toàn bộ khu vực cần cuộn.
outer.pack(fill="both", expand=True)

canvas = tk.Canvas( #màn hình ảo bắt buộc phải có khi tạo thanh cuộn
    outer,
    bg=BG,
    highlightthickness=0
)
scrollbar = ttk.Scrollbar( #tạo thanh cuộn
    outer,
    orient="vertical", #cuộn theo chiều dọc
    command=canvas.yview, #cuộn đến đâu canvas chayh theo tới đó
    style="Vertical.TScrollbar" #áp dụng màu đã thiết lập sẵn
)
canvas.configure(yscrollcommand=scrollbar.set)

scrollbar.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)

content = tk.Frame(#khung chứa nội dung thật
    canvas,
    bg=BG,
    padx=22,
    pady=18
)
canvas_window = canvas.create_window(
    (0, 0),
    window=content,
    anchor="nw"
)
def update_scroll_region(event=None): #cập nhật vùng có thể cuộn
    canvas.configure(scrollregion=canvas.bbox("all"))
def resize_content(event): #định nghĩa hàm xử lý khi kích thước khung nhìn thay đổi
    canvas.itemconfigure(canvas_window, width=event.width)
def mousewheel(event): #định nghĩa hàm bắt sự kiện lăn con chuột
    canvas.yview_scroll(int(-event.delta / 120), "units") #-event.delta lấy thông số mức độ lăn

content.bind("<Configure>", update_scroll_region) #thay đổi kích thước content
canvas.bind("<Configure>", resize_content) #thay đổi kích thước canvas
canvas.bind_all("<MouseWheel>", mousewheel) #áp dụng con chuột ở bất kỳ đâu thay vì phải trỏ đúng vào thanh cuộn

#TIÊU ĐỀ VÀ Ô NHẬP VĂN BẢN
tk.Label(
    content,
    text="Phân tích văn bản tiếng Việt",
    font=(FONT, 22, "bold"),
    bg=BG,
    fg=DEEP_NAVY
).pack(anchor="w")

tk.Label(
    content,
    text=("Nhập một câu tiếng Việt để xem quá trình xử lý ngôn ngữ tự nhiên"),
    font=(FONT, 10),
    bg=BG,
    fg=NAVY
).pack(anchor="w", pady=(5, 14))

input_card = tk.Frame( #tạo frame chứa dữ liệu bên trong
    content,
    bg=WHITE,
    padx=14,
    pady=12,
    highlightbackground=BORDER,
    highlightthickness=1
)
input_card.pack(fill="x", pady=(0, 15))

input_row = tk.Frame(input_card, bg=WHITE) #khung con làm hàng ngang chứa ô nhập văn bản
input_row.pack(fill="x")

input_entry = tk.Entry( #nhập liệu
    input_row,
    font=(FONT, 12),
    bg=WHITE,
    fg=DARK,
    insertbackground=BLUE, #đổi màu con trỏ chuột nháy bên trong ô nhập thành màu xanh dương
    relief="flat" #xóa bỏ kiểu viền nổi 3D mặc định của Entry
)
input_entry.pack( #để co dãn như mấy lệnh ở trên
    side="left",
    fill="x",
    expand=True,
    ipady=13,
    padx=5
)

input_entry.insert(
    0,
    "Nguyễn An mua điện thoại tại Shopee."
)

#THANH QUY TRÌNH CÁC BƯỚC
pipeline = tk.Frame(#khung chứa bọc các bước
    content,
    bg=WHITE,
    padx=9,
    pady=10,
    highlightbackground=BORDER,
    highlightthickness=1
)
pipeline.pack(fill="x", pady=(0, 17))

steps = [
    ("1", "Chuẩn hóa văn bản", "Loại ký tự thừa", BLUE, LIGHT_BLUE),
    ("2", "Tách từ", "Tách các từ trong câu", NAVY, LIGHT_NAVY),
    ("3", "POS Tagging", "Gán nhãn từ loại", DEEP_NAVY, LIGHT_BLUE),
    ("4", "Dependency Parsing", "Phân tích quan hệ phụ thuộc", FOREST, LIGHT_GREEN),
    ("5", "Triple Extraction", "Trích xuất bộ ba S - R - O", BLUE, LIGHT_NAVY),
]
for i, (number, title, subtitle, accent, tint) in enumerate(steps): #vòng lặp để vẽ các bước
    step = tk.Frame(
        pipeline,
        bg=tint,
        padx=9,
        pady=9,
        highlightbackground=BORDER,
        highlightthickness=1
    )
    step.pack(side="left", fill="x", expand=True)
    top = tk.Frame(step, bg=tint) #khung hàng ngang nhỏ bên trong để chứa số thứ tự và tên bước nằm cùng một hàng
    top.pack(fill="x")

    tk.Label( #nhãn hiển thị số thứ tự
        top,
        text=number,
        font=(FONT, 10, "bold"),
        bg=accent,
        fg=WHITE,
        width=2,
        pady=3
    ).pack(side="left", padx=(0, 7))

    tk.Label( #nhãn hiển thị tên tiêu đề
        top,
        text=title,
        font=(FONT, 9, "bold"),
        bg=tint,
        fg=DEEP_NAVY,
        anchor="w",
        wraplength=150
    ).pack(side="left", fill="x", expand=True)

    tk.Label(# nhãn hiển thị dòng mô tả chi tiết phía dưới
        step,
        text=subtitle,
        font=(FONT, 8),
        bg=tint,
        fg=NAVY,
        anchor="w",
        wraplength=170
    ).pack(fill="x", pady=(5, 0))
    if i < len(steps) - 1: #kiểm tra xem pải bước cuối k
        tk.Label( #vẽ mũi tên
            pipeline,
            text="›",
            font=(FONT, 19, "bold"),
            bg=WHITE,
            fg=BLUE
        ).pack(side="left", padx=4)

#KẾT QUẢ PHÂN TÍCH
tk.Label(
    content,
    text="Kết quả phân tích",
    font=(FONT, 16, "bold"),
    bg=BG,
    fg=DEEP_NAVY
).pack(anchor="w", pady=(0, 10))

#HÀNG CHUẨN HÓA, TÁCH TỪ, POS
top_row = tk.Frame(content, bg=BG) #tạo một khung hàng ngang
top_row.pack(fill="x", pady=(0, 13))
top_row.columnconfigure(0, weight=10, uniform="top") #chia hàng top_row thành 3 cột
top_row.columnconfigure(1, weight=10, uniform="top") #uniform="top" ép buộc 3 cột này có kích thước đồng đều tuyệt đối theo chiều ngang
top_row.columnconfigure(2, weight=11, uniform="top")

normalized_panel = make_panel( #gọi hàm helper đã viết ở trên
    top_row,
    "1. Văn bản sau khi chuẩn hóa",
    BLUE,
    LIGHT_BLUE
)
normalized_panel.grid(
    row=0, column=0, sticky="nsew", padx=(0, 6) #sticky="nsew" ép panel co giãn tràn đầy bốn hướng
)
normalized_label = tk.Label( #nhãn bên trong panel chuẩn hóa dùng để hiển thị đoạn văn bản sau khi đã lọc bỏ ký tự thừa
    normalized_panel,
    text="",
    font=(FONT, 10),
    bg=LIGHT_BLUE,
    fg=DEEP_NAVY,
    anchor="nw",
    justify="left",
    padx=12,
    pady=12,
    wraplength=330
)
normalized_label.pack(fill="x", padx=10, pady=10)

token_panel = make_panel(
    top_row,
    "2. Tách từ",
    NAVY,
    LIGHT_NAVY
)
token_panel.grid(
    row=0, column=1, sticky="nsew", padx=6
)
token_container = tk.Frame(token_panel, bg=WHITE) #tạo một khung trắng bên trong panel tách từ để chứa danh sách các từ sau khi tách
token_container.pack(fill="both", expand=True, padx=9, pady=10)

pos_panel = make_panel(
    top_row,
    "3. POS Tagging",
    DEEP_NAVY,
    LIGHT_BLUE
)
pos_panel.grid(
    row=0, column=2, sticky="nsew", padx=(6, 0)
)

pos_table = make_table(
    pos_panel,
    ("word", "pos", "meaning"),
    ("Từ", "Nhãn từ loại", "Diễn giải"),
    (105, 85, 100)
)
pos_table.pack(fill="both", expand=True, padx=8, pady=8)

#HÀNG DƯỚI DEPEND VÀ TRIPLES
bottom_row = tk.Frame(content, bg=BG)
bottom_row.pack(fill="both", expand=True)
bottom_row.columnconfigure(0, weight=12, uniform="bottom")
bottom_row.columnconfigure(1, weight=13, uniform="bottom")

# 10.1 Dependency Parsing
dependency_panel = make_panel(
    bottom_row,
    "4. Dependency Parsing",
    FOREST,
    LIGHT_GREEN
)
dependency_panel.grid(
    row=0, column=0, sticky="nsew", padx=(0, 6)
)

tk.Label(
    dependency_panel,
    text="Cây phụ thuộc (Dependency Tree)",
    font=(FONT, 9, "bold"),
    bg=WHITE,
    fg=DEEP_NAVY
).pack(anchor="w", padx=12, pady=(10, 3))

tree_canvas = tk.Canvas(
    dependency_panel,
    height=165,
    bg=WHITE,
    highlightthickness=0
)
tree_canvas.pack(fill="x", padx=8, pady=(0, 5))

tk.Label(
    dependency_panel,
    text="Quan hệ phụ thuộc",
    font=(FONT, 9, "bold"),
    bg=WHITE,
    fg=DEEP_NAVY
).pack(anchor="w", padx=12, pady=(5, 5))

dependency_table = make_table(
    dependency_panel,
    ("dependent", "relation", "head"),
    ("Từ phụ thuộc", "Quan hệ", "Từ gốc"),
    (145, 95, 130)
)
dependency_table.pack(
    fill="both", expand=True, padx=8, pady=(0, 10)
)

# 10.2 Triple Extraction
triple_panel = make_panel(
    bottom_row,
    "5. Triple Extraction",
    BLUE,
    LIGHT_NAVY
)
triple_panel.grid(
    row=0, column=1, sticky="nsew", padx=(6, 0)
)

tk.Label(
    triple_panel,
    text="Các bộ ba trích xuất được",
    font=(FONT, 9, "bold"),
    bg=WHITE,
    fg=DEEP_NAVY
).pack(anchor="w", padx=12, pady=(10, 5))

triple_table = make_table(
    triple_panel,
    ("subject", "relation", "object"),
    ("Chủ thể (Subject)", "Quan hệ (Relation)", "Đối tượng (Object)"),
    (145, 120, 145)
)
triple_table.pack(fill="x", padx=9, pady=(0, 10))

delete_triple_button = tk.Button(
    triple_panel,
    text="Xóa bộ ba",
    font=(FONT, 10, "bold"),
    bg="#C0392B",
    fg=WHITE,
    activebackground="#922B21",
    activeforeground=WHITE,
    relief="flat",
    padx=14,
    pady=8,
    cursor="hand2"
)
delete_triple_button.pack(
    anchor="e",
    padx=9,
    pady=(0, 10)
)

#KẾT QUẢ CUỐI CÙNG
final_panel = tk.Frame(
    triple_panel,
    bg=LIGHT_BLUE,
    padx=10,
    pady=10,
    highlightbackground=BORDER,
    highlightthickness=1
)
final_panel.pack(fill="x", padx=9, pady=(0, 9))

tk.Label(
    final_panel,
    text="Kết quả cuối cùng",
    font=(FONT, 10, "bold"),
    bg=LIGHT_BLUE,
    fg=DEEP_NAVY
).pack(anchor="w", pady=(0, 8))

final_flow = tk.Frame(final_panel, bg=LIGHT_BLUE)
final_flow.pack(fill="x")

subject_label = tk.Label(
    final_flow,
    text="—",
    font=(FONT, 9, "bold"),
    bg=WHITE,
    fg=BLUE,
    padx=8,
    pady=10,
    wraplength=140
)
subject_label.pack(side="left", fill="x", expand=True)

tk.Label(
    final_flow,
    text="→",
    font=(FONT, 13, "bold"),
    bg=LIGHT_BLUE,
    fg=NAVY
).pack(side="left", padx=4)

relation_label = tk.Label(
    final_flow,
    text="—",
    font=(FONT, 9, "bold"),
    bg=FOREST,
    fg=WHITE,
    padx=8,
    pady=10,
    wraplength=120
)
relation_label.pack(side="left", fill="x", expand=True)

tk.Label(
    final_flow,
    text="→",
    font=(FONT, 13, "bold"),
    bg=LIGHT_BLUE,
    fg=NAVY
).pack(side="left", padx=4)

object_label = tk.Label(
    final_flow,
    text="—",
    font=(FONT, 9, "bold"),
    bg=BLUE,
    fg=WHITE,
    padx=8,
    pady=10,
    wraplength=140
)
object_label.pack(side="left", fill="x", expand=True)

note_label = tk.Label(
    triple_panel,
    text="",
    font=(FONT, 9),
    bg=LIGHT_NAVY,
    fg=DEEP_NAVY,
    justify="left",
    anchor="w",
    padx=10,
    pady=9,
    wraplength=550
)
note_label.pack(fill="x", padx=9, pady=(0, 10))

analyze_button = tk.Button(
    input_row,
    text="➤  Phân tích",
    font=(FONT, 10, "bold"),
    bg=BLUE,
    fg=WHITE,
    activebackground=NAVY,
    activeforeground=WHITE,
    relief="flat",
    padx=22,
    pady=12,
    cursor="hand2"
)
analyze_button.pack(side="left", padx=(12, 0))

export_button = tk.Button(
    input_row,
    text="Xuất JSON",
    font=(FONT, 10, "bold"),
    bg=FOREST,
    fg=WHITE,
    activebackground=NAVY,
    activeforeground=WHITE,
    relief="flat",
    padx=18,
    pady=12,
    cursor="hand2"
)
export_button.pack(side="left", padx=(8, 0))
