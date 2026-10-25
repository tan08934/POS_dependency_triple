import tkinter as tk

from . import app_state
from .ui_components import (
    BLUE, NAVY, DEEP_NAVY, FOREST,
    LIGHT_BLUE, LIGHT_GREEN, WHITE, FONT,
    replace_rows
)
from .ui_widgets import (
    tree_canvas, token_container, normalized_label,
    pos_table, dependency_table, triple_table,
    subject_label, relation_label, object_label, note_label
)

#VẼ CÂY PHỤ THUỘC
def draw_dependency_tree(dependencies=None):
    tree_canvas.delete("all")

    if not dependencies:
        tree_canvas.create_text(
            250, 75,
            text="Chưa có dữ liệu Dependency",
            font=(FONT, 10),
            fill=NAVY
        )
        return

    #chuyển dữ liệu đầu vào về danh sách dict thống nhất
    nodes = []
    for i, item in enumerate(dependencies, start=1):
        if isinstance(item, dict):
            node = item
        else:
            node = vars(item) if hasattr(item, "__dict__") else {}

        nodes.append({
            "id": node.get("id", i),
            "token": str(node.get("token", node.get("form", ""))),
            "head": node.get("head", 0),
            "relation": str(node.get("deprel", node.get("relation", "")))
        })
    if not nodes:
        return

    #cây vẽ theo thứ tự token trong câu
    node_count = len(nodes)
    node_width = 100
    gap = 28
    left_margin = 65
    top_y = 20
    node_y = 90
    node_height = 42

    required_width = (
        left_margin * 2
        + node_count * node_width
        + max(0, node_count - 1) * gap
    )
    width = max(tree_canvas.winfo_width(), required_width, 500)

    tree_canvas.configure(
        scrollregion=(0, 0, width, 165)
    )

    positions = {}
    total_nodes_width = (
        node_count * node_width
        + max(0, node_count - 1) * gap
    )
    start_x = max(
        left_margin,
        (width - total_nodes_width) // 2
    )

    for i, node in enumerate(nodes):
        positions[node["id"]] = (
            start_x + i * (node_width + gap) + node_width // 2
        )

    #vẽ các cạnh trước để cạnh không đè lên nội dung nút
    for node in nodes:
        head = node["head"]
        relation = node["relation"]

        try:
            head_id = int(head)
        except (TypeError, ValueError):
            head_id = 0

        dependent_id = node["id"]

        if head_id == 0 or head_id not in positions:
            continue

        parent_x = positions[head_id]
        child_x = positions.get(dependent_id)

        if child_x is None:
            continue

        #đường nối từ từ gốc tới từ phụ thuộc
        tree_canvas.create_line(
            parent_x, node_y,
            child_x, node_y + node_height,
            fill=FOREST,
            width=1.5,
            arrow=tk.LAST,
            arrowshape=(7, 9, 3)
        )
        mid_x = (parent_x + child_x) // 2
        tree_canvas.create_text(
            mid_x,
            node_y - 8,
            text=relation,
            font=(FONT, 8),
            fill=FOREST
        )

    #vẽ các nút token
    for node in nodes:
        x = positions[node["id"]]
        token = node["token"]

        try:
            is_root = int(node["head"]) == 0
        except (TypeError, ValueError):
            is_root = str(node["relation"]).lower() == "root"

        fill_color = LIGHT_GREEN if is_root else LIGHT_BLUE
        outline_color = FOREST if is_root else BLUE

        tree_canvas.create_rectangle(
            x - node_width // 2,
            node_y + node_height,
            x + node_width // 2,
            node_y + node_height + 32,
            fill=fill_color,
            outline=outline_color
        )

        tree_canvas.create_text(
            x,
            node_y + node_height + 16,
            text=token,
            font=(FONT, 9, "bold"),
            fill=DEEP_NAVY,
            width=node_width - 8
        )

        if is_root:
            tree_canvas.create_text(
                x,
                top_y,
                text="ROOT",
                font=(FONT, 8, "bold"),
                fill=FOREST
            )

#HÀM HỖ TRỢ CHUYỂN KẾT QUẢ PIPELINE
POS_MEANINGS = {
    "ADJ": "Tính từ",
    "ADP": "Giới từ",
    "ADV": "Trạng từ",
    "AUX": "Trợ động từ",
    "CCONJ": "Liên từ đẳng lập",
    "DET": "Từ hạn định",
    "INTJ": "Thán từ",
    "NOUN": "Danh từ",
    "NUM": "Số từ",
    "PART": "Tiểu từ",
    "PRON": "Đại từ",
    "PROPN": "Danh từ riêng",
    "PUNCT": "Dấu câu",
    "SCONJ": "Liên từ phụ thuộc",
    "SYM": "Ký hiệu",
    "VERB": "Động từ",
    "X": "Khác"
}

def get_field(item, *names, default=""):
    """Lấy trường dữ liệu từ dict hoặc object."""
    if isinstance(item, dict):
        for name in names:
            if name in item:
                return item[name]
    else:
        for name in names:
            if hasattr(item, name):
                return getattr(item, name)

    return default

def triple_to_row(triple):
    """
    Chuẩn hóa một bộ ba về dạng:
    (subject, relation, object).

    Hỗ trợ dict, tuple/list và object có thuộc tính.
    """
    if isinstance(triple, dict):
        subject = get_field(
            triple, "subject", "subj", "S", "s"
        )
        relation = get_field(
            triple, "relation", "predicate", "pred", "R", "r"
        )
        obj = get_field(
            triple, "object", "obj", "O", "o"
        )

    elif isinstance(triple, (tuple, list)):
        if len(triple) >= 3:
            subject, relation, obj = triple[:3]
        else:
            return (
                str(triple),
                "—",
                "—"
            )
    else:
        subject = get_field(
            triple, "subject", "subj", "S", "s"
        )
        relation = get_field(
            triple, "relation", "predicate", "pred", "R", "r"
        )
        obj = get_field(
            triple, "object", "obj", "O", "o"
        )

        #hỗ trợ trường hợp object là dataclass hoặc object khác
        if not subject and not relation and not obj:
            if hasattr(triple, "__dict__"):
                fields = vars(triple)
                return (
                    str(fields.get("subject", fields.get("subj", "—"))),
                    str(fields.get("relation", fields.get("predicate", "—"))),
                    str(fields.get("object", fields.get("obj", "—")))
                )
            return (str(triple), "—", "—")

    return (
        str(subject) if subject not in (None, "") else "—",
        str(relation) if relation not in (None, "") else "—",
        str(obj) if obj not in (None, "") else "—"
    )

def update_token_display(tokens):
    """Hiển thị danh sách token trong panel Tách từ."""
    for widget in token_container.winfo_children():
        widget.destroy()

    if not tokens:
        tk.Label(
            token_container,
            text="Không có token",
            font=(FONT, 9),
            bg=WHITE,
            fg=NAVY
        ).pack(anchor="w")
        return

    for token in tokens:
        # Giữ nguyên token thực tế của pipeline
        # Chỉ thay dấu gạch dưới khi hiển thị để dễ đọc
        display_token = str(token).replace("_", " ")

        tk.Label(
            token_container,
            text=display_token,
            font=(FONT, 9),
            bg=LIGHT_BLUE,
            fg=BLUE,
            padx=8,
            pady=6
        ).pack(side="left", padx=3, pady=4)

def display_results(result, normalized_text):
    """Đưa kết quả pipeline thật lên giao diện."""

    app_state.current_result = result
    tokens = result.get("tokens", []) or []
    pos_tags = result.get("pos")
    if pos_tags is None:
        pos_tags = []
    dependencies = result.get("dependency", []) or []
    triples = result.get("triples", []) or []
    print("Kiểu dữ liệu:", type(triples))
    print("Dữ liệu bộ ba:", repr(triples))
    print("Số lượng:", len(triples))

    # 1. Văn bản chuẩn hóa
    normalized_label.configure(text=normalized_text)

    # 2. Tách từ
    update_token_display(tokens)

    # 3. POS Tagging
    pos_rows = []

    for i, token in enumerate(tokens):
        if i < len(pos_tags):
            pos = pos_tags[i]

            # Phòng trường hợp pipeline trả về object/dict thay vì chuỗi
            if isinstance(pos, dict):
                pos = pos.get(
                    "upos",
                    pos.get("pos", pos.get("tag", "—"))
                )
            elif not isinstance(pos, (str, int, float)):
                pos = get_field(
                    pos, "upos", "pos", "tag", default="—"
                )

            pos = str(pos)
        else:
            pos = "—"

        meaning = POS_MEANINGS.get(pos, "Chưa có diễn giải")
        pos_rows.append((str(token), pos, meaning))

    replace_rows(pos_table, pos_rows)

    # 4. Dependency Parsing
    id_to_token = {}

    for i, node in enumerate(dependencies, start=1):
        node_id = get_field(node, "id", default=i)
        token = get_field(node, "token", "form", default="")

        try:
            id_to_token[int(node_id)] = str(token)
        except (TypeError, ValueError):
            id_to_token[node_id] = str(token)

    dependency_rows = []

    for i, node in enumerate(dependencies, start=1):
        node_id = get_field(node, "id", default=i)
        token = get_field(node, "token", "form", default="")
        head = get_field(node, "head", default=0)
        relation = get_field(
            node, "deprel", "relation", default="—"
        )

        try:
            head_id = int(head)
        except (TypeError, ValueError):
            head_id = 0

        if head_id == 0:
            head_token = "ROOT"
        else:
            head_token = id_to_token.get(
                head_id,
                id_to_token.get(head, str(head))
            )

        dependency_rows.append((
            str(token),
            str(relation),
            str(head_token)
        ))

    replace_rows(dependency_table, dependency_rows)
    draw_dependency_tree(dependencies)

        # 5. Triple Extraction

    triple_rows = [triple_to_row(t) for t in triples]

    # Lưu bộ ba để hỗ trợ chỉnh sửa và xuất JSON
    app_state.current_triples = [
        {
            "subject": subject,
            "relation": relation,
            "object": obj
        }
        for subject, relation, obj in triple_rows
    ]

    # Hiển thị bộ ba trên bảng
    replace_rows(triple_table, triple_rows)

    # Hiển thị bộ ba đầu tiên ở khu vực kết quả cuối cùng
    if triple_rows:
        subject, relation, obj = triple_rows[0]

        subject_label.configure(text=subject)
        relation_label.configure(text=relation)
        object_label.configure(text=obj)

        note_label.configure(
            text=f"Phân tích hoàn tất. "
                 f"Tổng số token: {len(tokens)} | "
                 f"Tổng số quan hệ phụ thuộc: {len(dependencies)} | "
                 f"Tổng số bộ ba trích xuất: {len(triple_rows)}."
        )

    else:
        subject_label.configure(text="—")
        relation_label.configure(text="—")
        object_label.configure(text="—")

        note_label.configure(
            text=f"Đã phân tích xong nhưng không trích xuất được bộ ba. "
                 f"Tổng số token: {len(tokens)} | "
                 f"Tổng số quan hệ phụ thuộc: {len(dependencies)}."
        )
        
