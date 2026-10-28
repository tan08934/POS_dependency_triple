from typing import List, Dict, Set, Tuple, Optional


#Định nghĩa hằng số toàn cục (tránh khởi tạo lại nhiều lần trong hàm)
# Set chứa nhãn bổ nghĩa danh từ (Đã loại bỏ 'appos' để tránh bị phình to bởi các cụm chêm xen, chú thích)
NOUN_MODIFIERS = {'nmod', 'nmod:poss', 'amod', 'det', 'compound', 'flat', 'nummod', 'case'}
# Set chứa nhãn bổ nghĩa động từ
VERB_MODIFIERS = {'advmod', 'advmod:neg', 'aux', 'aux:pass', 'compound:svc', 'mark'}


# Tập nhãn chủ ngữ
SUBJ_PASSIVE_LABELS = {'nsubj:pass'}
SUBJ_ACTIVE_LABELS  = {'nsubj', 'nsubj:outer', 'csubj', 'csubj:pass'}
NOMINAL_POS = {'NOUN', 'PROPN', 'PRON'}


# Các nhãn không dùng làm đối số trong quy tắc dự phòng (Đã mở rộng chặn các nhãn obl để chống trạng ngữ đầu câu)
EXCLUDED_ARG_LABELS = {'obj', 'iobj', 'nsubj', 'nsubj:outer', 'nsubj:pass', 'csubj', 'csubj:pass', 'obl:tmod', 'obl:lmod', 'obl:agent', 'punct'}


PASSIVE_INDICATORS = ('bởi', 'do')
TIME_LOC_PREPS = ('vào', 'lúc', 'tại', 'ở', 'ngày', 'năm')
# Từ khóa nhận diện câu mệnh lệnh / cầu khiến khuyết chủ ngữ
IMPERATIVE_MARKERS = ('hãy', 'đừng', 'chớ', 'xin')




class TokenNode:
    def __init__(self, token_id: int, form: str, head: int, deprel: str, pos: str = None):
        # Vị trí của từ trong câu
        self.id = token_id
        # Từ, cụm từ
        self.form = form
        self.head = head
        # Nhãn quan hệ phụ thuộc
        self.deprel = deprel
        self.pos = pos




def build_children_mapping(parsed_sentence: List[TokenNode]) -> Tuple[Dict[int, List[TokenNode]], Dict[int, TokenNode]]:
    """
    Tạo bảng ánh xạ ID nút gốc -> Danh sách các nút con.
    Có thể hiểu như sau, đoạn code soạn sẵn một danh sách quan hệ cha con ngay từ đầu,
    nên khi cần tìm con của từ nào thì xét ở dòng đó là xong.
    (Được tối ưu tích hợp bảng tra cứu nhanh Nút theo ID O(1))
    """
    children_by_head: Dict[int, List[TokenNode]] = {}
    node_by_id: Dict[int, TokenNode] = {}
   
    for node in parsed_sentence:
        node_by_id[node.id] = node
        if node.head not in children_by_head:
            children_by_head[node.head] = []
        children_by_head[node.head].append(node)
       
    return children_by_head, node_by_id




def extract_modifier_phrase(
    head_id: int,
    node_by_id: Dict[int, TokenNode],      # Thêm vào để tối ưu tra cứu O(1)
    children_by_head: Dict[int, List[TokenNode]],
    phrase_type: str = "NOUN"              # Tham số xác định loại cụm (mặc định noun nếu k truyền vào)
) -> str:
    """
    Gom các từ bổ trợ xung quanh từ chính (head_id) bằng DFS.
    """
    allowed_relations = NOUN_MODIFIERS if phrase_type == "NOUN" else VERB_MODIFIERS
   
    # Tìm trong danh sách parsed_sentence xem nút nào có id trùng với head_id
    # Xong r lấy ra lưu vào biến head_node, còn tìm không thấy thì trả về None
    head_node = node_by_id.get(head_id)
    if not head_node:
        return ""


    collected_nodes = [head_node]
    """
    Cơ chế của visited_ids.
    Nếu Malt bị lỗi rồi tạo ra một vòng lặp như từ A -> B, B -> A, thì nó sẽ lặp mãi.
    Nhờ có visited_ids thì biết là đã thăm rồi nên dừng lại không lặp nữa.
    """
    visited_ids = {head_id}
   
    # Đi duyệt con
    def traverse(current_id: int):
        children = children_by_head.get(current_id, [])
        for child in children:
            if child.deprel in allowed_relations and child.id not in visited_ids:
                visited_ids.add(child.id)
                collected_nodes.append(child)
                traverse(child.id)


    traverse(head_id)


    # Sắp xếp lại theo đúng trật tự tự nhiên của từ trong câu gốc
    sorted_nodes = sorted(collected_nodes, key=lambda node: node.id)
    return " ".join([node.form for node in sorted_nodes])




def extract_triples_openie(parsed_sentence: List[TokenNode]) -> List[List[str]]:
    """
    Hàm chính trích xuất danh sách bộ ba [Subject, Relation, Object].
    Xử lý phân loại vai trò đối số (obj, iobj, obl) và đảo câu bị động.
    """
    # Tạo danh sách chứa bộ 3
    predicted_triples: List[List[str]] = []
    # Set này lưu dưới dạng (S, R, O), bộ 3 nào có r thì k thêm nữa
    seen_triples: Set[Tuple[str, str, str]] = set()
   
    # Khởi tạo cây cú pháp và bảng tra cứu nút O(1) (Dùng đúng tên build_children_mapping)
    children_by_head, node_by_id = build_children_mapping(parsed_sentence)
   
    # 1. Tìm các nút có thể đóng vai trò quan hệ R
    relation_candidates = [
        node for node in parsed_sentence
        if node.pos == "VERB"
        or (node.pos in {"ADJ", "NOUN", "PROPN"} and node.deprel == "root")
    ]

    for node in parsed_sentence:
        print(
            f"ID={node.id}, FORM={node.form}, UPOS={node.pos}, "
            f"HEAD={node.head}, DEPREL={node.deprel}"
        )
    for verb in relation_candidates:
        print(f"ID={verb.id}, FORM={verb.form}, DEPREL={verb.deprel}")
    for verb in relation_candidates:
        # R: lấy động từ cùng các từ bổ trợ của nó.
        rel_str = extract_modifier_phrase(
            verb.id,
            node_by_id,
            children_by_head,
            phrase_type="VERB"
        )


        subj_str = ""
        direct_obj = ""
        indirect_obj = ""
        agent_str = ""


        # Chỉ xét các nút con trực tiếp của động từ.
        children = children_by_head.get(verb.id, [])


        # Xác định bị động trước vòng lặp để k phụ thuộc thứ tự token
        is_passive = any(
            child.deprel in SUBJ_PASSIVE_LABELS
            for child in children
        )


        # 2. Tìm S và O theo nhãn dependency
        for child in children:
            if child.deprel in SUBJ_PASSIVE_LABELS:
                subj_str = extract_modifier_phrase(
                    child.id, node_by_id, children_by_head, "NOUN"
                )


            elif child.deprel in SUBJ_ACTIVE_LABELS and not subj_str:
                subj_str = extract_modifier_phrase(
                    child.id, node_by_id, children_by_head, "NOUN"
                )


            elif child.deprel == "obj":
                direct_obj = extract_modifier_phrase(
                    child.id, node_by_id, children_by_head, "NOUN"
                )


            elif child.deprel == "iobj":
                indirect_obj = extract_modifier_phrase(
                    child.id, node_by_id, children_by_head, "NOUN"
                )


            elif child.deprel.startswith("obl"):
                phrase = extract_modifier_phrase(
                    child.id, node_by_id, children_by_head, "NOUN"
                )
                phrase_lower = phrase.lower()


                is_agent = (
                    child.deprel == "obl:agent"
                    or any(
                        phrase_lower.startswith(prep + " ") or phrase_lower == prep
                        for prep in PASSIVE_INDICATORS
                    )
                )


                if is_passive and is_agent:
                    clean_agent = phrase


                    for prep in PASSIVE_INDICATORS:
                        if clean_agent.lower().startswith(prep):
                            clean_agent = clean_agent[len(prep):].strip()
                            break


                    agent_str = clean_agent


                elif (
                    child.deprel not in {"obl:tmod", "obl:lmod"}
                    and not direct_obj
                    and not any(
                        phrase_lower.startswith(prep)
                        for prep in TIME_LOC_PREPS
                    )
                ):
                    direct_obj = phrase


        # 3. DỰ PHÒNG S cho câu chủ động
        if not is_passive and not subj_str:
            subject_candidates = [
                child for child in children
                if child.pos in NOMINAL_POS
                and child.id < verb.id
                and child.deprel not in EXCLUDED_ARG_LABELS
                and not child.deprel.startswith("obl")
                and not any(child.form.lower().startswith(prep) for prep in TIME_LOC_PREPS)
            ]


            if subject_candidates:
                subject_node = max(
                    subject_candidates,
                    key=lambda node: node.id
                )


                subj_str = extract_modifier_phrase(
                    subject_node.id,
                    node_by_id,
                    children_by_head,
                    phrase_type="NOUN"
                )
       
        # S kế thừa từ động từ cha khi động từ này là conj/xcomp/advcl/parataxis
        if not is_passive and not subj_str:
            ancestor = verb
            depth = 0
            while depth < 10:
                if ancestor.deprel not in {"conj", "xcomp", "advcl", "parataxis"}:
                    break
                ancestor = node_by_id.get(ancestor.head)
                if ancestor is None:
                    break
                inherited = [c for c in children_by_head.get(ancestor.id, [])
                             if c.deprel in SUBJ_ACTIVE_LABELS]
                if inherited:
                    subj_str = extract_modifier_phrase(
                        inherited[0].id, node_by_id, children_by_head, "NOUN"
                    )
                    break
                depth += 1


        # 4. DỰ PHÒNG O cho câu chủ động
        if not is_passive and not direct_obj and not indirect_obj:
            object_candidates = sorted(
                [
                    child for child in children
                    if child.pos in NOMINAL_POS
                    and child.id > verb.id
                    and child.deprel not in EXCLUDED_ARG_LABELS
                    and not child.deprel.startswith("obl")
                ],
                key=lambda node: node.id
            )


            for object_node in object_candidates:
                candidate_phrase = extract_modifier_phrase(
                    object_node.id,
                    node_by_id,
                    children_by_head,
                    phrase_type="NOUN"
                )


                if (
                    candidate_phrase
                    and not candidate_phrase.lower().startswith(
                        TIME_LOC_PREPS
                    )
                ):
                    direct_obj = candidate_phrase
                    break


        # 5. Xác định vai trò cuối cùng, có xử lý câu bị động
        if is_passive and agent_str:
            final_subj = agent_str
            final_obj = subj_str
        else:
            final_subj = subj_str
            final_obj = direct_obj or indirect_obj


        # Xử lý đặc biệt cho câu mệnh lệnh / cầu khiến khuyết chủ ngữ
        if not is_passive and not final_subj:
            rel_lower = rel_str.lower()
            if any(rel_lower.startswith(marker) for marker in IMPERATIVE_MARKERS) or any(child.deprel == "advmod" and child.form.lower() in IMPERATIVE_MARKERS for child in children):
                final_subj = "Bạn"


        triple_key = (
            final_subj.strip(),
            rel_str.strip(),
            final_obj.strip()
        )


        # Loại bỏ bộ ba trùng lặp và đảm bảo các thành phần khác rỗng
        if all(triple_key) and triple_key not in seen_triples:
            seen_triples.add(triple_key)
            predicted_triples.append(list(triple_key))


    return predicted_triples

