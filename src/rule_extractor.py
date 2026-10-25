"""
Đoạn mã gồm lớp TokenNode, hàm phụ trợ (kèm gom từ bổ nghĩa) và hàm xử lý chính.
Vì sao cần đi gom từ bổ nghĩa?
-> Với câu "Tôi giao gói hàng màu đen",  root = giao thì có quan hệ với từ gói hàng,
nhưng từ màu đen vẫn thuộc gói hàng mà không được xét đến, nên cần đi gom những từ này
để trả về đủ "gói hàng màu đen".
"""
from typing import List, Dict, Set, Tuple, Optional

class TokenNode:
    def __init__(self, token_id: int, form: str, head: int, deprel: str, pos: str = None):
        #vị trí của từ trong câu
        self.id = token_id
        #từ, cụm từ
        self.form = form
        self.head = head
        #nhãn quan hệ phụ thuộc
        self.deprel = deprel
        self.pos = pos

def build_children_mapping(parsed_sentence: List[TokenNode]) -> Dict[int, List[TokenNode]]:
    """
    Tạo bảng ánh xạ ID nút gốc -> Danh sách các nút con.
    Có thể hiểu như sau, đoạn code soạn sẵn một danh sách quan hệ cha con ngay từ đầu,
    nên khi cần tìm con của từ nào thì xét ở dòng đó là xong.
    """
    children_by_head: Dict[int, List[TokenNode]] = {}
    for node in parsed_sentence:
        if node.head not in children_by_head:
            children_by_head[node.head] = []
        children_by_head[node.head].append(node)
    return children_by_head


def extract_modifier_phrase(
    head_id: int, 
    parsed_sentence: List[TokenNode], 
    children_by_head: Dict[int, List[TokenNode]], 
    phrase_type: str = "NOUN" #tham số xác định loại cụm (mặc định noun nếu k truyền vào)
) -> str:
    """
    Gom các từ bổ trợ xung quanh từ chính (head_id) bằng DFS.
    """
    #set chứa nhãn bổ nghĩa danh từ
    NOUN_MODIFIERS = {
        'nmod', 'nmod:poss', 'amod', 'det', 'compound', 
        'flat', 'appos', 'nummod', 'case'
    }
    
    #động từ
    VERB_MODIFIERS = {
        'advmod', 'advmod:neg', 'aux', 'aux:pass', 'compound:svc', 'mark'
    }

    allowed_relations = NOUN_MODIFIERS if phrase_type == "NOUN" else VERB_MODIFIERS
    
    #tìm trong danh sách parsed_sentence xem nút nào có id trùng với head_id
    #xong r lấy ra lưu vào biến head_node, còn tìm không thấy thì trả về None
    head_node = next((node for node in parsed_sentence if node.id == head_id), None)
    if not head_node:
        return ""

    collected_nodes = [head_node]
    visited_ids = {head_id}
    """ 
    Cơ chế của visited_ids.
    Nếu Malt bị lỗi rồi tạo ra một vòng lặp như từ A -> B, B -> A, thì nó sẽ lặp mãi.
    Nhờ có visited_ids thì biết là đã thăm rồi nên dừng lại không lặp nữa.
    """
    #đi duyệt con
    def traverse(current_id: int):
        children = children_by_head.get(current_id, [])
        for child in children:
            if child.deprel in allowed_relations and child.id not in visited_ids:
                visited_ids.add(child.id)
                collected_nodes.append(child)
                traverse(child.id)

    traverse(head_id)

    #sắp xếp lại theo đúng trật tự tự nhiên của từ trong câu gốc
    sorted_nodes = sorted(collected_nodes, key=lambda node: node.id)
    return " ".join([node.form for node in sorted_nodes])


def extract_triples_openie(parsed_sentence: List[TokenNode]) -> List[List[str]]:
    """
    Hàm chính trích xuất danh sách bộ ba [Subject, Relation, Object].
    Xử lý phân loại vai trò đối số (obj, iobj, obl) và đảo câu bị động.
    """
    #tạo danh sách chứa bộ 3
    predicted_triples: List[List[str]] = []
    #set này lưu dưới dạng (S, R, O), bộ 3 nào có r thì k thêm nữa
    seen_triples: Set[Tuple[str, str, str]] = set()
    
    children_by_head = build_children_mapping(parsed_sentence)
    
    #tập nhãn
    SUBJ_PASSIVE_LABELS = {'nsubj:pass'}
    SUBJ_ACTIVE_LABELS  = {'nsubj', 'nsubj:outer', 'csubj', 'csubj:pass'}
    
 # 1. Tìm các nút có thể đóng vai trò quan hệ R.
    relation_candidates = [
        node for node in parsed_sentence
        if node.pos == "VERB"
        or (node.pos == "ADJ" and node.deprel == "root")
    ]

    nominal_pos = {"NOUN", "PROPN", "PRON"}

    # Các nhãn không dùng làm đối số trong quy tắc dự phòng.
    excluded_arg_labels = {
        "obj", "iobj",
        "nsubj", "nsubj:outer", "nsubj:pass",
        "csubj", "csubj:pass",
        "obl:tmod", "obl:lmod", "obl:agent",
        "punct"
    }

    PASSIVE_INDICATORS = ("bởi", "do", "từ")
    TIME_LOC_PREPS = ("vào", "lúc", "tại", "ở", "ngày", "năm")

    for verb in relation_candidates:
        # R: lấy động từ cùng các từ bổ trợ của nó.
        rel_str = extract_modifier_phrase(
            verb.id,
            parsed_sentence,
            children_by_head,
            phrase_type="VERB"
        )

        subj_str = ""
        direct_obj = ""
        indirect_obj = ""
        agent_str = ""

        # Chỉ xét các nút con trực tiếp của động từ.
        children = children_by_head.get(verb.id, [])

        # Xác định bị động trước vòng lặp để không phụ thuộc thứ tự token.
        is_passive = any(
            child.deprel in SUBJ_PASSIVE_LABELS
            for child in children
        )

        # 2. Tìm S và O theo nhãn dependency.
        for child in children:
            if child.deprel in SUBJ_PASSIVE_LABELS:
                subj_str = extract_modifier_phrase(
                    child.id, parsed_sentence, children_by_head, "NOUN"
                )

            elif child.deprel in SUBJ_ACTIVE_LABELS and not subj_str:
                subj_str = extract_modifier_phrase(
                    child.id, parsed_sentence, children_by_head, "NOUN"
                )

            elif child.deprel == "obj":
                direct_obj = extract_modifier_phrase(
                    child.id, parsed_sentence, children_by_head, "NOUN"
                )

            elif child.deprel == "iobj":
                indirect_obj = extract_modifier_phrase(
                    child.id, parsed_sentence, children_by_head, "NOUN"
                )

            elif child.deprel.startswith("obl"):
                phrase = extract_modifier_phrase(
                    child.id, parsed_sentence, children_by_head, "NOUN"
                )
                phrase_lower = phrase.lower()

                is_agent = (
                    child.deprel == "obl:agent"
                    or any(
                        phrase_lower.startswith(prep)
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

        # 3. DỰ PHÒNG S cho câu chủ động:
        # Nếu không tìm được S qua nhãn chuẩn, thử tìm danh từ/đại từ
        # đứng trước động từ và nối trực tiếp với động từ.
        if not is_passive and not subj_str:
            subject_candidates = [
                child for child in children
                if child.pos in nominal_pos
                and child.id < verb.id
                and child.deprel not in excluded_arg_labels
            ]

            if subject_candidates:
                # Chọn danh từ/đại từ gần động từ nhất ở phía trước.
                subject_node = max(
                    subject_candidates,
                    key=lambda node: node.id
                )

                subj_str = extract_modifier_phrase(
                    subject_node.id,
                    parsed_sentence,
                    children_by_head,
                    phrase_type="NOUN"
                )
        # S kế thừa từ động từ cha khi động từ này là conj/xcomp/advcl/parataxis
        if not is_passive and not subj_str:
            ancestor = verb
            for _ in range(5):
                if ancestor.deprel not in {"conj", "xcomp", "advcl", "parataxis"}:
                    break
                ancestor = next((n for n in parsed_sentence if n.id == ancestor.head), None)
                if ancestor is None:
                    break
                inherited = [c for c in children_by_head.get(ancestor.id, [])
                             if c.deprel in SUBJ_ACTIVE_LABELS]
                if inherited:
                    subj_str = extract_modifier_phrase(
                        inherited[0].id, parsed_sentence, children_by_head, "NOUN"
                    )
                    break

        # 4. DỰ PHÒNG O cho câu chủ động:
        # Nếu chưa tìm được O, thử tìm danh từ/đại từ đứng sau động từ
        # và nối trực tiếp với động từ.
        if not is_passive and not direct_obj and not indirect_obj:
            object_candidates = sorted(
                [
                    child for child in children
                    if child.pos in nominal_pos
                    and child.id > verb.id
                    and child.deprel not in excluded_arg_labels
                ],
                key=lambda node: node.id
            )

            for object_node in object_candidates:
                candidate_phrase = extract_modifier_phrase(
                    object_node.id,
                    parsed_sentence,
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

        # 5. Xác định vai trò cuối cùng, có xử lý câu bị động.
        if is_passive and agent_str:
            # Câu bị động có tác thể:
            # tác thể -> S; thành phần bị tác động -> O.
            final_subj = agent_str
            final_obj = subj_str
        else:
            final_subj = subj_str
            final_obj = direct_obj or indirect_obj

        triple_key = (
            final_subj.strip(),
            rel_str.strip(),
            final_obj.strip()
        )

        # Loại bỏ bộ ba trùng lặp.
        if triple_key not in seen_triples:
            seen_triples.add(triple_key)
            predicted_triples.append(list(triple_key))

    return predicted_triples