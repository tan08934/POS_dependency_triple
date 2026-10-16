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
    
    relation_candidates: List[TokenNode] = []
    #set ghi nhớ ID của các đtu đã tìm thấy
    seen_verb_ids: Set[int] = set()
    
    for node in parsed_sentence:
        if node.id in seen_verb_ids:
            continue
        #ưu tiên đtu hoặc tính từ vị từ làm root (ADJ + root)
        is_verb = node.pos == "VERB"
        is_pred_adj = (node.pos == "ADJ" and node.deprel == "root")
        
        if is_verb or is_pred_adj:
            relation_candidates.append(node)
            seen_verb_ids.add(node.id)

    #duyệt trích xuất bộ 3
    for verb in relation_candidates:
        #gom cụm
        rel_str = extract_modifier_phrase(verb.id, parsed_sentence, children_by_head, phrase_type="VERB")
        
        subj_str = ""
        obj_str = ""
        agent_str = ""
        direct_obj = ""    #tân ngữ trực tiếp (obj)
        indirect_obj = ""  #tân ngữ gián tiếp (iobj)
        
        #lấy danh sách con nối với verb
        children = children_by_head.get(verb.id, [])
        is_passive = False
        
        for child in children:
            #nếu nút con có nhãn bị động thì đánh dấu is_passive = True và gom cụm dtu đó lưu vào subj_str
            if child.deprel in SUBJ_PASSIVE_LABELS:
                is_passive = True
                subj_str = extract_modifier_phrase(child.id, parsed_sentence, children_by_head, phrase_type="NOUN")
            
            #nếu có nhãn chủ động và chưa có subj_str
            elif child.deprel in SUBJ_ACTIVE_LABELS and not subj_str:
                subj_str = extract_modifier_phrase(child.id, parsed_sentence, children_by_head, phrase_type="NOUN")
                
            #nếu nút con là tân ngữ trực tiếp (obj) hoặc gián tiếp (iobj) thì gom cụm lưu vào direct_obj hoặc indirect_obj
            elif child.deprel == "obj":
                direct_obj = extract_modifier_phrase(child.id, parsed_sentence, children_by_head, phrase_type="NOUN")
            elif child.deprel == "iobj":
                indirect_obj = extract_modifier_phrase(child.id, parsed_sentence, children_by_head, phrase_type="NOUN")
            
            #xử lý mấy từ liên quan bị động
            elif child.deprel.startswith("obl"):
                phrase = extract_modifier_phrase(child.id, parsed_sentence, children_by_head, phrase_type="NOUN")
                phrase_lower = phrase.lower()
                
                #tập từ chỉ tác thể bị động
                PASSIVE_INDICATORS = ("bởi", "do", "từ")
                
                #kiểm tra nếu là tác thể bị động (nhãn obl:agent hoặc bắt đầu bằng từ chỉ tác thể)
                is_agent = child.deprel == "obl:agent" or any(phrase_lower.startswith(prep) for prep in PASSIVE_INDICATORS)
                
                if is_passive and is_agent:
                    #lược bỏ các từ nối bị động để lấy tác thể sạch (ví dụ: do Công ty A -> Công ty A)
                    clean_agent = phrase
                    for prep in PASSIVE_INDICATORS:
                        if clean_agent.lower().startswith(prep):
                            clean_agent = clean_agent[len(prep):].strip()
                            break
                    agent_str = clean_agent
                
                #nếu k phải tác thể thì chỉ lấy làm direct_obj
                elif child.deprel not in ["obl:tmod", "obl:lmod"] and not direct_obj:
                    #bỏ qua nếu cụm bắt đầu bằng giới từ chỉ thời gian, địa điểm rõ ràng
                    TIME_LOC_PREPS = ("vào", "lúc", "tại", "ở", "ngày", "năm")
                    if not any(phrase_lower.startswith(prep) for prep in TIME_LOC_PREPS):
                        direct_obj = phrase

        #xử lý logic ngữ nghĩa
        if is_passive and agent_str:
            #đảo vai trò câu bị động: tthe -> Subject, thành phần bị tác động -> Object
            final_subj = agent_str
            final_obj = subj_str
        else:
            final_subj = subj_str
            final_obj = direct_obj or indirect_obj

        #đóng gói
        if final_subj and rel_str and final_obj:
            triple_key = (final_subj.strip(), rel_str.strip(), final_obj.strip())
            if triple_key not in seen_triples:
                seen_triples.add(triple_key)
                predicted_triples.append(list(triple_key))

    return predicted_triples