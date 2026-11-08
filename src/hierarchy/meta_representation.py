class MetaRepresentation:
    def __init__(self):
        """
        Quản lý siêu biểu diễn (Meta Representation) duy trì mạng lưới quan hệ
        đa tầng giữa các cấp độ trừu tượng khác nhau (từ low-level đến high-level concepts).
        """
        self.relations = {}

    def link(self, parent_concept, child_concept, relation_type="composed_of"):
        """
        Thiết lập mối quan hệ phân cấp giữa khái niệm cha (vĩ mô) và khái niệm con (vi mô).
        """
        if parent_concept not in self.relations:
            self.relations[parent_concept] = []
        
        self.relations[parent_concept].append({
            "child": child_concept,
            "type": relation_type
        })

    def get_children(self, parent_concept):
        """
        Truy xuất các thành phần con cấu thành nên một khái niệm vĩ mô.
        """
        return self.relations.get(parent_concept, [])