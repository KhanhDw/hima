class ContextMemory:
    def __init__(self):
        """
        Bộ nhớ ngữ cảnh (Context Memory) lưu trữ trạng thái bối cảnh hiện tại 
        (ví dụ: mã nhận diện nhánh, trạng thái môi trường hiện tại) để định hướng 
        cho mạng xử lý đúng hướng khi gặp các tín hiệu lặp lại.
        """
        self.current_context = {}

    def set_context(self, key, value):
        """
        Thiết lập hoặc cập nhật một giá trị bối cảnh.
        """
        self.current_context[key] = value

    def get_context(self, key):
        """
        Lấy giá trị của một bối cảnh cụ thể.
        """
        return self.current_context.get(key, None)

    def clear(self):
        """
        Xóa toàn bộ ngữ cảnh khi đổi tác vụ hoặc kết thúc phiên.
        """
        self.current_context = {}