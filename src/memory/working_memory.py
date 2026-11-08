class WorkingMemory:
    def __init__(self, capacity=5):
        """
        Bộ nhớ làm việc (Working Memory) đóng vai trò RAM ngắn hạn,
        giới hạn số lượng mục thông tin được giữ active trong tầm nhìn trực tiếp của mạng.
        """
        self.capacity = capacity
        self.slots = []

    def load(self, item):
        """
        Đưa một item (token/latent vector) mới vào bộ nhớ làm việc. 
        Nếu vượt quá dung lượng (capacity), item cũ nhất sẽ bị đẩy ra (FIFO).
        """
        self.slots.append(item)
        if len(self.slots) > self.capacity:
            self.slots.pop(0)

    def get_contents(self):
        """
        Trích xuất toàn bộ nội dung đang có trong RAM ngắn hạn.
        """
        return self.slots

    def clear(self):
        """
        Làm sạch bộ nhớ làm việc.
        """
        self.slots = []