class GlobalWorkspace:
    def __init__(self):
        """
        Không gian làm việc toàn cục (Global Workspace) đóng vai trò trung tâm 
        chia sẻ thông tin và phát sóng ý thức (broadcast) giữa các module nhận thức.
        """
        self.current_conscious_content = None
        self.subconscious_queue = []

    def broadcast(self, content, source_module):
        """
        Phát sóng thông tin quan trọng từ một module ra toàn bộ hệ thống.
        """
        self.current_conscious_content = {
            "source": source_module,
            "data": content
        }
        return self.current_conscious_content

    def get_current_focus(self):
        """
        Truy xuất nội dung đang chiếm giữ tâm trí (ý thức) hiện tại của hệ thống.
        """
        return self.current_conscious_content