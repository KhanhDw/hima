class CoordinationProtocol:
    def __init__(self):
        """
        Giao thức phối hợp (Coordination Protocol) quản lý việc phân chia nhiệm vụ,
        chia sẻ trạng thái mục tiêu giữa các tác nhân trong mạng lưới.
        """
        self.shared_tasks = {}

    def propose_task(self, task_id, task_description, target_agent_id):
        """
        Đề xuất hoặc giao nhiệm vụ cho một agent cụ thể trong hệ thống đa tác nhân.
        """
        self.shared_tasks[task_id] = {
            "description": task_description,
            "assigned_to": target_agent_id,
            "status": "pending"
        }

    def update_task_status(self, task_id, status):
        """
        Cập nhật trạng thái tiến độ thực hiện nhiệm vụ chung.
        """
        if task_id in self.shared_tasks:
            self.shared_tasks[task_id]["status"] = status