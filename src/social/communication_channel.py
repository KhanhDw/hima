class CommunicationChannel:
    def __init__(self):
        """
        Kênh giao tiếp (Communication Channel) quản lý việc phát và nhận 
        thông điệp giữa các tác nhân (agents) khác nhau trong mạng lưới xã hội.
        """
        self.message_mailbox = {}

    def send_message(self, sender_id, receiver_id, message_content):
        """
        Gửi một thông điệp từ tác nhân này sang tác nhân khác.
        """
        if receiver_id not in self.message_mailbox:
            self.message_mailbox[receiver_id] = []
            
        self.message_mailbox[receiver_id].append({
            "sender": sender_id,
            "content": message_content
        })

    def fetch_messages(self, agent_id):
        """
        Thu thập toàn bộ các thông điệp đến mà một tác nhân chưa đọc, 
        sau đó làm sạch hộp thư của agent đó.
        """
        messages = self.message_mailbox.get(agent_id, [])
        self.message_mailbox[agent_id] = []
        return messages