class Chunker:
    def __init__(self, frequency_threshold=3):
        """
        Bộ gom nhóm (Chunker) phát hiện các chuỗi trạng thái lặp lại thường xuyên 
        và gộp chúng thành một khái niệm trừu tượng (chunk) duy nhất để tiết kiệm không gian xử lý.
        """
        self.frequency_threshold = frequency_threshold
        self.sequence_counts = {}
        self.chunks = {}
        self.chunk_counter = 0

    def observe_sequence(self, sequence_tuple):
        """
        Ghi nhận một chuỗi trạng thái (dưới dạng tuple) và đếm tần suất xuất hiện.
        Nếu vượt ngưỡng, tự động kết tinh thành một khái niệm (chunk) mới.
        """
        # Chuyển đổi an toàn sang tuple để có thể hash được trong dictionary
        seq_key = tuple(tuple(item) if isinstance(item, list) else item for item in sequence_tuple)
        
        self.sequence_counts[seq_key] = self.sequence_counts.get(seq_key, 0) + 1

        # Nếu đạt tần suất quen thuộc, biến chuỗi thành một chunk trừu tượng
        if self.sequence_counts[seq_key] >= self.frequency_threshold:
            # Kiểm tra xem chuỗi này đã có chunk ID chưa
            for cid, cval in self.chunks.items():
                if cval == seq_key:
                    return cid
                    
            chunk_id = f"chunk_{self.chunk_counter}"
            self.chunks[chunk_id] = seq_key
            self.chunk_counter += 1
            return chunk_id
            
        return None

    def get_chunks(self):
        """
        Trả về toàn bộ các khái niệm trừu tượng đã được học và gom nhóm.
        """
        return self.chunks