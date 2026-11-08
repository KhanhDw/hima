class SubRegion:
    def __init__(self, region_id, input_indices, output_index):
        self.region_id = region_id
        self.input_indices = input_indices
        self.output_index = output_index
        self.is_frozen = False
        self.recent_performance = []

    def update_performance(self, is_correct):
        self.recent_performance.append(1 if is_correct else 0)
        # Theo dõi qua cửa sổ 100 lần xuất hiện của task để đảm bảo độ chín muồi
        if len(self.recent_performance) > 100:
            self.recent_performance.pop(0)

    @property
    def accuracy(self):
        # Cần ít nhất 60 mẫu đánh giá thực tế mới bắt đầu xét đóng băng
        if not self.recent_performance or len(self.recent_performance) < 60:
            return 0.0
        return sum(self.recent_performance) / len(self.recent_performance) * 100.0


class SubRegionManager:
    def __init__(self, freeze_threshold=98.0):
        self.regions = {}
        self.freeze_threshold = freeze_threshold

    def register_region(self, region_id, input_indices, output_index):
        self.regions[region_id] = SubRegion(region_id, input_indices, output_index)

    def update_region_performance(self, target, is_correct):
        for region_id, region in self.regions.items():
            if region.output_index == target:
                region.update_performance(is_correct)
                if not region.is_frozen and region.accuracy >= self.freeze_threshold:
                    region.is_frozen = True
                    print(f"\n[SUB-REGION CONSOLIDATION] Tiểu vùng '{region_id}' đạt độ chính xác {region.accuracy:.1f}%. Tiến hành đóng băng cấu trúc synapse cục bộ!")

    def is_target_frozen(self, target):
        for region in self.regions.values():
            if region.output_index == target and region.is_frozen:
                return True
        return False