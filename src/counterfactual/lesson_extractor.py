class LessonExtractor:
    def __init__(self):
        """
        Bộ trích xuất bài học (Lesson Extractor) chuyển hóa các phân tích nhân quả
        thành tri thức kinh nghiệm để cập nhật chiến lược ứng xử cho hệ thống.
        """
        self.learned_lessons = []

    def extract_lesson(self, cause_effect_insight):
        """
        Trích xuất bài học từ phân tích nhân quả và lưu trữ vào kho kinh nghiệm.
        """
        if not cause_effect_insight:
            return None

        lesson = {
            "condition": "alternative_action_evaluation",
            "recommended_action": cause_effect_insight.get("alternative_action"),
            "confidence": 0.8 if cause_effect_insight.get("is_better") else 0.2
        }
        
        self.learned_lessons.append(lesson)
        return lesson