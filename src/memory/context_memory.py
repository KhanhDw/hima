import time

class ContextMemory:
    def __init__(self):
        self.current_context = {}
        self.history = []
        self.max_history = 100

    def set_context(self, key, value, scope="global"):
        self.current_context.setdefault(scope, {})[key] = value

    def get_context(self, key, scope="global", default=None):
        return self.current_context.get(scope, {}).get(key, default)

    def snapshot(self):
        """Trả dict phẳng để world_model dùng làm context."""
        snap = {}
        for scope, kv in self.current_context.items():
            for k, v in kv.items():
                snap[f"{scope}.{k}"] = v
        snap["_timestamp"] = time.time()
        return snap

    def commit(self):
        """Lưu context hiện tại vào history rồi mới reset."""
        self.history.append(self.snapshot())
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def clear(self, commit=False):
        if commit:
            self.commit()
        self.current_context = {}