# MemSys Core - Modern Memory System

class MemSys:
    def __init__(self, max_short_term=50, max_long_term=1000):
        self.short_term = []
        self.long_term = []
        self.episodic = []
        self.max_short_term = max_short_term
        self.max_long_term = max_long_term

    def remember(self, item, memory_type="short"):
        if memory_type == "short":
            self.short_term.append(item)
            if len(self.short_term) > self.max_short_term:
                self._consolidate_to_long_term()
        elif memory_type == "long":
            self.long_term.append(item)
        elif memory_type == "episodic":
            self.episodic.append(item)

    def _consolidate_to_long_term(self):
        if self.short_term:
            item = self.short_term.pop(0)
            self.long_term.append(item)
            if len(self.long_term) > self.max_long_term:
                self.long_term.pop(0)

    def recall(self, query=None, memory_type="all"):
        if memory_type == "short":
            return self.short_term[-10:]
        elif memory_type == "long":
            return self.long_term[-20:]
        return self.short_term[-5:] + self.long_term[-10:]

    def clear_short_term(self):
        self.short_term = []