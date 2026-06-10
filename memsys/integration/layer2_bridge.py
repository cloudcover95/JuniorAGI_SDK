# MemSys Integration with Layer 2 Contextual Brain

class MemSysLayer2Bridge:
    def __init__(self, memsys, layer2_brain):
        self.memsys = memsys
        self.brain = layer2_brain

    def remember_with_context(self, item):
        self.memsys.remember(item, "short")
        self.brain.add_context(item)

    def recall_for_generation(self, query):
        context = self.memsys.recall(query)
        return self.brain.generate_with_context(query)