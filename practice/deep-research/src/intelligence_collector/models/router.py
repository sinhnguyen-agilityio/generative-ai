class ModelRouter:
    def __init__(self, primary, fallback): self.primary, self.fallback = primary, fallback
    def invoke(self, role, prompt):
        try: return self.primary(role, prompt)
        except Exception: return self.fallback(role, prompt)

