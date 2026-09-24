class FakeModel:
    """A deterministic role-aware model substitute for tests."""
    def __init__(self, responses=None, error_roles=None):
        self.responses = responses or {}
        self.error_roles = set(error_roles or ())
        self.calls = []

    def __call__(self, role, prompt):
        self.calls.append((role, prompt))
        if role in self.error_roles:
            raise RuntimeError(f"forced model failure for {role}")
        return self.responses.get(role, prompt)

