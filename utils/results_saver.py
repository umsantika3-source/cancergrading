class ResultsSaver:
    """In-memory accumulator for metrics across all experiments."""

    def __init__(self):
        self._results = []

    def add(self, eval_dict):
        self._results.append(eval_dict)

    def all(self):
        return list(self._results)
