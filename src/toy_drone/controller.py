class Controller():
    # TODO: docstrings
    def __init__(self, feedback_law, feedforward_law, reference_size):
        self._feedback_law = feedback_law
        self._feedforward_law = feedforward_law
        self._reference_size = reference_size

    def compute_controls(self, estimate, reference):
        return self._feedback_law(estimate, reference) + self._feedforward_law()

    def get_reference_size(self):
        return self._reference_size
