class Controller():
    # TODO: docstrings
    def __init__(self, feedback_law, feedforward_law):
        self._feedback_law = feedback_law
        self._feedforward_law = feedforward_law

    def compute_controls(self, error):
        return self._feedback_law(error) + self._feedforward_law()
