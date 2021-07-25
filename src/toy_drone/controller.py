class Controller():
    """
    Controller Interface class for combining feedback with feedforward laws.
    """

    def __init__(self, feedback_law, feedforward_law, reference_size=1):
        """
        __init__ Constructor.

        Args:
            feedback_law: Feedback law as a function of state and reference
            feedforward_law: Feedforward law as a function with no inputs (will be expanded)
            reference_size: Size of the reference that is used in the feedback law
        """
        self._feedback_law = feedback_law
        self._feedforward_law = feedforward_law
        self._reference_size = reference_size

    def compute_controls(self, state, reference):
        """
        compute_controls Compute the controls based on feedback and feedforward laws.

        Args:
            state: State at which control is computed
            reference: Reference for error computation

        Returns:
            Control
        """
        return self._feedback_law(state, reference) + self._feedforward_law()

    def get_reference_size(self):
        return self._reference_size
