class LeadOptimizerRequestError(ValueError):
    """A user-correctable request or input problem that should be returned."""

    def __init__(
        self,
        message: str,
        status: str = "unsupported",
        metadata: dict | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.metadata = metadata or {}
