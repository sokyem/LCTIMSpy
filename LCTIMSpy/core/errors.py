"""Exception hierarchy. Every error LCTIMSpy raises on purpose is an LCTIMSError."""


class LCTIMSError(Exception):
    """Base class for all LCTIMSpy errors."""


class SdkUnavailableError(LCTIMSError):
    """The Bruker TDF-SDK (pyTDFSDK + timsdata library) cannot be loaded."""

    def __init__(self, message: str, remediation: str | None = None):
        super().__init__(message)
        self.remediation = remediation


class DatasetError(LCTIMSError):
    """A path is not a readable Bruker timsTOF ``.d`` folder."""


class CacheError(LCTIMSError):
    """The extraction cache is missing, corrupt, or cannot be written."""


class CancelledError(LCTIMSError):
    """A long-running operation was cancelled through its CancelToken."""
