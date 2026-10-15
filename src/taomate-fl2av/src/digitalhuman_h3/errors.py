class H3Error(RuntimeError):
    """Base error for project failures."""


class UnresolvedOfficialConfigError(H3Error):
    """Raised when execution needs a value not yet resolved from official files."""


class ProvenanceError(H3Error):
    """Raised when model or data provenance is incomplete or inconsistent."""


class SchemaValidationError(H3Error):
    """Raised when a portable artifact violates its schema contract."""
