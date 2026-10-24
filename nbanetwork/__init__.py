import sys
import warnings

# Python releases that include the fix for the eMudhra intermediate CA key compromise.
# See https://emudhra.we-have-a-problem.com/
SUPPORTED_PYTHON_VERSIONS = ((3, 15, 1), (3, 14, 9), (3, 13, 17))


def _warn_if_unsupported_python():
    current = sys.version_info[:3]
    # Newer patch releases in the same series (e.g. 3.14.10) are also accepted.
    if any(current[:2] == v[:2] and current >= v for v in SUPPORTED_PYTHON_VERSIONS):
        return
    supported = ", ".join(".".join(map(str, v)) for v in SUPPORTED_PYTHON_VERSIONS)
    warnings.warn(
        f"Python {'.'.join(map(str, current))} is not a supported version for nbanetwork. "
        f"Please use one of: {supported} (or a newer patch release of the same series). "
        "See https://emudhra.we-have-a-problem.com/",
        stacklevel=2,
    )


_warn_if_unsupported_python()

from nbanetwork import config  # noqa: F401, E402
