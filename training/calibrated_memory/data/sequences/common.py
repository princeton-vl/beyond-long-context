from __future__ import annotations

IGNORE_INDEX = -100

# Minimal control tokens for binary tasks.
# STREAM_QUERY_SEPARATOR marks the boundary between the encoded stream prefix
# and each subsequent query, LABEL_SEPARATOR precedes the ground-truth label,
# and QUERY_END_SEPARATOR delimits individual queries.
STREAM_QUERY_SEPARATOR = 0
LABEL_SEPARATOR = 1
QUERY_END_SEPARATOR = 2
CANDIDATE_SEPARATOR = 3
YES_TOKEN = 4
NO_TOKEN = 5
UNCERTAIN_TOKEN = 6

LABEL_TOKENS = (YES_TOKEN, NO_TOKEN, UNCERTAIN_TOKEN)

# Stream vocab must start above the reserved control ids.
TOKEN_OFFSET = UNCERTAIN_TOKEN + 1


def validate_stream_token_offset(value: int) -> int:
    """Ensure dataset builders keep stream vocab above reserved ids."""

    if value < TOKEN_OFFSET:
        raise ValueError(
            f"token_offset must be >= {TOKEN_OFFSET} to avoid control-token collisions; got {value}"
        )
    return value
