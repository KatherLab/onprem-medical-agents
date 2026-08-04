import logging


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
THIRD_PARTY_LOGGERS = (
    "litellm",
    "LiteLLM",
    "openai",
    "httpx",
    "httpcore",
)


def setup_logging(
    log_level: int | str = logging.INFO,
    third_party_level: int | str = logging.WARNING,
) -> logging.Logger:
    """Configure root logging once and keep noisy dependency loggers quiet."""
    logging.basicConfig(level=log_level, format=LOG_FORMAT, force=True)

    for name in THIRD_PARTY_LOGGERS:
        lib_logger = logging.getLogger(name)
        lib_logger.handlers.clear()
        lib_logger.setLevel(third_party_level)
        lib_logger.propagate = True

    return logging.getLogger(__name__)
