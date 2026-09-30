import logging


def _rank_zero_only(fn):
    """Fallback decorator when pytorch_lightning is not available.
    In single-GPU inference this is a no-op wrapper."""
    try:
        from pytorch_lightning.utilities import rank_zero_only
        return rank_zero_only(fn)
    except ImportError:
        return fn


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""
    logger = logging.getLogger(name)

    logging_levels = (
        "debug", "info", "warning", "error",
        "exception", "fatal", "critical",
    )
    for level in logging_levels:
        setattr(logger, level, _rank_zero_only(getattr(logger, level)))

    return logger
