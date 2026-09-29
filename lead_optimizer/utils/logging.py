"""Logging helpers for lead optimizer runtime traces."""

from functools import wraps
import logging
import sys

from ..errors import LeadOptimizerRequestError


LOGGER_NAME = "LeadOptimizer"
CONSOLE_LOG_FORMAT = "%(asctime)s [LeadOptimizer] [%(levelname)s] %(message)s"
DATE_FORMAT = "%m-%d %H:%M:%S"


def _console_formatter() -> logging.Formatter:
    return logging.Formatter(fmt=CONSOLE_LOG_FORMAT, datefmt=DATE_FORMAT)


class _MaxLevelFilter(logging.Filter):
    def __init__(self, max_level: int):
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level


def get_logger() -> logging.Logger:
    """Return the shared logger, split between stdout and stderr by severity."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    has_stdout = any(
        getattr(handler, "_lead_optimizer_stdout", False)
        for handler in logger.handlers
    )
    if not has_stdout:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.addFilter(_MaxLevelFilter(logging.INFO))
        handler.setFormatter(_console_formatter())
        handler._lead_optimizer_stdout = True
        logger.addHandler(handler)

    has_stderr = any(
        getattr(handler, "_lead_optimizer_stderr", False)
        for handler in logger.handlers
    )
    if not has_stderr:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.WARNING)
        handler.setFormatter(_console_formatter())
        handler._lead_optimizer_stderr = True
        logger.addHandler(handler)
    return logger


def log_run_exceptions(func):
    """Return user-correctable failures and log unexpected failures."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except LeadOptimizerRequestError as exc:
            logger = get_logger()
            logger.warning("'LeadOptimizer' request not executed: %s", exc)
            agent = args[0] if args else None
            tool_chest = getattr(agent, "tool_chest", None)
            planner = getattr(agent, "planner", None)
            responses = (
                agent._collect_llm_responses()
                if hasattr(agent, "_collect_llm_responses")
                else []
            )
            metadata = {
                "tool": None,
                "available_tools": (
                    tool_chest.available() if tool_chest is not None else []
                ),
                "planner": (
                    getattr(planner, "name", planner.__class__.__name__)
                    if planner is not None
                    else None
                ),
                "plan": None,
                "status": exc.status,
                "stopping_reason": "request_not_executed",
                "stopping_reason_message": str(exc),
                "target_status": exc.status,
                "validation_status": "not_run",
                "property_scored": False,
                "property_target_evaluated": False,
                "target_hits": None,
                "scored_output_count": 0,
                "partial_output_note": None,
                "action_notes": [str(exc)],
                "partial_output": False,
                "input_type": "csv",
                "molecule_count": 0,
                "row_count": 0,
                "failed_input_ids": [],
                "message": str(exc),
                "error_category": "request_error",
                "input_warnings": [],
                "skipped_input_rows": 0,
                "request_schema": "OptimizationRequest",
                "outputs": {"csv": None, "debug_csvs": [], "log": None},
                "responses": responses,
            }
            metadata.update(getattr(exc, "metadata", {}) or {})
            return str(exc), metadata
        except Exception as exc:
            logger = get_logger()
            logger.error("'LeadOptimizer' run failed: %s", exc)
            logger.debug("run failure traceback:", exc_info=True)
            agent = args[0] if args else None
            tool_chest = getattr(agent, "tool_chest", None)
            planner = getattr(agent, "planner", None)
            responses = (
                agent._collect_llm_responses()
                if hasattr(agent, "_collect_llm_responses")
                else []
            )
            message = (
                "LeadOptimizer failed unexpectedly before it could complete "
                "the request. No output CSV was confirmed."
            )
            metadata = {
                "tool": None,
                "available_tools": (
                    tool_chest.available() if tool_chest is not None else []
                ),
                "planner": (
                    getattr(planner, "name", planner.__class__.__name__)
                    if planner is not None
                    else None
                ),
                "plan": None,
                "status": "failed",
                "stopping_reason": "unexpected_error",
                "stopping_reason_message": str(exc),
                "target_status": "failed",
                "validation_status": "not_run",
                "property_scored": False,
                "property_target_evaluated": False,
                "target_hits": None,
                "scored_output_count": 0,
                "partial_output_note": None,
                "action_notes": [message],
                "partial_output": False,
                "input_type": "csv",
                "molecule_count": 0,
                "row_count": 0,
                "failed_input_ids": [],
                "message": message,
                "error_category": "unexpected_error",
                "error_type": exc.__class__.__name__,
                "error": str(exc),
                "input_warnings": [],
                "skipped_input_rows": 0,
                "request_schema": "OptimizationRequest",
                "outputs": {"csv": None, "debug_csvs": [], "log": None},
                "responses": responses,
            }
            return message, metadata

    return wrapper
