import logging


def _configure_quiet_http_logs():
	"""Reduce third-party HTTP client log noise for pageindex workflows."""
	for logger_name in ("httpx", "httpcore", "openai"):
		logging.getLogger(logger_name).setLevel(logging.WARNING)


_configure_quiet_http_logs()

from .page_index import *
from .page_index_txt import txt_to_tree