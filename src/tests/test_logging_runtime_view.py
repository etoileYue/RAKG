import importlib
import logging
import logging.handlers
import os
import unittest
import uuid
from pathlib import Path
from unittest import mock


class LoggingRuntimeViewTests(unittest.TestCase):
    def test_debug_logger_defaults_to_file_only(self):
        debug_file = f"debug-runtime-{uuid.uuid4().hex}.log"
        debug_path = Path("logs") / debug_file
        self.addCleanup(lambda: debug_path.exists() and debug_path.unlink())
        if debug_path.exists():
            debug_path.unlink()

        with mock.patch.dict(
            os.environ,
            {
                "RAKG_LOGGER_NAME": f"AgentLog.runtime.{uuid.uuid4().hex}",
                "RAKG_DEBUG_FILE": debug_file,
            },
            clear=False,
        ):
            import src.pipeline.shared as shared_module

            shared_module = importlib.reload(shared_module)
            handlers = shared_module.debug_logger.get_logger().handlers
            self.assertFalse(any(type(handler) is logging.StreamHandler for handler in handlers))
            self.assertTrue(
                any(isinstance(handler, logging.handlers.RotatingFileHandler) for handler in handlers)
            )

            shared_module.debug_logger.debug("debug-file-only-message")

        self.assertTrue(debug_path.exists())
        self.assertIn("debug-file-only-message", debug_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
