"""Tests for DropClientResetFilter: suppression of harmless asyncio noise."""
import logging
import tempfile
import unittest

from agent.common.log import DropClientResetFilter, logging_config

RESET_MESSAGE = "Exception in callback _ProactorBasePipeTransport._call_connection_lost()"


def make_record(logger_name="asyncio", message=RESET_MESSAGE, error=None):
    exc_info = (type(error), error, error.__traceback__) if error is not None else None
    return logging.LogRecord(name=logger_name, level=logging.ERROR, pathname="agent/server_openvino.py",
                             lineno=1, msg=message, args=(), exc_info=exc_info)


def make_reset_error(winerror=10054):
    error = ConnectionResetError(10054, "Удаленный хост принудительно разорвал существующее подключение")
    error.winerror = winerror  # not present on non-Windows builds; set for determinism
    return error


class DropClientResetFilterCase(unittest.TestCase):
    def setUp(self):
        self.log_filter = DropClientResetFilter()

    def test_drops_proactor_reset(self):
        record = make_record(error=make_reset_error())
        self.assertFalse(self.log_filter.filter(record))

    def test_keeps_records_of_other_loggers(self):
        record = make_record(logger_name="agent.server_openvino", error=make_reset_error())
        self.assertTrue(self.log_filter.filter(record))

    def test_keeps_other_exceptions(self):
        record = make_record(error=TimeoutError("timed out"))
        self.assertTrue(self.log_filter.filter(record))

    def test_keeps_reset_with_other_winerror(self):
        record = make_record(error=make_reset_error(winerror=None))
        self.assertTrue(self.log_filter.filter(record))

    def test_keeps_reset_from_other_callbacks(self):
        record = make_record(message="Exception in callback Future-1<finished>", error=make_reset_error())
        self.assertTrue(self.log_filter.filter(record))

    def test_keeps_records_without_exc_info(self):
        record = make_record()
        self.assertTrue(self.log_filter.filter(record))


class LoggingConfigCase(unittest.TestCase):
    def test_filter_is_attached_to_asyncio_logger(self):
        with tempfile.TemporaryDirectory() as logs_dir:
            config = logging_config(logs_dir)
        self.assertIn("drop_client_reset", config["filters"])
        self.assertEqual(["drop_client_reset"], config["loggers"]["asyncio"]["filters"])


if __name__ == '__main__':
    unittest.main()
