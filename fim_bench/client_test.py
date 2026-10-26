"""Tests of the agent server HTTP client."""

import http.client
import unittest
from unittest import mock

from fim_bench import client


class WaitReadyCase(unittest.TestCase):
    """Tests of the readiness polling."""

    def test_tolerates_a_server_that_closes_connections_while_starting(self):
        """A starting uvicorn accepts TCP and closes without a status (BadStatusLine)."""
        responses = [
            http.client.BadStatusLine("GET /v1/models HTTP/1.1"),
            http.client.NotConnected(),
            mock.mock_open(read_data=b'{"data": [{"id": "m"}]}').return_value.__enter__.return_value,
        ]
        responses[-1].status = 200

        def urlopen(url, timeout=5.0):
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        with mock.patch.object(client.urllib.request, "urlopen", urlopen), \
                mock.patch.object(client.time, "sleep"):
            self.assertTrue(client.wait_ready(1234, timeout_s=600.0))

    def test_gives_up_after_the_timeout(self):
        with mock.patch.object(client.urllib.request, "urlopen",
                               side_effect=http.client.BadStatusLine("x")), \
                mock.patch.object(client.time, "sleep"):
            self.assertFalse(client.wait_ready(1234, timeout_s=0.0))


class GetModelIdsCase(unittest.TestCase):
    """Tests of the model list probing."""

    def test_silent_on_bad_status_line(self):
        with mock.patch.object(client.urllib.request, "urlopen",
                               side_effect=http.client.BadStatusLine("GET /v1/models HTTP/1.1")):
            self.assertEqual([], client.get_model_ids(1234))


if __name__ == "__main__":
    unittest.main()
