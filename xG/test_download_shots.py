"""Focused checks for the requested exception boundary and shot extraction."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from download_shots import fetch, shot_rows


class ExportTests(unittest.TestCase):
    def test_download_failure_logs_url_then_next_request_succeeds(self):
        responses = [subprocess.CompletedProcess([], 22, b"", b"HTTP 503"),
                     subprocess.CompletedProcess([], 0, b'{"ok": true}', b"")]
        with tempfile.TemporaryDirectory() as directory, patch("download_shots.time.sleep"), \
                patch("download_shots.subprocess.run", side_effect=responses), \
                patch("download_shots.sys.stderr") as stderr:
            failures = []
            cache = Path(directory) / "game.gz"
            self.assertIsNone(fetch("https://example.com/fail", cache, failures))
            self.assertFalse(cache.exists())
            self.assertEqual(failures[0]["url"], "https://example.com/fail")
            self.assertIn("HTTP 503", failures[0]["exception"])
            self.assertTrue(stderr.write.called)
            self.assertEqual(fetch("https://example.com/ok", cache, failures), {"ok": True})

    def test_invalid_json_and_local_curl_errors_propagate(self):
        with tempfile.TemporaryDirectory() as directory, patch("download_shots.time.sleep"):
            cache = Path(directory) / "game.gz"
            for response, error in [
                (subprocess.CompletedProcess([], 0, b"not JSON", b""), json.JSONDecodeError),
                (subprocess.CompletedProcess([], 23, b"", b"write error"), subprocess.CalledProcessError),
            ]:
                with patch("download_shots.subprocess.run", return_value=response):
                    with self.assertRaises(error):
                        fetch("https://example.com", cache, [])

    def test_goal_shootout_and_missing_values(self):
        game = {"id": 1, "season": 20252026, "gameType": 2, "gameDate": "2025-10-07",
                "gameState": "OFF", "homeTeam": {"id": 1, "abbrev": "AAA"},
                "awayTeam": {"id": 2, "abbrev": "BBB"}, "plays": [
                    {"eventId": 3, "typeDescKey": "goal", "sortOrder": 1,
                     "periodDescriptor": {"number": 5, "periodType": "SO"},
                     "situationCode": "0101", "details": {"scoringPlayerId": 42, "xCoord": 0}},
                    {"typeDescKey": "stoppage"}]}
        row, = shot_rows(game)
        self.assertEqual(row["shooter_id"], 42)
        self.assertEqual(row["is_shootout"], 1)
        self.assertEqual(row["situationCode"], "0101")
        self.assertEqual(row["x_coord"], 0)
        self.assertIsNone(row["y_coord"])
        self.assertIsNone(row["goalie_id"])
        self.assertIsNone(row["shooter_name"])


if __name__ == "__main__":
    unittest.main()
