"""CPU-only checks for reproducible histories and qualification failure detection."""
import copy
import hashlib
import importlib.util
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("qualify_cache", Path(__file__).with_name("qualify-cache.py"))
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


class Tokenizer:
    def tokenize(self, messages, model, thinking):
        data = json.dumps(messages, sort_keys=True).encode()
        return [int.from_bytes(data[i:i+4], "little") % 129280 for i in range(0,len(data),4)]


class HistoriesTest(unittest.TestCase):
    def test_same_seed_produces_same_history_and_tokens(self):
        a = q.make_history(Tokenizer(), 7812, 0, 4096, "model", True)
        b = q.make_history(Tokenizer(), 7812, 0, 4096, "model", True)
        self.assertEqual(a,b)
        self.assertLessEqual(abs(len(a["ids"])-4096),256)

    def test_other_seed_and_session_have_independent_facts(self):
        a = q.make_history(Tokenizer(), 1, 0, 4096, "model", True)
        for seed,index in ((2,0),(1,1)):
            b = q.make_history(Tokenizer(), seed, index, 4096, "model", True)
            self.assertNotEqual(a["expected"],b["expected"])
            self.assertNotEqual(a["sha256"],b["sha256"])

    def test_tool_update_supersedes_only_the_release(self):
        a = q.make_history(Tokenizer(), 1, 0, 4096, "model", True)
        b = q.extend_history(Tokenizer(), a, "model", True)
        self.assertNotEqual(a["expected"]["fixed_in"],b["expected"]["fixed_in"])
        for key in ("case","document","alert"):
            self.assertEqual(a["expected"][key],b["expected"][key])
        self.assertEqual(b["messages"][:len(a["messages"])],a["messages"])

    def test_stale_facts_are_rejected(self):
        r = {"response":{"text":'<think>checking</think>{"fixed_in":"old"}'},
             "meta":{"finish_reason":{"type":"stop"}}}
        with self.assertRaises(AssertionError):
            q.check_facts(r,{"fixed_in":"new"})
        q.check_facts(r,{"fixed_in":"old"})


class ArtifactsTest(unittest.TestCase):
    def test_diagnostics_allowlist_excludes_unknown_and_nested_configuration(self):
        info = {"context_length": 1048576, "api_key": "server-secret",
            "admin_api_key": "admin-secret", "future_credential": "future-secret",
            "dtype": {"secret": "nested-secret"},
            "server_args": {"max_running_requests": 4, "api_key": "nested-api-secret",
                "admin_api_key": "nested-admin-secret", "storage": {"password": "password-secret"}},
            "internal_states": [{"api_key": "worker-secret"}]}
        original = copy.deepcopy(info)
        self.assertEqual(q.server_diagnostics(info), {
            "context_length": 1048576, "server_args": {"max_running_requests": 4}})
        self.assertEqual(info, original)

    def test_prepare_artifacts_do_not_contain_client_or_server_credentials(self):
        secrets = ("client-secret", "server-secret", "admin-secret", "future-secret")
        info = {"context_length": 1048576, "max_running_requests": 4,
            "api_key": secrets[1], "admin_api_key": secrets[2], "future_credential": secrets[3]}
        item = {"ids": [1, 2, 3], "sha256": hashlib.sha256(json.dumps([1, 2, 3]).encode()).hexdigest()}
        fixtures = {"sessions": [item], "updated": item, "timed": [item], "pressure": item}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root/"fixtures.json", root/"output"
            source.write_text(json.dumps(fixtures))
            with patch("sys.argv", ["qualify-cache.py", "--prepare-only",
                    "--fixtures", str(source), "--output", str(output), "--api-key", secrets[0]]), \
                 patch.object(q.Client, "query", return_value=info):
                q.main()
            self.assertEqual(json.loads((output/"server.json").read_text()), {
                "context_length": 1048576, "max_running_requests": 4})
            for artifact in output.iterdir():
                for secret in secrets:
                    self.assertNotIn(secret, artifact.read_text(), artifact.name)


class StreamTest(unittest.TestCase):
    def stream(self, events):
        return io.BytesIO(b"".join(b"data: "+json.dumps(e).encode()+b"\n\n" for e in events)
                            +b"data: [DONE]\n\n")

    def run_stream(self, events, tokens=2, fixed=False):
        c = q.Client("http://localhost")
        with patch.object(c,"open",return_value=self.stream(events)), \
             patch.object(q.time,"monotonic",side_effect=itertools.count()):
            return c.generate([1,2],"salt",tokens,fixed)

    def test_decode_timing_excludes_time_before_first_token(self):
        r = self.run_stream([
            {"text":"a","meta_info":{"completion_tokens":1,"finish_reason":None}},
            {"text":"ab","meta_info":{"completion_tokens":2,"finish_reason":{"type":"length"}}}],fixed=True)
        self.assertEqual(r["ttft_s"],1)
        self.assertEqual(r["decode_tok_s"],1)
        self.assertEqual(r["completion_tokens"],2)

    def test_stream_error_is_not_a_successful_response(self):
        with self.assertRaises(RuntimeError):
            self.run_stream([{"error":{"message":"device failed"}}])

    def test_missing_finish_reason_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError,"finish reason"):
            self.run_stream([{"meta_info":{"completion_tokens":1}}])

    def test_short_fixed_length_generation_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError,"Incomplete"):
            self.run_stream([{"meta_info":{"completion_tokens":1,"finish_reason":{"type":"stop"}}}],fixed=True)

    def test_retracted_request_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError,"retracted"):
            self.run_stream([{"meta_info":{"completion_tokens":1,"num_retractions":1,
                              "finish_reason":{"type":"stop"}}}])


if __name__ == "__main__":
    unittest.main()
