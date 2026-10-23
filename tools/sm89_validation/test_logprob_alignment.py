import unittest
from model_client import analyse_input_logprobs


class AlignmentTest(unittest.TestCase):
    def setUp(self):
        self.ids = list(range(200))
        self.entries = [[-0.1, i, None] for i in self.ids[136:]]

    def check(self, entries):
        return analyse_input_logprobs(self.ids, 136, entries)

    def test_full_numeric_window(self):
        self.assertTrue(self.check(self.entries)['passed'])

    def test_pinned_server_boundary_sentinel(self):
        self.entries[0][0] = None
        got = self.check(self.entries)
        self.assertTrue(got['passed'], got)
        self.assertEqual(got['positions'], list(range(137, 200)))
        self.assertEqual(len(got['logprobs']), 63)

    def test_shifted_numeric_window(self):
        self.assertTrue(self.check(self.entries[1:])['passed'])

    def test_interior_null_is_failure(self):
        self.entries[7][0] = None
        self.assertFalse(self.check(self.entries)['passed'])

    def test_wrong_token_is_failure(self):
        self.entries[7][1] = -1
        self.assertFalse(self.check(self.entries)['passed'])

    def test_nonfinite_is_failure(self):
        self.entries[7][0] = float('nan')
        self.assertFalse(self.check(self.entries)['passed'])

    def test_shifted_null_is_failure(self):
        self.entries[1][0] = None
        self.assertFalse(self.check(self.entries[1:])['passed'])


if __name__ == '__main__':
    unittest.main()
