#!/usr/bin/env python3
"""Standalone full-model validation client for the SM89 PR3 server (stdlib only).

Run once per arm against an otherwise identical server:

    python3 model_client.py --arm portable --output OUT_PORTABLE
    python3 model_client.py --arm dual     --output OUT_DUAL
    python3 model_client.py --compare OUT_PORTABLE OUT_DUAL

Every HTTP exchange is saved under <output>/exchanges/, report.json is rewritten
after every check, and the exit status is nonzero if any check failed.  There are
no sessions, no streaming and no skipped checks: an exception is a failed check.
See CLIENT_NOTES.md for scope and the input-logprob alignment rules.
"""
import argparse
import dataclasses
import http.client
import json
import math
import re
import statistics
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_BASE_URL = 'http://127.0.0.1:18209'
DEFAULT_MODEL = 'sm89-pr3'
MAX_TIMEOUT_S = 600
MAX_RESPONSE_BYTES = 256 << 20
SEED = 20261023
CHECK_MAX_TOKENS = 96
THROUGHPUT_TOKENS = 128
LOGPROB_WINDOW = 64
LOGPROB_POSITIVE_EPSILON = 1e-3
MAX_FILLER_TOKENS = 60000
RETRIEVAL_TOLERANCE = 0.10
RETRIEVAL_ADJUST_TOLERANCE = 0.05
RETRIEVAL_ATTEMPTS = 3
ESTIMATED_CHARS_PER_TOKEN = 4.8

# Settings the task statement fixes for both arms.  Everything else in
# /server_info (attention backend, expert placement, ...) is only recorded.
EXPECTED_SERVER = dict(
    disable_cuda_graph=False,
    speculative_algorithm=None,
    context_length=65536,
    max_running_requests=4,
)

JSON_SYSTEM = ('You are a precise assistant. Reply with one JSON object only, without markdown '
               'and without any explanation.')

# Non-ASCII text is written with escapes so this file stays plain ASCII.
GRUESSE = 'Grüße, 東京, Привет'  # Grusse, Tokyo (kanji), Privet (Cyrillic)
CAFE = 'café ñandú — \U0001F642'  # accents, em dash, emoji outside the BMP
TOKYO_KANJI = '東京'

# (label, user prompt, exact expected JSON object)
BASIC_CASES = [
    ('fact-capital', 'What is the capital city of France? Reply with {"capital": "<city name>"}.',
     dict(capital='Paris')),
    ('fact-boiling-point',
     'At standard atmospheric pressure, at what temperature in degrees Celsius does pure water boil? '
     'Reply with {"celsius": <integer, not a string>}.',
     dict(celsius=100)),
    ('fact-gold-symbol',
     'What is the chemical element symbol for gold? Reply with {"symbol": "<symbol>"}.',
     dict(symbol='Au')),
    ('json-flat-types',
     'Return a JSON object with exactly three keys: "colors" is the list ["red", "green", "blue"] in that '
     'order, "count" is the integer 3, and "primary" is the boolean true.',
     dict(colors=['red', 'green', 'blue'], count=3, primary=True)),
    ('json-nested',
     'Return a JSON object with exactly two keys: "user" is an object with "name" equal to the string "Ada" '
     'and "langs" equal to the list ["en", "fr"]; "active" is the boolean false.',
     dict(user=dict(name='Ada', langs=['en', 'fr']), active=False)),
    ('arithmetic-expression',
     'Compute 347 * 29 + 18. Reply with {"result": <integer, not a string>}.',
     dict(result=10081)),
    ('arithmetic-parentheses',
     'Compute (1234 + 5678) * 3 - 17. Reply with {"result": <integer, not a string>}.',
     dict(result=20719)),
    ('arithmetic-word-problem',
     'A shop sells pens at 3 pens for 4 dollars. How many dollars do 18 pens cost? '
     'Reply with {"dollars": <integer, not a string>}.',
     dict(dollars=24)),
    ('unicode-copy-multiscript',
     'Copy this text exactly, character for character, into the "text" field of a JSON object: ' + GRUESSE,
     dict(text=GRUESSE)),
    ('unicode-copy-emoji',
     'Copy this text exactly, character for character, into the "text" field of a JSON object: ' + CAFE,
     dict(text=CAFE)),
    ('unicode-romanize',
     'The Japanese city name ' + TOKYO_KANJI + ' is written in English with Latin letters as one word. '
     'Reply with {"city": "<that word>"}.',
     dict(city='Tokyo')),
]

OWNERS = ['Mira', 'Oren', 'Lina', 'Tavi']
COHORTS = (1, 2, 4)

TOOL = dict(type='function', function=dict(
    name='lookup_inventory',
    description='Look up available stock at a warehouse.',
    parameters=dict(
        type='object',
        properties=dict(sku=dict(type='string'), warehouse=dict(type='string')),
        required=['sku', 'warehouse'],
        additionalProperties=False)))
TOOL_SLOTS = [('SKU-527', 'north'), ('SKU-690', 'south'), ('SKU-853', 'east'), ('SKU-1016', 'west')]

# (label, target prompt tokens, ledger name, unique code)
RETRIEVAL_CASES = [
    ('retrieval-4k', 4000, 'HERON', 'QX7-4081-MRV'),
    ('retrieval-16k', 16000, 'CINDER', 'LV3-9526-TNB'),
    ('retrieval-35k', 35000, 'LARKSPUR', 'HD8-2674-WPC'),
]

PROBE_TEXTS = dict(
    natural_prose=(
        'The old lighthouse at the edge of the bay had been dark for nearly a decade when the village '
        'council finally voted to restore it. Volunteers arrived every Saturday with paint, rope and '
        'thermoses of tea, and by the end of the first month the iron railing along the gallery was '
        'no longer red with rust. A retired engineer named Halvard studied the original drawings and '
        'discovered that the great lens had been assembled from more than three hundred separate prisms, '
        'each of which had to be cleaned by hand. Children from the school next door were allowed to '
        'watch, provided they stayed behind the yellow line and promised not to touch the glass. When '
        'the lamp was lit again on a clear autumn evening, fishermen far out on the water said that '
        'the beam looked softer than they remembered, and that it made the whole coast seem a little '
        'closer to home. Nobody could say exactly why, but the council agreed the effort had been worth it.'),
    code_json_unicode=(
        'Review the following snippet and the data record below.\n'
        'def moving_average(values, window):\n'
        '    total = sum(values[:window])\n'
        '    result = [total / window]\n'
        '    for i in range(window, len(values)):\n'
        '        total += values[i] - values[i - window]\n'
        '        result.append(total / window)\n'
        '    return result\n\n'
        '{"id": 48213, "tags": ["alpha", "beta"], "ratio": 0.3125, "note": "Grüße aus München, '
        '東京へようこそ, Привет, café"}\n\n'
        'The record lists 3 widgets at 12.50 each and 7 gadgets at 4.25 each, so the total is 67.25.'),
)

THROUGHPUT_PROMPTS = [
    'The history of the lighthouse on the northern coast begins',
    'In a small workshop at the edge of the city, an engineer',
    'A practical guide to growing tomatoes on a balcony starts with',
    'The river valley was quiet that morning, and the first traveler',
]


# ---------------------------------------------------------------- JSON helpers

def jsonable(value):
    """Make a value strictly JSON-serialisable (NaN/Infinity become strings)."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                   encoding='utf-8')
    tmp.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


FENCE = re.compile(r'^```[A-Za-z0-9_-]*\s*\n(.*?)\n?```\s*$', re.S)


def parse_json_object(text):
    """First JSON object in an answer; tolerates a markdown fence. Returns (object, extra_text)."""
    body = text.strip()
    fenced = FENCE.match(body)
    if fenced:
        body = fenced.group(1).strip()
    start = body.find('{')
    if start < 0:
        raise ValueError('no JSON object in answer')
    value, end = json.JSONDecoder().raw_decode(body, start)
    return value, (body[:start] + body[end:]).strip()


def same_json(actual, expected):
    """Semantic equality: key order and whitespace are irrelevant, JSON types are exact."""
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and actual.keys() == expected.keys()
                and all(same_json(actual[k], v) for k, v in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(same_json(a, e) for a, e in zip(actual, expected)))
    return type(actual) is type(expected) and actual == expected


def judge_json(text, expected):
    verdict = dict(expected=expected, answer_text=text)
    try:
        actual, extra = parse_json_object(text)
    except ValueError as exc:
        return dict(verdict, passed=False, error=f'answer is not a JSON object: {exc}')
    return dict(verdict, passed=same_json(actual, expected), actual=actual, extra_text=extra)


def message_text(choice):
    return (choice.get('message') or {}).get('content') or ''


# ----------------------------------------------------------------- HTTP client

class RequestError(Exception):
    pass


@dataclasses.dataclass
class Exchange:
    label: str
    file: str
    status: int
    wall_s: float
    value: object


class Client:
    """One fresh HTTP connection per request; every exchange is written to disk."""

    def __init__(self, base_url, timeout_s, exchange_dir):
        parts = urlsplit(base_url)
        if parts.scheme not in ('http', 'https') or not parts.hostname:
            raise ValueError(f'unsupported base URL: {base_url}')
        self.https = parts.scheme == 'https'
        self.host = parts.hostname
        self.port = parts.port or (443 if self.https else 80)
        self.prefix = parts.path.rstrip('/')
        self.timeout_s = timeout_s
        self.exchange_dir = exchange_dir
        self.lock = threading.Lock()
        self.sequence = 0

    def connect(self):
        cls = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=self.timeout_s)

    def call(self, label, route, body=None):
        """GET when body is None, otherwise POST JSON. Raises RequestError on any non-200 outcome."""
        with self.lock:
            self.sequence += 1
            number = self.sequence
        file = f'{number:04d}-' + re.sub(r'[^A-Za-z0-9_.-]+', '_', label) + '.json'
        record = dict(label=label, method='GET' if body is None else 'POST', route=route,
                      request=body, utc=utc_now())
        status = text = error = None
        start = time.monotonic()
        connection = self.connect()
        try:
            payload = None if body is None else json.dumps(body).encode('utf-8')
            connection.request(record['method'], self.prefix + route, payload,
                               {'Content-Type': 'application/json'})
            response = connection.getresponse()
            status = response.status
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise RequestError('response exceeds size limit')
            text = raw.decode('utf-8', errors='replace')
        except Exception as exc:  # timeouts, resets and refusals are recorded, then re-raised below
            error = f'{type(exc).__name__}: {exc}'
        finally:
            connection.close()
        wall_s = time.monotonic() - start
        record.update(status=status, wall_s=wall_s, error=error, response_text=text)
        write_json(self.exchange_dir / file, record)
        if error:
            raise RequestError(f'{label}: {error} (exchange {file})')
        if status != 200:
            raise RequestError(f'{label}: HTTP {status}: {text[:300]} (exchange {file})')
        try:
            value = json.loads(text)
        except ValueError as exc:
            raise RequestError(f'{label}: response is not JSON: {exc} (exchange {file})')
        return Exchange(label, file, status, wall_s, value)


# -------------------------------------------------------------------- reporting

class Report:
    def __init__(self, path, header):
        self.path = path
        self.lock = threading.RLock()
        self.data = dict(header, status='RUNNING', total_checks=0, passed_checks=0, failed_checks=0,
                         checks=[], throughput=[])
        self.save()

    def add_check(self, label, passed, **details):
        with self.lock:
            self.data['checks'].append(dict(label=label, passed=bool(passed), details=details))
            checks = self.data['checks']
            self.data.update(total_checks=len(checks),
                             passed_checks=sum(c['passed'] for c in checks),
                             failed_checks=sum(not c['passed'] for c in checks))
            self.save()

    def save(self):
        with self.lock:
            write_json(self.path, self.data)

    def failed_labels(self):
        with self.lock:
            return [c['label'] for c in self.data['checks'] if not c['passed']]


def guarded(report, label, fn):
    """Run fn() -> dict with a 'passed' key and record it; any exception is a failed check."""
    try:
        details = dict(fn())
        passed = bool(details.pop('passed'))
    except Exception as exc:
        passed = False
        details = dict(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    report.add_check(label, passed, **details)
    return passed


# ----------------------------------------------------------- input-logprob logic

def analyse_input_logprobs(ids, start, entries):
    """Validate meta_info.input_token_logprobs and align it to the supplied IDs.

    Entries are [logprob, token_id, text]. For logprob_start_len=start the server reports the
    logprob of token ids[o] under the prefix ids[:o] for consecutive positions, with o equal to
    either `start` or `start + 1` depending on the SGLang version, and possibly one trailing entry
    beyond the last input token. The offset is therefore detected from the token IDs that the
    server echoes back and must match ids exactly; nothing is assumed.
    """
    problems = []
    if not isinstance(entries, list) or not entries:
        return dict(passed=False, problems=['input_token_logprobs missing or empty'])
    logprobs, token_ids = [], []
    for k, entry in enumerate(entries):
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            problems.append(f'entry {k} is not [logprob, token_id, ...]: {entry!r}')
            continue
        logprob, token = entry[0], entry[1]
        if k == 0 and logprob is None:
            pass  # pinned SGLang inserts an unscored boundary token at logprob_start_len
        elif isinstance(logprob, bool) or not isinstance(logprob, (int, float)) or not math.isfinite(logprob):
            problems.append(f'entry {k} logprob is not a finite number: {logprob!r}')
        elif logprob > LOGPROB_POSITIVE_EPSILON:
            problems.append(f'entry {k} logprob is positive: {logprob!r}')
        if type(token) is not int:
            problems.append(f'entry {k} token id is not an integer: {token!r}')
        logprobs.append(logprob)
        token_ids.append(token)

    offset = trailing = None
    if len(token_ids) == len(entries):
        for candidate in (start, start + 1):
            covered = len(ids) - candidate
            extra = len(token_ids) - covered
            if covered > 0 and 0 <= extra <= 1 and token_ids[:covered] == ids[candidate:]:
                offset, trailing = candidate, extra
                break
    if offset is None:
        problems.append(f'echoed token ids do not match supplied ids at offset {start} or {start + 1}')
        covered = 0
    else:
        covered = len(ids) - offset
        boundary_sentinel = logprobs[0] is None
        if boundary_sentinel:
            if offset != start:
                problems.append('unscored boundary token is not at logprob_start_len')
            offset += 1
            covered -= 1
            logprobs = logprobs[1:]
            token_ids = token_ids[1:]
        if covered < LOGPROB_WINDOW - 1:
            problems.append(f'only {covered} aligned positions, expected at least {LOGPROB_WINDOW - 1}')
    return dict(passed=not problems, problems=problems, entry_count=len(entries),
                aligned_offset=offset, trailing_entries=trailing,
                positions=list(range(offset, offset + covered)) if offset is not None else [],
                logprobs=logprobs[:covered], token_ids=token_ids[:covered])


# --------------------------------------------------------- retrieval text filler

class Lcg:
    def __init__(self, seed):
        self.state = (seed * 2654435761 + 12345) & 0xFFFFFFFF

    def pick(self, items):
        self.state = (self.state * 1664525 + 1013904223) & 0xFFFFFFFF
        return items[(self.state >> 8) % len(items)]


SUBJECTS = ['The harbor keeper', 'A traveling botanist', 'The village baker', 'An old cartographer',
            'The night ferry captain', 'A young apprentice', 'The orchard manager', 'A retired schoolteacher']
ACTIONS = ['recorded the weather over the', 'repaired the wooden fence beside the', 'walked slowly along the',
           'described the quiet sound of the', 'counted the lanterns hanging above the',
           'painted a careful sketch of the']
PLACES = ['river bridge', 'salt marsh', 'market square', 'hill pasture', 'stone chapel', 'lighthouse stairs',
          'winter garden', 'clay kiln']
ENDINGS = ['before the morning bell rang.', 'while the fog lifted from the valley.',
           'and then shared bread with the neighbors.', 'as the first swallows returned.',
           'and noted the tide in a small green book.', 'despite a stubborn wind from the north.']


def filler_paragraph(index):
    """Deterministic digit-free prose; the same index always yields the same paragraph."""
    rng = Lcg(index)
    sentences = [f'{rng.pick(SUBJECTS)} {rng.pick(ACTIONS)} {rng.pick(PLACES)} {rng.pick(ENDINGS)}'
                 for _ in range(6)]
    return ' '.join(sentences)


def build_document(target_chars, needle):
    paragraphs, total, index = [], 0, 0
    while total < target_chars or len(paragraphs) < 2:
        paragraph = filler_paragraph(index)
        paragraphs.append(paragraph)
        total += len(paragraph) + 2
        index += 1
    paragraphs.insert(len(paragraphs) // 2, needle)
    return '\n\n'.join(paragraphs)


# -------------------------------------------------------------------- validation

class Validation:
    def __init__(self, client, report, out_dir, arm, model):
        self.client = client
        self.report = report
        self.out_dir = out_dir
        self.arm = arm
        self.model = model
        self.run_id = time.strftime('%H%M%S')
        self.probe_lock = threading.Lock()
        self.probe_ids = {}
        self.probe_store = dict(arm=arm, model=model, probes={})

    # ---- request helpers
    def rid(self, label):
        return re.sub(r'[^A-Za-z0-9_.-]+', '_', f'fv-{self.arm}-{label}-{self.run_id}')

    def chat(self, label, messages, max_tokens=CHECK_MAX_TOKENS, **extra):
        body = dict(model=self.model, messages=messages, temperature=0, max_tokens=max_tokens, seed=SEED,
                    chat_template_kwargs=dict(thinking=False), rid=self.rid(label))
        body.update(extra)
        exchange = self.client.call(label, '/v1/chat/completions', body)
        return exchange, exchange.value['choices'][0]

    def tokenize(self, label, messages):
        body = dict(model=self.model, messages=messages, chat_template_kwargs=dict(thinking=False))
        exchange = self.client.call(label, '/tokenize', body)
        tokens, count = exchange.value.get('tokens'), exchange.value.get('count')
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int for t in tokens):
            raise ValueError('/tokenize returned no integer token list')
        if count is not None and count != len(tokens):
            raise ValueError(f'/tokenize count {count} differs from token list length {len(tokens)}')
        return tokens, exchange.file

    def json_case(self, label, messages, expected):
        exchange, choice = self.chat(label, messages)
        verdict = judge_json(message_text(choice), expected)
        verdict.update(finish_reason=choice.get('finish_reason'), exchange=exchange.file)
        return verdict

    def run_together(self, name, count, worker):
        """Run worker(i) for i < count at the same moment; workers record their own checks."""
        barrier = threading.Barrier(count)

        def run(i):
            try:
                barrier.wait(timeout=60)
            except threading.BrokenBarrierError:
                self.report.add_check(f'{name}-start-barrier-{i}', False, error='start barrier broken')
            try:
                worker(i)
            except Exception as exc:
                self.report.add_check(f'{name}-worker-{i}-crashed', False,
                                      error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())

        with ThreadPoolExecutor(max_workers=count) as pool:
            list(pool.map(run, range(count)))

    # ---- sections
    def check_server_info(self):
        info = {}
        try:
            info = self.client.call('server-info', '/server_info').value
            self.report.data['server_info'] = info
            self.report.add_check('server-info:fetch', isinstance(info, dict))
        except Exception as exc:
            self.report.add_check('server-info:fetch', False, error=f'{type(exc).__name__}: {exc}')
        info = info if isinstance(info, dict) else {}
        for key, wanted in EXPECTED_SERVER.items():
            present = key in info
            actual = info.get(key)
            self.report.add_check(f'server-info:{key}', present and actual == wanted,
                                  expected=wanted, actual=actual, present=present)

    def check_basic(self):
        for label, prompt, expected in BASIC_CASES:
            messages = [dict(role='system', content=JSON_SYSTEM), dict(role='user', content=prompt)]
            guarded(self.report, label, lambda: self.json_case(label, messages, expected))

    @staticmethod
    def cohort_facts(n):
        facts = []
        for i in range(n):
            code = 1000 + (n * 7919 + (i + 1) * 3571) % 9000
            new_code = 1000 + (code * 7 + 1234) % 9000
            if new_code == code:
                new_code = 1000 + (new_code + 1) % 9000
            facts.append(dict(owner=OWNERS[i], code=str(code), new_code=str(new_code)))
        return facts

    @staticmethod
    def turn_script(fact, turn):
        """(user prompt, expected answer) for the three stateless turns of one conversation."""
        if turn == 0:
            return (f'For this conversation only, remember owner={fact["owner"]} and code={fact["code"]}. '
                    'Reply with {"ack": true}.', dict(ack=True))
        if turn == 1:
            return ('What owner and code did I give you? Use exactly the JSON keys owner and code, '
                    'both strings.', dict(owner=fact['owner'], code=fact['code']))
        return (f'Change my code to {fact["new_code"]}. Report the original and the current code as strings '
                'under the keys original_code and current_code.',
                dict(original_code=fact['code'], current_code=fact['new_code']))

    def check_concurrency(self):
        """N concurrent stateless conversations; each re-sends its history with the canonical
        assistant answers so every turn is an independent exact check and prompts match across arms."""
        for n in COHORTS:
            facts = self.cohort_facts(n)
            histories = [[dict(role='system', content=JSON_SYSTEM)] for _ in range(n)]
            for turn in range(3):
                expected = []
                for i in range(n):
                    prompt, answer = self.turn_script(facts[i], turn)
                    histories[i].append(dict(role='user', content=prompt))
                    expected.append(answer)

                def worker(i, n=n, turn=turn, expected=expected):
                    label = f'N{n}-T{turn}-slot{i}'
                    guarded(self.report, label,
                            lambda: self.json_case(label, list(histories[i]), expected[i]))

                self.run_together(f'N{n}-T{turn}', n, worker)
                for i in range(n):
                    histories[i].append(dict(role='assistant', content=json.dumps(expected[i])))

    def check_tools(self):
        for n in COHORTS:
            def worker(i, n=n):
                sku, warehouse = TOOL_SLOTS[i]
                wanted = dict(sku=sku, warehouse=warehouse)
                available = 17 + 6 * i
                messages = [dict(role='system', content='Use the provided inventory tool. Do not invent inventory.'),
                            dict(role='user', content=f'Call lookup_inventory for sku {sku} at warehouse {warehouse}.')]
                state = {}

                def tool_call():
                    exchange, choice = self.chat(f'N{n}-tool-call-{i}', messages, max_tokens=128,
                                                 tools=[TOOL], tool_choice='auto')
                    message = choice.get('message') or {}
                    calls = message.get('tool_calls') or []
                    result = dict(exchange=exchange.file, finish_reason=choice.get('finish_reason'),
                                  expected=wanted, content=message.get('content'))
                    if choice.get('finish_reason') != 'tool_calls' or len(calls) != 1:
                        return dict(result, passed=False, error='expected finish_reason tool_calls with exactly one call',
                                    tool_calls=calls)
                    call = calls[0]
                    function = call.get('function') or {}
                    if call.get('type') != 'function' or function.get('name') != 'lookup_inventory':
                        return dict(result, passed=False, error='wrong tool call', tool_calls=calls)
                    actual = json.loads(function.get('arguments') or '')
                    passed = same_json(actual, wanted)
                    if passed:  # only a correct parsed call seeds the model-sourced round trip
                        state.update(message=message, call=call)
                    return dict(result, passed=passed, actual=actual)

                def tool_result():
                    if 'call' in state:
                        source, message, call = 'model', state['message'], state['call']
                    else:  # still exercise the tool-result path with a canonical call
                        source, message = 'canonical', dict(content=None)
                        call = dict(id=f'call_{n}_{i}', type='function',
                                    function=dict(name='lookup_inventory', arguments=json.dumps(wanted)))
                    followup = messages + [
                        dict(role='assistant', content=message.get('content'), tool_calls=[call]),
                        dict(role='tool', tool_call_id=call['id'], content=json.dumps(dict(wanted, available=available))),
                        dict(role='user', content='How many units are available? Reply only with a JSON object '
                                                  'with the integer key available.')]
                    exchange, choice = self.chat(f'N{n}-tool-result-{i}', followup, tools=[TOOL],
                                                 tool_choice='none')
                    verdict = judge_json(message_text(choice), dict(available=available))
                    return dict(verdict, exchange=exchange.file, tool_call_source=source,
                                finish_reason=choice.get('finish_reason'))

                guarded(self.report, f'N{n}-tool-call-{i}', tool_call)
                guarded(self.report, f'N{n}-tool-result-{i}', tool_result)

            self.run_together(f'N{n}-tool', n, worker)

    def check_logprob_probes(self):
        for name, text in PROBE_TEXTS.items():
            messages = [dict(role='user', content=text)]
            for repeat in range(2):
                guarded(self.report, f'logprob-probe-{name}-repeat{repeat}',
                        lambda: self.run_probe(name, messages, repeat))

    def run_probe(self, name, messages, repeat):
        with self.probe_lock:
            if name not in self.probe_ids:
                ids, tokenize_file = self.tokenize(f'logprob-probe-{name}-tokenize', messages)
                if len(ids) <= LOGPROB_WINDOW:
                    raise ValueError(f'probe has only {len(ids)} tokens; needs more than {LOGPROB_WINDOW}')
                self.probe_ids[name] = ids
                self.probe_store['probes'][name] = dict(
                    messages=messages, ids=ids, tokenize_exchange=tokenize_file,
                    logprob_start_len=max(0, len(ids) - LOGPROB_WINDOW), repeats=[])
        ids = self.probe_ids[name]
        start = max(0, len(ids) - LOGPROB_WINDOW)
        label = f'logprob-probe-{name}-repeat{repeat}'
        body = dict(input_ids=ids, rid=self.rid(label), stream=False,
                    sampling_params=dict(temperature=0, max_new_tokens=1),
                    return_logprob=True, top_logprobs_num=0, logprob_start_len=start)
        exchange = self.client.call(label, '/generate', body)
        meta = (exchange.value.get('meta_info') or {}) if isinstance(exchange.value, dict) else {}
        analysis = analyse_input_logprobs(ids, start, meta.get('input_token_logprobs'))
        problems = list(analysis['problems'])
        if meta.get('prompt_tokens') != len(ids):
            problems.append(f'server prompt_tokens {meta.get("prompt_tokens")!r} != {len(ids)} supplied ids')
        if meta.get('completion_tokens') != 1:
            problems.append(f'completion_tokens {meta.get("completion_tokens")!r} != 1')
        analysis.update(problems=problems, passed=not problems)
        with self.probe_lock:
            self.probe_store['probes'][name]['repeats'].append(dict(
                repeat=repeat, exchange=exchange.file, rid=body['rid'],
                raw_input_token_logprobs=meta.get('input_token_logprobs'),
                output_ids=exchange.value.get('output_ids'), analysis=analysis))
            write_json(self.out_dir / 'logprob_probes.json', self.probe_store)
        return dict(passed=not problems, problems=problems, exchange=exchange.file, ids_count=len(ids),
                    logprob_start_len=start, entry_count=analysis.get('entry_count'),
                    aligned_offset=analysis.get('aligned_offset'),
                    trailing_entries=analysis.get('trailing_entries'))

    def retrieval_messages(self, filler_chars, ledger, code):
        needle = (f'Archive note: the vault access code for the ledger named {ledger} is {code}. '
                  'Keep this code exactly as written.')
        document = build_document(filler_chars, needle)
        question = (f'Question: what is the exact vault access code for the ledger named {ledger}? Copy it '
                    'character for character from the document above. Reply with JSON only: '
                    '{"code": "<the exact code>"}')
        return [dict(role='system', content=JSON_SYSTEM),
                dict(role='user', content=f'Read this document.\n\n{document}\n\n{question}')]

    def check_retrieval(self):
        for label, target, ledger, code in RETRIEVAL_CASES:
            guarded(self.report, label, lambda: self.retrieval_case(label, target, ledger, code))

    def retrieval_case(self, label, target, ledger, code):
        chars = int(target * ESTIMATED_CHARS_PER_TOKEN)
        attempts = []
        for attempt in range(1, RETRIEVAL_ATTEMPTS + 1):
            messages = self.retrieval_messages(chars, ledger, code)
            tokens, tokenize_file = self.tokenize(f'{label}-tokenize{attempt}', messages)
            attempts.append(dict(attempt=attempt, filler_chars=chars, tokens=len(tokens), exchange=tokenize_file))
            if abs(len(tokens) - target) <= RETRIEVAL_ADJUST_TOLERANCE * target:
                break
            if attempt < RETRIEVAL_ATTEMPTS:
                chars = int(chars * target / len(tokens))
        counted = len(tokens)
        if counted >= MAX_FILLER_TOKENS or abs(counted - target) > RETRIEVAL_TOLERANCE * target:
            raise ValueError(f'could not size prompt: target {target}, tokenized {counted}, attempts {attempts}')
        content = messages[1]['content']
        exchange, choice = self.chat(label, messages)
        usage_tokens = (exchange.value.get('usage') or {}).get('prompt_tokens')
        verdict = judge_json(message_text(choice), dict(code=code))
        usage_ok = isinstance(usage_tokens, int) and abs(usage_tokens - counted) <= 0.01 * counted
        verdict.update(
            passed=verdict['passed'] and usage_ok, exchange=exchange.file, finish_reason=choice.get('finish_reason'),
            target_tokens=target, tokenized_tokens=counted, usage_prompt_tokens=usage_tokens,
            usage_matches_tokenize_within_1pct=usage_ok, sizing_attempts=attempts,
            needle_char_fraction=round(content.find(code) / len(content), 3))
        return verdict

    def check_throughput(self):
        for repeat in range(6):
            self.throughput_round(repeat)
        self.report.data['throughput_medians'] = {
            str(n): statistics.median(row['aggregate_tokens_per_s']
                                      for row in self.report.data['throughput']
                                      if row['n'] == n and row['repeat'] > 0)
            for n in COHORTS
        }
        self.report.save()

    def throughput_round(self, repeat):
        for n in COHORTS:
            rows = [None] * n

            def worker(i, n=n, rows=rows):
                label = f'throughput-R{repeat}-N{n}-slot{i}'
                try:
                    body = dict(text=THROUGHPUT_PROMPTS[i], rid=self.rid(label), stream=False,
                                sampling_params=dict(temperature=0, max_new_tokens=THROUGHPUT_TOKENS,
                                                     ignore_eos=True))
                    exchange = self.client.call(label, '/generate', body)
                    meta = exchange.value.get('meta_info') or {}
                    tokens = meta.get('completion_tokens')
                    finish = (meta.get('finish_reason') or {}).get('type')
                    rows[i] = dict(slot=i, exchange=exchange.file, latency_s=exchange.wall_s,
                                   completion_tokens=tokens, finish_reason=finish,
                                   server_e2e_latency_s=meta.get('e2e_latency'),
                                   tokens_per_s=(tokens / exchange.wall_s) if isinstance(tokens, int) else None,
                                   ok=tokens == THROUGHPUT_TOKENS and finish == 'length')
                except Exception as exc:
                    rows[i] = dict(slot=i, ok=False, error=f'{type(exc).__name__}: {exc}')

            start = time.monotonic()
            self.run_together(f'throughput-N{n}', n, worker)
            wall_s = time.monotonic() - start
            total = sum(r.get('completion_tokens') or 0 for r in rows if r)
            summary = dict(n=n, repeat=repeat, warmup=repeat == 0, cohort_wall_s=wall_s, total_completion_tokens=total,
                           aggregate_tokens_per_s=total / wall_s if wall_s > 0 else None, requests=rows)
            self.report.data['throughput'].append(summary)
            self.report.add_check(f'throughput-R{repeat}-N{n}', all(r and r['ok'] for r in rows),
                                  expected_tokens_each=THROUGHPUT_TOKENS, **summary)

    def run_all(self, suite='full'):
        sections = [('server-info', self.check_server_info), ('basic', self.check_basic),
                    ('concurrency', self.check_concurrency), ('tools', self.check_tools),
                    ('logprob-probes', self.check_logprob_probes), ('retrieval', self.check_retrieval),
                    ('throughput', self.check_throughput)]
        if suite == 'short':
            sections = [(name, fn) for name, fn in sections
                        if name in ('server-info', 'basic', 'logprob-probes', 'throughput')]
        for name, section in sections:
            try:
                section()
            except Exception as exc:
                self.report.add_check(f'section:{name}:crashed', False,
                                      error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())


# ----------------------------------------------------------- cross-arm comparison

def position_deltas(a, b):
    """Compare two analysed repeats by absolute token position (never by list index)."""
    left = dict(zip(a['positions'], zip(a['logprobs'], a['token_ids'])))
    right = dict(zip(b['positions'], zip(b['logprobs'], b['token_ids'])))
    common = sorted(left.keys() & right.keys())
    deltas = [abs(left[p][0] - right[p][0]) for p in common]
    return dict(common_positions=len(common),
                max_abs_delta=max(deltas) if deltas else None,
                mean_abs_delta=sum(deltas) / len(deltas) if deltas else None,
                token_id_mismatches=sum(left[p][1] != right[p][1] for p in common))


def passing_repeats(probe):
    return [r['analysis'] for r in probe.get('repeats', []) if r['analysis'].get('passed')]


def compare_arms(dir_a, dir_b):
    """Offline comparison of the saved probes of two arms. Reports differences; applies no tolerance."""
    stores = [json.loads((Path(d) / 'logprob_probes.json').read_text(encoding='utf-8')) for d in (dir_a, dir_b)]
    result = dict(arms=[s.get('arm') for s in stores], probes={})
    ok = True
    for name in sorted(set(stores[0]['probes']) | set(stores[1]['probes'])):
        probes = [s['probes'].get(name) for s in stores]
        if not all(probes):
            result['probes'][name] = dict(error='probe missing in one arm')
            ok = False
            continue
        runs = [passing_repeats(p) for p in probes]
        entry = dict(ids_equal=probes[0]['ids'] == probes[1]['ids'], id_count=[len(p['ids']) for p in probes])
        if not all(runs):
            entry['error'] = 'no passing repeat in one arm'
            ok = False
        else:
            entry['cross_arm'] = position_deltas(runs[0][0], runs[1][0])
            entry['offsets'] = [r[0]['aligned_offset'] for r in runs]
            for arm, arm_runs in zip(result['arms'], runs):
                if len(arm_runs) > 1:
                    entry[f'intra_arm_repeat_{arm}'] = position_deltas(arm_runs[0], arm_runs[1])
            ok = ok and entry['ids_equal'] and entry['cross_arm']['common_positions'] >= LOGPROB_WINDOW - 1 \
                and entry['cross_arm']['token_id_mismatches'] == 0
        result['probes'][name] = entry
    result['comparable'] = ok
    print(json.dumps(jsonable(result), indent=2))
    return 0 if ok else 1


# --------------------------------------------------------------------------- main

def parse_args(argv):
    parser = argparse.ArgumentParser(description='SM89 PR3 full-model validation client (stdlib only).')
    parser.add_argument('--base-url', default=DEFAULT_BASE_URL)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--output', type=Path, help='empty or new directory for all retained data')
    parser.add_argument('--arm', help='variant label recorded in the output')
    parser.add_argument('--suite', choices=['full', 'short'], default='full')
    parser.add_argument('--timeout', type=float, default=MAX_TIMEOUT_S,
                        help=f'per-HTTP-request timeout in seconds, at most {MAX_TIMEOUT_S}')
    parser.add_argument('--compare', nargs=2, metavar=('DIR_A', 'DIR_B'),
                        help='offline: compare logprob_probes.json of two finished arm directories')
    args = parser.parse_args(argv)
    if args.compare:
        return args
    if args.output is None or args.arm is None:
        parser.error('--output and --arm are required unless --compare is used')
    if not 0 < args.timeout <= MAX_TIMEOUT_S:
        parser.error(f'--timeout must be in (0, {MAX_TIMEOUT_S}]')
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error('--output must be a new or empty directory')
    return args


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.compare:
        return compare_arms(*args.compare)
    exchange_dir = args.output / 'exchanges'
    exchange_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = Report(args.output / 'report.json', dict(
        arm=args.arm, suite=args.suite, model=args.model, base_url=args.base_url, started_utc=utc_now(), per_request_timeout_s=args.timeout,
        scope='Functional/quality validation of one serving arm; see CLIENT_NOTES.md. Throughput is recorded, not gated.'))
    client = Client(args.base_url, args.timeout, exchange_dir)
    try:
        Validation(client, report, args.output, args.arm, args.model).run_all(args.suite)
        status = 'PASS' if report.data['total_checks'] and not report.data['failed_checks'] else 'FAIL'
    except BaseException as exc:  # includes KeyboardInterrupt: never leave a RUNNING report behind
        report.add_check('client:aborted', False, error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        status = 'FAIL'
    report.data.update(status=status, finished_utc=utc_now(), seconds=time.monotonic() - started)
    report.save()
    print(json.dumps(dict(
        status=status, arm=args.arm, total_checks=report.data['total_checks'],
        passed_checks=report.data['passed_checks'], failed_checks=report.data['failed_checks'],
        failed_labels=report.failed_labels(), seconds=round(report.data['seconds'], 1), output=str(args.output))))
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
