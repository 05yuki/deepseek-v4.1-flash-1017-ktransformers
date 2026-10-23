"""Summarize matched model reports without turning numerical differences into passes."""
import argparse
import json
import statistics
from pathlib import Path

CONFIG_KEYS = [
    'model_path', 'context_length', 'dtype', 'quantization', 'kv_cache_dtype',
    'max_running_requests', 'max_total_tokens', 'chunked_prefill_size',
    'disable_cuda_graph', 'speculative_algorithm', 'kt_method', 'kt_cpuinfer',
    'kt_threadpool_count', 'kt_num_gpu_experts', 'kt_max_deferred_experts_per_token',
    'language_model_only', 'attention_backend',
]


def summarize(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    groups = {}
    for n in (1, 2, 4):
        rows = [r for r in data['throughput'] if r['n'] == n and not r['warmup']]
        if len(rows) != 5 or any(not all(q['ok'] for q in r['requests']) for r in rows):
            raise ValueError(f'{path}: expected five successful measured N{n} rounds')
        values = [r['aggregate_tokens_per_s'] for r in rows]
        groups[str(n)] = dict(median=statistics.median(values), minimum=min(values),
                              maximum=max(values), trials=values)
    return data, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    base, before = summarize(args.baseline)
    cand, after = summarize(args.candidate)
    mismatches = {k: [base['server_info'].get(k), cand['server_info'].get(k)]
                  for k in CONFIG_KEYS
                  if k not in base['server_info'] or k not in cand['server_info']
                  or base['server_info'][k] != cand['server_info'][k]}
    result = dict(
        baseline=str(args.baseline), candidate=str(args.candidate),
        baseline_status=base['status'], candidate_status=cand['status'],
        configuration_mismatches=mismatches,
        metric='aggregate completion tokens / client cohort wall seconds; includes prompt processing',
        protocol='one excluded warmup and five measured rounds; 128 output tokens per request',
        results={n: dict(before=before[n], after=after[n],
                         improvement_percent=100*(after[n]['median']/before[n]['median']-1))
                 for n in before},
        note='Functional status and configuration only. Numerical comparison and source/build provenance require separate review.',
    )
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    if mismatches or base['status'] != 'PASS' or cand['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
