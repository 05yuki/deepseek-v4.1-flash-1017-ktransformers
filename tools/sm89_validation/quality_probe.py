"""Matched, fixed WikiText-2 excerpts; this is not a full WikiText benchmark."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from model_client import Client, analyse_input_logprobs

CORPUS_SHA='eb3abb55e9edaae45dbfe3dadc7f4574ffa6b80ff2b1af42d4d4e95ebfb2813b'  # canonical row IDs and text
# Fixed before evaluating these excerpts. Screening guard only, not an equivalence claim.
MAX_NLL_INCREASE=0.02  # nats/token; exp(.02)-1 = approximately 2.02% perplexity increase

def compare(before,after):
    assert before['status']==after['status']=='PASS'
    assert before['corpus_sha256']==after['corpus_sha256']==CORPUS_SHA
    assert len(before['chunks'])==len(after['chunks']) and before['chunks']
    all_d=[];rows=[]
    for a,b in zip(before['chunks'],after['chunks']):
        assert (a['name'],a['ids'])==(b['name'],b['ids'])
        assert a['positions']==b['positions'] and a['token_ids']==b['token_ids']
        d=[x-y for x,y in zip(a['logprobs'],b['logprobs'])]
        all_d.extend(d)
        rows.append(dict(name=a['name'],positions=len(d),nll_before=a['nll'],nll_after=b['nll'],
                         nll_delta=statistics.mean(d),mean_abs_delta=statistics.mean(map(abs,d)),max_abs_delta=max(map(abs,d))))
    nll0=-statistics.mean([x for c in before['chunks'] for x in c['logprobs']])
    nll1=-statistics.mean([x for c in after['chunks'] for x in c['logprobs']])
    return dict(positions=len(all_d),chunks=rows,nll_before=nll0,nll_after=nll1,nll_delta=nll1-nll0,
                perplexity_before=math.exp(nll0),perplexity_after=math.exp(nll1),
                perplexity_change_percent=100*math.expm1(nll1-nll0),
                max_nll_increase=MAX_NLL_INCREASE,screen_pass=nll1-nll0<=MAX_NLL_INCREASE,
                note='Fixed short-context test excerpts, not full WikiText-2 or broad task-quality equivalence.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--compare',nargs=2,type=Path)
    p.add_argument('--base-url',default='http://127.0.0.1:18209')
    p.add_argument('--checkpoint')
    p.add_argument('--arm',default='unknown')
    p.add_argument('--plan-only',action='store_true')
    a=p.parse_args()
    if a.compare:
        out=compare(*[json.loads(x.read_text(encoding='utf-8')) for x in a.compare])
        a.output.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out))
        return 0 if out['screen_pass'] else 1
    if not a.checkpoint:
        p.error('--checkpoint is required for collection')
    from transformers import AutoTokenizer
    corpus=HERE/'quality-corpus.json'
    data=json.loads(corpus.read_text(encoding='utf-8'))
    canonical=json.dumps([{'offset': s['offset'], 'rows': s['rows']} for s in data['segments']], ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    assert hashlib.sha256(canonical).hexdigest()==CORPUS_SHA, 'Evaluation text changed'
    tokenizer=AutoTokenizer.from_pretrained(a.checkpoint,trust_remote_code=True,local_files_only=True)
    if a.plan_only:
        plan=[]
        for seg in data['segments']:
            count=len(tokenizer.encode(''.join(r['text'] for r in seg['rows']),add_special_tokens=False))
            assert count>=1024,(seg['offset'],count)
            plan.append(dict(offset=seg['offset'],tokens=count,chunks=min(4,count//1024)))
        result=dict(corpus_sha256=CORPUS_SHA,segments=plan,scored_positions=1023*sum(x['chunks'] for x in plan),max_nll_increase=MAX_NLL_INCREASE)
        a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result));return 0
    a.output.mkdir(exist_ok=False)
    (a.output/'exchanges').mkdir()
    client=Client(a.base_url,600,a.output/'exchanges')
    receipt=dict(arm=a.arm,corpus_sha256=CORPUS_SHA,source=data['source'],license=data['license'],
                 protocol='First up to four complete 1024-token chunks per fixed segment; 1023 scored positions/chunk, no chat template, no special tokens.',
                 screen_max_nll_increase=MAX_NLL_INCREASE,chunks=[])
    for seg in data['segments']:
        text=''.join(r['text'] for r in seg['rows'])
        ids=tokenizer.encode(text,add_special_tokens=False)
        count=min(4,len(ids)//1024)
        assert count>=1,(seg['offset'],len(ids))
        for i in range(count):
            name=f"rows{seg['offset']}-chunk{i}"
            selected=ids[i*1024:(i+1)*1024]
            body=dict(input_ids=selected,rid=f"quality-{a.arm}-{name}-{int(time.time())}",stream=False,
                      sampling_params=dict(temperature=0,max_new_tokens=1),return_logprob=True,
                      top_logprobs_num=0,logprob_start_len=0)
            ex=client.call(name,'/generate',body)
            rows=ex.value['meta_info']['input_token_logprobs']
            aligned=analyse_input_logprobs(selected,0,rows)
            assert aligned['passed'],aligned
            assert aligned['positions']==list(range(1,1024)),aligned['positions'][:4]
            receipt['chunks'].append(dict(name=name,ids=selected,exchange=ex.file,
                                         positions=aligned['positions'],token_ids=aligned['token_ids'],
                                         logprobs=aligned['logprobs'],nll=-statistics.mean(aligned['logprobs'])))
            (a.output/'report.json').write_text(json.dumps(receipt,indent=2))
            print(name+' complete',flush=True)
    receipt['status']='PASS';receipt['scored_positions']=sum(len(c['positions']) for c in receipt['chunks'])
    receipt['nll']=-statistics.mean(x for c in receipt['chunks'] for x in c['logprobs'])
    (a.output/'report.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k!='chunks'}))
    return 0

if __name__=='__main__':raise SystemExit(main())
