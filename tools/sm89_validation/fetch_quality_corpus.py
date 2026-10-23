"""Retain four deterministic WikiText-2 test slices and their retrieval provenance."""
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

out=Path(__file__).resolve().parent/'quality-corpus.json'
assert not out.exists(),'Do not silently replace the evaluation corpus'
segments=[]
for offset in (0,1000,2000,3000):
    params=dict(dataset='Salesforce/wikitext',config='wikitext-2-raw-v1',split='test',offset=offset,length=100)
    url='https://datasets-server.huggingface.co/rows?'+urllib.parse.urlencode(params)
    with urllib.request.urlopen(url,timeout=60) as r:raw=r.read()
    obj=json.loads(raw)
    rows=[dict(index=r['row_idx'],text=r['row']['text']) for r in obj['rows']]
    assert len(rows)==100 and rows[0]['index']==offset
    segments.append(dict(offset=offset,url=url,response_sha256=hashlib.sha256(raw).hexdigest(),rows=rows))
result=dict(dataset='Salesforce/wikitext',config='wikitext-2-raw-v1',split='test',
            source='https://huggingface.co/datasets/Salesforce/wikitext',license='CC-BY-SA-3.0 / GFDL (dataset card)',segments=segments)
out.write_bytes(json.dumps(result,ensure_ascii=False,indent=2).encode())
print(json.dumps(dict(path=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),segments=len(segments),chars=[sum(len(r['text']) for r in s['rows']) for s in segments])))
