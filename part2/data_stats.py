"""Q4: observed token statistics before and after the actual processing pipeline."""
import json
from pathlib import Path
from statistics import mean
from transformers import T5TokenizerFast
from load_data import get_tokenizer, PREFIX

base=T5TokenizerFast.from_pretrained('google-t5/t5-small')
processed=get_tokenizer()
output={'definitions':{'before':'Base tokenizer, no EOS/prefix; observed token-ID vocabulary per split',
                      'after':'Extended tokenizer; NL prefix and EOS; SQL EOS; no padding or truncation',
                      'length':'Number of token IDs, not whitespace-delimited words'},'before':{},'after':{}}
for split in ['train','dev']:
    nl=Path(f'data/{split}.nl').read_text().splitlines()
    sql=Path(f'data/{split}.sql').read_text().splitlines()
    for stage,tok in [('before',base),('after',processed)]:
        texts=[PREFIX+x.strip() for x in nl] if stage=='after' else nl
        nids=tok(texts,add_special_tokens=stage=='after')['input_ids']
        sids=tok([x.strip() for x in sql],add_special_tokens=stage=='after')['input_ids']
        output[stage][split]={
            'examples':len(nl),'mean_nl_length':mean(map(len,nids)),
            'mean_sql_length':mean(map(len,sids)),
            'nl_vocabulary':len({v for x in nids for v in x}),
            'sql_vocabulary':len({v for x in sids for v in x}),
            'max_nl_length':max(map(len,nids)),'max_sql_length':max(map(len,sids)),
            'unknown_nl_examples':sum(tok.unk_token_id in x for x in nids),
            'unknown_sql_examples':sum(tok.unk_token_id in x for x in sids),
        }
        if stage=='after':
            assert all(tok.unk_token_id not in x for x in nids+sids)
            decoded=tok.batch_decode(sids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            decoded=[' '.join(x.replace('< =','<=').replace('> =','>=').split()) for x in decoded]
            Path(f'results/tokenizer_roundtrip_{split}.sql').write_text('\n'.join(decoded)+'\n')
output['tokenizer_length']=len(processed)
Path('logs/data_stats.json').write_text(json.dumps(output,indent=2))
print(json.dumps(output,indent=2))
