"""Reproducible, overlapping diagnostic signatures for development-set errors."""
from pathlib import Path
from collections import Counter
import argparse,json,pickle
import sqlglot
from sqlglot import exp
from utils import compute_record_F1

p=argparse.ArgumentParser();p.add_argument('--stem',required=True);p.add_argument('--output',default='logs/error_analysis.json');args=p.parse_args()
gold=Path('data/dev.sql').read_text().splitlines()
nl=Path('data/dev.nl').read_text().splitlines()
pred=Path(f'results/{args.stem}.sql').read_text().splitlines()
gt,_=pickle.load(open('records/ground_truth_dev.pkl','rb'))
records,errors=pickle.load(open(f'records/{args.stem}.pkl','rb'))
assert len(nl)==len(pred)==len(records)==len(errors)==466

def features(sql):
    try:
        tree=sqlglot.parse_one(sql,read='sqlite')
        strings=Counter(x.this for x in tree.find_all(exp.Literal) if x.is_string)
        numbers=Counter(x.this for x in tree.find_all(exp.Literal) if not x.is_string)
        structure=(Counter(x.name.lower() for x in tree.find_all(exp.Table)),
                   Counter(x.name.lower() for x in tree.find_all(exp.Column)),
                   Counter(type(x).__name__ for x in tree.walk() if isinstance(x,exp.AggFunc)))
        return strings,numbers,structure
    except Exception:
        return None

cats={k:[] for k in ['execution','string_literals','numeric_literals','relational_structure','other_semantic']}
all_cases=[]
for i,(g,q) in enumerate(zip(gold,pred)):
    mismatch=set(gt[i])!=set(records[i])
    gs,ps=features(g),features(q)
    tags=[]
    if errors[i]: tags.append('execution')
    if mismatch and not errors[i] and gs and ps:
        if gs[0]!=ps[0]:tags.append('string_literals')
        if gs[1]!=ps[1]:tags.append('numeric_literals')
        if gs[2]!=ps[2]:tags.append('relational_structure')
    if mismatch and not tags: tags.append('other_semantic')
    if tags:
        entry={'index_0based':i,'line_1based':i+1,'nl':nl[i], 'gold_sql':g,'predicted_sql':q,
               'record_f1':float(compute_record_F1([gt[i]],[records[i]])),
               'gold_record_count':len(set(gt[i])),'predicted_record_count':len(set(records[i])),
               'execution_error':errors[i],'categories':tags,
               'string_difference':None if gs is None or ps is None else {
                    'gold_only':dict(gs[0]-ps[0]),'predicted_only':dict(ps[0]-gs[0])},
               'numeric_difference':None if gs is None or ps is None else {
                    'gold_only':dict(gs[1]-ps[1]),'predicted_only':dict(ps[1]-gs[1])}}
        all_cases.append(entry)
        for tag in tags:cats[tag].append(i)

# Prefer clear, single-signature examples before mixed cases.
examples={}
for tag,indices in cats.items():
    cases=[x for x in all_cases if x['index_0based'] in indices]
    cases.sort(key=lambda x:(len(x['categories']),len(x['gold_sql'])+len(x['predicted_sql'])))
    if cases:examples[tag]=cases[0]
summary={'total':466,'record_mismatches':sum(set(x)!=set(y) for x,y in zip(gt,records)),
         'execution_errors':sum(bool(x) for x in errors),
         'empty_gold_record_sets':sum(not x for x in gt),
         'sql_nonmatch_but_records_match':sum(g!=q and set(x)==set(y) for g,q,x,y in zip(gold,pred,gt,records)),
         'category_counts':{k:len(v) for k,v in cats.items()},
         'category_indices_0based':cats,'examples':examples,'all_cases':all_cases,
         'definitions':{'denominator':'All 466 dev queries; categories can overlap',
                        'execution':'Any nonempty SQLite execution error, including timeout',
                        'string_literals':'Executable record mismatch with different string literal multisets',
                        'numeric_literals':'Executable record mismatch with different numeric literal multisets',
                        'relational_structure':'Executable record mismatch with different table/column/aggregate multisets',
                        'other_semantic':'Remaining record mismatches; includes predicate or role wiring differences'},
         'scope':'Automated diagnostic signatures, not an exhaustive manual causal attribution'}
Path(args.output).write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k in ['record_mismatches','category_counts','empty_gold_record_sets','sql_nonmatch_but_records_match']},indent=2))
