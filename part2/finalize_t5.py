"""Select greedy vs beam-4 on dev, then perform exactly one hidden-test run."""
import json,pickle,shutil,hashlib
from pathlib import Path
from argparse import Namespace
import torch
from load_data import load_t5_data
from train_t5 import eval_epoch, test_inference, metrics_dict
from t5_utils import load_model_from_checkpoint
from utils import compute_metrics,save_queries_and_records,set_random_seeds

if Path('results/t5_ft_experiment_test.sql').exists() or Path('records/t5_ft_experiment_test.pkl').exists():
    raise FileExistsError('Final test artifacts already exist; use build_full_report.py to rebuild documents only.')
torch.set_num_threads(6);set_random_seeds(42)
args=Namespace(finetune=True,experiment_name='experiment',num_beams=1,max_new_tokens=640)
_,dev,test=load_t5_data(16,16)
model=load_model_from_checkpoint(args,best=True)
path1=Path('results/t5_ft_experiment_dev_beam1.sql')
assert path1.exists(),'Complete training before finalizing.'
# Rerun the existing greedy SQL with the final execution budget; no regeneration.
save_queries_and_records(path1.read_text().splitlines(),str(path1),'records/t5_ft_experiment_dev_beam1.pkl')
sql_em,record_em,f1,errors=compute_metrics('data/dev.sql',str(path1),'records/ground_truth_dev.pkl','records/t5_ft_experiment_dev_beam1.pkl')
old=json.loads(Path('logs/t5_ft_experiment_dev_beam1.json').read_text())
greedy={**old,'sql_em':sql_em,'record_em':record_em,'record_f1':float(f1),'sql_error_rate':sum(bool(x) for x in errors)/len(errors)}
Path('logs/t5_ft_experiment_dev_beam1.json').write_text(json.dumps(greedy,indent=2))
print('FINAL_GREEDY',json.dumps(greedy),flush=True)
args.num_beams=4
beam=metrics_dict(eval_epoch(args,model,dev,'data/dev.sql','results/t5_ft_experiment_dev_beam4.sql',
                             'records/ground_truth_dev.pkl','records/t5_ft_experiment_dev_beam4.pkl'))
Path('logs/t5_ft_experiment_dev_beam4.json').write_text(json.dumps(beam,indent=2))
print('FINAL_BEAM4',json.dumps(beam),flush=True)
args.num_beams=4 if beam['record_f1']>greedy['record_f1'] else 1
chosen=beam if args.num_beams==4 else greedy
for folder,ext in [('results','sql'),('records','pkl')]:
    shutil.copyfile(f'{folder}/t5_ft_experiment_dev_beam{args.num_beams}.{ext}',f'{folder}/t5_ft_experiment_dev.{ext}')
best=Path('checkpoints/ft_experiments/experiment/best')
selection={'selected_beams':args.num_beams,'checkpoint_selection':json.loads((best/'selection.json').read_text()),
           'dev':chosen,'decoding_experiments':{'greedy':greedy,'beam4':beam},
           'sql_query_timeout_seconds':120,'hidden_test_metrics':None,
           'test_metrics_status':'Unavailable locally: hidden references require Gradescope evaluation',
           'test_examples':len(test.dataset),'test_generation_runs':1,
           'checkpoint_sha256':hashlib.sha256((best/'model.safetensors').read_bytes()).hexdigest()}
config={'prefix':'translate English to SQL: ','num_beams':args.num_beams,'do_sample':False,
        'max_new_tokens':640,'length_penalty':1.0,'decoder_start_token_id':0,
        'eos_token_id':1,'pad_token_id':0,'added_token':'<',
        'postprocessing':'Collapse whitespace; repair < = and > = into <= and >='}
(best/'inference_config.json').write_text(json.dumps(config,indent=2))
# No test scores are used in any model or decoding decision.
test_inference(args,model,test,'results/t5_ft_experiment_test.sql','records/t5_ft_experiment_test.pkl')
Path('logs/final_selection.json').write_text(json.dumps(selection,indent=2))
(best/'final_selection.json').write_text(json.dumps(selection,indent=2))
if Path('environment.txt').exists():
    shutil.copyfile('environment.txt',best/'environment.txt')
print('FINAL_SELECTION',json.dumps(selection),flush=True)
