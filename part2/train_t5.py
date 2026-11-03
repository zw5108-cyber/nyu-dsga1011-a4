"""Reproducible T5 fine-tuning; development selects models, test runs explicitly."""
import argparse
import json
import time
from pathlib import Path
import torch
from torch import nn
from tqdm import tqdm
from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint
from load_data import load_t5_data, get_tokenizer
from utils import compute_metrics, save_queries_and_records, set_random_seeds

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PAD_IDX = 0


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--finetune', action='store_true')
    p.add_argument('--optimizer_type', default='AdamW', choices=['AdamW'])
    p.add_argument('--learning_rate', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--scheduler_type', default='linear', choices=['none','cosine','linear'])
    p.add_argument('--num_warmup_epochs', type=int, default=1)
    p.add_argument('--max_n_epochs', type=int, default=30)
    p.add_argument('--patience_epochs', type=int, default=15)
    p.add_argument('--experiment_name', default='experiment')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--test_batch_size', type=int, default=16)
    p.add_argument('--eval_every', type=int, default=5)
    p.add_argument('--num_beams', type=int, default=1)
    p.add_argument('--max_new_tokens', type=int, default=640)
    p.add_argument('--eval_only', action='store_true')
    p.add_argument('--run_test', action='store_true')
    return p.parse_args()


def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = total_tokens = 0
    for encoder, mask, decoder, targets, _ in tqdm(train_loader, desc='train'):
        optimizer.zero_grad(set_to_none=True)
        encoder, mask, decoder, targets = [x.to(DEVICE) for x in (encoder, mask, decoder, targets)]
        logits = model(input_ids=encoder, attention_mask=mask, decoder_input_ids=decoder).logits
        non_pad = targets != PAD_IDX
        loss = nn.functional.cross_entropy(logits[non_pad], targets[non_pad])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        n = non_pad.sum().item()
        total_loss += loss.item() * n
        total_tokens += n
    return total_loss / total_tokens


@torch.inference_mode()
def validation_loss(model, loader):
    model.eval()
    loss_sum = token_sum = 0
    for encoder, mask, decoder, targets, _ in loader:
        encoder, mask, decoder, targets = [x.to(DEVICE) for x in (encoder, mask, decoder, targets)]
        logits = model(input_ids=encoder, attention_mask=mask, decoder_input_ids=decoder).logits
        keep = targets != PAD_IDX
        loss_sum += nn.functional.cross_entropy(logits[keep], targets[keep], reduction='sum').item()
        token_sum += keep.sum().item()
    return loss_sum / token_sum


def decode_sql(ids):
    # Special PAD/EOS tokens are stripped; '<' is an ordinary added token.
    texts = get_tokenizer().batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [' '.join(x.replace('< =','<=').replace('> =','>=').split()) for x in texts]


@torch.inference_mode()
def generate_queries(args, model, loader):
    model.eval()
    queries = []
    for batch in tqdm(loader, desc=f'generate beams={args.num_beams}'):
        encoder, mask = batch[:2]
        ids = model.generate(input_ids=encoder.to(DEVICE), attention_mask=mask.to(DEVICE),
                             num_beams=args.num_beams, max_new_tokens=args.max_new_tokens,
                             do_sample=False, length_penalty=1.0,
                             decoder_start_token_id=0, eos_token_id=1, pad_token_id=0)
        queries.extend(decode_sql(ids))
    assert len(queries) == len(loader.dataset)
    return queries


def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    loss = validation_loss(model, dev_loader)
    queries = generate_queries(args, model, dev_loader)
    save_queries_and_records(queries, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, errors = compute_metrics(gt_sql_pth, model_sql_path, gt_record_path, model_record_path)
    error_rate = sum(bool(x) for x in errors) / len(errors)
    return loss, record_f1, record_em, sql_em, error_rate


def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    if Path(model_sql_path).exists() or Path(model_record_path).exists():
        raise FileExistsError('Test outputs already exist; refusing an accidental repeated test run.')
    save_queries_and_records(generate_queries(args, model, test_loader), model_sql_path, model_record_path)


def metrics_dict(values):
    return dict(zip(['loss','record_f1','record_em','sql_em','sql_error_rate'], map(float, values)))


def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    history = []
    best_f1 = -1
    best_epoch = 0
    checkpoint_dir = Path('checkpoints') / 'ft_experiments' / args.experiment_name
    for epoch in range(1, args.max_n_epochs + 1):
        started = time.time()
        entry = {'epoch':epoch,'train_loss':train_epoch(args, model, train_loader, optimizer, scheduler)}
        if epoch % args.eval_every == 0 or epoch == args.max_n_epochs:
            stem = f't5_ft_{args.experiment_name}_epoch{epoch:02d}_dev'
            values = eval_epoch(args, model, dev_loader, 'data/dev.sql',f'results/{stem}.sql',
                                'records/ground_truth_dev.pkl',f'records/{stem}.pkl')
            entry.update(metrics_dict(values))
            if entry['record_f1'] > best_f1:
                best_f1, best_epoch = entry['record_f1'], epoch
                save_model(str(checkpoint_dir), model, best=True)
                (checkpoint_dir/'best'/'selection.json').write_text(json.dumps({'epoch':epoch,**entry},indent=2))
        else:
            entry['loss'] = validation_loss(model, dev_loader)
        entry['seconds'] = time.time()-started
        history.append(entry)
        Path(f'logs/{args.experiment_name}_history.json').write_text(json.dumps(history,indent=2))
        print(json.dumps(entry),flush=True)
        if best_epoch and epoch-best_epoch >= args.patience_epochs:
            break
    save_model(str(checkpoint_dir), model, best=False)
    (checkpoint_dir/'config.json').write_text(json.dumps(vars(args),indent=2))
    return history


def main():
    args=get_args()
    assert args.finetune, 'This run implements Q4-Q7; scratch training is optional extra credit.'
    torch.set_num_threads(6)
    set_random_seeds(42)
    for folder in ['logs','results','records']:
        Path(folder).mkdir(exist_ok=True)
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size,args.test_batch_size)
    if not args.eval_only:
        model = initialize_model(args)
        print('PARAMETERS',sum(p.numel() for p in model.parameters()),flush=True)
        optimizer,scheduler=initialize_optimizer_and_scheduler(args,model,len(train_loader))
        train(args,model,train_loader,dev_loader,optimizer,scheduler)
        del model,optimizer,scheduler
        torch.cuda.empty_cache()
    model=load_model_from_checkpoint(args,best=True)
    stem=f't5_ft_{args.experiment_name}_dev_beam{args.num_beams}'
    values=eval_epoch(args,model,dev_loader,'data/dev.sql',f'results/{stem}.sql',
                      'records/ground_truth_dev.pkl',f'records/{stem}.pkl')
    metrics=metrics_dict(values)
    Path(f'logs/{stem}.json').write_text(json.dumps(metrics,indent=2))
    print('DEV_FINAL',json.dumps(metrics),flush=True)
    if args.run_test:
        test_inference(args,model,test_loader,f'results/t5_ft_{args.experiment_name}_test.sql',
                        f'records/t5_ft_{args.experiment_name}_test.pkl')
        Path(f'logs/{args.experiment_name}_test_generation.json').write_text(json.dumps(
            {'args':vars(args),'examples':len(test_loader.dataset),'dev_metrics':metrics},indent=2))

if __name__ == '__main__':
    main()
