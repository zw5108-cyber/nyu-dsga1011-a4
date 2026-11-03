"""T5 text-to-SQL data, including a lossless token for SQL's less-than sign."""
from pathlib import Path
from functools import lru_cache
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import T5TokenizerFast, AddedToken

PAD_IDX = 0
PREFIX = 'translate English to SQL: '

@lru_cache(maxsize=1)
def get_tokenizer():
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    tokenizer.add_tokens([AddedToken('<', normalized=False)])
    return tokenizer


def load_lines(path):
    return Path(path).read_text().splitlines()


class T5Dataset(Dataset):
    def __init__(self, data_folder, split):
        self.tokenizer = get_tokenizer()
        self.split = split
        self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        self.texts = load_lines(Path(data_folder) / f'{split}.nl')
        self.sqls = None if split == 'test' else load_lines(Path(data_folder) / f'{split}.sql')
        self.encoder_ids = tokenizer([PREFIX + x.strip() for x in self.texts],
                                     add_special_tokens=True)['input_ids']
        self.target_ids = None if self.sqls is None else tokenizer(
            [x.strip() for x in self.sqls], add_special_tokens=True)['input_ids']
        if self.sqls is not None:
            assert len(self.sqls) == len(self.texts)
            assert max(map(len, self.target_ids)) <= 640, 'Unexpected SQL length; do not silently truncate.'
        assert max(map(len, self.encoder_ids)) <= 128

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoder = torch.tensor(self.encoder_ids[idx], dtype=torch.long)
        if self.target_ids is None:
            return encoder
        target = torch.tensor(self.target_ids[idx], dtype=torch.long)
        decoder = torch.cat([torch.tensor([PAD_IDX]), target[:-1]])
        return encoder, decoder, target


def normal_collate_fn(batch):
    encoder_ids, decoder_inputs, targets = zip(*batch)
    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    targets = pad_sequence(targets, batch_first=True, padding_value=PAD_IDX)
    return encoder_ids, (encoder_ids != PAD_IDX).long(), decoder_inputs, targets, torch.zeros((len(batch),1),dtype=torch.long)


def test_collate_fn(batch):
    encoder_ids = pad_sequence(batch, batch_first=True, padding_value=PAD_IDX)
    return encoder_ids, (encoder_ids != PAD_IDX).long(), torch.zeros((len(batch),1),dtype=torch.long)


def get_dataloader(batch_size, split):
    return DataLoader(T5Dataset('data', split), batch_size=batch_size,
                      shuffle=split == 'train',
                      collate_fn=test_collate_fn if split == 'test' else normal_collate_fn)


def load_t5_data(batch_size, test_batch_size):
    return (get_dataloader(batch_size, 'train'), get_dataloader(test_batch_size, 'dev'),
            get_dataloader(test_batch_size, 'test'))


def load_prompting_data(data_folder):
    return tuple(load_lines(Path(data_folder)/name) for name in
                 ['train.nl','train.sql','dev.nl','dev.sql','test.nl'])
