# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset


class GCL4SRData(Dataset):

    def __init__(self, args, data, test_neg_items=None):
        self.args = args
        self.data = data
        self.test_neg_items = test_neg_items
        self.max_len = args.max_seq_length
        self.uid_list = data[0]
        self.part_sequence = data[1]
        self.part_sequence_target = data[2]
        self.part_sequence_length = data[3]
        self.length = len(data[0])

    def __getitem__(self, index):

        input_ids = self.part_sequence[index]
        target_pos = self.part_sequence_target[index]
        user_id = self.uid_list[index]

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids

        input_ids = input_ids[-self.max_len:]

        assert len(input_ids) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
            )

        return cur_tensors

    def __len__(self):
        return self.length
