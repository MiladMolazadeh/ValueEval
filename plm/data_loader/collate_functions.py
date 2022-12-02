"""
    script for collate fns
"""

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(method_type="sequence"):
    """Factory Method for collate fn"""
    functions = {
        "sequence": sequence_collate,
    }

    return functions[method_type]


def sequence_collate(batch) -> dict:
    """
    for dataloader
    Args:
        batch: batch of data

    Returns:

    """
    input_ids, attention_ids, labels = zip(*batch)
    padded_xs = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_masks = pad_sequence(attention_ids, batch_first=True, padding_value=0)

    return {'input_ids': padded_xs,
            'attention_mask': padded_masks,
            'labels': torch.tensor(labels)}
