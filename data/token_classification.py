import logging
import os

import torch
import json
import glob
from tqdm import tqdm
from torch.utils.data import Dataset

import sys

class TokenClassificationDataset(Dataset):
    def __init__(self, data_dir, tokenizer, pad_token_label_id, mode):
        self.tokenizer = tokenizer
        with open(f"{data_dir}/label_list.json", "r") as f:
            self.label_list = list(json.load(f).values())
        self.pad_token_label_id = pad_token_label_id
        self.mode = mode

        self.list_annot_files_path = get_annotations_files_path(data_dir, mode)
        self.label2idx = {label: i for i, label in enumerate(self.label_list)}
        self.idx2label = {i: label for i, label in enumerate(self.label_list)}

    def __len__(self):
        return len(self.list_annot_files_path)

    def __getitem__(self, index):
        annot_file_path = self.list_annot_files_path[index]

        example = read_example_from_json(annot_file_path, self.mode)

        # Convert example to features
        feature = convert_annotation_to_features(
            example,
            self.label2idx,
            max_seq_length=512,
            tokenizer=self.tokenizer,
            cls_token_at_end=False,
            cls_token=self.tokenizer.cls_token,
            cls_token_segment_id=0,
            sep_token=self.tokenizer.sep_token,
            sep_token_extra=False,
            pad_on_left=False,
            pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
            pad_token_segment_id=0,
            pad_token_label_id=self.pad_token_label_id,
        )

        # Convert features to tensors
        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        input_mask = torch.tensor(feature.input_mask, dtype=torch.long)
        segment_ids = torch.tensor(feature.segment_ids, dtype=torch.long)
        label_ids = torch.tensor(feature.label_ids, dtype=torch.long)
        bboxes = torch.tensor(feature.normalized_boxes, dtype=torch.long)

        # Mask indicating the position of the first token of each subtokenized word
        first_token_mask = (label_ids != -100).to(torch.bool)

        return {
            "file_name": feature.file_name,
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": segment_ids,
            "labels": label_ids,
            "bbox": bboxes,
            "first_token_mask": first_token_mask,
        }


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, normalized_boxes, actual_boxes, file_name, page_size):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        self.normalized_boxes = normalized_boxes
        self.actual_boxes = actual_boxes
        self.file_name = file_name
        self.page_size = page_size


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
            self,
            input_ids,
            input_mask,
            segment_ids,
            label_ids,
            normalized_boxes,
            actual_boxes,
            file_name,
            page_size,
    ):
        assert (
                0 <= all(normalized_boxes) <= 1000
        ), "Error with input bbox ({}): the coordinate value is not between 0 and 1000".format(
            normalized_boxes
        )
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.normalized_boxes = normalized_boxes
        self.actual_boxes = actual_boxes
        self.file_name = file_name
        self.page_size = page_size


def read_example_from_json(json_file, mode):
    with open(json_file, 'r') as file:
        json_content = json.load(file)

    docid = json_content['docid']
    page_nb = json_content['page']
    page_size = json_content['page_width'], json_content['page_height']
    words = json_content['words']
    labels = json_content['labels']
    normalized_boxes = json_content['boxes']
    actual_boxes = json_content['actual_boxes']

    return InputExample(guid=f"{mode}-{docid}-{page_nb}",
                        words=words,
                        labels=labels,
                        normalized_boxes=normalized_boxes,
                        actual_boxes=actual_boxes,
                        file_name=f"{docid}_{page_nb}",
                        page_size=page_size)


def get_annotations_files_path(data_dir, mode):
    """
    Get the path of the annotation files for a given mode (train, test)
    Args:
        data_dir: The directory where the data is stored
        mode: The mode (train, test)
    """
    list_annot_files = []
    annotations_path = os.path.join(data_dir, mode, 'annotations')
    if annotations_path[-1] == '/':
        annotations_path = annotations_path[:-1]
    # guid_index = 0

    list_annot_files = glob.glob(f'{annotations_path}/*.json')

    return list_annot_files


def convert_annotation_to_features(
        example,
        label_map,
        max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=0,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        cls_token_box=[0, 0, 0, 0],
        sep_token_box=[1000, 1000, 1000, 1000],
        pad_token_box=[0, 0, 0, 0],
        pad_token_segment_id=0,
        pad_token_label_id=-1,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    file_name = example.file_name
    page_size = example.page_size
    width, height = page_size

    tokens = []
    token_boxes = []
    actual_boxes = []
    label_ids = []

    for word, label, box, actual_bbox in zip(
            example.words, example.labels, example.normalized_boxes, example.actual_boxes
    ):
        word_tokens = tokenizer.tokenize(word)
        if word == '':
            continue
        tokens.extend(word_tokens)
        token_boxes.extend([box] * len(word_tokens))
        actual_boxes.extend([actual_bbox] * len(word_tokens))
        if len([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1)) != len(word_tokens):
            saved_word = word
        # Use the real label id for the first token of the word, and padding ids for the remaining tokens
        label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1)
                         )

    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = 3 if sep_token_extra else 2
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[: (max_seq_length - special_tokens_count)]
        token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]
        actual_boxes = actual_boxes[: (max_seq_length - special_tokens_count)]
        label_ids = label_ids[: (max_seq_length - special_tokens_count)]

    tokens += [sep_token]
    token_boxes += [sep_token_box]
    actual_boxes += [[0, 0, width, height]]
    label_ids += [pad_token_label_id]
    if sep_token_extra:
        # roberta uses an extra separator b/w pairs of sentences
        tokens += [sep_token]
        token_boxes += [sep_token_box]
        actual_boxes += [[0, 0, width, height]]
        label_ids += [pad_token_label_id]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    if cls_token_at_end:
        tokens += [cls_token]
        token_boxes += [cls_token_box]
        actual_boxes += [[0, 0, width, height]]
        label_ids += [pad_token_label_id]
        segment_ids += [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        token_boxes = [cls_token_box] + token_boxes
        actual_boxes = [[0, 0, width, height]] + actual_boxes
        label_ids = [pad_token_label_id] + label_ids
        segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = (
                             [0 if mask_padding_with_zero else 1] * padding_length
                     ) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        label_ids = ([pad_token_label_id] * padding_length) + label_ids
        token_boxes = ([pad_token_box] * padding_length) + token_boxes
    else:
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        label_ids += [pad_token_label_id] * padding_length
        token_boxes += [pad_token_box] * padding_length

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(token_boxes) == max_seq_length

    return InputFeatures(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         label_ids=label_ids,
                         normalized_boxes=token_boxes,
                         actual_boxes=actual_boxes,
                         file_name=file_name,
                         page_size=page_size)
