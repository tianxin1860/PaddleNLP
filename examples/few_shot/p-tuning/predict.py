# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys
import random
import time
import json
from functools import partial

from tqdm import tqdm

import numpy as np
import paddle
import paddle.nn.functional as F

import paddlenlp as ppnlp
from model import ErnieForPretraining
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset

from data import create_dataloader, transform_fn_dict
from data import convert_example, convert_chid_example
from evaluate import do_evaluate, do_evaluate_chid


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task_name",
        required=True,
        type=str,
        help="The task_name to be evaluated")
    parser.add_argument(
        "--p_embedding_num", type=int, default=1, help="number of p-embedding")
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--extend_batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. "
        "Sequences longer than this will be truncated, sequences shorter will be padded."
    )
    parser.add_argument(
        "--init_from_ckpt",
        type=str,
        default=None,
        help="The path of checkpoint to be loaded.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The path of checkpoint to be loaded.")
    parser.add_argument(
        "--seed", type=int, default=1000, help="random seed for initialization")
    parser.add_argument(
        '--device',
        choices=['cpu', 'gpu'],
        default="gpu",
        help="Select which device to train model, defaults to gpu.")
    parser.add_argument(
        '--save_steps',
        type=int,
        default=10000,
        help="Inteval steps to save checkpoint")

    args = parser.parse_args()
    return args


def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


@paddle.no_grad()
def do_predict(model, tokenizer, data_loader, label_normalize_dict):
    model.eval()

    normed_labels = [
        normalized_lable
        for origin_lable, normalized_lable in label_normalize_dict.items()
    ]

    origin_labels = [
        origin_lable
        for origin_lable, normalized_lable in label_normalize_dict.items()
    ]

    label_length = len(normed_labels[0])

    y_pred_labels = []
    y_pred_probs = []

    for batch in tqdm(data_loader):
        src_ids, token_type_ids, masked_positions = batch

        max_len = src_ids.shape[1]
        new_masked_positions = []
        # masked_positions: [bs, label_length]
        for bs_index, mask_pos in enumerate(masked_positions.numpy()):
            for pos in mask_pos:
                new_masked_positions.append(bs_index * max_len + pos)
        # new_masked_positions: [bs * label_length, 1]
        new_masked_positions = np.array(new_masked_positions).astype('int32')
        new_masked_positions = paddle.to_tensor(new_masked_positions)

        prediction_scores, _ = model(
            input_ids=src_ids,
            token_type_ids=token_type_ids,
            masked_positions=new_masked_positions)

        softmax_fn = paddle.nn.Softmax()
        prediction_probs = softmax_fn(prediction_scores)

        batch_size = len(src_ids)
        vocab_size = prediction_probs.shape[1]

        # prediction_probs: [batch_size, label_lenght, vocab_size]
        prediction_probs = paddle.reshape(
            prediction_probs, shape=[batch_size, -1, vocab_size]).numpy()

        # [label_num, label_length]
        label_ids = np.array(
            [tokenizer(label)["input_ids"][1:-1] for label in normed_labels])

        y_pred = np.ones(shape=[batch_size, len(label_ids)])

        # Calculate joint distribution of candidate labels
        for index in range(label_length):
            y_pred *= prediction_probs[:, index, label_ids[:, index]]

        y_pred_probs.append(y_pred)
        # Get max probs label's index
        y_pred_index = np.argmax(y_pred, axis=-1)

        for index in y_pred_index:
            y_pred_labels.append(origin_labels[index])

    y_pred_probs = np.concatenate(y_pred_probs, axis=0)
    model.train()
    # y_pred: probs
    return y_pred_labels, y_pred_probs


@paddle.no_grad()
def do_predict_chid(model, tokenizer, data_loader, label_normalize_dict):
    """
        FewCLUE `chid` dataset is specical when evaluate: input slots have 
        additional `candidate_label_ids`, so need to customize the
        evaluate function.
    """

    model.eval()

    normed_labels = [
        normalized_lable
        for origin_lable, normalized_lable in label_normalize_dict.items()
    ]

    label_length = len(normed_labels[0])

    y_pred_all = []
    y_pred_probs = []

    for batch in data_loader:
        src_ids, token_type_ids, masked_positions, candidate_label_ids = batch

        # [bs * label_length, vocab_size]
        # prediction_probs = model.predict(
        #     input_ids=src_ids,
        #     token_type_ids=token_type_ids,
        #     masked_positions=masked_positions)

        max_len = src_ids.shape[1]
        new_masked_positions = []
        # masked_positions: [bs, label_length]
        for bs_index, mask_pos in enumerate(masked_positions.numpy()):
            for pos in mask_pos:
                new_masked_positions.append(bs_index * max_len + pos)
        # new_masked_positions: [bs * label_length, 1]
        new_masked_positions = np.array(new_masked_positions).astype('int32')
        new_masked_positions = paddle.to_tensor(new_masked_positions)

        prediction_scores, _ = model(
            input_ids=src_ids,
            token_type_ids=token_type_ids,
            masked_positions=new_masked_positions)
        softmax_fn = paddle.nn.Softmax()
        prediction_probs = softmax_fn(prediction_scores)

        batch_size = len(src_ids)
        vocab_size = prediction_probs.shape[1]

        # prediction_probs: [batch_size, label_lenght, vocab_size]
        prediction_probs = paddle.reshape(
            prediction_probs, shape=[batch_size, -1, vocab_size]).numpy()

        candidate_num = candidate_label_ids.shape[1]

        # [batch_size, candidate_num(7)]
        y_pred = np.ones(shape=[batch_size, candidate_num])

        for label_idx in range(candidate_num):
            # [bathc_size, label_length(4)] 
            single_candidate_label_ids = candidate_label_ids[:, label_idx, :]
            # Calculate joint distribution of candidate labels
            for index in range(label_length):
                # [batch_size,]
                slice_word_ids = single_candidate_label_ids[:, index].numpy()

                batch_single_token_prob = []
                for bs_index in range(batch_size):
                    # [1, 1]
                    single_token_prob = prediction_probs[
                        bs_index, index, slice_word_ids[bs_index]]
                    batch_single_token_prob.append(single_token_prob)

                y_pred[:, label_idx] *= np.array(batch_single_token_prob)

        # Get max probs label's index
        y_pred_index = np.argmax(y_pred, axis=-1)
        y_pred_all.extend(y_pred_index)
        y_pred_probs.append(y_pred)

    y_pred_probs = np.concatenate(y_pred_probs, axis=0)

    model.train()
    
    return y_pred_all, y_pred_probs


predict_file = {
    "bustm": "bustm_predict.json",
    "chid": "chidf_predict.json",
    "cluewsc": "cluewscf_predict.json",
    "csldcp": "csldcp_predict.json",
    "csl": "cslf_predict.json",
    "eprstmt": "eprstmt_predict.json",
    "iflytek": "iflytekf_predict.json",
    "ocnli": "ocnlif_predict.json",
    "tnews": "tnewsf_predict.json"
}

unlabeled_file_dict = {
    "bustm": "unlabeled.json",
    "chid": "unlabeled.json",
    "cluewsc": "unlabeled.json",
    "csldcp": "unlabeled.json",
    "csl": "unlabeled.json",
    "eprstmt": "unlabeled.json",
    "iflytek": "unlabeled.json",
    "ocnli": "unlabeled.json",
    "tnews": "unlabeled.json"
}

def write_iflytek(task_name, output_file, pred_labels, probs, is_test=True, min_prob=0.7):
    if is_test:
        test_ds, train_few_all = load_dataset(
            "fewclue", name="iflytek", splits=("test", "train_few_all"))

        def label2id(train_few_all):
            label2id = {}
            for example in train_few_all:
                label = example["label_des"]
                label_id = example["label"]
                if label not in label2id:
                    label2id[label] = str(label_id)
            return label2id

        label2id_dict = label2id(train_few_all)

        test_example = {}
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, example in enumerate(test_ds):
                test_example["id"] = example["id"]
                test_example["label"] = label2id_dict[pred_labels[idx]]

                str_test_example = json.dumps(test_example) + "\n"
                f.write(str_test_example)
    else:
        test_ds, train_few_all = load_dataset(
            "fewclue", name="iflytek", splits=("unlabeled", "train_few_all"))

        def label2id(train_few_all):
            label2id = {}
            for example in train_few_all:
                label = example["label_des"]
                label_id = example["label"]
                if label not in label2id:
                    label2id[label] = str(label_id)
            return label2id

        label2id_dict = label2id(train_few_all)

        test_example = {}
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, example in enumerate(test_ds):
                test_example["id"] = example["id"]
                test_example["label"] = label2id_dict[pred_labels[idx]]
                test_example["label_des"] = pred_labels[idx]
                test_example["sentence"] = example["sentence"]

                prob = max(probs[idx])
                if prob >= min_prob:
                    str_test_example = str(test_example)
                    f.write(str_test_example + "\n")
                else:
                    continue
        return None


def write_bustm(task_name, output_file, pred_labels, probs, is_test=True, min_prob=0.7):
    if is_test:
        test_ds = load_dataset("fewclue", name="bustm", splits=("test"))
        test_example = {}
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, example in enumerate(test_ds):
                test_example["id"] = example["id"]
                test_example["label"] = pred_labels[idx]
                str_test_example = json.dumps(test_example) + "\n"
                f.write(str_test_example)
    else:
        test_ds = load_dataset("fewclue", name="bustm", splits=("unlabeled"))
        #data_file = "/home/tianxin04/.paddlenlp/datasets/FewCLUE/fewclue_" + task_name + "/unlabeled_demo.json"
        #test_ds = load_dataset("fewclue", name="bustm", data_files=data_file)

        test_example = {}
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, example in enumerate(test_ds):
                test_example["id"] = example["id"]
                test_example["label"] = pred_labels[idx]
                test_example["sentence1"] = example["sentence1"]
                test_example["sentence2"] = example["sentence2"]

                prob = max(probs[idx])
                if prob >= min_prob:
                    str_test_example = str(test_example)
                    f.write(str_test_example + "\n")
                else:
                    continue
        return None

def write_csldcp(task_name, output_file, pred_labels, probs, is_test=True, min_prob=0.7):
    if is_test:
        test_ds = load_dataset("fewclue", name="csldcp", splits=("test"))
        test_example = {}

        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, example in enumerate(test_ds):
                test_example["id"] = example["id"]
                test_example["label"] = pred_labels[idx]
                # {"id": 0, "label": "力学"}
                str_test_example = "\"{}\": {}, \"{}\": \"{}\"".format(
                    "id", test_example['id'], "label", test_example["label"])
                f.write("{" + str_test_example + "}\n")
    else:
        test_ds = load_dataset("fewclue", name="csldcp", splits=("unlabeled"))
        test_example = {}

        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, example in enumerate(test_ds):
                test_example["id"] = example["id"]
                test_example["label"] = pred_labels[idx]
                test_example["content"] = example["content"]
                # {"id": 0, "label": "力学"}

                prob = max(probs[idx])
                if prob >= min_prob:
                    str_test_example = str(test_example)
                    f.write(str_test_example + "\n")
                else:
                    continue

                # str_test_example = "\"{}\": {}, \"{}\": \"{}\"".format(
                #     "id", test_example['id'], "label", test_example["label"])
                # f.write("{" + str_test_example + "}\n")
        return None


def write_tnews(task_name, output_file, pred_labels, probs, is_test=True, min_prob=0.7):

    if is_test:
        test_ds, train_few_all = load_dataset(
            "fewclue", name="tnews", splits=("test", "train_few_all"))

        def label2id(train_few_all):
            label2id = {}
            for example in train_few_all:
                label = example["label_desc"]
                label_id = example["label"]
                if label not in label2id:
                    label2id[label] = str(label_id)
            return label2id

        label2id_dict = label2id(train_few_all)

        test_example = {}
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, example in enumerate(test_ds):
                test_example["id"] = example["id"]
                test_example["label"] = label2id_dict[pred_labels[idx]]

                str_test_example = json.dumps(test_example) + "\n"
                f.write(str_test_example)
    else:
        test_ds, train_few_all = load_dataset(
            "fewclue", name="tnews", splits=("unlabeled", "train_few_all"))

        def label2id(train_few_all):
            label2id = {}
            for example in train_few_all:
                label = example["label_desc"]
                label_id = example["label"]
                if label not in label2id:
                    label2id[label] = str(label_id)
            return label2id

        label2id_dict = label2id(train_few_all)

        test_example = {}
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, example in enumerate(test_ds):
                test_example["id"] = example["id"]
                test_example["label"] = label2id_dict[pred_labels[idx]]
                test_example["label_desc"] = pred_labels[idx]
                test_example['sentence'] = example['sentence']
                
                prob = max(probs[idx])
                if prob >= min_prob:
                    str_test_example = str(test_example)
                    f.write(str_test_example + "\n")
                else:
                    continue
        return None


def write_cluewsc(task_name, output_file, pred_labels, probs, is_test=True, min_prob=0.7):
    test_ds = load_dataset("fewclue", name="cluewsc", splits=("test"))
    test_example = {}
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(test_ds):
            test_example["id"] = example["id"]
            test_example["label"] = pred_labels[idx]
            # {"id": 0, "label": "力学"}
            str_test_example = "\"{}\": {}, \"{}\": \"{}\"".format(
                "id", test_example['id'], "label", test_example["label"])
            f.write("{" + str_test_example + "}\n")


def write_eprstmt(task_name, output_file, pred_labels, probs, is_test=True, min_prob=0.7):
    # predict for test.json
    if is_test:
        test_ds = load_dataset("fewclue", name="eprstmt", splits=("test"))
        test_example = {}
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, example in enumerate(test_ds):
                test_example["id"] = example["id"]
                test_example["label"] = pred_labels[idx]

                str_test_example = json.dumps(test_example)
                f.write(str_test_example + "\n")
        return None
    else:
        #predict for unlabeled.json
        test_ds = load_dataset("fewclue", name="eprstmt", splits=("unlabeled"))
        #test_ds = load_dataset("fewclue", name="eprstmt", data_files="/home/tianxin04/.paddlenlp/datasets/FewCLUE/fewclue_eprstmt/unlabeled_demo.json")

        all_examples = []
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, example in enumerate(test_ds):
                test_example = {}
                test_example["id"] = example["id"]
                test_example["sentence"] = example["sentence"]
                test_example["label"] = pred_labels[idx]

                prob = max(probs[idx])
                if prob >= min_prob:
                    str_test_example = str(test_example)
                    f.write(str_test_example + "\n")
                    all_examples.append(test_example)
                else:
                    continue
        return all_examples

def write_ocnli(task_name, output_file, pred_labels, probs, is_test=True, min_prob=0.7):
    if is_test:
        test_ds = load_dataset("fewclue", name="ocnli", splits=("test"))
        test_example = {}
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, example in enumerate(test_ds):
                test_example["id"] = example["id"]
                test_example["label"] = pred_labels[idx]
                str_test_example = json.dumps(test_example)
                f.write(str_test_example + "\n")
    else:
        test_ds = load_dataset("fewclue", name="ocnli", splits=("unlabeled"))
        test_example = {}
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, example in enumerate(test_ds):
                test_example["id"] = example["id"]
                test_example["label"] = pred_labels[idx]
                test_example["sentence1"] =  example["sentence1"]
                test_example["sentence2"] =  example["sentence2"]

                prob = max(probs[idx])
                if prob >= min_prob:
                    str_test_example = str(test_example)
                    f.write(str_test_example + "\n")
                else:
                    continue

                # str_test_example = json.dumps(test_example)
                # f.write(str_test_example + "\n")
        return None


def write_csl(task_name, output_file, pred_labels, probs, is_test=True, min_prob=0.7):
    if is_test:
        test_ds = load_dataset("fewclue", name="csl", splits=("test"))
        test_example = {}
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, example in enumerate(test_ds):
                test_example["id"] = example["id"]
                test_example["label"] = pred_labels[idx]
                str_test_example = json.dumps(test_example)
                f.write(str_test_example + "\n")
    else:
        test_ds = load_dataset("fewclue", name="csl", splits=("unlabeled"))
        test_example = {}
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, example in enumerate(test_ds):
                test_example["id"] = example["id"]
                test_example["label"] = pred_labels[idx]
                test_example["abst"] = example["abst"]
                test_example["keyword"] = example["keyword"]

                prob = max(probs[idx])
                if prob >= min_prob:
                    str_test_example = str(test_example)
                    f.write(str_test_example + "\n")
                else:
                    continue
                # str_test_example = json.dumps(test_example)
                # f.write(str_test_example + "\n")
        return None


def write_chid(task_name, output_file, pred_labels, probs, is_test=True, min_prob=0.7):
    if is_test:
        test_ds = load_dataset("fewclue", name="chid", splits=("test"))
        test_example = {}
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, example in enumerate(test_ds):
                test_example["id"] = example["id"]
                test_example["answer"] = pred_labels[idx]
                str_test_example = "\"{}\": {}, \"{}\": {}".format(
                    "id", test_example['id'], "answer", test_example["answer"])
                f.write("{" + str_test_example + "}\n")
    else:
        test_ds = load_dataset("fewclue", name="chid", splits=("unlabeled"))
        test_example = {}
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, example in enumerate(test_ds):
                test_example["id"] = example["id"]
                test_example["answer"] = pred_labels[idx]
                test_example["candidates"] = example["candidates"]
                test_example["content"] = example["content"]
       
                prob = max(probs[idx])
                if prob >= 0.005:
                    str_test_example = str(test_example)
                    f.write(str_test_example + "\n")
                else:
                    continue
        return None

write_fn = {
    "bustm": write_bustm,
    "iflytek": write_iflytek,
    "csldcp": write_csldcp,
    "tnews": write_tnews,
    "cluewsc": write_cluewsc,
    "eprstmt": write_eprstmt,
    "ocnli": write_ocnli,
    "csl": write_csl,
    "chid": write_chid
}

if __name__ == "__main__":
    args = parse_args()

    paddle.set_device(args.device)
    set_seed(args.seed)

    label_normalize_json = os.path.join("./label_normalized",
                                        args.task_name + ".json")

    label_norm_dict = None
    with open(label_normalize_json) as f:
        label_norm_dict = json.load(f)

    convert_example_fn = convert_example if args.task_name != "chid" else convert_chid_example
    predict_fn = do_predict if args.task_name != "chid" else do_predict_chid

    # Load test_ds for FewCLUE leaderboard
    test_ds = load_dataset("fewclue", name=args.task_name, splits=("test"))

    # Task related transform operations, eg: numbert label -> text_label, english -> chinese
    transform_fn = partial(
        transform_fn_dict[args.task_name],
        label_normalize_dict=label_norm_dict,
        is_test=True)

    # Some fewshot_learning strategy is defined by transform_fn
    # Note: Set lazy=False to transform example inplace immediately,
    # because transform_fn should only be executed only once when 
    # iterate multi-times for train_ds
    test_ds = test_ds.map(transform_fn, lazy=False)

    model = ErnieForPretraining.from_pretrained('ernie-1.0')
    tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained('ernie-1.0')

    # Load parameters of best model on test_public.json of current task
    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.init_from_ckpt)
    else:
        raise ValueError(
            "Please set --params_path with correct pretrained model file")

    if args.task_name != "chid":
        # [src_ids, token_type_ids, masked_positions, masked_lm_labels]
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_ids
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
            Stack(dtype="int64"),  # masked_positions
        ): [data for data in fn(samples)]
    else:
        # [src_ids, token_type_ids, masked_positions, masked_lm_labels, candidate_labels_ids]
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_ids
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
            Stack(dtype="int64"),  # masked_positions
            Stack(dtype="int64"),  # candidate_labels_ids [candidate_num, label_length]
        ): [data for data in fn(samples)]

    trans_func = partial(
        convert_example_fn,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        p_embedding_num=args.p_embedding_num,
        is_test=True)

    test_data_loader = create_dataloader(
        test_ds,
        mode='eval',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    y_pred_labels = predict_fn(model, tokenizer, test_data_loader,
                               label_norm_dict)

    output_file = os.path.join(args.output_dir, predict_file[args.task_name])

    write_fn[args.task_name](args.task_name, output_file, y_pred_labels)
