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

import numpy as np
import paddle
import paddle.nn.functional as F

import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup

from data import read_fn_dict, convert_example, create_dataloader
from model import ErnieForPretraining, ErnieMLMCriterion

# yapf: disable
parser = argparse.ArgumentParser()

parser.add_argument("--task_data_dir", default="../datasets/", type=str, help="The default dir contain datasets")
parser.add_argument("--task_name", required=True, type=str, help="The task_name to be evaluated")
parser.add_argument("--train_set_index", type=str, default="0", help="training set index")
parser.add_argument("--p_embedding_num", type=int, default=1, help="number of p-embedding")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--save_dir", default='./checkpoint', type=str, help="The output directory where the model checkpoints will be written.")
parser.add_argument("--output_dir", default='./result/', type=str, help="The output directory where the model checkpoints will be written.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion", default=0.0, type=float, help="Linear warmup proption over the training process.")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
parser.add_argument("--seed", type=int, default=1000, help="random seed for initialization")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument('--save_steps', type=int, default=10000, help="Inteval steps to save checkpoint")

args = parser.parse_args()
# yapf: enable


def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


@paddle.no_grad()
def do_evaluate(model, tokenizer, data_loader, label_normalize_dict):
    model.eval()

    total_num = 0
    correct_num = 0

    normed_labels = [
        normalized_lable
        for origin_lable, normalized_lable in label_normalize_dict.items()
    ]

    label_length = len(normed_labels[0])

    for batch in data_loader:
        src_ids, token_type_ids, masked_positions, masked_lm_labels = batch

        # [bs * label_length, vocab_size]
        prediction_probs = model.predict(
            input_ids=src_ids,
            token_type_ids=token_type_ids,
            masked_positions=masked_positions)

        batch_size = len(src_ids)
        vocab_size = prediction_probs.shape[1]

        #prediction_probs: [batch_size, label_lenght, vocab_size]
        prediction_probs = paddle.reshape(
            prediction_probs, shape=[batch_size, -1, vocab_size]).numpy()

        # [label_num, label_length]
        label_ids = np.array(
            [tokenizer(label)["input_ids"][1:-1] for label in normed_labels])

        y_pred = np.ones(shape=[batch_size, len(label_ids)])

        # calculate joint distribution of candidate labels
        for index in range(label_length):
            y_pred *= prediction_probs[:, index, label_ids[:, index]]

        # get max probs label's index
        y_pred_index = np.argmax(y_pred, axis=-1)

        y_true_index = []
        for masked_lm_label in masked_lm_labels.numpy():
            label_text = "".join(
                tokenizer.convert_ids_to_tokens(list(masked_lm_label)))

            label_index = normed_labels.index(label_text)
            y_true_index.append(label_index)

        y_true_index = np.array(y_true_index)

        total_num += len(y_true_index)
        correct_num += (y_true_index == y_pred_index).sum()

    return 100 * correct_num / total_num, total_num


def do_train():
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)

    train_set = os.path.join(args.task_data_dir, args.task_name,
                             "train_" + args.train_set_index + ".json")

    dev_set = os.path.join(args.task_data_dir, args.task_name,
                           "dev_" + args.train_set_index + ".json")
    public_test_set = os.path.join(args.task_data_dir, args.task_name,
                                   "test_public.json")

    label_normalize_json = os.path.join("./label_normalized",
                                        args.task_name + ".json")

    label_norm_dict = None
    with open(label_normalize_json) as f:
        label_norm_dict = json.load(f)

    read_fn = read_fn_dict[args.task_name]
    train_ds = load_dataset(
        read_fn,
        data_path=train_set,
        lazy=False,
        label_normalize_dict=label_norm_dict)

    dev_ds = load_dataset(
        read_fn,
        data_path=dev_set,
        lazy=False,
        label_normalize_dict=label_norm_dict)

    public_test_ds = load_dataset(
        read_fn,
        data_path=public_test_set,
        lazy=False,
        label_normalize_dict=label_norm_dict)

    model = ErnieForPretraining.from_pretrained('ernie-1.0')

    tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained('ernie-1.0')

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        p_embedding_num=args.p_embedding_num)

    # [src_ids, token_type_ids, masked_positions, masked_lm_labels]
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
        Stack(dtype="int64"),  # masked_positions
        Stack(dtype="int64"),  # masked_lm_labels
    ): [data for data in fn(samples)]

    train_data_loader = create_dataloader(
        train_ds,
        mode='train',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    dev_data_loader = create_dataloader(
        dev_ds,
        mode='eval',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    public_test_data_loader = create_dataloader(
        public_test_ds,
        mode='eval',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
        print("warmup from:{}".format(args.init_from_ckpt))

    mlm_loss_fn = ErnieMLMCriterion()

    num_training_steps = len(train_data_loader) * args.epochs

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_proportion)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    global_step = 0
    tic_train = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for step, batch in enumerate(train_data_loader, start=1):

            src_ids = batch[0]
            token_type_ids = batch[1]
            masked_positions = batch[2]
            masked_lm_labels = batch[3]

            prediction_scores = model(
                input_ids=src_ids,
                token_type_ids=token_type_ids,
                masked_positions=masked_positions)

            loss = mlm_loss_fn(prediction_scores, masked_lm_labels)

            global_step += 1
            if global_step % 10 == 0 and rank == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss,
                       10 / (time.time() - tic_train)))
                tic_train = time.time()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

        dev_accuracy, total_num = do_evaluate(model, tokenizer, dev_data_loader,
                                              label_norm_dict)
        print("epoch:{}\tdev_accuracy:{:.3f}\ttotal_num:{}".format(
            epoch, dev_accuracy, total_num))
        test_accuracy, total_num = do_evaluate(
            model, tokenizer, public_test_data_loader, label_norm_dict)
        print("epoch:{}\ttest_accuracy:{:.3f}\ttotal_num:{}".format(
            epoch, test_accuracy, total_num))

        if rank == 0:
            save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_param_path = os.path.join(save_dir, 'model_state.pdparams')
            paddle.save(model.state_dict(), save_param_path)
            tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    do_train()
