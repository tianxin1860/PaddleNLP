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
import math
from functools import partial
import gc

from tqdm import tqdm
import numpy as np
import paddle
import paddle.nn.functional as F

import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup

from model import ErnieForPretraining, ErnieMLMCriterion
from data import create_dataloader, transform_fn_dict
from data import convert_example, convert_chid_example
from evaluate import do_evaluate, do_evaluate_chid
from predict import write_fn, do_predict, do_predict_chid, predict_file, unlabeled_file_dict
from self_train import ensemble, ensemble_dict


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task_name",
        required=True,
        type=str,
        help="The task_name to be evaluated")
    parser.add_argument(
        "--language_model",
        required=True,
        type=str,
        help="The model name to be used")
    parser.add_argument(
        "--index",
        required=True,
        type=str,
        default="0",
        help="must be in [0, 1, 2, 3, 4, all]")
    parser.add_argument(
        "--p_embedding_num", type=int, default=1, help="number of p-embedding")
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--predict_batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--pattern_id",
        default=0,
        type=int,
        help="which pattern to be used")
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--save_dir",
        default='./checkpoint',
        type=str,
        help="The output directory where the model checkpoints will be written.")
    parser.add_argument(
        "--output_dir",
        default='./output',
        type=str,
        help="The output directory where to save output")
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. "
        "Sequences longer than this will be truncated, sequences shorter will be padded."
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--warmup_proportion",
        default=0.0,
        type=float,
        help="Linear warmup proption over the training process.")
    parser.add_argument(
        "--init_from_ckpt",
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
        default=10000000000,
        help="Inteval steps to save checkpoint")
    parser.add_argument(
        '--eval_steps',
        type=int,
        default=50,
        help="Inteval steps to save checkpoint")
    parser.add_argument(
        '--max_not_better_num',
        type=int,
        default=20,
        help="Inteval steps to save checkpoint")
    parser.add_argument(
        '--confidence',
        type=float,
        default=1.0,
        help="Inteval steps to save checkpoint")
    parser.add_argument(
        '--min_prob',
        type=float,
        default=0.7,
        help="Inteval steps to save checkpoint")

    args = parser.parse_args()
    return args


def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def do_train(args, iter_num=0, unlabeled_file=None, history_max_acc=0.0, best_checkpoint=None, last_train=False, pretrained_model=None):
    print("[start training] iter_num:{}, unlabeled_file:{}, history_max_acc:{}, best_checkpoint:{}, last_train:{}".format(iter_num, unlabeled_file, history_max_acc, best_checkpoint, last_train))

    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)

    label_normalize_json = os.path.join("./label_normalized",
                                        args.task_name + ".json")

    label_norm_dict = None
    with open(label_normalize_json, 'r', encoding="utf-8") as f:
        label_norm_dict = json.load(f)

    convert_example_fn = convert_example if args.task_name != "chid" else convert_chid_example
    evaluate_fn = do_evaluate if args.task_name != "chid" else do_evaluate_chid
    predict_fn = do_predict if args.task_name != "chid" else do_predict_chid

    # train_ds, dev_ds, test_ds, unlabeled_ds = load_dataset(
    #     "fewclue",
    #     name=args.task_name,
    #     splits=("train_" + args.index, "test_public", "test", "unlabeled"))

    train_ds, dev_ds, test_ds, unlabeled_ds = load_dataset(
        "fewclue",
        name=args.task_name,
        splits=("train_" + args.index, "dev_" + args.index, "test", "unlabeled"))


    if unlabeled_file:
        print("load_unlabeled_file:{}".format(unlabeled_file))
        tmp_unlabeled_ds = load_dataset("fewclue", name=args.task_name, data_files=unlabeled_file)
        print("self_training_unlabeled_example:{}".format(len(tmp_unlabeled_ds)))
        train_ds.new_data += tmp_unlabeled_ds.new_data
        print("using extended train_ds:{}".format(len(train_ds)))
   
    # Task related transform operations, eg: numbert label -> text_label, english -> chinese
    transform_fn = partial(
        transform_fn_dict[args.task_name], label_normalize_dict=label_norm_dict, pattern_id=args.pattern_id)

    # Task related transform operations, eg: numbert label -> text_label, english -> chinese
    predict_transform_fn = partial(
        transform_fn_dict[args.task_name],
        label_normalize_dict=label_norm_dict,
        pattern_id=args.pattern_id,
        is_test=True)

    # Some fewshot_learning strategy is defined by transform_fn
    # Note: Set lazy=True to transform example inplace immediately,
    # because transform_fn should only be executed only once when 
    # iterate multi-times for train_ds
    train_ds = train_ds.map(transform_fn, lazy=False)
    dev_ds = dev_ds.map(transform_fn, lazy=False)
    test_ds = test_ds.map(predict_transform_fn, lazy=False)
    unlabeled_ds = unlabeled_ds.map(predict_transform_fn, lazy=False)

    #model = ErnieForPretraining.from_pretrained('ernie-1.0')
    #tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained('ernie-1.0')

    if not pretrained_model:
        model = ppnlp.transformers.BertForPretraining.from_pretrained(
            args.language_model)
    else:
        model = pretrained_model

    tokenizer = ppnlp.transformers.BertTokenizer.from_pretrained(
            args.language_model)

    if args.task_name != "chid":
        # [src_ids, token_type_ids, masked_positions, masked_lm_labels]
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_ids
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
            Stack(dtype="int64"),  # masked_positions
            Stack(dtype="int64"),  # masked_lm_labels
        ): [data for data in fn(samples)]

        # [src_ids, token_type_ids, masked_positions]
        predict_batchify_fn = lambda samples, fn=Tuple(
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
            Stack(dtype="int64"),  # masked_lm_labels
            Stack(dtype="int64"),  # candidate_labels_ids [candidate_num, label_length]
        ): [data for data in fn(samples)]

        # [src_ids, token_type_ids, masked_positions, masked_lm_labels, candidate_labels_ids]
        predict_batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_ids
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
            Stack(dtype="int64"),  # masked_positions
            Stack(dtype="int64"),  # candidate_labels_ids [candidate_num, label_length]
        ): [data for data in fn(samples)]

    trans_func = partial(
        convert_example_fn,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        p_embedding_num=args.p_embedding_num)

    trans_func_test = partial(
        convert_example_fn,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        p_embedding_num=args.p_embedding_num,
        is_test=True)

    train_data_loader = create_dataloader(
        train_ds,
        mode='train',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    dev_data_loader = create_dataloader(
        dev_ds,
        mode='eval',
        batch_size=args.predict_batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    test_data_loader = create_dataloader(
        test_ds,
        mode='eval',
        batch_size=args.predict_batch_size,
        batchify_fn=predict_batchify_fn,
        trans_fn=trans_func_test)

    unlabeled_data_loader = create_dataloader(
        unlabeled_ds,
        mode='eval',
        batch_size=args.predict_batch_size,
        batchify_fn=predict_batchify_fn,
        trans_fn=trans_func_test)

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
        print("warmup from:{}".format(args.init_from_ckpt))

    if os.path.exists(best_checkpoint) and os.path.isfile(best_checkpoint):
        state_dict = paddle.load(best_checkpoint)
        model.set_dict(state_dict)
        print("model paramters warmup from:{}".format(best_checkpoint))

    mlm_loss_fn = ErnieMLMCriterion()

    num_training_steps = len(train_data_loader) * args.epochs

    if last_train:
        args.epoch = 10
        args.learning_rate = args.learning_rate / 2.0
        print("last train learning_rate:{}".format(args.learning_rate))

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_proportion)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]

    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)

    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
        grad_clip=clip)

    global_step = 0
    max_dev_acc = 0.0
    best_epoch = 0
    not_better_num_step = 0
    tic_train = time.time()

    if iter_num >= 1 and last_train == False:
        args.epochs = 1

    for epoch in range(1, args.epochs + 1):
        for step, batch in tqdm(enumerate(train_data_loader, start=1)):
            model.train()

            src_ids = batch[0]
            token_type_ids = batch[1]
            masked_positions = batch[2]
            masked_lm_labels = batch[3]

            max_len = src_ids.shape[1]
            new_masked_positions = []
            # masked_positions: [bs, label_length]
            for bs_index, mask_pos in enumerate(masked_positions.numpy()):
                for pos in mask_pos:
                    new_masked_positions.append(bs_index * max_len + pos)
            # new_masked_positions: [bs * label_length, 1]
            new_masked_positions = np.array(new_masked_positions).astype(
                'int32')
            new_masked_positions = paddle.to_tensor(new_masked_positions)

            prediction_scores, _ = model(
                input_ids=src_ids,
                token_type_ids=token_type_ids,
                masked_positions=new_masked_positions)

            loss = mlm_loss_fn(prediction_scores, masked_lm_labels)

            if math.isnan(loss) and iter_num > 0:
                # self-training meet nan
                # Stop training
                print("Self-training stop at iter_num because loss nan:{} ********************************".format(iter_num))
                return {'unlabeled_file': None,
                    'history_max_acc': history_max_acc, 
                    'best_checkpoint': best_checkpoint,
                    'last_train': True}
            else:
                pass

            global_step += 1
            if global_step % 10 == 0 and rank == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss,
                       10 / (time.time() - tic_train)))
                tic_train = time.time()

            if global_step % args.eval_steps == 0 and rank == 0 and iter_num >= 1:
                dev_accuracy, total_num = evaluate_fn(model, tokenizer, dev_data_loader,
                                        label_norm_dict)
                print("epoch:{}, global_step:{}, dev_accuracy:{:.3f}, total_num:{}, iter_num:{}, last_train:{}".format(
                    epoch, global_step, dev_accuracy, total_num, iter_num, last_train))

                # when new_max_acc, predict test.json and unlabeled.json
                if dev_accuracy > history_max_acc:
                    print("[meet better performance] dev_accuracy:{}\thistory_max_acc:{}".format(dev_accuracy, history_max_acc))
                    y_pred_labels, probs = predict_fn(model, tokenizer, test_data_loader,
                                            label_norm_dict)

                    if not os.path.exists(args.output_dir):
                        os.makedirs(args.output_dir)
                    output_file = os.path.join(args.output_dir,
                                            "index" + args.index + "_" + str(epoch) +
                                            "epoch_" + str(iter_num) + "iter_" + str(global_step) + "step_" + predict_file[args.task_name])
                                            
                    print("[save predict_result]{}".format(output_file))

                    _ = write_fn[args.task_name](args.task_name, output_file, y_pred_labels, probs)

                    max_dev_acc = dev_accuracy
                    best_epoch = epoch
                    best_step = global_step

                    if rank == 0:
                        save_dir = os.path.join(args.save_dir, "model_iter{}_epoch{}_step{}".format(iter_num, epoch, global_step))
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        save_param_path = os.path.join(save_dir, 'model_state.pdparams')
                        print("[save checkpoint]{}".format(save_param_path))
                        paddle.save(model.state_dict(), save_param_path)
                        tokenizer.save_pretrained(save_dir)

                    # predict unlabeled.json
                    # load best_epoch checkpoint
                    best_checkpoint = os.path.join(args.save_dir, "model_iter{}_epoch{}_step{}".format(iter_num, best_epoch, best_step), 'model_state.pdparams')
                    assert os.path.isfile(best_checkpoint), "best_checkpoint {} not exist".format(best_checkpoint)
                    print("start load parameters from best_checkpoint:{}".format(best_checkpoint))
                    state_dict = paddle.load(best_checkpoint)
                    print("start set parameters to model")
                    model.set_dict(state_dict)

                    # predict unlabeled_data
                    print("predicting unlabel_data......")
                    y_pred_labels, probs = predict_fn(model, tokenizer, unlabeled_data_loader, label_norm_dict)
                    output_file = os.path.join(args.output_dir,
                                    "index" + args.index + "_" + str(best_epoch) +
                                    "epoch_" + str(iter_num) + "iter_" + str(best_step) + "step_" + unlabeled_file_dict[args.task_name])
                    print("[save unlabeled_result]{}".format(output_file))
                    unlabeled_examples = write_fn[args.task_name](args.task_name, output_file, y_pred_labels, probs, is_test=False, min_prob=args.min_prob)

                    return {'unlabeled_file': output_file,
                            'history_max_acc': max_dev_acc, 
                            'best_checkpoint': best_checkpoint,
                            'last_train': False,
                            'pretrained_model': model}
                else:
                    # if continuously 50 steps not geneate better performance, then end self-traning
                    not_better_num_step += 1
                    print("not_better_num_step:{}".format(not_better_num_step))
                    print("current dev_accuracy:{}\thistory_max_acc:{}".format(dev_accuracy, history_max_acc))

                    if not_better_num_step >= args.max_not_better_num:
                        # Stop training
                        print("Self-training stop at iter_num:{} ********************************".format(iter_num))
               
                        return {'unlabeled_file': None,
                            'history_max_acc': history_max_acc, 
                            'best_checkpoint': best_checkpoint,
                            'last_train': True,
                            'pretrained_model': model}
                    else:
                        pass

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

        if iter_num == 0 or last_train:
            dev_accuracy, total_num = evaluate_fn(model, tokenizer, dev_data_loader,
                                                label_norm_dict)
            print("epoch:{}, global_step:{}, dev_accuracy:{:.3f}, total_num:{}, iter_num:{}, last_train:{}".format(
                epoch, global_step, dev_accuracy, total_num, iter_num, last_train))

            if dev_accuracy > max_dev_acc:
                y_pred_labels, probs = predict_fn(model, tokenizer, test_data_loader,
                                        label_norm_dict)

                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir)
                output_file = os.path.join(args.output_dir,
                                        "index" + args.index + "_" + str(epoch) +
                                        "epoch_" + str(iter_num) + "iter_" + str(global_step) + "step_" + predict_file[args.task_name])
                                        
                print("[save predict_result]{}".format(output_file))
                _ = write_fn[args.task_name](args.task_name, output_file, y_pred_labels, probs)

                max_dev_acc = dev_accuracy
                best_epoch = epoch
                best_step = global_step

                if rank == 0:
                    save_dir = os.path.join(args.save_dir, "model_iter{}_epoch{}_step{}".format(iter_num, epoch, global_step))
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_param_path = os.path.join(save_dir, 'model_state.pdparams')
                    print("[save checkpoint]{}".format(save_param_path))
                    paddle.save(model.state_dict(), save_param_path)
                    tokenizer.save_pretrained(save_dir)
        
    if max_dev_acc > history_max_acc:
        # load best_epoch checkpoint
        best_checkpoint = os.path.join(args.save_dir, "model_iter{}_epoch{}_step{}".format(iter_num, best_epoch, best_step), 'model_state.pdparams')
        assert os.path.isfile(best_checkpoint), "best_checkpoint {} not exist".format(best_checkpoint)
        print("start load parameters from best_checkpoint:{}".format(best_checkpoint))
        state_dict = paddle.load(best_checkpoint)
        print("start set parameters to model")
        model.set_dict(state_dict)

        # predict unlabeled_data
        print("predicting unlabel_data......")
        y_pred_labels, probs = predict_fn(model, tokenizer, unlabeled_data_loader, label_norm_dict)
        output_file = os.path.join(args.output_dir,
                        "index" + args.index + "_" + str(best_epoch) +
                        "epoch_" + str(iter_num) + "iter_" + str(best_step) + "step_" + unlabeled_file_dict[args.task_name])
        print("[save unlabeled_result]{}".format(output_file))
        unlabeled_examples = write_fn[args.task_name](args.task_name, output_file, y_pred_labels, probs, is_test=False, min_prob=args.min_prob)

        if last_train:
            print("Last train use labeled data finished:{} ************************".format(iter_num))
            return None
        else:
            # print("return from first_train")
            return {'unlabeled_file': output_file,
                'history_max_acc': max_dev_acc, 
                'best_checkpoint': best_checkpoint,
                'last_train': False,
                'pretrained_model': model}
    else:
        # Stop training
        print("Stop training at iter_num:{} ********************************".format(iter_num))
        return None



if __name__ == "__main__":
    args = parse_args()

    kwargs = {
        'unlabeled_file': None,
        'history_max_acc': 0.0,
        'best_checkpoint': "",
        'last_train': False,
        'pretrained_model': None,
    }

    iter_num = 0
    all_unlabdled_files = []
    while True:
        if kwargs:
            if kwargs['unlabeled_file']:
                all_unlabdled_files.append(kwargs['unlabeled_file'])
                print("all_unlabdled_files:{}".format(all_unlabdled_files))
            else:
                if kwargs['last_train']:
                    print("return unlabeled_file None")
                    kwargs = do_train(args, iter_num, **kwargs)
                    break
                else:
                    pass

            if len(all_unlabdled_files) > 1:
                # ensemble unlabeled.json
                print("start ensemble:{}".format(all_unlabdled_files))
                ensembled_unlabeled_json = ensemble(all_unlabdled_files, ensemble_dict[args.task_name], iter_num=iter_num, confidence=args.confidence)
                kwargs['unlabeled_file'] = ensembled_unlabeled_json
                all_unlabdled_files.append(ensembled_unlabeled_json)

            kwargs = do_train(args, iter_num, **kwargs)
            gc.collect()
            iter_num += 1
        else:
            break