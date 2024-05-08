#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning a 🤗 Transformers model for question answering using 🤗 Accelerate.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from custom_gpt2_lora_switch import GPT2ForQuestionAnswering
import datasets
import evaluate
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils_qa import postprocess_qa_predictions
import torch.nn as nn
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
logging.basicConfig(level = logging.INFO)
import wandb

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
logger = logging.getLogger(__name__)

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def setup_wandb(name):
    wandb.init(project="assignment-IEC",entity="chandrahul0320", name=name)
    # wandb.config.update(args)

# def log_into_wandb(loss, optimizer, epoch):
#     # violators, loss, trn metrics, tst metrics, epoch, scaler
#     wandb_dict = {}
    
#     dic = {'epoch': epoch, 'loss': loss, 'epoch': epoch,'lr': optimizer.param_groups[0]['lr']}
#     wandb_dict.update({
#         k : v for k,v in dic.items() if v is not None
#     })
#     wandb.log(wandb_dict)

def log_into_wandb(dic_loss, optimizer, epoch):
    # violators, loss, trn metrics, tst metrics, epoch, scaler
    wandb_dict = {}
    
    dic = dic_loss

    dic_loss['epoch'] = epoch
    dic_loss['lr'] = optimizer.param_groups[0]['lr']
    wandb_dict.update({
        k : v for k,v in dic.items() if v is not None
    })
    wandb.log(wandb_dict)



def save_prefixed_metrics(results, output_dir, file_name: str = "all_results.json", metric_key_prefix: str = "eval"):
    """
    Save results while prefixing metric names.

    Args:
        results: (:obj:`dict`):
            A dictionary of results.
        output_dir: (:obj:`str`):
            An output directory.
        file_name: (:obj:`str`, `optional`, defaults to :obj:`all_results.json`):
            An output file name.
        metric_key_prefix: (:obj:`str`, `optional`, defaults to :obj:`eval`):
            A metric name prefix.
    """
    # Prefix all keys with metric_key_prefix + '_'
    for key in list(results.keys()):
        if not key.startswith(f"{metric_key_prefix}_"):
            results[f"{metric_key_prefix}_{key}"] = results.pop(key)

    with open(os.path.join(output_dir, file_name), "w") as f:
        json.dump(results, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Question Answering task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "--p_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "--is_lora",
        type=int,
        default=0,
        help="The name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "--disable_wandb",
        type=int,
        default=0,
        help="The name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "--half",
        type=int,
        default=0,
        help="The name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "--only_eval",
        type=int,
        default=0,
        help="The name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--preprocessing_num_workers", type=int, default=1, help="A csv or a json file containing the training data."
    )
    parser.add_argument("--do_predict", action="store_true", help="To do prediction on the question answering model")
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the Prediction data."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=384,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_seq_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="constant",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help="When splitting up a long document into chunks how much stride to take between chunks.",
    )
    parser.add_argument(
        "--a_bit",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--w_bit",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help="The total number of n-best predictions to generate when looking for an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help=(
            "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        ),
    )
    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, some of the examples do not have an answer.",
    )
    parser.add_argument(
        "--max_answer_length",
        type=int,
        default=30,
        help=(
            "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_predict_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of prediction examples to this",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    args = parser.parse_args()

    # Sanity checks
    if (
        args.dataset_name is None
        and args.train_file is None
        and args.validation_file is None
        and args.test_file is None
    ):
        raise ValueError("Need either a dataset name or a training/validation/test file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if args.test_file is not None:
            extension = args.test_file.split(".")[-1]
            assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()
    
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
            extension = args.train_file.split(".")[-1]
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
            extension = args.validation_file.split(".")[-1]
        if args.test_file is not None:
            data_files["test"] = args.test_file
            extension = args.test_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, field="data")
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, trust_remote_code=args.trust_remote_code)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    
    
    tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=True, trust_remote_code=args.trust_remote_code) 
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.cls_token = tokenizer.bos_token

    # if args.model_name_or_path:
    #     model = AutoModelForQuestionAnswering.from_pretrained(
    #         args.model_name_or_path,
    #         from_tf=bool(".ckpt" in args.model_name_or_path),
    #         config=config,
    #         trust_remote_code=args.trust_remote_code,
    #     )

    #! Add weight.weight for these keys
    #! attn.c_proj.weight, attn.c_attn.weight, mlp.c_fc.weight, mlp.c_proj

    if (not args.disable_wandb):
        setup_wandb(args.p_name)
    
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.w_bits = args.w_bit
    config.a_bits = args.a_bit
    config.is_lora = bool(args.is_lora)

    # assert 5==6, "error!"
    model = GPT2ForQuestionAnswering(config, num_loras=3)
    
    if (not args.only_eval):
        dic = torch.load('pytorch_model.bin',map_location='cpu')                 
        new_dic = {}
        for k in dic.keys():
            new_dic['transformer.' + k] = dic[k]
        
        dic = new_dic
        new_dic = {}

        weight_weight_keys = ['attn.c_proj.weight', 'attn.c_attn.weight', 'mlp.c_fc.weight', 'mlp.c_proj']
        for k in dic.keys():
            if any([x in k for x in weight_weight_keys]):
                new_dic[k + ".weight"] = dic[k]
            else:
                new_dic[k] = dic[k]
                
        for k in new_dic.keys():
            print(k)

        print("--------------------")

        for name, p in model.named_parameters():
            if name in new_dic:
                print("YES: ", name)
            else:
                print("NO: ", name)
        
        
        model.load_state_dict(new_dic,strict=False)
        
    else:
        dic = torch.load('switch_nodistil_32_8_4_2_pytorch_model.bin',map_location='cpu')                 
        model.load_state_dict(dic,strict=True)


    
    column_names = raw_datasets["train"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )

    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    # Training preprocessing
    def prepare_train_features(examples):
        eos_token_id = tokenizer.eos_token_id
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        l = len(tokenized_examples['input_ids'])
        for i in range(l):
            tokenized_examples['input_ids'][i] = [eos_token_id] + tokenized_examples['input_ids'][i]
            tokenized_examples['attention_mask'][i] = [1] + tokenized_examples['attention_mask'][i]
            tokenized_examples['offset_mapping'][i] = [(0,0)] + tokenized_examples['offset_mapping'][i]


        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            sequence_ids = tokenized_examples.sequence_ids(i)
            sequence_ids = [None] + sequence_ids


            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    if "train" not in raw_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = raw_datasets["train"]
    if args.max_train_samples is not None:
        # We will select sample from whole data if argument is specified
        train_dataset = train_dataset.select(range(args.max_train_samples))

    # Create train feature from dataset
    
    train_dataset = train_dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on train dataset",
    )
    if args.max_train_samples is not None:
        # Number of samples might increase during Feature Creation, We select only specified max samples
        train_dataset = train_dataset.select(range(args.max_train_samples))

    # Validation preprocessing
    def prepare_validation_features(examples):
        
        eos_token_id = tokenizer.eos_token_id
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )
        l = len(tokenized_examples['input_ids'])
        for i in range(l):
            tokenized_examples['input_ids'][i] = [eos_token_id] + tokenized_examples['input_ids'][i]
            tokenized_examples['attention_mask'][i] = [1] + tokenized_examples['attention_mask'][i]
            tokenized_examples['offset_mapping'][i] = [(0,0)] + tokenized_examples['offset_mapping'][i]

        
        
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            sequence_ids = [None] + sequence_ids
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples


    eval_examples = raw_datasets["validation"]


    eval_dataset = eval_examples.map(
        prepare_validation_features,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on validation dataset",
    )

    if args.max_eval_samples is not None:
        # During Feature creation dataset samples might increase, we will select required samples again
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)

    if (args.half):
        train_dataset = train_dataset[:len(train_dataset)//2]

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )

    eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])
    eval_dataloader = DataLoader(
        eval_dataset_for_model, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )


    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=args.version_2_with_negative,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
            null_score_diff_threshold=args.null_score_diff_threshold,
            output_dir=args.output_dir,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = evaluate.load("squad_v2" if args.version_2_with_negative else "squad")

    # Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
    def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        """
        Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

        Args:
            start_or_end_logits(:obj:`tensor`):
                This is the output predictions of the model. We can only enter either start or end logits.
            eval_dataset: Evaluation dataset
            max_len(:obj:`int`):
                The maximum length of the output tensor. ( See the model.eval() part for more details )
        """

        step = 0
        # create a numpy array and fill it with -100.
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
        for i, output_logit in enumerate(start_or_end_logits):  # populate columns
            # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
            # And after every iteration we have to change the step

            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]

            if step + batch_size < len(dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

            step += batch_size

        return logits_concat

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.

    if config.is_lora:    
        allowed = ['qa_outputs','ln_f','lora','ln_1','ln_2','wpe','wte']
        for name, p in model.named_parameters():
            if not any([x in name for x in allowed]):
                p.requires_grad = False

        for name, p in model.named_parameters():
            if p.requires_grad:
                print(name)

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if any([x in n for x in allowed])]
            }
        ]

    else:

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()]
            }
        ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True


    args.num_warmup_steps = len(train_dataloader)*0.1
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * 1,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * 1,
    )

    # Prepare everything with our `accelerator`.
    

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)


    # Train!
    total_batch_size = args.per_device_train_batch_size * 1 * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0
    starting_epoch = 0

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    model = model.to('cuda:0')
    run_loss = 0

    w_bit_list = [8,4]
    a_bit_list = [8,4]

    total_steps = 0
    
    run_loss_main = 0
    run_loss_dist = 0
    run_loss_logit = 0

    run_loss_32 = 0
    run_loss_8_logit = 0
    run_loss_4_logit = 0
    
    run_loss_8_dist = 0
    run_loss_4_dist = 0

    for epoch in range(starting_epoch, args.num_train_epochs):
        if not args.only_eval:
            
            model.train()
            
            total_loss = 0
            active_dataloader = train_dataloader
            for step, batch in enumerate(active_dataloader):

                total_loss_i = 0.0
                total_dist_i = 0.0
                total_logit_i = 0.0

                batch = {k:batch[k].to('cuda:0') for k in batch.keys()}
                current_loss_logit = 0
                current_loss_dist = 0
                
                w_bit, a_bit, bit_index = 32, 32, 0
                outputs = model(**batch, w_bit_list = [w_bit]*12, a_bit_list = [a_bit]*12, lora_i = [bit_index]*12)

                loss = outputs.loss
                start_logit_teacher = outputs.start_logits.detach()
                end_logit_teacher = outputs.end_logits.detach()
                loss.backward()

                l = loss.item()
                total_loss_i += l
                total_logit_i += l
                run_loss_32 = 0.8*run_loss_32 + 0.2*l
                # for name, p in model.named_parameters():
                #     if 'h.11.attn.c_proj.weight.weight' in name:
                #         print(name, p.grad)
                #     if 'h.11.mlp.c_proj.lora_B_list.1.weight' in name:
                #         print(name, p.grad)

                # print("-------------------")                

                for bit_index in range(len(w_bit_list)):
                    w_bit, a_bit = w_bit_list[bit_index], a_bit_list[bit_index]
                    outputs = model(**batch, w_bit_list = [w_bit]*12, a_bit_list = [a_bit]*12, lora_i = [bit_index+1]*12)
                    
                    # print(start_logit_teacher.shape, outputs.start_logits.shape, end_logit_teacher.shape, outputs.end_logits.shape)

                    # print(start_logit_teacher[0,0], outputs.start_logits[0,0], end_logit_teacher[0,0], outputs.end_logits[0,0])

                    # dist_loss = nn.MSELoss()(start_logit_teacher,outputs.start_logits) + nn.MSELoss()(end_logit_teacher,outputs.end_logits)
                    dist_loss = 0.0
                    loss = outputs.loss + dist_loss
                    
                    loss.backward()

                    # for name, p in model.named_parameters():
                    #     if 'h.11.attn.c_proj.weight.weight' in name:
                    #         print(name, p.grad)
                    #     if 'h.11.mlp.c_proj.lora_B_list.1.weight' in name:
                    #         print(name, p.grad)

                    # assert 5==6+1,"FALSE!"
                    
                    l_logit = outputs.loss.item()
                    # l_distil = dist_loss.item()
                    l_distil = dist_loss

                    total_loss_i += (l_logit + l_distil)
                    total_logit_i += l_logit
                    total_dist_i += l_distil

                    if (bit_index == 0):
                        run_loss_8_logit = 0.8*run_loss_8_logit + 0.2*(l_logit)
                        run_loss_8_dist = 0.8*run_loss_8_dist + 0.2*(l_distil)
                    else:
                        run_loss_4_logit = 0.8*run_loss_4_logit + 0.2*(l_logit)
                        run_loss_4_dist = 0.8*run_loss_4_dist + 0.2*(l_distil)
                    

                run_loss_dist = 0.8*run_loss_dist + 0.2*(total_dist_i)
                run_loss_logit = 0.8*run_loss_logit + 0.2*(total_logit_i)
                run_loss_main = 0.8*run_loss_main + 0.2*total_loss_i
                        
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                progress_bar.update(1)
                completed_steps += 1
                if (step%10==0):
                    
                    if not args.disable_wandb:
                        log_into_wandb({
                            'run_loss_main' : run_loss_main,
                            'run_loss_dist' : run_loss_dist,
                            'run_loss_logit' : run_loss_logit,
                            'run_loss_32' : run_loss_32,
                            'run_loss_8_logit' : run_loss_8_logit,
                            'run_loss_4_logit' : run_loss_4_logit,
                            'run_loss_8_dist' : run_loss_8_dist,
                            'run_loss_4_dist' : run_loss_4_dist
                        }, optimizer, total_steps)
                        print(run_loss_main)

                if completed_steps >= args.max_train_steps:
                    break

                total_steps += 1
                

        # Evaluation
        # logger.info("***** Running Evaluation *****")
        # logger.info(f"  Num examples = {len(eval_dataset)}")
        # logger.info(f"  Batch size = {args.per_device_eval_batch_size}")

        all_start_logits = []
        all_end_logits = []

        model.eval()
        
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                batch = {k:batch[k].to('cuda:0') for k in batch.keys()}
                w_l = [32,32,4,4,4,4,4,4,8,8,32,32]
                a_l = [32,32,4,4,4,4,4,4,8,8,32,32]
                l_l = [0,0,2,2,2,2,2,2,1,1,0,0]
                # w_l = [4]*12
                # a_l = [4]*12
                # l_l = [2]*12
                outputs = model(**batch, w_bit_list=w_l, a_bit_list = a_l, lora_i = l_l)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

                all_start_logits.append(start_logits.cpu().numpy())
                all_end_logits.append(end_logits.cpu().numpy())

        max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor

        # concatenate the numpy array
        start_logits_concat = create_and_fill_np_array(all_start_logits, eval_dataset, max_len)
        end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len)

        # delete the list of numpy arrays
        del all_start_logits
        del all_end_logits

        outputs_numpy = (start_logits_concat, end_logits_concat)
        prediction = post_processing_function(eval_examples, eval_dataset, outputs_numpy)
        eval_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
        print(eval_metric)
        logger.info(f"Evaluation metrics: {eval_metric}")


        
        

        # --- COPY ---
        

        
        assert 5==6+1, "FALSE!"
   
    # if args.with_tracking:
    #     log = {
    #         "squad_v2" if args.version_2_with_negative else "squad": eval_metric,
    #         "train_loss": total_loss.item() / len(train_dataloader),
    #         "epoch": epoch,
    #         "step": completed_steps,
    #     }
  
    

if __name__ == "__main__":
    main()