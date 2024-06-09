#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import math
import sys
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    get_scheduler,
)
import transformers
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from accelerate import Accelerator
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_prompt_dataset
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters
from utils.model.model_utils import create_hf_model
from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_grad, safe_get_full_optimizer_state
from torch.utils.data import Dataset
from transformers import Trainer
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import numpy as np
IGNORE_INDEX = -100

def set_first_false_to_true(mask_tensor):
    # 找到每行第一个False的位置
    
    first_false_indices = (~mask_tensor).cumsum(dim=1) == 1

    # 将每行的第一个False置为True
    mask_tensor[first_false_indices] = True
    return mask_tensor

@dataclass
class MyDataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    pad_token_id: int
    max_seq_len: int
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        """batch_first: 默认batch在第一个维度，padding_value:不够的填充"""
        """input_ids:list，以里面最多的为基准"""
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        """label也是"""
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        # torch.cuda.empty_cache()

        # 默认是右padding
        if input_ids.size(1) < self.max_seq_len:
            padding_size = self.max_seq_len - input_ids.size(1)
            padding = torch.full((input_ids.size(0), padding_size),self.pad_token_id, dtype=input_ids.dtype)
            input_ids = torch.cat((input_ids, padding), dim=1)
        if labels.size(1) < self.max_seq_len:
            padding_size = self.max_seq_len - labels.size(1)
            padding = torch.full((labels.size(0), padding_size),IGNORE_INDEX, dtype=labels.dtype)
            labels = torch.cat((labels, padding), dim=1)
        
        attention_mask=input_ids.ne(self.pad_token_id)
        attention_mask=set_first_false_to_true(attention_mask) # 右padding 保留第一个eos_id为true
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )


class MyDataset(Dataset):
    def __init__(self, data_prefix, seq_length, pad_id):
        super(MyDataset, self).__init__()
        """这边要求data_prefix为完整的路径，但不包括后缀"""
        """比如：/llama/our/data"""
        """后面会根据需要自动的添加上/llama/our/data.idx"""
        """后面会根据需要自动的添加上/llama/our/data.bin"""
        """后面会根据需要自动的添加上/llama/our/data.dis"""
        self.idx_file_path = f"{data_prefix}.idx"
        self.bin_file_path = f"{data_prefix}.bin"
        self.dis_file_path = f"{data_prefix}.dis"
        self.seq_length = seq_length
        self.pad_id = pad_id

        self.index_start_pos = None  # 每个样本的起始位置
        self.index_length = None  # 每个样本的长度
        self._load_index()
        self._load_bin()
        self._load_dis()

        self._check()

    def _check(self):
        """验证数据是否正确"""
        assert self.index_length[-1] + self.index_start_pos[-1] == len(self.bin_buffer), \
            "数据错误校验错误！"

    def _load_index(self):
        """文件所占的字节大小"""
        file_size = os.stat(self.idx_file_path).st_size
        """样本总数"""
        assert file_size % 10 == 0  # 2B的length，8B的start pos
        self.total_sample = file_size // 10
        with open(self.idx_file_path, "rb") as f:
            self.index_start_pos = np.frombuffer(f.read(self.total_sample * 8), dtype=np.uint64).tolist()
            self.index_length = np.frombuffer(f.read(self.total_sample * 2), dtype=np.uint16).tolist()
            # print(self.index_length)

    def _load_bin(self):
        """以内存映射的方式进行加载大文件"""
        self.bin_buffer = np.memmap(self.bin_file_path, dtype=np.uint16, mode='r')

    def _load_dis(self):
        """仅当有多种类别的数据混合有效"""
        self.distributed = torch.load(self.dis_file_path)
        if len(self.distributed) != 0:
            assert sum(self.distributed) == self.total_sample

    def __len__(self):
        return self.total_sample

    def __getitem__(self, idx):
        """为了节省时间，采用动态长度"""
        start_idx = self.index_start_pos[idx]
        length = self.index_length[idx]
        if length > self.seq_length:
            """如果超出最大长度，则使用最大长度"""
            """否则使用原生长度"""
            length = self.seq_length
        data = torch.as_tensor(self.bin_buffer[start_idx:start_idx + length].tolist(), dtype=torch.long)
        labels = data.clone()
        """注意，此时都是没有padding的"""
        return dict(input_ids=data, labels=labels)


def _make_supervised_data_module(args,train_data_prefix, eval_data_prefix, pad_id=0) -> Dict:
    train_dataset = MyDataset(data_prefix=train_data_prefix, seq_length=args.max_seq_len, pad_id=pad_id)
    eval_dataset = MyDataset(data_prefix=eval_data_prefix, seq_length=args.max_seq_len, pad_id=pad_id)
    train_data_collator = MyDataCollatorForSupervisedDataset(pad_token_id=pad_id,max_seq_len=args.max_seq_len)
    eval_data_collator = MyDataCollatorForSupervisedDataset(pad_token_id=pad_id,max_seq_len=args.max_seq_len)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, train_data_collator=train_data_collator, eval_data_collator = eval_data_collator)


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        nargs='*',
                        default=['Dahoas/rm-static'],
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')
    parser.add_argument('--pretrain_train_data_path',
                        type=str,
                        default='/usr/KnowLM-main/pretrain/data/chinese',
                        help='Path to the training data.')
    parser.add_argument('--pretrain_test_data_path',
                        type=str,
                        default='/usr/KnowLM-main/pretrain/data/chinese',
                        help='Path to the testing data.')
    
    parser.add_argument('--data_split',
                        type=str,
                        default='2,4,4',
                        help='Comma-separated list of proportions for training'
                        'phase 1, 2, and 3 data. For example the split `6,2,2`'
                        'will use 60%% of data for phase 1, 20%% for phase 2'
                        'and 20%% for phase 3.')
    parser.add_argument(
        '--sft_only_data_path',
        nargs='*',
        default=[],
        help='Path to the dataset for only using in SFT phase.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to new pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument("--total_cards",
                        type=int,
                        help="total_cards for distributed training on gpus")
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.001,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=100,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Validate settings
    if args.gradient_checkpointing and args.lora_dim > 0:
        assert (
            not args.only_optimize_lora
        ), "--gradient_checkpointing and --only_optimize_lora cannot be enabled at the same time."

    return args



def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage)
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    torch.distributed.barrier()
    
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    tokenizer.pad_token = tokenizer.eos_token
    # make sure tokenizer is right pad in our logic
    tokenizer.padding_side = 'right'

    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config=None,
                            disable_dropout=args.disable_dropout)
    
    
    
    data_module = _make_supervised_data_module(args,train_data_prefix=args.pretrain_train_data_path,eval_data_prefix=args.pretrain_test_data_path,
                                               pad_id=tokenizer.eos_token_id)
    
    
    train_dataset = data_module['train_dataset']
    eval_dataset = data_module['eval_dataset']
    train_data_collator = data_module['train_data_collator']
    eval_data_collator = data_module['eval_data_collator']
    
    train_batch = train_data_collator([train_dataset[i] for i in range(5)])
    print_rank_0(train_batch,  args.global_rank)
    print_rank_0(tokenizer.decode(train_dataset[0]["input_ids"]),args.global_rank)
    print_rank_0(tokenizer.decode(train_dataset[1]["input_ids"]),args.global_rank)

    eval_batch = eval_data_collator([eval_dataset[i] for i in range(2)])
    print_rank_0(eval_batch,  args.global_rank)
    print_rank_0(tokenizer.decode(eval_dataset[0]["input_ids"]),args.global_rank)
    print_rank_0(tokenizer.decode(eval_dataset[1]["input_ids"]),args.global_rank)


    print_rank_0('len_train_sample:{}'.format(len(train_dataset)), args.global_rank)
    print_rank_0('len_eval_sample:{}'.format(len(eval_dataset)),  args.global_rank)

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=train_data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=eval_data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)
    

    def evaluation(model, eval_dataloader):
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            
            batch = to_device(batch, device)
            # print(batch)
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            losses += loss.float()
            # if step + 1 >= 20:
            #     break

        losses = losses / (step + 1)
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        return perplexity

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Train!
   
    training_step_losses = []
    batch_size =(args.total_cards*args.per_device_train_batch_size)
    save_samples = [10000,100000]
    save_steps = [math.ceil(samples/batch_size) for samples in save_samples]
    save_dict = dict(zip(save_steps,save_samples))
    print_rank_0(
            "save msg:",
            args.global_rank)
    print_rank_0(
            save_dict,
            args.global_rank)
    

    # 保存需要冻结的节点的权重
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    
    print_rank_0("***** Running training *****", args.global_rank)
    print_rank_0(
        f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank)
    perplexity = evaluation(model, eval_dataloader)
    print_rank_0(f"ppl: {perplexity}", args.global_rank)

    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    
    
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        model.train()
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), unit="batch"):
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            # 只进行梯度累积
            model.backward(loss)
            # model.step()

            if step % 100 == 0:
                print_rank_0(
                    f"Epoch {epoch+1}/{args.num_train_epochs}, Step {step+1}/{len(train_dataloader)}, Loss {loss.item()}"
                )
            if (step+1) in save_steps:
                    
                print_rank_0(
                    f"***** Evaluating perplexity, Epoch {epoch+1}/{args.num_train_epochs} *****",
                    args.global_rank)
                perplexity = evaluation(model, eval_dataloader)
                print_rank_0(f"ppl {save_dict[step + 1]}: {perplexity}", args.global_rank)
                
                for n, lp in model.named_parameters():
                    # # 1. gradient lookup
                    # For zero1 and zero2, gradient lookup must be called after `backward` and before `step`
                    # For zero3, gradient lookup must be called after `backward`
                    hp_grad = safe_get_full_grad(lp)
                    print_rank_0(n,args.global_rank)
                    print_rank_0(hp_grad,args.global_rank)

                    # # 2. fp32 and optim states can probably be called anywhere in the training loop, but will be updated after `step`
                    # hp = safe_get_full_fp32_param(lp)
                    # exp_avg = safe_get_full_optimizer_state(lp, "exp_avg")
                    # exp_avg_sq = safe_get_full_optimizer_state(lp, "exp_avg_sq")
                    
                    save_dir = os.path.join(args.output_dir, 'grad-mul-param_checkpoint_{}'.format(save_dict[step + 1]))
                    os.makedirs(save_dir,exist_ok=True)
                    save_path = os.path.join(save_dir, '{}.pt'.format(n.replace('module.','')))
                    
                    # 使用torch.save()保存张量到文件
                    grad_mul_param_tensor = torch.mul(hp_grad,lp)
                    torch.save(grad_mul_param_tensor.bfloat16(), save_path)

                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                
                if save_dict[step + 1] == 100000:
                    break
                

    # Evaluate perplexity on the validation set.

    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    # print_rank_0(
    #     f"***** Evaluating perplexity, Epoch {epoch+1}/{args.num_train_epochs} *****",
    #     args.global_rank)
    # perplexity = evaluation(model, eval_dataloader)
    # print_rank_0(f"ppl: {perplexity}", args.global_rank)
    # # model.tput_timer.update_epoch_count()
    
    # if hasattr(torch.cuda, 'empty_cache'):
    #     torch.cuda.empty_cache()

    # if args.output_dir is not None:
    #     print_rank_0('saving the final model ...', args.global_rank)
    #     model = convert_lora_to_linear_layer(model)

    #     if args.global_rank == 0:
    #         save_hf_format(model, tokenizer, args)

    #     if args.zero_stage == 3:
    #         # For zero stage 3, each gpu only has a part of the model, so we need a special save function
    #         save_zero_three_model(model,
    #                               args.global_rank,
    #                               args.output_dir,
    #                               zero_stage=args.zero_stage)

if __name__ == "__main__":
    main()
