# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import math
import torch
from torch import nn
import torch.nn.functional as F
from deepspeed.compression.helper import recursive_getattr, recursive_setattr
import deepspeed
import os
from utils.utils import print_rank_0
class LinearLayer_LoRA(nn.Module):
    # an simple implementation of LoRA
    # for now only support Linear Layer
    def __init__(self,
                 weight,
                 lora_dim=0,
                 lora_scaling=1,
                 lora_droppout=0,
                 bias=None):
        super(LinearLayer_LoRA, self).__init__()
        self.weight = weight
        self.bias = bias

        if lora_dim <= 0:
            raise ValueError(
                "You are training to use LoRA, whose reduced dim should be larger than 1"
            )

        try:
            # for zero stage 3
            rows, columns = weight.ds_shape
        except:
            rows, columns = weight.shape
        self.lora_right_weight = nn.Parameter(torch.zeros(
            columns,
            lora_dim))  # apply transpose so in forward we do not need to
        self.lora_left_weight = nn.Parameter(torch.zeros(lora_dim, rows))
        self.lora_scaling = lora_scaling / lora_dim

        if lora_droppout > 0:
            self.lora_dropout = nn.Dropout(lora_droppout)
        else:
            self.lora_dropout = nn.Identity()

        self.reset_parameters()
        # disable the original weight gradient
        self.weight.requires_grad = False
        # fuse LoRA to the original weight
        self.fuse_lora = False

    def eval(self):
        self.lora_dropout.eval()

    #   self.fuse_lora_weight()

    def train(self, mode=True):
        self.lora_dropout.train(mode)
        # self.unfuse_lora_weight()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_right_weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_left_weight)

    def fuse_lora_weight(self):
        if not self.fuse_lora:
            self.weight.data += self.lora_scaling * torch.matmul(
                self.lora_left_weight.t(), self.lora_right_weight.t())
        self.fuse_lora = True

    def unfuse_lora_weight(self):
        if self.fuse_lora:
            self.weight.data -= self.lora_scaling * torch.matmul(
                self.lora_left_weight.t(), self.lora_right_weight.t())
        self.fuse_lora = False

    def forward(self, input):
        if self.fuse_lora:
            return F.linear(input, self.weight, self.bias)
        else:
            return F.linear(
                input, self.weight,
                self.bias) + (self.lora_dropout(input) @ self.lora_right_weight
                              @ self.lora_left_weight) * self.lora_scaling




class LinearLayer_SVD(nn.Module):
    # def __init__(self, input_dim, output_dim, rank, svd_U, svd_Vh, bias=None):
    def __init__(self,
                 weight,
                 svd_U,
                 svd_Vh,
                 svd_dim=0,
                 svd_scaling=1,
                 svd_droppout=0,
                 bias=None):
        super(LinearLayer_SVD, self).__init__()
        self.weight = weight
        self.bias = bias
        """
        初始化SVD线性层。
        :param input_dim: 输入特征维度。
        :param output_dim: 输出特征维度。
        :param svd_dim: SVD的秩。
        :param svd_U: 初始化的U矩阵（冻结）。
        :param svd_Vh: 初始化的V矩阵（冻结）。
        :param bias: 可选的偏置项。
        """
        if svd_dim <= 0:
            raise ValueError(
                "You are training to use LoRA, whose reduced dim should be larger than 1"
            )

        try:
            # for zero stage 3
            rows, columns = weight.ds_shape
        except:
            rows, columns = weight.shape
        
        if svd_U.shape != (rows, svd_dim) or svd_Vh.shape != (svd_dim, columns):
            raise ValueError("Dimensions of svd_U or svd_Vh are not compatible with specified svd_dim and input/output dimensions.")

        self.svd_U = nn.Parameter(svd_U, requires_grad=False)  # 冻结U矩阵
        self.svd_Vh = nn.Parameter(svd_Vh, requires_grad=False)  # 冻结V矩阵
        self.sigma = nn.Parameter(torch.zeros(svd_dim), requires_grad=True)  # 对角线上的sigma值
        self.svd_scaling = svd_scaling

        if svd_droppout > 0:
            self.svd_dropout = nn.Dropout(svd_droppout)
        else:
            self.svd_dropout = nn.Identity()

        self.reset_parameters()
        # disable the original weight gradient
        self.weight.requires_grad = False
        # fuse svd to the original weight
        self.fuse_svd = False
        self.negfuse_svd = False

    def eval(self):
        self.svd_dropout.eval()
        # self.fuse_svd_weight()

    def train(self, mode=True):
        self.svd_dropout.train(mode)
        # self.unfuse_svd_weight()

    def reset_parameters(self):
        nn.init.zeros_(self.sigma)

    def fuse_svd_weight(self):
        
        sigma_matrix = torch.diag(self.sigma)
        if not self.fuse_svd and not self.negfuse_svd:
            self.weight.data += 10 * self.svd_scaling * (self.svd_U @ sigma_matrix @ self.svd_Vh)
        self.fuse_svd = True

    def unfuse_svd_weight(self):
        
        sigma_matrix = torch.diag(self.sigma)
        if self.fuse_svd:
            self.weight.data -= 10 * self.svd_scaling * (self.svd_U @ sigma_matrix @ self.svd_Vh)
        self.fuse_svd = False
    
    def fuse_negtive_svd_weight(self):
        
        sigma_matrix = torch.diag(self.sigma)
        if not self.negfuse_svd and not self.fuse_svd:
            self.weight.data -= 10 * self.svd_scaling * (self.svd_U @ sigma_matrix @ self.svd_Vh)
        self.negfuse_svd = True
    
    def unfuse_negtive_svd_weight(self):
        
        sigma_matrix = torch.diag(self.sigma)
        if self.negfuse_svd:
            self.weight.data += 10 * self.svd_scaling * (self.svd_U @ sigma_matrix @ self.svd_Vh)
        self.negfuse_svd = False
    

    def reverse_singular(self):
        with torch.no_grad():
            self.sigma.data = -self.sigma.data

    
    
    def save_sigma(self,samples,name,args):
        if args.global_rank == 0:
            save_path = os.path.join(args.output_dir, 'sigma_checkpoint_{}/{}_sigma.pt'.format(samples,name.replace('module.','').replace('model.','')))
            os.makedirs(os.path.dirname(save_path),exist_ok=True)
            # # 使用torch.save()保存张量到文件
            print_rank_0(f'Checkpoint {samples} name: {name}',args.global_rank)
            # print_rank_0(f'Parameter sigma: {self.sigma}',args.global_rank)
            torch.save(self.sigma, save_path)

    def forward(self, input):
        """
        定义前向传播。
        :param input: 输入数据。
        """
        if self.fuse_svd or self.negfuse_svd:
            return F.linear(input, self.weight, self.bias)
        else:
            sigma_matrix = torch.diag(self.sigma)
            return F.linear(
                input, self.weight,
                self.bias) + (self.svd_dropout(input) @ self.svd_Vh.t()
                              @ sigma_matrix.t() @ self.svd_U.t()) * self.svd_scaling

# convert the linear layer to LoRA
def convert_linear_layer_to_lora(model,
                                 part_module_name,
                                 lora_dim=0,
                                 lora_scaling=1,
                                 lora_droppout=0):
    repalce_name = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and part_module_name in name:
            repalce_name.append(name)
    for name in repalce_name:
        module = recursive_getattr(model, name)
        tmp = LinearLayer_LoRA(
            module.weight, lora_dim, lora_scaling, lora_droppout,
            module.bias).to(module.weight.device).to(module.weight.dtype)
        recursive_setattr(model, name, tmp)
    return model


# convert the linear layer to SVD
def convert_linear_layer_to_svd(model,
                                 part_module_name,
                                 svd_path,
                                 svd_dim=0,
                                 svd_scaling=1,
                                 svd_droppout=0):
    repalce_name = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and part_module_name in name:
            repalce_name.append(name)
    # print_rank_0('replace name :---------------------{}'.format(repalce_name))
    for name in repalce_name:
        module = recursive_getattr(model, name)
        # 加载SVD特征向量矩阵U
        svd_U_save_path = '{}/{}.weight_filtered_U.pt'.format(svd_path,name.replace('module.','').replace('model.',''))
        svd_U_weight = torch.load(svd_U_save_path,map_location=module.weight.device)
        
        # 加载SVD特征向量矩阵Vh
        svd_Vh_save_path = '{}/{}.weight_filtered_Vh.pt'.format(svd_path,name.replace('module.','').replace('model.',''))
        svd_Vh_weight = torch.load(svd_Vh_save_path,map_location=module.weight.device)
        
        tmp = LinearLayer_SVD(
            module.weight, svd_U_weight, svd_Vh_weight, svd_dim, svd_scaling, svd_droppout,
            module.bias).to(module.weight.device).to(module.weight.dtype)
        recursive_setattr(model, name, tmp)
    return model

def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == deepspeed.runtime.zero.
        partition_parameters.ZeroParamStatus.NOT_AVAILABLE
    ]


# convert the LoRA layer to linear layer
def convert_lora_to_linear_layer(model):
    repalce_name = []
    for name, module in model.named_modules():
        if isinstance(module, LinearLayer_LoRA):
            repalce_name.append(name)
    for name in repalce_name:
        module = recursive_getattr(model, name)
        zero_stage_3 = hasattr(module.weight, 'ds_id')
        with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([
                module.weight, module.bias, module.lora_left_weight,
                module.lora_right_weight
        ]),
                                               modifier_rank=0,
                                               enabled=zero_stage_3):
            module.fuse_lora_weight()
    return model


# convert the SVD layer to linear layer
def convert_svd_to_linear_layer(model):
    print('fused_svd_weight')
    repalce_name = []
    for name, module in model.named_modules():
        if isinstance(module, LinearLayer_SVD):
            repalce_name.append(name)
    for name in repalce_name:
        module = recursive_getattr(model, name)
        zero_stage_3 = hasattr(module.weight, 'ds_id')
        with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([
                module.weight, module.bias, module.svd_U, module.sigma,
                module.svd_Vh
        ]),
                                               modifier_rank=0,
                                               enabled=zero_stage_3):
            module.fuse_svd_weight()
    return model

# convert the LORA layer unfuse
def convert_lora_unfuse(model):
    repalce_name = []
    for name, module in model.named_modules():
        if isinstance(module, LinearLayer_LoRA):
            repalce_name.append(name)
    for name in repalce_name:
        module = recursive_getattr(model, name)
        zero_stage_3 = hasattr(module.weight, 'ds_id')
        with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([
                module.weight, module.bias, module.lora_left_weight,
                module.lora_right_weight
        ]),
                                               modifier_rank=0,
                                               enabled=zero_stage_3):
            module.unfuse_lora_weight()
    return model


# convert the SVD layer unfuse
def convert_svd_unfuse(model):
    print('unfused_svd_weight')
    repalce_name = []
    for name, module in model.named_modules():
        if isinstance(module, LinearLayer_SVD):
            repalce_name.append(name)
    for name in repalce_name:
        module = recursive_getattr(model, name)
        zero_stage_3 = hasattr(module.weight, 'ds_id')
        with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([
                module.weight, module.bias, module.svd_U, module.sigma,
                module.svd_Vh
        ]),
                                               modifier_rank=0,
                                               enabled=zero_stage_3):
            module.unfuse_svd_weight()
    return model


# convert the SVD layer negtive fuse
def convert_neg_svd_fuse(model):
    print('fused_neg_svd_weight')
    repalce_name = []
    for name, module in model.named_modules():
        if isinstance(module, LinearLayer_SVD):
            repalce_name.append(name)
    for name in repalce_name:
        module = recursive_getattr(model, name)
        zero_stage_3 = hasattr(module.weight, 'ds_id')
        with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([
                module.weight, module.bias, module.svd_U, module.sigma,
                module.svd_Vh
        ]),
                                               modifier_rank=0,
                                               enabled=zero_stage_3):
            module.fuse_negtive_svd_weight()
    return model



# convert the SVD layer negtive unfuse
def convert_neg_svd_unfuse(model):
    print('unfused_svd_weight')
    repalce_name = []
    for name, module in model.named_modules():
        if isinstance(module, LinearLayer_SVD):
            repalce_name.append(name)
    for name in repalce_name:
        module = recursive_getattr(model, name)
        zero_stage_3 = hasattr(module.weight, 'ds_id')
        with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([
                module.weight, module.bias, module.svd_U, module.sigma,
                module.svd_Vh
        ]),
                                               modifier_rank=0,
                                               enabled=zero_stage_3):
            module.unfuse_negtive_svd_weight()
    return model


# save the SVD layer sigma
def save_svd_sigma(model,samples,args):
    repalce_name = []
    for name, module in model.named_modules():
        if isinstance(module, LinearLayer_SVD):
            repalce_name.append(name)
    for name in repalce_name:
        module = recursive_getattr(model, name)
        zero_stage_3 = hasattr(module.weight, 'ds_id')
        # print_rank_0('zero_stage_3:----------------------{}'.format(zero_stage_3),args.global_rank)
        with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([
                module.sigma,
        ]),
                                               modifier_rank=0,
                                               enabled=zero_stage_3):
            module.save_sigma(samples,name,args)
    return model




def only_optimize_lora_parameters(model):
    # turn off the gradient of all the parameters except the LoRA parameters
    for name, param in model.named_parameters():
        if "lora_right_weight" in name or "lora_left_weight" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def only_optimize_svd_parameters(model):
    # turn off the gradient of all the parameters except the svd diagram parameters
    for name, param in model.named_parameters():
        if "sigma" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model





