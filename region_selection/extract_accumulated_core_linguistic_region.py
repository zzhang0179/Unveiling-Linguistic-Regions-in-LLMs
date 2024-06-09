import torch
from transformers import AutoModelForCausalLM
import csv
from tqdm import tqdm
import random
import os


def jaccard_similarity(tensor1, tensor2):
    # Compute the intersection (common elements)
    intersection = torch.logical_and(tensor1, tensor2).sum()
    # Compute the union (all unique elements)
    union = torch.logical_or(tensor1, tensor2).sum()
    # Calculate the Jaccard Index (IoU)
    iou = intersection.float() / union.float()

    return iou.item()

def compare_bool_matrix(bool_dict1, bool_dict2):
    params_diff_similarity = {}
    for (name1, bool_matrix1), (name2, bool_matrix2) in zip(bool_dict1.items(), bool_dict2.items()):
        assert name1 == name2
        params_diff_similarity[name1] = jaccard_similarity(bool_matrix1,bool_matrix2)
    return params_diff_similarity

def logical_and_bool_matrix(bool_dict1, bool_dict2):
    params_diff_similarity = {}
    for (name1, bool_matrix1), (name2, bool_matrix2) in zip(bool_dict1.items(), bool_dict2.items()):
        assert name1 == name2
        params_diff_similarity[name1] = torch.logical_and(bool_matrix1,bool_matrix2)
    return params_diff_similarity

def logical_or_bool_matrix(bool_dict1, bool_dict2):
    params_diff_similarity = {}
    for (name1, bool_matrix1), (name2, bool_matrix2) in zip(bool_dict1.items(), bool_dict2.items()):
        assert name1 == name2
        params_diff_similarity[name1] = torch.logical_or(bool_matrix1,bool_matrix2)
    return params_diff_similarity

def calculate_row_bool_matrix(bool_dict1, num):
    params_top_row = {}
    params_top_col = {}
    params_bottom_row = {}
    params_bottom_col = {}
    
    for (name1, bool_matrix1) in bool_dict1.items():
        row_sums = torch.sum(bool_matrix1, dim=1)
        col_sums = torch.sum(bool_matrix1, dim=0)
        top_rows_indices = torch.topk(row_sums, k=num).indices.tolist()
        top_cols_indices = torch.topk(col_sums, k=num).indices.tolist()
        min_rows_indices = torch.topk(row_sums, k=num, largest=False).indices.tolist()
        min_cols_indices = torch.topk(col_sums, k=num, largest=False).indices.tolist()
        params_top_row[name1] = top_rows_indices
        params_top_col[name1] = top_cols_indices
        params_bottom_row[name1] = min_rows_indices
        params_bottom_col[name1] = min_cols_indices
    return params_top_row,params_top_col,params_bottom_row,params_bottom_col

def get_diff_tensor(tensor_base,tensor_change):
    # 计算两个张量的差异
    assert tensor_change.shape == tensor_base.shape
    tensor_diff = torch.abs(tensor_change - tensor_base)
    tensor_diff = tensor_diff / torch.abs(tensor_base)
    return tensor_diff

def get_top_bottom_tensor(tensor_diff,k):
    # 计算需要记录的点的数量（最大/最小/中间的3%）
    num_points = int(k * tensor_diff.numel())

    # 找到差异幅度最大的前3%的点
    max_points = tensor_diff.view(-1).topk(num_points).indices
    bool_sensor_max = torch.zeros(tensor_diff.shape, dtype=torch.bool)
    bool_sensor_max.view(-1)[max_points] = True

    # 找到差异幅度最小的前3%的点
    min_points = tensor_diff.view(-1).topk(num_points, largest=False).indices
    bool_sensor_min = torch.zeros(tensor_diff.shape, dtype=torch.bool)
    bool_sensor_min.view(-1)[min_points] = True

    # 随机找3%的点
    bool_sensor_random = torch.zeros(tensor_diff.shape, dtype=torch.bool)
    # 随机选择要设置为True的元素的索引
    random_points = random.sample(range(bool_sensor_random.numel()), num_points)
    # 将选定的索引位置设置为True
    bool_sensor_random.view(-1)[random_points] = True

    return bool_sensor_max,bool_sensor_min,bool_sensor_random
        
def accumulate_matrix(param_dict1, param_dict2):
    params_diff_accumulate = {}
    for (name1, param_matrix1), (name2, param_matrix2) in zip(param_dict1.items(), param_dict2.items()):
        assert name1 == name2
        params_diff_accumulate[name1] = param_matrix1 + param_matrix2
    return params_diff_accumulate

# 比较每个模型的每层参数变化幅度的平均值
def compare_parameters(model1, model2):
    params_diff = {}
    with tqdm(total=400) as pbar:
        for (name1, params1), (name2, params2) in zip(model1.named_parameters(), model2.named_parameters()):
            pbar.update(1)
            if 'layers.' not in name1:
                continue
            assert name1 == name2  # 确保参数名称一致
            params_diff[name1] = get_diff_tensor(params1,params2)
    return params_diff

# 模型名称和路径
model_name = "llama2"
original_model_path = 'path_to_llama2_base'  # 原版llama模型

# 加载原版llama模型和再训练的llama模型
original_model = AutoModelForCausalLM.from_pretrained(original_model_path)

language_list = ['Arabic','Spanish','Russian','Chinese','Korean','Vietnamese']
top_k_params_dict = {}
bottom_k_params_dict = {}
random_k_params_dict = {}


k = 0.01 # ratio

for samples in [10000,100000]:
    with tqdm(total=400) as pbar:

        for (name, params) in original_model.named_parameters():
            if 'layers.' not in name:
                continue
            grad_tensor = torch.zeros_like(params).cpu()
            pbar.update(1)
            for language in language_list:
                #将六国的grad-mul-params取abs后相加
                file_dir = 'path_to_save_{}/grad-mul-param_checkpoint_{}'.format(language,samples)
                save_path = os.path.join(file_dir,'{}.pt'.format(name.replace('module.','')))
                grad_tensor += torch.load(save_path).abs().cpu()
            
            bool_sensor_max,bool_sensor_min,bool_sensor_random = get_top_bottom_tensor(grad_tensor,k)
            top_k_params_dict[name] = bool_sensor_max
            bottom_k_params_dict[name] = bool_sensor_min
            random_k_params_dict[name] = bool_sensor_random


        # 将输出结果保存到CSV文件
        output_file = "six-countries-accumulated-grad-mul-param_top-bottom-{}-{}-{}.csv".format(model_name,k,samples)

        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Parameter Name", f"Parameters Difference top {k} Similarity", f"Parameters Difference bottom {k} Similarity", f"Parameters Difference random {k} Similarity"])
            for (name, diff_top),(name2, diff_bottom),(name3,diff_random) in zip(top_k_params_dict.items(),bottom_k_params_dict.items(),random_k_params_dict.items()):
                writer.writerow([name, (diff_top.sum()/diff_top.numel()).item(), (diff_bottom.sum()/diff_bottom.numel()).item(), (diff_random.sum()/diff_random.numel()).item()])


        os.makedirs('six-countries-accumulated-{}-grad-mul-param/{}/top{}'.format(model_name,samples,k),exist_ok=True)
        for key,values in top_k_params_dict.items():
            # 将布尔矩阵转换为字节类型的张量
            save_path = os.path.join('six-countries-accumulated-{}-grad-mul-param/{}/top{}'.format(model_name,samples,k),"{}.pt".format(key))
            # 使用torch.save()保存张量到文件
            torch.save(values, save_path)

        os.makedirs('six-countries-accumulated-{}-grad-mul-param/{}/bottom{}'.format(model_name,samples,k),exist_ok=True)
        for key,values in bottom_k_params_dict.items():
            save_path = os.path.join('six-countries-accumulated-{}-grad-mul-param/{}/bottom{}'.format(model_name,samples,k),"{}.pt".format(key))
            torch.save(values, save_path)

        if samples == 100000: # 如果已经保存过random k 就不用再筛选
            continue

        os.makedirs('six-countries-accumulated-{}-grad-mul-param/{}/random{}'.format(model_name,samples,k),exist_ok=True)
        for key,values in random_k_params_dict.items():
            save_path = os.path.join('six-countries-accumulated-{}-grad-mul-param/{}/random{}'.format(model_name,samples,k),"{}.pt".format(key))
            torch.save(values, save_path)
