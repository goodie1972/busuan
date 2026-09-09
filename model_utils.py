# -*- coding:utf-8 -*-
"""
模型工具模块，包含与模型加载、预测相关的功能
"""
import os
import torch
import joblib
import numpy as np
from model import LstmCRFModel

# 模型和路径配置
name_path = {
    "dlt": {
        "name": "大乐透",
        "path": "./model/dlt/",
        "model_file": "dlt_model.pth",
        "scaler_X_file": "scaler_X.pkl",
        "train_script": "./scripts/dlt/train_dlt_model.py",
        "fetch_script": "./scripts/dlt/fetch_dlt_data.py"
    },
    "ssq": {
        "name": "双色球",
        "path": "./model/ssq/",
        "model_file": "ssq_model.pth",
        "scaler_X_file": "scaler_X.pkl",
        "train_script": "./scripts/ssq/train_ssq_model.py",
        "fetch_script": "./scripts/ssq/fetch_ssq_data.py"
    }
}

def load_pytorch_model(model_path, input_dim, hidden_dim, output_dim, output_seq_length, lottery_type):
    """
    加载 PyTorch 模型及缩放器
    """
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # 加载红球模型
    red_model = LstmCRFModel(input_dim, hidden_dim, output_dim['red'], output_seq_length['red'], num_layers=1, dropout=0.3)
    red_model.load_state_dict(checkpoint['red_model'])
    red_model.eval()

    # 加载蓝球模型
    blue_model = LstmCRFModel(input_dim, hidden_dim, output_dim['blue'], output_seq_length['blue'], num_layers=1, dropout=0.3)
    blue_model.load_state_dict(checkpoint['blue_model'])
    blue_model.eval()

    # 加载缩放器
    scaler_X_path = os.path.join(os.path.dirname(model_path), name_path[lottery_type]['scaler_X_file'])
    if not os.path.exists(scaler_X_path):
        raise FileNotFoundError(f"特征缩放器文件不存在：{scaler_X_path}")
    scaler_X = joblib.load(scaler_X_path)

    return red_model, blue_model, scaler_X

def load_resources_pytorch(lottery_type):
    """
    根据彩票类型加载模型和资源
    """
    if lottery_type not in name_path:
        raise ValueError(f"不支持的彩票类型：{lottery_type}，请检查输入。")
    if lottery_type == "dlt":
        hidden_dim = 128
        output_dim = {
            'red': 35,
            'blue': 12
        }
        output_seq_length = {
            'red': 5,
            'blue': 2
        }
    elif lottery_type == "ssq":
        hidden_dim = 128
        output_dim = {
            'red': 33,
            'blue': 16
        }
        output_seq_length = {
            'red': 6,
            'blue': 1
        }

    model_path = os.path.join(name_path[lottery_type]['path'], name_path[lottery_type]['model_file'])
    scaler_path = os.path.join(name_path[lottery_type]['path'], name_path[lottery_type]['scaler_X_file'])

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在：{model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"特征缩放器文件不存在：{scaler_path}")

    # 从scaler_X文件中获取input_dim
    scaler_X = joblib.load(scaler_path)
    input_dim = scaler_X.n_features_in_

    red_model, blue_model, scaler_X = load_pytorch_model(
        model_path, input_dim, hidden_dim, output_dim, output_seq_length, lottery_type
    )

    return red_model, blue_model, scaler_X

def sample_crf_sequences(crf_model, emissions, mask, num_samples=1, temperature=1.0):
    """
    从CRF模型中采样序列
    """
    batch_size, seq_length, num_tags = emissions.size()
    emissions = emissions.cpu().numpy()
    mask = mask.cpu().numpy()

    sampled_sequences = []

    for i in range(batch_size):
        seq_mask = mask[i]
        seq_emissions = emissions[i][:seq_mask.sum()]
        seq_sample = []
        for t, emission in enumerate(seq_emissions):
            emission = emission / temperature
            probs = np.exp(emission - np.max(emission))
            probs /= probs.sum()
            sampled_tag = np.random.choice(num_tags, p=probs)
            seq_sample.append(sampled_tag)
        sampled_sequences.append(seq_sample)

    return sampled_sequences

def process_predictions(red_predictions, blue_predictions, lottery_type):
    """
    处理预测结果，确保号码在有效范围内且为整数
    """
    if lottery_type == "dlt":
        # 大乐透前区：1-35，后区：1-12
        front_numbers = [min(max(int(num) + 1, 1), 35) for num in red_predictions[:5]]
        back_numbers = [min(max(int(num) + 1, 1), 12) for num in blue_predictions[:2]]

        # 确保前区号码唯一
        front_numbers = list(set(front_numbers))
        while len(front_numbers) < 5:
            additional_num = np.random.randint(1, 36)
            if additional_num not in front_numbers:
                front_numbers.append(additional_num)
        front_numbers = sorted(front_numbers)[:5]

        # 随机交换前区号码以增加多样性
        if np.random.rand() > 0.5:
            idx1, idx2 = np.random.choice(5, 2, replace=False)
            front_numbers[idx1], front_numbers[idx2] = front_numbers[idx2], front_numbers[idx1]

    elif lottery_type == "ssq":
        # 双色球红球：1-33，蓝球：1-16
        front_numbers = [min(max(int(num) + 1, 1), 33) for num in red_predictions[:6]]
        back_number = min(max(int(blue_predictions[0]) + 1, 1), 16)

        # 确保红球号码唯一
        front_numbers = list(set(front_numbers))
        while len(front_numbers) < 6:
            additional_num = np.random.randint(1, 34)
            if additional_num not in front_numbers:
                front_numbers.append(additional_num)
        front_numbers = sorted(front_numbers)[:6]

        # 随机交换红球号码以增加多样性
        if np.random.rand() > 0.5:
            idx1, idx2 = np.random.choice(6, 2, replace=False)
            front_numbers[idx1], front_numbers[idx2] = front_numbers[idx2], front_numbers[idx1]

    else:
        raise ValueError("不支持的彩票类型！请选择 'ssq' 或 'dlt'。")

    if lottery_type == "dlt":
        return front_numbers + back_numbers
    elif lottery_type == "ssq":
        return front_numbers + [back_number]

def randomize_numbers(numbers, lottery_type):
    """
    为预测号码增加随机性，以产生更多样化的结果
    """
    import random
    
    if lottery_type == "dlt":
        # 大乐透: 前区5个红球(1-35)，后区2个蓝球(1-12)
        red_numbers = numbers[:5]
        blue_numbers = numbers[5:]
        
        # 为前区号码增加随机性，但保持号码在合法范围内
        for i in range(len(red_numbers)):
            if random.random() < 0.3:  # 30%的几率修改号码
                offset = random.randint(-2, 2)
                red_numbers[i] = max(1, min(35, red_numbers[i] + offset))
        
        # 确保前区号码唯一
        while len(set(red_numbers)) < 5:
            for i in range(len(red_numbers)):
                if red_numbers.count(red_numbers[i]) > 1:
                    red_numbers[i] = random.randint(1, 35)
                    break
        
        # 为后区号码增加随机性
        for i in range(len(blue_numbers)):
            if random.random() < 0.3:
                offset = random.randint(-1, 1)
                blue_numbers[i] = max(1, min(12, blue_numbers[i] + offset))
                
        # 确保后区号码唯一
        while len(set(blue_numbers)) < 2:
            for i in range(len(blue_numbers)):
                if blue_numbers.count(blue_numbers[i]) > 1:
                    blue_numbers[i] = random.randint(1, 12)
                    break
        
        return sorted(red_numbers) + sorted(blue_numbers)
        
    elif lottery_type == "ssq":
        # 双色球: 红球6个(1-33)，蓝球1个(1-16)
        red_numbers = numbers[:6]
        blue_number = numbers[6]
        
        # 为红球号码增加随机性
        for i in range(len(red_numbers)):
            if random.random() < 0.3:  # 30%的几率修改号码
                offset = random.randint(-2, 2)
                red_numbers[i] = max(1, min(33, red_numbers[i] + offset))
        
        # 确保红球号码唯一
        while len(set(red_numbers)) < 6:
            for i in range(len(red_numbers)):
                if red_numbers.count(red_numbers[i]) > 1:
                    red_numbers[i] = random.randint(1, 33)
                    break
        
        # 为蓝球增加随机性
        if random.random() < 0.3:
            offset = random.randint(-1, 1)
            blue_number = max(1, min(16, blue_number + offset))
            
        return sorted(red_numbers) + [blue_number]
    
    else:
        return numbers  # 未知类型，返回原始号码 