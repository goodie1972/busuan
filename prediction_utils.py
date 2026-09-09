import numpy as np
import random

def process_predictions(red_predictions, blue_predictions, lottery_type):
    """
    处理预测结果，确保号码在有效范围内且为整数
    :param red_predictions: list, 红球预测的类别索引
    :param blue_predictions: list, 蓝球预测的类别索引
    :param lottery_type: 'ssq' 或 'dlt'
    :return: list, 预测的开奖号码
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
    
    Args:
        numbers: 原始预测号码列表
        lottery_type: 'ssq' 或 'dlt'
    
    Returns:
        处理后的号码列表
    """
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

def sample_crf_sequences(crf_model, emissions, mask, num_samples=1, temperature=1.0):
    """
    从CRF模型中采样序列
    
    Args:
        crf_model: CRF模型
        emissions: 发射概率
        mask: 掩码
        num_samples: 采样数量
        temperature: 温度参数，控制随机性
        
    Returns:
        采样的序列列表
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