#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双色球LSTM-CRF模型训练脚本
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torchcrf import CRF
import matplotlib.pyplot as plt
import logging
import time
import joblib

# 确保当前目录在sys.path中
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
if project_dir not in sys.path:
    sys.path.append(project_dir)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('ssq_train')

# 全局参数
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# 模型定义
class LSTMCRF(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, output_seq_length):
        super(LSTMCRF, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_seq_length = output_seq_length
        
        # LSTM层
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim * output_seq_length)
        
        # CRF层
        self.crf = CRF(output_dim, batch_first=True)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        emissions = self.fc(lstm_out[:, -1])
        emissions = emissions.view(-1, self.output_seq_length, self.output_dim)
        return emissions

    def loss(self, emissions, tags, mask=None):
        return -self.crf(emissions, tags, mask=mask, reduction='mean')

    def decode(self, emissions, mask=None):
        return self.crf.decode(emissions, mask=mask)

def load_data(data_path):
    """加载数据"""
    logger.info(f"加载数据: {data_path}")
    try:
        # 尝试使用不同的编码读取数据
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
        
        for encoding in encodings:
            try:
                logger.info(f"尝试使用 {encoding} 编码读取数据...")
                df = pd.read_csv(data_path, encoding=encoding)
                logger.info(f"数据加载成功，共 {len(df)} 条记录")
                return df
            except UnicodeDecodeError:
                continue
        
        # 如果所有尝试都失败
        logger.error("所有编码尝试均失败")
        sys.exit(1)
    except Exception as e:
        logger.error(f"加载数据时出错: {str(e)}")
        sys.exit(1)

def preprocess_data(df):
    """预处理数据"""
    logger.info("预处理数据...")
    
    # 提取特征和标签
    # 双色球: 6个红球(1-33) + 1个蓝球(1-16)
    red_cols = [col for col in df.columns if col.startswith('红球_')][:6]
    blue_cols = [col for col in df.columns if col.startswith('蓝球')][:1]
    
    # 确保数据按期数降序排列
    df = df.sort_values('期数', ascending=False).reset_index(drop=True)
    
    # 创建特征
    X_data = []
    y_red_data = []
    y_blue_data = []
    
    # 使用滑动窗口创建序列数据
    sequence_length = 10
    
    for i in range(len(df) - sequence_length):
        # 使用过去的sequence_length期作为特征
        features = []
        for j in range(sequence_length):
            row_features = []
            for col in red_cols + blue_cols:
                row_features.append(df.iloc[i+j+1][col])
            features.append(row_features)
        
        X_data.append(features)
        y_red_data.append([df.iloc[i][col]-1 for col in red_cols])  # 减1使标签从0开始
        y_blue_data.append([df.iloc[i][col]-1 for col in blue_cols])  # 减1使标签从0开始
    
    X = np.array(X_data)
    y_red = np.array(y_red_data)
    y_blue = np.array(y_blue_data)
    
    # 标准化特征
    X_reshaped = X.reshape(-1, X.shape[-1])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(X.shape)
    
    logger.info(f"处理后的特征形状: {X_scaled.shape}")
    logger.info(f"红球标签形状: {y_red.shape}")
    logger.info(f"蓝球标签形状: {y_blue.shape}")
    
    return X_scaled, y_red, y_blue, scaler

def train_model(X_train, y_train, input_dim, hidden_dim, output_dim, output_seq_length, n_epochs=100, batch_size=32, use_gpu=False):
    """训练模型"""
    logger.info("开始训练模型...")
    
    # 创建训练数据集
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.LongTensor(y_train)
    train_dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
    # 初始化模型
    model = LSTMCRF(input_dim, hidden_dim, output_dim, output_seq_length)
    
    # 使用GPU（如果可用且要求使用）
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"使用GPU训练: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("使用CPU训练")
        
    model.to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    losses = []
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # 前向传播
            emissions = model(batch_X)
            mask = torch.ones(emissions.size()[:2], dtype=torch.uint8).to(device)

            # 计算损失
            loss = model.loss(emissions, batch_y, mask=mask)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # 每轮训练后记录平均损失
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{n_epochs}, 损失: {avg_loss:.4f}")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(script_dir, 'training_loss.png'))
    
    return model, device

def save_model(model, scaler, model_dir):
    """保存模型"""
    logger.info(f"保存模型到: {model_dir}")
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 保存模型
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))
    
    # 保存缩放器
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    
    logger.info("模型保存完成")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练双色球预测模型')
    parser.add_argument('--gpu', action='store_true', help='使用GPU训练')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    args = parser.parse_args()
    
    # 检查GPU是否可用
    if args.gpu:
        if torch.cuda.is_available():
            logger.info(f"GPU可用: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA版本: {torch.version.cuda}")
        else:
            logger.warning("GPU不可用，将使用CPU训练")
            args.gpu = False
    
    # 数据路径
    data_path = os.path.join(script_dir, 'ssq_history.csv')
    
    # 如果数据文件不存在，尝试使用备用路径
    if not os.path.exists(data_path):
        logger.warning(f"找不到数据文件: {data_path}")
        data_path = os.path.join(project_dir, 'scripts', 'ssq', 'ssq_history.csv')
        if not os.path.exists(data_path):
            logger.error(f"找不到备用数据文件: {data_path}")
            sys.exit(1)
    
    # 加载和预处理数据
    df = load_data(data_path)
    X, y_red, y_blue, scaler = preprocess_data(df)
    
    # 设置模型参数
    input_dim = X.shape[2]  # 输入特征维度
    hidden_dim = 128  # 隐藏层维度
    red_output_dim = 33  # 红球输出维度 (1-33)
    blue_output_dim = 16  # 蓝球输出维度 (1-16)
    red_output_length = 6  # 红球序列长度
    blue_output_length = 1  # 蓝球序列长度
    
    # 训练红球模型
    logger.info("训练红球模型...")
    red_model, device = train_model(
        X, y_red, 
        input_dim, hidden_dim, 
        red_output_dim, red_output_length, 
        n_epochs=args.epochs, 
        use_gpu=args.gpu
    )

    # 训练蓝球模型
    logger.info("训练蓝球模型...")
    blue_model, device = train_model(
        X, y_blue, 
        input_dim, hidden_dim, 
        blue_output_dim, blue_output_length, 
        n_epochs=args.epochs, 
        use_gpu=args.gpu
    )

    # 保存模型
    model_dir = os.path.join(project_dir, 'model', 'ssq')
    
    # 创建组合模型字典
    combined_model = {
        'red_model': red_model.state_dict(),
        'blue_model': blue_model.state_dict(),
        'metadata': {
            'timestamp': time.time(),
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'red_output_dim': red_output_dim,
            'blue_output_dim': blue_output_dim,
            'red_seq_length': red_output_length,
            'blue_seq_length': blue_output_length
        }
    }
    
    # 确保目录存在
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 保存组合模型
    torch.save(combined_model, os.path.join(model_dir, 'ssq_model.pth'))
    
    # 保存缩放器
    joblib.dump(scaler, os.path.join(model_dir, 'scaler_X.pkl'))
    
    logger.info(f"训练完成，模型已保存到 {model_dir}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    logger.info(f"总训练时间: {end_time - start_time:.2f} 秒")
