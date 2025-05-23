import torch
import torch.nn as nn
from train_AQ import MLP4  #定义模型结构
import pandas as pd

def predict(dataset_path):
    # 初始化模型
    model = MLP4()
    model.load_state_dict(torch.load('model_AQ.pth'))  #读取预训练模型
    model.eval()

    # 读取新特征数据，跳过第一行
    new_features = pd.read_excel(dataset_path, engine='openpyxl', skiprows=1)

    # 选择需要的特征列，假设特征在第2到第20列
    new_features = new_features.iloc[:, 1:].values

    # 转换为 PyTorch 张量
    new_features_tensor = torch.tensor(new_features, dtype=torch.float32)

    # 使用模型进行预测
    with torch.no_grad():
        predicted_output = model(new_features_tensor)

    # 输出预测结果
    print("Predicted Output:")
    print(predicted_output)

    # 保存预测结果到 Excel 文件
    output_df = pd.DataFrame(predicted_output.numpy())
    output_df.to_excel("OUTPUT.xlsx", index=False)  #定义输出路径

if __name__ == "__main__":
    dataset_path = 'predictDATA.xlsx'  #定义输入数据集
    predict(dataset_path)
