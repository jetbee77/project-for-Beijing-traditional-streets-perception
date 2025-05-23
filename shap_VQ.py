import torch
import torch.nn as nn
import pandas as pd
import shap  # 确保这里导入的是 SHAP 库
import matplotlib.pyplot as plt

# ========== 定义你的 MLP 模型结构 ==========
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(7, 9)
        self.fc2 = nn.Linear(9, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ========== 1. 加载数据 ==========
data = pd.read_excel('traindata_VQ_new.xlsx', engine='openpyxl')
X = data.iloc[1:, 1:8].values  # 第二行以后，第二列以后
y = data.iloc[1:, 0].values   # 第二行以后，第一列

# ========== 2. 加载训练好的模型 ==========
model = MLP()
model.load_state_dict(torch.load('model_VQ.pth'))  # 读取权重
model.eval()  # 设为评估模式

# ========== 3. 定义模型预测函数（供SHAP使用） ==========
def model_forward(x_numpy):
    x_tensor = torch.tensor(x_numpy, dtype=torch.float32)
    with torch.no_grad():
        output = model(x_tensor).numpy()
    return output

# ========== 4. 创建SHAP解释器 ==========
background = X[:100]  # 取前100个样本作为背景数据
explainer = shap.Explainer(model_forward, background)

# 解释所有数据（或者你可以只解释一部分，比如测试集）
shap_values = explainer(X)

# ========== 5. 可视化特征贡献（总结图） ==========
feature_names = data.columns[1:]  # 特征名是第二列及之后
shap.summary_plot(shap_values.values, X, feature_names=feature_names)

# ========== 6. 绘制SHAP值的条形图 ==========
# 绘制特征重要性条形图
shap.summary_plot(shap_values.values, X, plot_type="bar", feature_names=feature_names)

# （可选）保存SHAP图为文件
# plt.savefig('shap_bar_plot.png')
# plt.show()
