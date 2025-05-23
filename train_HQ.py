import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(8, 9)
        self.fc2 = nn.Linear(9, 1)  #神经网络结构

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(X_train, y_train, model, criterion, optimizer, scheduler, num_epochs=20000): #设置epochs次数
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            train_losses.append(loss.item())

        scheduler.step()

    return train_losses

def evaluate_model(X_test, y_test, model, criterion):
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        print(f'Test set loss: {test_loss.item():.4f}')

def calculate_metrics(y_true, y_pred):   #计算权重
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

def visualize_results(y_true, y_pred, label=''):
    plt.scatter(y_true, y_pred, label=label)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')

    # Plot regression line
    x_line = np.linspace(min(y_true), max(y_true), 100)
    y_line = x_line
    plt.plot(x_line, y_line, color='red', label='Regression Line')

    plt.legend()
    plt.show()

if __name__ == "__main__":
    data = pd.read_excel('traindata_HQ_new.xlsx', engine='openpyxl')   #读取数据集
    X = data.iloc[1:, 1:9].values  #读取第二行后所有&读取第二列后所有
    y = data.iloc[1:, 0].values   #读取第二行后所有&读取第一列

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Train and evaluate MLP model
    model = MLP()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  #学习率
    scheduler = lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.5)   #步长

    train_losses = train_model(X_train_tensor, y_train_tensor, model, criterion, optimizer, scheduler)
    evaluate_model(X_test_tensor, y_test_tensor, model, criterion)

    # Calculate metrics for MLP
    with torch.no_grad():
        y_pred_mlp = model(X_test_tensor).numpy()

    mse_mlp, r2_mlp = calculate_metrics(y_test, y_pred_mlp)
    print(f'MLP - MSE: {mse_mlp:.4f}, R^2: {r2_mlp:.4f}')

    torch.save(model.state_dict(), 'model_HQ.pth')

    plt.plot(train_losses, label='Training Loss (MLP)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Train and evaluate Multiple Regression model
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    y_pred_regression = regression_model.predict(X_test)

    # Calculate metrics for Multiple Regression
    mse_regression, r2_regression = calculate_metrics(y_test, y_pred_regression)
    print(f'Multiple Regression - MSE: {mse_regression:.4f}, R^2: {r2_regression:.4f}')

    # Train and evaluate Random Forest model
    forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
    forest_model.fit(X_train, y_train)
    y_pred_forest = forest_model.predict(X_test)

    # Calculate metrics for Random Forest
    mse_forest, r2_forest = calculate_metrics(y_test, y_pred_forest)
    print(f'Random Forest - MSE: {mse_forest:.4f}, R^2: {r2_forest:.4f}')

    # Visualize results for all models
    visualize_results(y_test, y_pred_mlp, label='MLP')
    visualize_results(y_test, y_pred_regression, label='Multiple Regression')
    visualize_results(y_test, y_pred_forest, label='Random Forest')
