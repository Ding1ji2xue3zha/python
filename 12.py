import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from time import time  # 明确导入 time 模块
import os
# 设置全局字体大小
plt.rcParams.update({'font.size': 42, 'font.family': 'Arial', 'font.weight': 'normal'})

# 读取数据集
data = pd.read_csv('1.csv')

# 清理特征名称，去除空格和特殊字符
#data.columns = [col.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").replace("/", "_") for col in data.columns]
feature_names = list(data.columns[:-1])  # 获取清理后的特征名称
print(f"清理后的特征名称: {feature_names}")

# 提取自变量和因变量
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# 使用 XGBRegressor 训练模型
model = XGBRegressor(objective='reg:squarederror',
                     eta=0.2,
                     max_depth=7,
                     gamma=0,
                     subsample=0.6,
                     colsample_bytree=0.6,
                     eval_metric='rmse')
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred_test = model.predict(X_test)

# 计算测试集均方误差（MSE）
test_mse = mean_squared_error(y_test, y_pred_test)
print("XGBoost 测试集均方误差 (MSE): %.6f" % test_mse)

# 计算决定系数（R^2）
test_r2 = r2_score(y_test, y_pred_test)
print("XGBoost 测试集决定系数 (R^2): %.6f" % test_r2)

# 计时开始
tic = time()

print('Computing partial dependence plots...')
# for feature in feature_names:
#     fig_xgb, ax_xgb = plt.subplots(figsize=(10, 6))
#     display_xgb = PartialDependenceDisplay.from_estimator(
#         model,
#         X_train,
#         features=[feature],
#         kind='average',
#         grid_resolution=20,
#         ax=ax_xgb
#     )
#     ax_xgb.set_ylim([0.25, 1.25])
#     ax_xgb.figure.canvas.draw()
#     for ax_ in display_xgb.axes_.ravel():
#         ax_.set_ylabel('Predicted qₑ(mg/g)', labelpad=10, fontsize=30)
#         ax_.set_xlabel(feature, fontsize=16)
#         ax_.tick_params(axis='both', which='major', labelsize=16)
#         for line in ax_.lines:
#             line.set_linewidth(4)
#             plt.show()  # 显示图像
#     plt.close(fig_xgb)


print(f"done in {time() - tic:.3f}s")




# 创建保存路径，如果路径不存在则创建
# save_path = r'D:\pytharm\person\部分依赖图'
# os.makedirs(save_path, exist_ok=True)

# 计算部分依赖图，使用清理后的特征名称
for feature in feature_names:
    fig_xgb, ax_xgb = plt.subplots(figsize=(8, 6))  # 调整图像大小
    display_xgb = PartialDependenceDisplay.from_estimator(
        model,
        X_train,
        features=[feature],
        kind='average',
        grid_resolution=20,
        ax=ax_xgb
    )
    ax_xgb.set_ylim([0.25, 1.25])
    ax_xgb.figure.canvas.draw()
    for ax_ in display_xgb.axes_.ravel():
        ax_.set_ylabel('Predicted degradation rate', labelpad=10, fontsize=24)  # 调整字体大小
        ax_.set_xlabel(feature, fontsize=24)
        ax_.tick_params(axis='both', which='major', labelsize=24)
        for line in ax_.lines:
            line.set_linewidth(3)  # 调整线条宽度
            plt.show()  # 显示图像
    # # 保存图像
    # feature_cleaned = feature.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    # fig_xgb.savefig(os.path.join(save_path, f'partial_dependence_xgb_{feature_cleaned}.png'),
    #                 bbox_inches='tight', dpi=100)
    # plt.close(fig_xgb)  # 关闭图像
