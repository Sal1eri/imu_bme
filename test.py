import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix

# 生成一个不平衡的武侠数据集
# 假设特征表示武功修炼时间、战斗胜率等，标签表示是否为高手
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.9, 0.1], random_state=42)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练一个逻辑回归模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 预测测试集
y_pred_prob = model.predict_proba(X_test)[:, 1]

# 计算 ROC 曲线和 AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

print(thresholds)
auc = roc_auc_score(y_test, y_pred_prob)

# 可视化结果
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("ROC 曲线")
plt.plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlabel("假阳性率")
plt.ylabel("真阳性率")
plt.legend(loc="lower right")

plt.subplot(1, 2, 2)
plt.title("AUC 值示意")
plt.fill_between(fpr, tpr, color='blue', alpha=0.3)
plt.plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {auc:.2f}")
plt.xlabel("假阳性率")
plt.ylabel("真阳性率")
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()

print(f"AUC: {auc:.2f}")