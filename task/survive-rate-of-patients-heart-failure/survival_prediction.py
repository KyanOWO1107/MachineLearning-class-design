import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

# 数据加载
data = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')

# 数据探索
print('数据集维度:', data.shape)
print('\n前5行数据:\n', data.head())
print('\n统计描述:\n', data.describe())
print('\n缺失值统计:\n', data.isnull().sum())

# 数据预处理
X = data.drop(['DEATH_EVENT', 'time'], axis=1)
y = data['DEATH_EVENT']

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 模型训练
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 预测与评估
y_pred = model.predict(X_test_scaled)
print('\n准确率:', accuracy_score(y_test, y_pred))
print('\n分类报告:\n', classification_report(y_test, y_pred))

# 特征重要性分析
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.coef_[0]})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print('\n特征重要性排序:\n', feature_importance)

# 在数据探索部分后添加可视化
plt.figure(figsize=(12,8))
sns.pairplot(data[['age','ejection_fraction','serum_creatinine','DEATH_EVENT']], hue='DEATH_EVENT')
plt.savefig('data_distribution.png')

# 在模型评估后添加交叉验证
cv = StratifiedKFold(n_splits=5)
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
print('\n交叉验证平均准确率:', cv_scores.mean())
print('交叉验证标准差:', cv_scores.std())

# 添加ROC曲线绘制
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_scaled)[:,1])
plt.figure(figsize=(10,6))
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')

# 在特征重要性部分添加可视化
plt.figure(figsize=(10,6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('特征重要性排序')
plt.tight_layout()
plt.savefig('feature_importance.png')