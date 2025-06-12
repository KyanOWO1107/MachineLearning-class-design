import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

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