import os
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# 初始化结果目录
os.makedirs('result', exist_ok=True)

# 数据加载
data = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']

# 模型配置
models = {
    'L2 Linear': RidgeClassifier(alpha=0.5),
    'Weighted Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=100)
}

# 5折分层交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True)

# 结果存储
report_content = "模型评估报告\n================\n"
plt.figure(figsize=(10, 8))

for name, model in models.items():
    pipeline = make_pipeline(
        StandardScaler(),
        model
    )
    
    # 交叉验证
    cv_results = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=['accuracy', 'precision', 'f1', 'roc_auc'],
        return_estimator=True
    )
    
    # 收集测试集预测
    y_tests, y_probas = [], []
    for estimator, (_, test_idx) in zip(cv_results['estimator'], cv.split(X, y)):
        try:
            probas = estimator.predict_proba(X.iloc[test_idx])[:,1]
        except AttributeError:
            probas = estimator.decision_function(X.iloc[test_idx])
        y_tests.append(y.iloc[test_idx])
        y_probas.append(probas)
    
    # 计算指标
    y_test_all = np.concatenate(y_tests)
    y_proba_all = np.concatenate(y_probas)
    
    # 存储报告
    report_content += f"\n{name}:\n"
    report_content += f"  准确率: {np.mean(cv_results['test_accuracy']):.4f} ± {np.std(cv_results['test_accuracy']):.4f}\n"
    report_content += f"  精确率: {np.mean(cv_results['test_precision']):.4f} ± {np.std(cv_results['test_precision']):.4f}\n"
    report_content += f"  F1值: {np.mean(cv_results['test_f1']):.4f} ± {np.std(cv_results['test_f1']):.4f}\n"
    report_content += f"  AUC值: {roc_auc_score(y_test_all, y_proba_all):.4f}\n"
    
    # 绘制ROC
    fpr, tpr, _ = roc_curve(y_test_all, y_proba_all)
    plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y_test_all, y_proba_all):.2f})')

# 输出报告
with open('result/report.txt', 'w', encoding='utf-8') as f:
    f.write(report_content)

# 保存ROC曲线
plt.plot([0,1],[0,1],'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curves Compare')
plt.legend()
plt.savefig('result/roc_curves.png')
print('DONE，Output files are saved in directory -> result')