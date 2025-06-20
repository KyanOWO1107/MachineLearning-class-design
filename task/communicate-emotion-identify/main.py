import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score

# 数据加载
def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t', names=['label', 'text'])
    return df['text'].tolist(), df['label'].tolist()

# 文本向量化
def vectorize_text(texts):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

# 模型训练与评估
def train_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))
    return model, y_pred

# 新增验证指标计算
def calculate_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        'accuracy': report['accuracy'],
        'macro_avg': report['macro avg'],
        'weighted_avg': report['weighted avg']
    }

# 新增混淆矩阵可视化
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'results/{model_name}_confusion_matrix.png')
    plt.close()

# 新增交叉验证流程
def cross_validation(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return {
        'mean_accuracy': scores.mean(),
        'std_accuracy': scores.std()
    }

# 修改主程序
if __name__ == '__main__':
    # 加载训练数据
    texts, labels = load_data('data/train.txt')
    
    # 特征工程
    X, vectorizer = vectorize_text(texts)
    y = labels
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 初始化模型
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced'),
        'SVM': SVC(class_weight='balanced'),
        'Random Forest': RandomForestClassifier(class_weight='balanced_subsample')
    }
    
    # 训练并评估模型
    results = {}
    for name, model in models.items():
        print(f'Training {name}')
        trained_model, preds = train_evaluate(model, X_train, X_test, y_train, y_test)
        results[name] = {
            'model': trained_model,
            'predictions': preds
        }
    # 新增开发集加载
    dev_texts, dev_labels = load_data('data/dev.txt')
    X_dev = vectorizer.transform(dev_texts)

    # 在模型训练后添加验证流程
    for name in results:
        # 开发集验证
        dev_preds = results[name]['model'].predict(X_dev)
        
        # 计算各项指标
        metrics = calculate_metrics(dev_labels, dev_preds)
        
        # 执行交叉验证
        cv_results = cross_validation(results[name]['model'], X_train, y_train)
        
        # 保存结果
        results[name].update({
            'dev_metrics': metrics,
            'cv_results': cv_results
        })
        
        # 生成可视化图表
        plot_confusion_matrix(dev_labels, dev_preds, name)

    # 新增指标对比可视化
    plt.figure(figsize=(12,6))
    for name, data in results.items():
        plt.bar(name, data['dev_metrics']['accuracy'], label=name)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.savefig('results/accuracy_comparison.png')
    plt.close()