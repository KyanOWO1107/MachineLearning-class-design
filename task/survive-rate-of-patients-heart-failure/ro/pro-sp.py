# 导入matplotlib库用于绘图
import matplotlib.pyplot as plt
# 导入seaborn库，一个基于matplotlib的数据可视化库
import seaborn as sns
# 导入pandas库，用于数据分析和操作
import pandas as pd
# 导入train_test_split函数，用于分割数据集
from sklearn.model_selection import train_test_split
# 导入线性回归模型
from sklearn.linear_model import LinearRegression, Ridge
# 导入随机森林回归模型
from sklearn.ensemble import RandomForestRegressor
# 导入评估模型性能的函数：均方误差和R^2分数
from sklearn.metrics import mean_squared_error, r2_score
# 导入Pipeline，用于创建一个处理/预测流水线，将多个步骤串联起来
from sklearn.pipeline import Pipeline
# 导入StandardScaler，用于特征缩放
from sklearn.preprocessing import StandardScaler
# 导入SelectKBest和f_regression，用于特征选择
from sklearn.feature_selection import SelectKBest, f_regression
# 导入fetch_ucirepo，用于获取UCI机器学习库的数据集
from ucimlrepo import fetch_ucirepo

# 定义一个函数evaluate_model，用于评估模型的性能
def evaluate_model(model, X_test, y_test):
    # 使用模型对测试集进行预测
    y_pred = model.predict(X_test)
    # 计算预测结果与实际结果的均方误差
    mse = mean_squared_error(y_test, y_pred)
    # 计算预测结果与实际结果的R^2分数
    r2 = r2_score(y_test, y_pred)
    # 返回均方误差和R^2分数
    return mse, r2

# 定义一个函数plot_comparison_and_residuals，用于绘制模型的预测结果与实际结果的对比图和残差图
def plot_comparison_and_residuals(y_test, y_pred, model_name):
    # 创建一个图形，大小为12x6英寸
    plt.figure(figsize=(12, 6))
    # 创建子图1，占1行2列的第1个位置
    plt.subplot(1, 2, 1)
    # 绘制实际值与预测值的散点图
    plt.scatter(y_test, y_pred, alpha=0.5)
    # 绘制一条斜率为1的参考线
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    # 设置x轴标签为'Actual'
    plt.xlabel('Actual')
    # 设置y轴标签为'Predicted'
    plt.ylabel('Predicted')
    # 设置标题，显示模型名称
    plt.title(f'Predictions vs Actuals for {model_name}')
    # 创建子图2，占1行2列的第2个位置
    plt.subplot(1, 2, 2)
    # 计算残差
    residuals = y_test - y_pred
    # 绘制残差的直方图和核密度估计图
    sns.histplot(residuals, kde=True, color='orange')
    # 设置残差图的标题
    plt.title(f'Residuals for {model_name}')
    # 设置残差图的x轴标签
    plt.xlabel('Residuals')
    # 设置残差图的y轴标签
    plt.ylabel('Density')
    # 调整子图布局以适应图形
    plt.tight_layout()
    # 保存图形到文件，并指定文件名
    plt.savefig(f'{model_name}_comparison_residuals.png')
    # 关闭图形以释放内存
    plt.close()

# 1. 获取数据集
# 使用fetch_ucirepo函数获取ID为519的数据集
heart_failure_clinical_records = fetch_ucirepo(id=519)
# 将数据集中的特征赋值给变量X
X = heart_failure_clinical_records.data.features
# 将数据集中的目标变量赋值给变量y，并使用squeeze()去除单维度的轴
y = heart_failure_clinical_records.data.targets.squeeze()

# 2. 特征预处理与特征选择
# 打印数据集中特征的缺失值情况
print("缺失值情况:")
print(X.isnull().sum())

# 创建SelectKBest实例，使用f_regression作为评分函数，选择5个特征
selector = SelectKBest(score_func=f_regression, k=5)
# 使用选择的特征对数据进行转换
X_new = selector.fit_transform(X, y)
# 获取选择的特征名称
selected_features = X.columns[selector.get_support(indices=False)]

# 3. 使用Pipeline进行标准化和特征选择
# 创建一个Pipeline，包含特征选择和标准化步骤
pipeline = Pipeline([
    ('selector', SelectKBest(score_func=f_regression, k=5)),
    ('scaler', StandardScaler())
])

# 4. 划分数据集为训练集和测试集
# 使用train_test_split函数分割数据集，测试集占20%，随机状态设置为42
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

# 5. 使用Pipeline对数据进行转换
# 使用Pipeline对训练集进行转换
X_train_transformed = pipeline.fit_transform(X_train, y_train)
# 使用Pipeline对测试集进行转换
X_test_transformed = pipeline.transform(X_test)

# 6. 训练和评估模型
# 创建一个包含不同模型的字典
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest Regression': RandomForestRegressor(n_estimators=100, random_state=42)
}
# 初始化一个字典来存储模型性能
model_performance = {}

# 遍历模型字典，训练每个模型并评估性能
for name, model in models.items():
    # 训练模型
    model.fit(X_train_transformed, y_train)
    # 使用模型对测试集进行预测
    y_pred = model.predict(X_test_transformed)
    # 评估模型性能
    mse, r2 = evaluate_model(model, X_test_transformed, y_test)
    # 将性能指标存储到字典中
    model_performance[name] = {'MSE': mse, 'R2': r2}
    # 打印模型的性能
    print(f"{name} performance: MSE: {mse:.4f}, R2: {r2:.4f}")
    # 绘制模型的预测结果与实际结果的对比图和残差图
    plot_comparison_and_residuals(y_test, y_pred, name)

# 7. 特征重要性可视化 - 随机森林
# 获取随机森林模型
rf_reg = models['Random Forest Regression']
# 训练随机森林模型
rf_reg.fit(X_train_transformed, y_train)
# 获取随机森林模型的特征重要性
feature_importances_rf = pd.Series(rf_reg.feature_importances_, index=selected_features)

# 创建图形，大小为10x6英寸
plt.figure(figsize=(10, 6))
# 绘制特征重要性的条形图
sns.barplot(x=feature_importances_rf, y=feature_importances_rf.index)
# 设置特征重要性图的标题
plt.title('Feature Importance - Random Forest Regression')
# 设置特征重要性图的x轴标签
plt.xlabel('Importance')
# 设置特征重要性图的y轴标签
plt.ylabel('Features')
# 调整图形布局
plt.tight_layout()
# 保存特征重要性图到文件
plt.savefig('feature_importance_rf.png')
# 关闭图形

# 8. 绘制线性回归和岭回归的系数
# 获取线性回归模型
linear_reg = models['Linear Regression']
# 获取岭回归模型
ridge_reg = models['Ridge Regression']

# 获取线性回归模型的系数
linear_coeffs = pd.Series(linear_reg.coef_, index=selected_features)
# 获取岭回归模型的系数
ridge_coeffs = pd.Series(ridge_reg.coef_, index=selected_features)

# 创建图形，大小为12x6英寸
plt.figure(figsize=(12, 6))
# 创建子图1，占1行2列的第1个位置
plt.subplot(1, 2, 1)
# 绘制线性回归系数的条形图
sns.barplot(x=linear_coeffs, y=linear_coeffs.index)
# 设置线性回归系数图的标题
plt.title('Coefficients - Linear Regression')

# 创建子图2，占1行2列的第2个位置
plt.subplot(1, 2, 2)
# 绘制岭回归系数的条形图
sns.barplot(x=ridge_coeffs, y=ridge_coeffs.index)
# 设置岭回归系数图的标题
plt.title('Coefficients - Ridge Regression')

# 调整图形布局
plt.tight_layout()
# 保存模型系数图到文件
plt.savefig('model_coefficients.png')
# 关闭图形

# 9. 打印所有模型的性能
# 遍历模型性能字典，打印每个模型的最终性能
for name, performance in model_performance.items():
    # 打印模型名称、MSE和R^2分数
    print(f"{name} Final Performance: MSE: {performance['MSE']:.4f}, R2: {performance['R2']:.4f}")
