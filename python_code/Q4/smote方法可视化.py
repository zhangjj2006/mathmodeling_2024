import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, roc_curve, precision_recall_curve
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置颜色方案
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def load_and_preprocess_data(file_path):
    """加载并预处理数据"""
    print("正在加载数据...")
    
    try:
        # 直接读取Excel文件
        df = pd.read_excel(file_path)
        print(f"成功读取数据，形状: {df.shape}")
        
    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        return None
    
    # 处理缺失值
    print("\n处理缺失值...")
    
    # 删除Unnamed列（如果存在）
    columns_to_drop = [col for col in df.columns if 'Unnamed' in str(col)]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        print(f"已删除列: {columns_to_drop}")
    
    # 处理孕妇BMI缺失值
    if '孕妇BMI' in df.columns and df['孕妇BMI'].isnull().sum() > 0:
        bmi_median = df['孕妇BMI'].median()
        df['孕妇BMI'] = df['孕妇BMI'].fillna(bmi_median)
        print(f"使用中位数 {bmi_median:.2f} 填充孕妇BMI缺失值")
    
    # 创建目标变量 - 是否有任何染色体异常
    if all(col in df.columns for col in ['T21', 'T18', 'T13']):
        df['染色体异常'] = df[['T21', 'T18', 'T13']].max(axis=1)
        print("使用现有的T21, T18, T13列创建目标变量")
    elif '染色体的非整倍体' in df.columns:
        # 如果没有T21, T18, T13列，但有色体的非整倍体列，则创建它们
        df['T21'] = 0
        df['T18'] = 0
        df['T13'] = 0
        
        for idx, value in df['染色体的非整倍体'].items():
            if pd.isna(value) or value == '':
                continue
            
            value_str = str(value).upper()
            if 'T21' in value_str or '21' in value_str:
                df.at[idx, 'T21'] = 1
            if 'T18' in value_str or '18' in value_str:
                df.at[idx, 'T18'] = 1
            if 'T13' in value_str or '13' in value_str:
                df.at[idx, 'T13'] = 1
        
        df['染色体异常'] = df[['T21', 'T18', 'T13']].max(axis=1)
        print("使用'染色体的非整倍体'列创建目标变量")
    else:
        print("错误: 无法找到创建目标变量所需的列")
        return None
    
    print(f"\n目标变量分布:")
    print(df['染色体异常'].value_counts())
    print(f"异常样本比例: {df['染色体异常'].mean()*100:.2f}%")
    
    return df

def prepare_features(df):
    """准备特征矩阵和目标向量"""
    # 可能的特征列 - 只使用数值型特征
    possible_features = [
        '孕妇年龄', '孕妇BMI', '怀孕次数', '生产次数',  # 基本信息
        'X染色体的Z值', '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值',  # Z值
        'GC含量', 'X染色体浓度', '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量',  # GC相关
        '原始读段数', '唯一比对的读段数', '重复读段的比例', '在参考基因组上比对的比例', '被过滤掉读段数的比例'  # 测序质量
    ]
    
    # 只保留实际存在且为数值型的列
    available_features = []
    for col in possible_features:
        if col in df.columns:
            # 检查是否为数值型
            if pd.api.types.is_numeric_dtype(df[col]):
                available_features.append(col)
            else:
                print(f"警告: 列 {col} 不是数值型，将被忽略")
    
    print(f"\n使用的数值型特征: {available_features}")
    
    if not available_features:
        print("错误: 没有找到任何数值型特征列")
        return None, None
    
    X = df[available_features]
    y = df['染色体异常']
    
    # 确保所有特征都是数值型
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            print(f"错误: 列 {col} 不是数值型，但被包含在特征中")
            return None, None
    
    return X, y

def handle_imbalanced_data(X, y, method='smote'):
    """处理不平衡数据"""
    print(f"\n处理不平衡数据 - 方法: {method}")
    print(f"原始数据分布: {Counter(y)}")
    
    # 确保X是数值型
    if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
        print("错误: 特征矩阵包含非数值型数据")
        return X, y
    
    if method == 'smote':
        sampler = SMOTE(random_state=42)
    elif method == 'adasyn':
        sampler = ADASYN(random_state=42)
    elif method == 'undersample':
        sampler = RandomUnderSampler(random_state=42)
    elif method == 'smoteenn':
        sampler = SMOTEENN(random_state=42)
    elif method == 'smotetomek':
        sampler = SMOTETomek(random_state=42)
    else:
        return X, y  # 不处理
    
    X_res, y_res = sampler.fit_resample(X, y)
    print(f"处理后数据分布: {Counter(y_res)}")
    
    return X_res, y_res

def plot_cv_results(cv_results, model_name, scoring_metrics):
    """绘制交叉验证结果"""
    n_folds = len(cv_results['test_accuracy'])
    x = np.arange(n_folds)
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, metric in enumerate(scoring_metrics):
        scores = cv_results[f'test_{metric}']
        ax.bar(x + i*width, scores, width, label=metric, alpha=0.8)
    
    ax.set_xlabel('交叉验证折数')
    ax.set_ylabel('分数')
    ax.set_title(f'{model_name} 交叉验证结果')
    ax.set_xticks(x + width*2)
    ax.set_xticklabels([f'Fold {i+1}' for i in range(n_folds)])
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_cv_results.png')
    plt.close()
    
    # 绘制箱线图
    fig, ax = plt.subplots(figsize=(10, 6))
    scores_data = [cv_results[f'test_{metric}'] for metric in scoring_metrics]
    ax.boxplot(scores_data, labels=scoring_metrics)
    ax.set_title(f'{model_name} 交叉验证分数分布')
    ax.set_ylabel('分数')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_cv_boxplot.png')
    plt.close()

def plot_roc_curves(models, X_test, y_test, model_names):
    """绘制ROC曲线"""
    plt.figure(figsize=(10, 8))
    
    for i, model in enumerate(models):
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        
        plt.plot(fpr, tpr, label=f'{model_names[i]} (AUC = {auc_score:.3f})', color=colors[i], linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='随机猜测 (AUC = 0.5)')
    plt.xlabel('假正率')
    plt.ylabel('真正率')
    plt.title('ROC曲线比较')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('roc_curves_comparison.png')
    plt.close()

def plot_precision_recall_curves(models, X_test, y_test, model_names):
    """绘制精确率-召回率曲线"""
    plt.figure(figsize=(10, 8))
    
    for i, model in enumerate(models):
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
        
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        avg_precision = np.mean(precision)
        
        plt.plot(recall, precision, label=f'{model_names[i]} (AP = {avg_precision:.3f})', color=colors[i], linewidth=2)
    
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('精确率-召回率曲线比较')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('precision_recall_curves_comparison.png')
    plt.close()

def plot_metrics_comparison(metrics_df):
    """绘制模型指标比较图"""
    metrics = metrics_df.columns[1:]  # 排除模型名列
    
    # 计算需要的行数和列数
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)  # 每行最多3列
    n_rows = (n_metrics + n_cols - 1) // n_cols  # 向上取整
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    
    # 如果只有一行或一列，确保axes是二维数组
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # 绘制每个指标的条形图
    for i, metric in enumerate(metrics):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        bars = ax.bar(range(len(metrics_df)), metrics_df[metric], 
                     color=colors[:len(metrics_df)], alpha=0.8)
        ax.set_xlabel('模型')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} 比较')
        ax.set_xticks(range(len(metrics_df)))
        ax.set_xticklabels(metrics_df['模型'], rotation=45)
        
        # 在柱子上添加数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(n_metrics, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png')
    plt.close()
    
    # 绘制雷达图
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # 准备数据
    categories = list(metrics)
    N = len(categories)
    
    # 计算每个模型的角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合雷达图
    
    # 绘制每个模型的雷达图
    for i, row in metrics_df.iterrows():
        values = row[metrics].values.flatten().tolist()
        values += values[:1]  # 闭合雷达图
        ax.plot(angles, values, linewidth=2, linestyle='solid', 
                label=row['模型'], color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # 添加类别标签
    plt.xticks(angles[:-1], categories)
    
    # 添加图例
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # 添加标题
    plt.title('模型性能雷达图比较', size=20, y=1.05)
    
    plt.tight_layout()
    plt.savefig('radar_chart_comparison.png')
    plt.close()
def train_and_evaluate_model(X, y, model_name='random_forest', use_cross_validation=True):
    """训练和评估模型"""
    # 检查数据是否有效
    if X is None or y is None or len(X) == 0 or len(y) == 0:
        print("错误: 无效的数据")
        return None, 0, 0
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 选择模型
    if model_name == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    elif model_name == 'logistic':
        model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    elif model_name == 'xgboost':
        # 计算正负样本比例
        scale_pos_weight = len(y[y==0])/len(y[y==1]) if len(y[y==1]) > 0 else 1
        model = XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight)
    elif model_name == 'svm':
        model = SVC(random_state=42, class_weight='balanced', probability=True)
    else:
        model = RandomForestClassifier(random_state=42)
    
    # 创建预处理管道
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])
    
    # 创建完整管道
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # 使用交叉验证
    cv_results = {}
    if use_cross_validation:
        print(f"\n对 {model_name} 进行交叉验证...")
        
        # 使用分层K折交叉验证
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 定义评分指标
        scoring_metrics = ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']
        
        # 执行交叉验证
        for metric in scoring_metrics:
            cv_scores = cross_val_score(clf, X, y, cv=cv, scoring=metric)
            cv_results[f'test_{metric}'] = cv_scores
            print(f"{model_name} 交叉验证 {metric}: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # 绘制交叉验证结果
        plot_cv_results(cv_results, model_name, scoring_metrics)
    
    # 训练最终模型
    clf.fit(X_train, y_train)
    
    # 预测
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    # 评估模型
    print(f"\n{model_name} 模型评估:")
    print(classification_report(y_test, y_pred))
    
    # 计算各种评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"AUC: {auc_score:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} 混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()
    
    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color=colors[0], lw=2, label=f'ROC曲线 (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率')
    plt.ylabel('真正率')
    plt.title(f'{model_name} ROC曲线')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{model_name}_roc_curve.png')
    plt.close()
    
    # 绘制精确率-召回率曲线
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, color=colors[1], lw=2, label=f'精确率-召回率曲线')
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title(f'{model_name} 精确率-召回率曲线')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{model_name}_precision_recall_curve.png')
    plt.close()
    
    return clf, {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'auc': auc_score, 'f1': f1}, cv_results

def compare_sampling_methods(X, y):
    """比较不同的采样方法"""
    methods = ['none', 'smote', 'adasyn', 'undersample', 'smoteenn', 'smotetomek']
    results = []
    
    for method in methods:
        print(f"\n=== 使用方法: {method} ===")
        X_res, y_res = handle_imbalanced_data(X, y, method)
        
        # 使用随机森林评估每种方法
        model, metrics, _ = train_and_evaluate_model(X_res, y_res, 'random_forest', use_cross_validation=False)
        results.append({
            'method': method,
            **metrics
        })
    
    # 比较结果
    results_df = pd.DataFrame(results)
    print("\n=== 不同采样方法比较 ===")
    print(results_df)
    
    # 绘制比较图
    plt.figure(figsize=(12, 8))
    metrics = ['accuracy', 'precision', 'recall', 'auc', 'f1']
    x_pos = np.arange(len(results_df))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        plt.bar(x_pos + i*width, results_df[metric], width, label=metric, alpha=0.8)
    
    plt.xlabel('采样方法')
    plt.ylabel('分数')
    plt.title('不同采样方法性能比较')
    plt.xticks(x_pos + width*2, results_df['method'])
    plt.legend()
    plt.tight_layout()
    plt.savefig('sampling_methods_comparison.png')
    plt.close()
    
    return results_df

def hyperparameter_tuning(X, y, model_name='random_forest'):
    """超参数调优"""
    print(f"\n=== 对 {model_name} 进行超参数调优 ===")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 创建预处理管道
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])
    
    # 根据模型选择不同的参数网格
    if model_name == 'random_forest':
        model = RandomForestClassifier(random_state=42, class_weight='balanced')
        param_dist = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
    elif model_name == 'xgboost':
        scale_pos_weight = len(y[y==0])/len(y[y==1]) if len(y[y==1]) > 0 else 1
        model = XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight)
        param_dist = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [3, 6, 9],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__subsample': [0.8, 0.9, 1.0]
        }
    else:
        print(f"暂不支持 {model_name} 的超参数调优")
        return None
    
    # 创建完整管道
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # 使用随机搜索进行超参数调优
    random_search = RandomizedSearchCV(
        clf, param_distributions=param_dist, n_iter=20, 
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='f1', random_state=42, n_jobs=-1
    )
    
    # 执行随机搜索
    random_search.fit(X_train, y_train)
    
    # 输出最佳参数
    print(f"最佳参数: {random_search.best_params_}")
    print(f"最佳交叉验证分数: {random_search.best_score_:.4f}")
    
    # 在测试集上评估最佳模型
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    
    print(f"测试集准确率: {accuracy:.4f}")
    print(f"测试集精确率: {precision:.4f}")
    print(f"测试集召回率: {recall:.4f}")
    print(f"测试集 AUC: {auc_score:.4f}")
    print(f"测试集 F1 分数: {f1:.4f}")
    
    return best_model, {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'auc': auc_score, 'f1': f1}

def main():
    """主函数"""
    # 文件路径
    input_file = "./python_code/Q4/处理后的女胎NIPT数据.xlsx"
    
    # 加载和预处理数据
    df = load_and_preprocess_data(input_file)
    if df is None:
        print("数据加载失败，程序退出")
        return
    
    # 准备特征和目标
    X, y = prepare_features(df)
    if X is None or y is None:
        print("特征准备失败，程序退出")
        return
    
    # 检查是否有任何非数值型数据
    print("\n检查数据类型...")
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            print(f"警告: 列 {col} 包含非数值型数据，将被删除")
            X = X.drop(columns=[col])
    
    # 如果删除了列，重新检查
    if len(X.columns) == 0:
        print("错误: 没有可用的数值型特征列")
        return
    
    print(f"最终使用的特征: {list(X.columns)}")
    
    # 比较不同的采样方法
    results_df = compare_sampling_methods(X, y)
    
    # 选择最佳方法
    best_method = results_df.loc[results_df['f1'].idxmax(), 'method']
    print(f"\n最佳采样方法: {best_method}")
    
    # 应用最佳采样方法
    X_res, y_res = handle_imbalanced_data(X, y, best_method)
    
    # 训练不同模型并进行交叉验证
    models = ['random_forest', 'logistic', 'xgboost', 'svm']
    trained_models = []
    model_metrics = []
    model_names = []
    
    for model_name in models:
        print(f"\n=== 训练 {model_name} 模型 ===")
        model, metrics, cv_results = train_and_evaluate_model(X_res, y_res, model_name, use_cross_validation=True)
        
        if model is not None:
            trained_models.append(model)
            model_metrics.append({'模型': model_name, **metrics})
            model_names.append(model_name)
    
    # 创建模型指标DataFrame
    metrics_df = pd.DataFrame(model_metrics)
    print("\n=== 不同模型性能比较 ===")
    print(metrics_df)
    
    # 绘制模型指标比较图
    plot_metrics_comparison(metrics_df)
    
    # 绘制ROC曲线比较
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.3, random_state=42, stratify=y_res
    )
    plot_roc_curves(trained_models, X_test, y_test, model_names)
    
    # 绘制精确率-召回率曲线比较
    plot_precision_recall_curves(trained_models, X_test, y_test, model_names)
    
    # 选择最佳模型
    best_model_name = metrics_df.loc[metrics_df['f1'].idxmax(), '模型']
    print(f"\n最佳模型: {best_model_name}")
    
    # 对最佳模型进行超参数调优
    best_model_tuned, tuned_metrics = hyperparameter_tuning(X_res, y_res, best_model_name)
    
    # 重新训练最终模型
    print(f"\n=== 训练最终模型: {best_model_name} ===")
    final_model, final_metrics, _ = train_and_evaluate_model(X_res, y_res, best_model_name, use_cross_validation=True)
    
    # 特征重要性（如果适用）
    if hasattr(final_model.named_steps['classifier'], 'feature_importances_'):
        feature_importances = final_model.named_steps['classifier'].feature_importances_
        features = X.columns
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': feature_importances
        }).sort_values('importance', ascending=False)
        
        print("\n特征重要性:")
        print(importance_df)
        
        # 绘制特征重要性图
        plt.figure(figsize=(12, 8))
        plt.barh(importance_df['feature'][:15], importance_df['importance'][:15], color=colors[0])
        plt.xlabel('重要性')
        plt.title('Top 15 特征重要性')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
    
    print("\n=== 分析完成 ===")

if __name__ == "__main__":
    main()