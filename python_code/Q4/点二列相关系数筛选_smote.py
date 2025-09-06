import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, make_scorer
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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
        'GC含量',  '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量',  # GC相关
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

def filter_features_point_biserial(X, y, threshold=0.1, top_k=None):
    """
    使用点二列相关系数进行初步特征筛选
    """
    print("\n=== 使用点二列相关系数进行初步特征筛选 ===")
    
    # 标准化特征（确保公平比较）
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    correlations = []
    p_values = []
    
    # 计算每个特征与目标变量的点二列相关系数
    for feature in X_scaled.columns:
        # 移除缺失值
        mask = ~(X_scaled[feature].isnull() | y.isnull())
        x_vals = X_scaled.loc[mask, feature]
        y_vals = y.loc[mask]
        
        # 计算点二列相关系数
        corr, p_val = pointbiserialr(x_vals, y_vals)
        correlations.append(corr)
        p_values.append(p_val)
    
    # 创建结果DataFrame
    corr_df = pd.DataFrame({
        'feature': X_scaled.columns,
        'correlation': correlations,
        'p_value': p_values,
        'abs_correlation': np.abs(correlations)
    }).sort_values('abs_correlation', ascending=False)
    
    print("特征与目标变量的点二列相关系数:")
    print(corr_df)
    
    # 绘制相关系数图
    plt.figure(figsize=(10, 8))
    plt.barh(corr_df['feature'], corr_df['correlation'])
    plt.xlabel('点二列相关系数')
    plt.title('特征与目标变量的点二列相关系数（标准化后）')
    plt.tight_layout()
    plt.savefig('point_biserial_correlations.png')
    plt.close()
    
    # 根据阈值或top_k筛选特征
    if top_k is not None:
        selected_features = corr_df.head(top_k)['feature'].tolist()
        print(f"\n选择前 {top_k} 个最相关的特征: {selected_features}")
    else:
        selected_features = corr_df[corr_df['abs_correlation'] >= threshold]['feature'].tolist()
        print(f"\n选择相关系数绝对值 >= {threshold} 的特征: {selected_features}")
    
    if not selected_features:
        print("警告: 没有特征满足筛选条件，使用所有特征")
        return X
    
    return X[selected_features]  # 返回原始数据，标准化只在筛选时使用

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
    elif method == 'smoteenn':
        sampler = SMOTEENN(random_state=42)
    else:
        return X, y  # 不处理
    
    X_res, y_res = sampler.fit_resample(X, y)
    print(f"处理后数据分布: {Counter(y_res)}")
    
    return X_res, y_res

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
    if use_cross_validation:
        print(f"\n对 {model_name} 进行交叉验证...")
        
        # 使用分层K折交叉验证
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 定义评分指标
        scoring = {
            'accuracy': 'accuracy',
            'f1': 'f1',
            'roc_auc': 'roc_auc',
            'precision': 'precision',
            'recall': 'recall'
        }
        
        # 执行交叉验证
        cv_results = cross_val_score(clf, X, y, cv=cv, scoring='f1')
        print(f"{model_name} 交叉验证 F1 分数: {cv_results.mean():.4f} (±{cv_results.std():.4f})")
        
        # 计算其他指标的交叉验证分数
        for metric_name, metric_scorer in scoring.items():
            cv_metric_results = cross_val_score(clf, X, y, cv=cv, scoring=metric_scorer)
            print(f"{model_name} 交叉验证 {metric_name}: {cv_metric_results.mean():.4f} (±{cv_metric_results.std():.4f})")
    
    # 训练最终模型
    clf.fit(X_train, y_train)
    
    # 预测
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    # 评估模型
    print(f"\n{model_name} 模型评估:")
    print(classification_report(y_test, y_pred))
    
    # 计算AUC
    auc_score = roc_auc_score(y_test, y_prob)
    print(f"AUC: {auc_score:.4f}")
    
    # 计算F1分数
    f1 = f1_score(y_test, y_pred)
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
    
    return clf, auc_score, f1

def compare_sampling_methods(X, y):
    """比较不同的采样方法"""
    methods = ['none', 'smote', 'smoteenn', 'adasyn']
    results = []
    
    for method in methods:
        print(f"\n=== 使用方法: {method} ===")
        X_res, y_res = handle_imbalanced_data(X, y, method)
        
        # 使用随机森林评估每种方法
        model, auc, f1 = train_and_evaluate_model(X_res, y_res, 'random_forest', use_cross_validation=False)
        results.append({
            'method': method,
            'auc': auc,
            'f1': f1,
            'distribution': Counter(y_res)
        })
    
    # 比较结果
    results_df = pd.DataFrame(results)
    print("\n=== 不同采样方法比较 ===")
    print(results_df[['method', 'auc', 'f1']])
    
    # 绘制比较图
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(results_df))
    plt.bar(x_pos - 0.2, results_df['auc'], 0.4, label='AUC')
    plt.bar(x_pos + 0.2, results_df['f1'], 0.4, label='F1 Score')
    plt.xlabel('采样方法')
    plt.ylabel('分数')
    plt.title('不同采样方法性能比较')
    plt.xticks(x_pos, results_df['method'])
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
    elif model_name == 'svm':
        model = SVC(random_state=42, class_weight='balanced', probability=True)
        param_dist = {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'classifier__kernel': ['rbf', 'linear']
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
    auc_score = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    
    print(f"测试集 AUC: {auc_score:.4f}")
    print(f"测试集 F1 分数: {f1:.4f}")
    
    return best_model, auc_score, f1

def embedded_feature_selection(X, y, model_name='random_forest', top_k=10):
    """
    使用嵌入式方法进行特征选择（基于树模型的特征重要性）
    
    参数:
    X: 特征矩阵
    y: 目标变量
    model_name: 使用的模型
    top_k: 选择前k个最重要的特征
    
    返回:
    筛选后的特征矩阵
    """
    print(f"\n=== 使用 {model_name} 进行嵌入式特征选择 ===")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 选择模型
    if model_name == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    elif model_name == 'xgboost':
        scale_pos_weight = len(y[y==0])/len(y[y==1]) if len(y[y==1]) > 0 else 1
        model = XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight)
    else:
        print(f"暂不支持 {model_name} 的嵌入式特征选择")
        return X
    
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
    
    # 训练模型
    clf.fit(X_train, y_train)
    
    # 获取特征重要性
    if hasattr(clf.named_steps['classifier'], 'feature_importances_'):
        feature_importances = clf.named_steps['classifier'].feature_importances_
        features = X.columns
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': feature_importances
        }).sort_values('importance', ascending=False)
        
        print("特征重要性排名:")
        print(importance_df)
        
        # 绘制特征重要性图
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'][:15], importance_df['importance'][:15])
        plt.xlabel('重要性')
        plt.title(f'{model_name} 特征重要性')
        plt.tight_layout()
        plt.savefig(f'{model_name}_feature_importance.png')
        plt.close()
        
        # 选择前k个最重要的特征
        selected_features = importance_df.head(top_k)['feature'].tolist()
        print(f"\n选择前 {top_k} 个最重要的特征: {selected_features}")
        
        return X[selected_features]
    else:
        print(f"{model_name} 模型不支持特征重要性分析")
        return X

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
    
    print(f"初始特征: {list(X.columns)}")
    
    # 第一步: 初步筛选 - 使用点二列相关系数
    X_filtered = filter_features_point_biserial(X, y, threshold=0.05, top_k=15)
    
    # 第二步: 精细筛选 - 使用嵌入式方法（基于树模型的特征重要性）
    X_final = embedded_feature_selection(X_filtered, y, model_name='random_forest', top_k=10)
    
    print(f"\n最终选择的特征: {list(X_final.columns)}")
    
    # 比较不同的采样方法
    results_df = compare_sampling_methods(X_final, y)
    
    # 选择最佳方法
    best_method = results_df.loc[results_df['f1'].idxmax(), 'method']
    print(f"\n最佳采样方法: {best_method}")
    
    # 应用最佳采样方法
    X_res, y_res = handle_imbalanced_data(X_final, y, best_method)
    
    # 训练不同模型并进行交叉验证
    models = ['random_forest', 'xgboost', 'svm']
    model_results = []
    
    for model_name in models:
        print(f"\n=== 训练 {model_name} 模型 ===")
        model, auc, f1 = train_and_evaluate_model(X_res, y_res, model_name, use_cross_validation=True)
        model_results.append({
            'model': model_name,
            'auc': auc,
            'f1': f1
        })
    
    # 比较不同模型
    model_results_df = pd.DataFrame(model_results)
    print("\n=== 不同模型性能比较 ===")
    print(model_results_df)
    
    # 选择最佳模型
    best_model_name = model_results_df.loc[model_results_df['f1'].idxmax(), 'model']
    print(f"\n最佳模型: {best_model_name}")
    
    # 对最佳模型进行超参数调优
    best_model_tuned, tuned_auc, tuned_f1 = hyperparameter_tuning(X_res, y_res, best_model_name)
    
    # 重新训练最终模型
    print(f"\n=== 训练最终模型: {best_model_name} ===")
    final_model, final_auc, final_f1 = train_and_evaluate_model(X_res, y_res, best_model_name, use_cross_validation=True)
    
    # 特征重要性（如果适用）
    if hasattr(final_model.named_steps['classifier'], 'feature_importances_'):
        feature_importances = final_model.named_steps['classifier'].feature_importances_
        features = X_final.columns
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': feature_importances
        }).sort_values('importance', ascending=False)
        
        print("\n最终模型特征重要性:")
        print(importance_df)
        
        # 绘制特征重要性图
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'][:15], importance_df['importance'][:15])
        plt.xlabel('重要性')
        plt.title('最终模型 - Top 15 特征重要性')
        plt.tight_layout()
        plt.savefig('final_feature_importance.png')
        plt.close()
    
    print("\n=== 分析完成 ===")

if __name__ == "__main__":
    main()
    