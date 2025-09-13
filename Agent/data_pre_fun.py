import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
"""
涉及到我们数据处理所需要的各种函数和类。
这些函数和类包括：

"""




def missing_value_stats(df, additional_missing_values=None):
    """
    统计每一列的缺失值数量和占比
    
    Args:
        df: pandas DataFrame
        additional_missing_values: 额外的缺失值表示，如 ['.', '?', '', ' ', 'NULL', 'null']
    """
    if additional_missing_values is None:
        additional_missing_values = ['.', '?', '', ' ', 'NULL', 'null', 'NA', 'n/a', 'N/A']
    df_copy = df.copy()
    # 将额外的缺失值标记替换为NaN
    for missing_val in additional_missing_values:
        df_copy = df_copy.replace(missing_val, np.nan)
    # 计算缺失值统计
    total = df_copy.isnull().sum()
    percent = df_copy.isnull().mean() * 100
    missing_dict = {
        col: {"缺失值数量": int(total[col]), "缺失值占比(%)": float(percent[col])}
        for col in df.columns
    }
    return missing_dict


def load_data(filepath):
    # 确保我们原始的数据格式是csv
    df = pd.read_csv(filepath)
    missing_info = missing_value_stats(df)
    unique_counts = df.nunique().to_dict()
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.apply(str).to_dict(),
        "missing_info": missing_info,
        "unique_counts": unique_counts
    }



# 在现有函数后面添加新函数
def feature_importance_random_forest(
    data_path: str,
    categorical_features: list[str],
    numeric_features: list[str],
    target_col: str,
    problem_type: str = "regression",
    n_estimators: int = 100,
    random_state: int = 42,
    n_jobs: int = -1
) -> dict[str, float]:
    """使用随机森林计算所有特征的重要性。

    该函数会自动处理数值特征和类别特征的缺失值，并对类别特征进行编码，
    然后使用随机森林计算每个特征相对于目标变量的重要性得分。

    Args:
        data_path (str): 数据文件的完整路径。
        categorical_features (list[str]): 类别特征的名称列表。
        numeric_features (list[str]): 数值特征的名称列表。
        target_col (str): 目标变量的列名。
        problem_type (str): 问题类型，"regression" 或 "classification"。
        n_estimators (int): 随机森林的树数量。
        random_state (int): 随机种子。
        n_jobs (int): 并行数。
    
    Returns:
        dict[str, float]: 一个字典，键是特征名，值是对应的随机森林重要性得分。
    """
    # 加载数据
    df = pd.read_csv(data_path)
    all_features = categorical_features + numeric_features
    available_features = [f for f in all_features if f in df.columns]
    available_categorical = [f for f in categorical_features if f in df.columns]
    available_numeric = [f for f in numeric_features if f in df.columns]
    if not available_features:
        return {}
    df_processed = df[available_features + [target_col]]



    # 1. 处理数值特征缺失值（用均值填充）
    for feature in available_numeric:
        df_processed[feature] = df_processed[feature].fillna(df_processed[feature].mean())
    # 2. 处理类别特征缺失值和编码
    for feature in available_categorical:
        # 用众数填充缺失值
        mode_value = df_processed[feature].mode()
        if len(mode_value) > 0:
            df_processed[feature] = df_processed[feature].fillna(mode_value[0])
        else:
            df_processed[feature] = df_processed[feature].fillna('-1')
        # 标签编码
        le = LabelEncoder()
        df_processed[feature] = le.fit_transform(df_processed[feature].astype(str))
    # 3. 准备特征矩阵和目标变量
    X = df_processed[available_features]
    y = df_processed[target_col]
    # 4. 根据问题类型选择随机森林模型
    if problem_type.lower() == "classification":
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs
        )
    else:
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs
        )
    # 5. 训练模型并获取特征重要性
    rf_model.fit(X, y)
    rf_importances = rf_model.feature_importances_
    
    # 6. 返回特征重要性字典
    importance_dict = {
        feature: float(importance) 
        for feature, importance in zip(available_features, rf_importances)
    }
    
    return importance_dict



def feature_importance_mutual_info(
    data_path: str,
    categorical_features: list[str],
    numeric_features: list[str],
    target_col: str,
    problem_type: str = "regression"
) -> dict[str, float]:
    """使用互信息法计算所有特征的重要性。

    该函数会自动处理数值特征和类别特征的缺失值，并对类别特征进行编码，
    然后使用互信息法计算每个特征相对于目标变量的重要性得分。

    Args:
        data_path (str): 数据文件的完整路径。
        categorical_features (list[str]): 类别特征的名称列表。
        numeric_features (list[str]): 数值特征的名称列表。
        target_col (str): 目标变量的列名。
        problem_type (str): 问题类型，"regression" 或 "classification"。
    
    Returns:
        dict[str, float]: 一个字典，键是特征名，值是对应的互信息重要性得分。
    """
    # 加载数据
    df = pd.read_csv(data_path)
    # 合并所有特征
    all_features = categorical_features + numeric_features
    # 过滤存在的特征
    available_features = [f for f in all_features if f in df.columns]
    available_categorical = [f for f in categorical_features if f in df.columns]
    available_numeric = [f for f in numeric_features if f in df.columns]
    if not available_features:
        return {}
    # 创建数据副本进行处理
    df_processed = df[available_features + [target_col]].copy()
    # 1. 处理数值特征缺失值（用均值填充）
    for feature in available_numeric:
        df_processed[feature] = df_processed[feature].fillna(df_processed[feature].mean())
    # 2. 处理类别特征缺失值和编码
    for feature in available_categorical:
        # 用众数填充缺失值
        mode_value = df_processed[feature].mode()
        if len(mode_value) > 0:
            df_processed[feature] = df_processed[feature].fillna(mode_value[0])
        else:
            df_processed[feature] = df_processed[feature].fillna('-1')
        # 标签编码
        le = LabelEncoder()
        df_processed[feature] = le.fit_transform(df_processed[feature].astype(str))
    # 3. 准备特征矩阵和目标变量
    X = df_processed[available_features]
    y = df_processed[target_col]
    
    # 4. 计算互信息得分
    if problem_type.lower() == "classification":
        mi_scores = mutual_info_classif(X, y, random_state=42)
    else:
        mi_scores = mutual_info_regression(X, y, random_state=42)
    # 5. 返回特征重要性字典
    importance_dict = {
        feature: float(score) 
        for feature, score in zip(available_features, mi_scores)
    }
    
    return importance_dict