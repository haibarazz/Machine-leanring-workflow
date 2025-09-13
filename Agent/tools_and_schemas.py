from typing import List
from pydantic import BaseModel, Field
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain_core.tools import tool

class TaskDefinition(BaseModel):
    """
    任务定义的结构化输出模型，对应 TaskState
    """
    target_col: str = Field(
        description="目标变量的列名，必须是数据集中存在的列名"
    )
    problem_type: str = Field(
        description="机器学习问题类型，必须是以下之一：classification, regression",
        enum=["classification", "regression"]
    )
    num_features: List[str] = Field(
        description="数值特征列表，包含所有数值型特征的列名"
    )
    cat_features: List[str] = Field(
        description="类别特征列表，包含所有类别型特征的列名"
    )
    evaluation_metric: List[str] = Field(
        description="评估指标列表，如 ['AUC', 'F1-score', 'MSE'] 等"
    )
class FeatureSelection(BaseModel):
    """
    特征选择的结构化输出模型
    """
    selected_features: List[str] = Field(
        description="筛选出的最重要的10个特征列表"
    )

class SearchQueryList(BaseModel):
    query: List[str] = Field(
        description="基于业务分析和重要特征生成的搜索查询列表"
    )



@tool
def search(query: str):
    """用于浏览网络进行搜索。"""
    # 可选的搜索工具: TavilySearchResults(max_results=3)
    search_tool = DuckDuckGoSearchResults(num_results=3) #这个配置根据需求来设置 output_format="list"
    return search_tool.invoke(query)