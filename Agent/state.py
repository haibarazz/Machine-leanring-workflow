from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import add_messages
from typing_extensions import Annotated
import operator
from dataclasses import dataclass, field
from typing_extensions import Annotated
import pandas as pd
import numpy as np

class SKWorkflowState(TypedDict):
    """
    工作流的全局核心状态，在所有节点间传递。
    它聚合了所有子任务所需的数据和产出。
    """
    # === 消息与控制流 ===
    # 用于记录 Agent 之间或与用户之间的对话，便于调试和观察
    messages: Annotated[List, add_messages]
    max_research_iterations: int
    current_iteration: int
    search_query: Annotated[list, operator.add]  # 特征衍生的时候提出的查询列表
    web_research_result: Annotated[list, operator.add]  # 特征衍生的查询结果
    data_path: str  # 数据路径
    task_description: str  # 初始任务描述路径
    initialization: Optional[Dict[str, Any]]  # 数据基础的信息，包括数据类型、缺失值等
    task_definition: Optional[Dict[str, Any]]  # 任务的定义。
    business_summary: Optional[str]  # 业务逻辑核心总结
    selected_features : Optional[list[str]]  # 特征选择的结果
    query_list: Optional[List[str]]  # 生成的查询列表



class TaskState(TypedDict):
    """
    分析出来任务
    """
    target_col: str
    problem_type: str
    num_features: List[str]  # 数值特征
    cat_features: List[str]  # 类别特征
    evaluation_metric: Annotated[list, operator.add] #可能有多个评估指标


class QueryGenerationState(TypedDict):
    query_list: list[str]


class WebSearchState(TypedDict):
    search_query: str
    id: str