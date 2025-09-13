import os
from pydantic import BaseModel, Field
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig


class Configuration(BaseModel):
    """这个是我们agent的各项控制参数配置类。"""

    # Fixed bug AttributeError: 'Configuration' object has no attribute 'reasoning_model'
    task_analysis_model: str = Field(
        default="moonshotai/Kimi-K2-Instruct",
        metadata={
            "description": "这个模型是用来做初始的这个任务定义解析"
        },
    )
    eda_analysis_model: str = Field(
        default="Qwen/Qwen2.5-72B-Instruct",
        metadata={
            "description": "这个模型是用来基于数据做初始的eda分析的"
        },
    )
    query_generator_model: str = Field(
        default="Qwen/Qwen2.5-72B-Instruct",
        metadata={"description": "用于生成搜索查询的模型"}
    )


    features_cross_generator_model: str = Field(
        default="Qwen/Qwen2.5-72B-Instruct",
        metadata={"description": "用于生成特征交叉建议的模型"}
    )

        # 特征衍生汇总模型
    feature_summary_model: str = Field(
        default="Qwen/Qwen2.5-72B-Instruct",
        metadata={"description": "用于汇总特征衍生策略的模型"}
    )

    number_of_initial_queries: int = Field(
        default=3,
        metadata={"description": "初始生成的搜索查询数量"}
    )
    number_of_one_queries: int = Field(
        default=3,
        metadata={"description": "对于每个搜寻结果，我们给出的具体的特征衍生操作方式"}
    )
    EDA_analysis: str = Field(
        default="F://python//机器学习工作流//Agent//result",
        metadata={
            "description": "这个是eda分析的输出路径"
        },
    )
    max_research_iterations: int = Field(
        default=3,
        metadata={"description": "我们进行特征衍生分析所进行的最大次数"},
    )


    # 工具函数模型的超参数配置
    rf_n_estimators: int = Field(default=100)
    rf_random_state: int = Field(default=42) 
    rf_n_jobs: int = Field(default=-1)


    @classmethod
    def from_runnable_config(
            cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)
