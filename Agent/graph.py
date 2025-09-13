import os
from langchain_openai import ChatOpenAI
from langchain_experimental.tools import PythonREPLTool
from .tools_and_schemas import TaskDefinition, FeatureSelection,SearchQueryList
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import Tool

from .state import (
    SKWorkflowState,
    TaskState,
    QueryGenerationState,
    WebSearchState
)
from .configuration import Configuration
from .prompts import (
    Task_Definitions_prompts,
    EDA_prompts,
    query_writer_instructions,
    web_searcher_instructions,
    feature_summary_instructions,  # 新增

)
from langchain_google_genai import ChatGoogleGenerativeAI
from .utils import (
    get_research_topic
)
from .data_pre_fun import   (
    load_data,
    feature_importance_mutual_info,
    feature_importance_random_forest

)
# 定义工具列表
tools_FI = [feature_importance_mutual_info, feature_importance_random_forest]
load_dotenv()

if os.getenv("API_KEY") is None:
    raise ValueError("API_KEY is not set")


def load_analysis(state: SKWorkflowState, config: RunnableConfig) -> dict:
    """
    在不动用任何业务知识的情况下，对原始数据进行一次纯粹的技术性扫描，形成一份“数据健康报告”。
    """
    data_path = state["data_path"]
    data_info = load_data(data_path)
    return {"initialization": data_info}

def business_summary_analysis(state: SKWorkflowState, config: RunnableConfig) -> dict:
    """
    对任务描述进行核心业务逻辑总结，压缩token但保留关键信息。
    """
    configurable = Configuration.from_runnable_config(config)
    
    task_description_path = state["task_description"]
    with open(task_description_path, 'r', encoding='utf-8') as f:
        task_description_content = f.read()
    
    llm = ChatOpenAI(
        base_url=os.getenv("API_URL"),
        api_key=os.getenv("API_KEY"),
        temperature=0.3,
        model_name=configurable.eda_analysis_model
    )
    summary_prompt = f"""
    请将以下任务描述总结为核心业务逻辑要点（300字以内）：
    原始描述：{task_description_content}
    要求：
    1. 保留关键业务场景和目标
    2. 提取潜在的业务逻辑规律
    3. 压缩表述但不丢失核心信息
    """
    result = llm.invoke(summary_prompt)
    # 保存业务总结到配置路径
    summary_path = os.path.join(configurable.EDA_analysis, "business_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(result.content)
    return {"business_summary": summary_path}

def task_definition_analysis(state: SKWorkflowState, config: RunnableConfig) -> TaskState:
    """
    让 Agent“读懂”自然语言的任务要求，将其转换为机器可以理解的结构化指令。
    提取出目标变量，知道分析出这个这个任务类型是什么，明确自己的评估指标。
    
    Args:
        state: 包含任务描述和数据初始化信息的工作流状态
        config: 配置信息，包含LLM模型设置
        
    Returns:
        包含 task_definition 的字典，其中 task_definition 符合 TaskState 结构
    """
    configurable = Configuration.from_runnable_config(config)
    
    task_description_path = state["task_description"]
    with open(task_description_path, 'r', encoding='utf-8') as f:
        task_description_content = f.read()
    dataset_info = state["initialization"]
    
    llm = ChatOpenAI(
        base_url=os.getenv("API_URL"),
        api_key=os.getenv("API_KEY"),	# app_key
        temperature=0.7,
        model_name=configurable.task_analysis_model,	# 模型名称
    )
    structured_llm = llm.with_structured_output(TaskDefinition)
    # 格式化提示词
    formatted_prompt = Task_Definitions_prompts.format(
        task_description_content=task_description_content,
        dataset_info=dataset_info
    )
    # 调用LLM获取结构化输出
    result = structured_llm.invoke(formatted_prompt)
    # 验证一下这个特征列表的完整性 
    all_columns = set(dataset_info["columns"])
    target_col = result.target_col
    predicted_features = set(result.num_features + result.cat_features)
    if target_col in all_columns:
        expected_features = all_columns - {target_col}
    else:
        expected_features = all_columns  # 如果目标列不在数据中，使用所有列
    # 如果特征数量不匹配，自动补全缺失的特征到数值特征列表
    missing_features = expected_features - predicted_features
    if missing_features:
        result.cat_features.extend(list(missing_features))
    # 将 Pydantic 模型转换为字典
    task_definition = {
        "target_col": result.target_col,
        "problem_type": result.problem_type,
        "num_features": result.num_features,
        "cat_features": result.cat_features,
        "evaluation_metric": result.evaluation_metric,
    }
    
    return {'task_definition': task_definition}


def eda_analysis(state: SKWorkflowState, config: RunnableConfig) -> dict:
    """
    基于随机森林特征重要性和业务逻辑，选择最具潜力的10个特征。
    """
    configurable = Configuration.from_runnable_config(config)
    
    # 获取随机森林特征重要性
    feature_importance = feature_importance_random_forest(
        state["data_path"],
        state["task_definition"]["cat_features"],
        state["task_definition"]["num_features"],
        state["task_definition"]["target_col"],
        state["task_definition"]["problem_type"],
        configurable.rf_n_estimators,
        configurable.rf_random_state,
        configurable.rf_n_jobs
    )
    
    missing_info = state["initialization"]['missing_info']
    
    # 读取业务总结文件内容
    business_summary_path = state["business_summary"]
    with open(business_summary_path, 'r', encoding='utf-8') as f:
        business_summary = f.read()
    
    llm = ChatOpenAI(
        base_url=os.getenv("API_URL"),
        api_key=os.getenv("API_KEY"),
        temperature=0.5,
        model=configurable.eda_analysis_model,
    )
    
    # 使用结构化输出
    structured_llm = llm.with_structured_output(FeatureSelection)
    
    formatted_prompt = EDA_prompts.format(
        feature_importance=feature_importance,
        missing_info=missing_info,
        business_summary=business_summary,
        target_col=state["task_definition"]["target_col"],
        problem_type=state["task_definition"]["problem_type"]
    )
    
    result = structured_llm.invoke(formatted_prompt)
    
    return {"selected_features": result.selected_features}


def generate_query(state: SKWorkflowState, config: RunnableConfig) -> QueryGenerationState:
    """
    基于业务总结和重要特征生成搜索查询，用于后续特征工程研究。
    """
    configurable = Configuration.from_runnable_config(config)
    
    # 读取业务总结
    business_summary_path = state["business_summary"]
    with open(business_summary_path, 'r', encoding='utf-8') as f:
        business_summary = f.read()
    
    llm = ChatOpenAI(
        base_url=os.getenv("API_URL"),
        api_key=os.getenv("API_KEY"),
        temperature=0.7,
        model=configurable.query_generator_model,
    )
    
    structured_llm = llm.with_structured_output(SearchQueryList)
    
    formatted_prompt = query_writer_instructions.format(
        business_summary=business_summary,
        selected_features=state["selected_features"],
        target_col=state["task_definition"]["target_col"],
        problem_type=state["task_definition"]["problem_type"],
        number_queries=configurable.number_of_initial_queries,
    )
    
    result = structured_llm.invoke(formatted_prompt)
    
    return {"query_list": result.query}

def continue_to_web_research(state: QueryGenerationState):
    # 任务分发中心，分给 web_research 来具体执行
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["query_list"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> dict:
    """执行网络搜索并总结特征工程策略。"""
    from .tools_and_schemas import search
    configurable = Configuration.from_runnable_config(config)
    # 执行搜索
    search_results = search.invoke(state["search_query"])
    # 使用LLM总结搜索结果
    llm = ChatOpenAI(
        base_url=os.getenv("API_URL"),
        api_key=os.getenv("API_KEY"),
        temperature=0.3,
        model=configurable.features_cross_generator_model,
    )
    
    formatted_prompt = web_searcher_instructions.format(
        research_topic=search_results,
        one_query_limit = configurable.number_of_one_queries
    )
    
    # 添加搜索结果到提示词
    full_prompt = f"{formatted_prompt}\n\n**搜索结果:**\n{search_results}\n\n请基于以上搜索结果进行分析总结。"
    
    result = llm.invoke(full_prompt)
    
    return {
        "search_query": [state["search_query"]],
        "web_research_result": [result.content],
    }


def feature_strategy_summary(state: SKWorkflowState, config: RunnableConfig) -> dict:
    """
    汇总所有web_research结果，生成最终的特征衍生策略报告。
    """
    configurable = Configuration.from_runnable_config(config)
    
    # 读取业务总结
    business_summary_path = state["business_summary"]
    with open(business_summary_path, 'r', encoding='utf-8') as f:
        business_summary = f.read()
    
    # 汇总所有搜索结果
    all_research_results = "\n\n".join([
        f"搜索查询: {query}\n结果: {result}" 
        for query, result in zip(state["search_query"], state["web_research_result"])
    ])
    
    llm = ChatOpenAI(
        base_url=os.getenv("API_URL"),
        api_key=os.getenv("API_KEY"),
        temperature=0.3,
        model=configurable.feature_summary_model,
    )
    
    formatted_prompt = feature_summary_instructions.format(
        business_summary=business_summary,
        selected_features=state["selected_features"],
        target_col=state["task_definition"]["target_col"],
        problem_type=state["task_definition"]["problem_type"],
        all_research_results=all_research_results
    )
    
    result = llm.invoke(formatted_prompt)
    
    # 保存汇总报告
    report_path = os.path.join(configurable.EDA_analysis, "feature_strategy_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(result.content)
    
    return {"feature_strategy_report": report_path}
# # Create our Agent Graph
# builder = StateGraph(OverallState, config_schema=Configuration)

# # Define the nodes we will cycle between
# builder.add_node("generate_query", generate_query)
# builder.add_node("web_research", web_research)
# builder.add_node("reflection", reflection)
# builder.add_node("finalize_answer", finalize_answer)

# # Set the entrypoint as `generate_query`
# # This means that this node is the first one called
# builder.add_edge(START, "generate_query")
# # Add conditional edge to continue with search queries in a parallel branch
# builder.add_conditional_edges(
#     "generate_query", continue_to_web_research, ["web_research"]
# )
# # Reflect on the web research
# builder.add_edge("web_research", "reflection")
# # Evaluate the research
# builder.add_conditional_edges(
#     "reflection", evaluate_research, ["web_research", "finalize_answer"]
# )
# # Finalize the answer
# builder.add_edge("finalize_answer", END)

# graph = builder.compile(name="pro-search-agent")
