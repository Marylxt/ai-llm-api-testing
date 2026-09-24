"""
Agent 轨迹评估：验证智能体是否按预期调用工具
"""
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from agentevals.trajectory.match import create_trajectory_match_evaluator


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。"""
    return f"{city}的天气是晴朗的"


# 创建 Agent
llm = ChatOllama(model="qwen2.5:7b", temperature=0)
agent = create_agent(model=llm, tools=[get_weather])


def test_weather_tool_called():
    """测试：Agent 是否调用了 get_weather 工具"""
    result = agent.invoke(
        {"messages": [HumanMessage(content="深圳天气怎么样？")]}
    )

    # 期望的轨迹：用户提问 → Agent 调用 get_weather → 工具返回 → Agent 总结
    reference_trajectory = [
        HumanMessage(content="深圳天气怎么样？"),
        AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "get_weather", "args": {"city": "深圳"}}],
        ),
        ToolMessage(content="深圳的天气是晴朗的", tool_call_id="call_1"),
        AIMessage(content="深圳的天气是晴朗的"),
    ]

    evaluator = create_trajectory_match_evaluator(trajectory_match_mode="strict")
    evaluation = evaluator(
        outputs=result["messages"],
        reference_outputs=reference_trajectory,
    )

    print(f"轨迹评估结果：{evaluation}")
    assert evaluation["score"] is True, "工具调用轨迹不匹配"