"""
Agent 任务完成度评估（自研规则评估）
不依赖 DeepEval，直接检查回答是否包含关键信息
"""
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain.tools import tool


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。"""
    weather_data = {"深圳": "28°C，多云", "广州": "30°C，晴"}
    return weather_data.get(city, f"{city}天气暂不可用")


llm = ChatOllama(model="qwen2.5:7b", temperature=0)
agent = create_agent(model=llm, tools=[get_weather], system_prompt="你是一个乐于助人的助手。")


def evaluate_task_completion(answer: str, expected_keywords: list) -> dict:
    """简单规则评估：检查回答是否包含所有预期关键词"""
    hit = [kw for kw in expected_keywords if kw in answer]
    score = len(hit) / len(expected_keywords)
    return {"score": round(score, 2), "hit": hit, "miss": [k for k in expected_keywords if k not in hit]}


if __name__ == "__main__":
    result = agent.invoke({"messages": [{"role": "user", "content": "深圳天气怎么样？"}]})
    answer = result["messages"][-1].content
    print(f"Agent 回答：{answer}")
    print("-" * 40)

    # 预期回答必须包含温度和多云
    evaluation = evaluate_task_completion(answer, ["28", "多云"])
    print(f"任务完成度评分：{evaluation['score']}")
    print(f"命中关键词：{evaluation['hit']}")
    print(f"遗漏关键词：{evaluation['miss']}")

    assert evaluation["score"] >= 0.7, "任务完成度不达标"
    print("✅ 评估通过")