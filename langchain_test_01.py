from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

@tool
def get_weather(city:str) -> str:
    """获取指定城市的天气信息。"""
    return f"{city}的天气是晴朗的"

llm = ChatOllama(model="qwen2.5:7b",temperature=0)
agent = create_agent(model=llm,tools=[get_weather])

result = agent.invoke({"messages":[{"role": "user","content":"深圳天气怎么样？"}]})
print(result["messages"][-1].content)