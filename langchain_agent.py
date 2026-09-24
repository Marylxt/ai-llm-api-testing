"""
LangChain 智能体开发示例
使用本地 Ollama 模型 + 自定义工具，演示 Agent 的工具调用与多步执行
"""
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain.tools import tool


# ========== 1. 定义工具 ==========
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。"""
    # 模拟数据，实际项目中可替换为真实 API 调用
    weather_data = {
        "深圳": "28°C，多云",
        "广州": "30°C，晴",
        "北京": "22°C，晴",
    }
    return weather_data.get(city, f"{city}的天气数据暂不可用")


@tool
def search_product(keyword: str) -> str:
    """根据关键词搜索商品，返回商品ID。"""
    products = {
        "MacBook": "P12345",
        "iPhone": "P67890",
        "耳机": "P11111",
    }
    for name, pid in products.items():
        if name in keyword:
            return f"找到商品 {name}，product_id={pid}"
    return f"未找到与 '{keyword}' 相关的商品"


@tool
def add_to_cart(product_id: str) -> str:
    """将指定商品加入购物车。"""
    if not product_id:
        return "错误：缺少 product_id"
    return f"商品 {product_id} 已加入购物车，cart_id=C001"


# ========== 2. 创建智能体 ==========
llm = ChatOllama(model="qwen2.5:7b", temperature=0)

agent = create_agent(
    model=llm,
    tools=[get_weather, search_product, add_to_cart],
    system_prompt="你是一个乐于助人的助手。你可以调用工具来帮助用户。",
)


# ========== 3. 运行智能体 ==========
if __name__ == "__main__":
    # 测试1：单步工具调用
    print("=" * 50)
    print("测试1：查询天气")
    print("=" * 50)
    result1 = agent.invoke(
        {"messages": [{"role": "user", "content": "深圳今天天气怎么样？"}]}
    )
    print(f"最终回答：{result1['messages'][-1].content}")

    # 测试2：多步工具调用（Agent 自主规划）
    print("\n" + "=" * 50)
    print("测试2：搜索并加购商品")
    print("=" * 50)
    result2 = agent.invoke(
        {"messages": [{"role": "user", "content": "帮我搜索 MacBook 并加入购物车"}]}
    )
    print(f"最终回答：{result2['messages'][-1].content}")