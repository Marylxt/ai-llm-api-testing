"""
Agent多步任务测试：验证状态追踪Prompt对上下文保持的影响
对比：无状态追踪 vs 有状态追踪
"""
import json
import requests

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen2.5:7b"

# ========== 模拟工具 ==========
def search_product(keyword):
    return {"product_id": "P12345", "name": "MacBook", "price": 9999}

def add_to_cart(product_id):
    if not product_id:
        return {"error": "缺少product_id"}
    return {"cart_id": "C001", "status": "added"}

def place_order(cart_id):
    if not cart_id:
        return {"error": "缺少cart_id，无法下单"}
    return {"order_id": "O001", "status": "success"}

TOOLS = {
    "search_product": search_product,
    "add_to_cart": add_to_cart,
    "place_order": place_order,
}

TOOL_DESC = """
可用工具：
- search_product(keyword): 搜索商品，返回product_id
- add_to_cart(product_id): 加入购物车，返回cart_id
- place_order(cart_id): 下单，返回order_id
"""

# ========== 两种Prompt ==========
PROMPT_WITHOUT_STATE = """
你是一个电商测试Agent。
""" + TOOL_DESC + """

用户指令：帮我买一台MacBook

历史执行记录：
{history}

【执行规则】
- 如果历史为空，调用 search_product
- 如果历史里有 product_id 但没有 cart_id，调用 add_to_cart
- 如果历史里有 cart_id 但没有 order_id，调用 place_order
- 不要重复调用已经成功过的工具

【输出格式要求】
必须严格输出JSON，字段名必须是 tool 和 args，不要使用 action 或其他字段名。
示例：{"tool": "search_product", "args": {"keyword": "MacBook"}}
只输出JSON，不要其他内容。
"""

PROMPT_WITH_STATE = """
你是一个电商测试Agent。
""" + TOOL_DESC + """

用户指令：帮我买一台MacBook

历史执行记录：
{history}

【重要】必须先输出当前状态：
- 已完成步骤：
- 已获取的关键信息：
- 剩余步骤：

【执行规则】
- 如果历史为空，调用 search_product
- 如果历史里有 product_id 但没有 cart_id，调用 add_to_cart
- 如果历史里有 cart_id 但没有 order_id，调用 place_order
- 不要重复调用已经成功过的工具

【输出格式要求】
状态输出完后，必须严格输出JSON，字段名必须是 tool 和 args。
示例：{"tool": "search_product", "args": {"keyword": "MacBook"}}
"""


# ========== JSON解析（兼容版） ==========
def extract_json(text):
    """提取JSON：兼容markdown代码块包裹"""
    text = text.replace("```json", "").replace("```", "")
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None


def parse_tool_call(call):
    """兼容多种格式解析工具调用"""
    tool_name = call.get("tool") or call.get("action")

    if "args" in call and isinstance(call["args"], dict):
        args = call["args"]
    else:
        args = {k: v for k, v in call.items() if k not in ("tool", "action")}

    return tool_name, args


# ========== 执行单步 ==========
def execute_step(prompt, step_name):
    print(f"\n{'='*50}")
    print(f"【{step_name}】")
    print(f"{'='*50}")
    resp = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=180,
    )
    output = resp.json().get("response", "").strip()
    print(f"模型输出：\n{output}")

    json_str = extract_json(output)
    if not json_str:
        print("⚠️ 未找到JSON")
        return None, output

    try:
        call = json.loads(json_str)
        tool_name, args = parse_tool_call(call)
        print(f"\n调用工具：{tool_name}({args})")

        if tool_name in TOOLS:
            result = TOOLS[tool_name](**args)
            print(f"工具返回：{result}")
            return result, output
        else:
            print(f"❌ 未知工具：{tool_name}")
            return None, output
    except Exception as e:
        print(f"❌ 解析失败：{e}")
        return None, output


# ========== 测试流程 ==========
def run_test(prompt_template, name, icon):
    print("\n\n" + icon * 25)
    print(name)
    print(icon * 25)
    history = []
    step_names = ["步骤1：搜索商品", "步骤2：加入购物车", "步骤3：下单"]
    for step, step_name in enumerate(step_names):
        prompt = prompt_template.replace("{history}", json.dumps(history, ensure_ascii=False))
        result, _ = execute_step(prompt, step_name)
        history.append({"step": step + 1, "result": result})
        if result is None or "error" in str(result):
            print(f"⚠️ 步骤{step+1}失败，终止后续步骤")
            break
    return history


# ========== 主程序 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("Agent多步任务测试：状态追踪Prompt对比实验")
    print("=" * 60)

    history_without = run_test(PROMPT_WITHOUT_STATE, "测试1：无状态追踪Prompt", "🔴")
    history_with = run_test(PROMPT_WITH_STATE, "测试2：有状态追踪Prompt", "🟢")

    print("\n\n" + "=" * 60)
    print("📊 对比结论")
    print("=" * 60)
    print(f"无状态追踪：完成{len(history_without)}步")
    for h in history_without:
        print(f"  步骤{h['step']}：{h['result']}")
    print(f"\n有状态追踪：完成{len(history_with)}步")
    for h in history_with:
        print(f"  步骤{h['step']}：{h['result']}")