"""
RAG 输出质量评估（自研评估脚本）
不依赖 DeepEval/RAGAS，用规则+关键词匹配评估输出质量
"""
import re
import requests

# ========== 1. 配置 ==========
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
GEN_MODEL = "qwen2.5:7b"

# ========== 2. 知识库 ==========
KNOWLEDGE_BASE = [
    "数字产品购买后30天内可以申请退款，超过30天不予退款。",
    "退款流程：登录账号→我的订单→申请退款→填写理由→提交，1-3个工作日到账。",
    "密码重置：在登录页面点击'忘记密码'，输入注册邮箱后查收重置链接，链接有效期24小时。",
    "支付方式支持微信支付、支付宝和信用卡支付。订单超过500元可分期付款（仅限信用卡）。",
    "客服热线：400-800-8888，服务时间9:00-18:00。",
]


# ========== 3. 极简检索 ==========
def retrieve(query, top_k=2):
    query_chars = set(query)
    scored = []
    for doc in KNOWLEDGE_BASE:
        doc_chars = set(doc)
        overlap = len(query_chars & doc_chars) / max(len(query_chars | doc_chars), 1)
        scored.append((overlap, doc))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in scored[:top_k]]


# ========== 4. 生成答案 ==========
def generate_answer(question, contexts):
    context_text = "\n".join(contexts)
    prompt = f"""请根据以下资料回答问题。如果资料中没有相关信息，请直接回答"我不知道"。

资料：
{context_text}

问题：{question}
答案："""
    data = {"model": GEN_MODEL, "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_URL, json=data, timeout=120)
    return response.json().get("response", "").strip()


# ========== 5. 评估函数（核心） ==========
def extract_keywords(text):
    """提取关键实体：数字、专有名词、核心名词"""
    # 提取数字
    numbers = set(re.findall(r'\d+', text))
    # 提取核心关键词（简单规则：2字以上的中文词组）
    words = set(re.findall(r'[\u4e00-\u9fa5]{2,}', text))
    return numbers | words


def evaluate_faithfulness(answer, contexts):
    """
    忠实度评估：答案中的关键实体，是否能在检索上下文中找到
    返回 0~1 之间的分数
    """
    if not answer or answer == "我不知道":
        return 1.0  # 拒答视为忠实

    context_text = " ".join(contexts)
    context_keywords = extract_keywords(context_text)
    answer_keywords = extract_keywords(answer)

    if not answer_keywords:
        return 0.0

    hit = len(answer_keywords & context_keywords)
    score = hit / len(answer_keywords)
    return round(score, 2)


def evaluate_refusal(answer):
    """拒答正确性：知识库外的问题，模型是否诚实拒答"""
    refusal_keywords = ["我不知道", "没有找到", "无法回答", "抱歉", "没有相关"]
    return any(kw in answer for kw in refusal_keywords)


# ========== 6. 测试数据 ==========
TEST_CASES = [
    {"question": "退款政策是什么？", "expect_refusal": False},
    {"question": "怎么重置密码？", "expect_refusal": False},
    {"question": "支持哪些支付方式？", "expect_refusal": False},
    {"question": "客服电话是多少？", "expect_refusal": False},
    {"question": "明天北京天气怎么样？", "expect_refusal": True},  # 知识库外，期望拒答
]

# ========== 7. 执行评估 ==========
print("=" * 60)
print("RAG 系统输出质量评估")
print("=" * 60)

results = []
for case in TEST_CASES:
    q = case["question"]
    contexts = retrieve(q)
    answer = generate_answer(q, contexts)

    faith_score = evaluate_faithfulness(answer, contexts)
    refusal_ok = evaluate_refusal(answer)

    results.append({
        "question": q,
        "answer": answer,
        "faithfulness": faith_score,
        "refusal_ok": refusal_ok,
        "expect_refusal": case["expect_refusal"],
    })

    print(f"\n【问题】{q}")
    print(f"【回答】{answer[:80]}...")
    print(f"【忠实度】{faith_score}  |  【拒答正确】{refusal_ok}")

# ========== 8. 汇总报告 ==========
print("\n" + "=" * 60)
print("评估汇总")
print("=" * 60)

avg_faith = sum(r["faithfulness"] for r in results) / len(results)
refusal_pass = sum(1 for r in results if r["refusal_ok"] == r["expect_refusal"])
total = len(results)

print(f"平均忠实度：{avg_faith:.2f}")
print(f"拒答准确率：{refusal_pass}/{total} ({refusal_pass / total * 100:.0f}%)")
print(f"\n测试通过：{refusal_pass}/{total}")