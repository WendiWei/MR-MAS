from openai import OpenAI

# 连接本地 Ollama
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
# api_key 必须要写，但随便写个字符串就行，Ollama 默认不会校验

# 构造消息
messages = [
    {"role": "user", "content": "请用中文介绍一下图灵奖。"}
]

# 调用本地模型
completion = client.chat.completions.create(
    model="gpt-oss:20b",   # Ollama 模型名
    messages=messages,
    stream=False
)

print(completion.choices[0].message.content)