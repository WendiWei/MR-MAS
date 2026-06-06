import openai
import time, tiktoken
from openai import OpenAI
import os, anthropic, json
import google.generativeai as genai

TOKENS_IN = dict()
def query_model(model_str, prompt, system_prompt, openai_api_key=None, gemini_api_key=None,  anthropic_api_key=None, tries=5, timeout=5.0, temp=None, print_cost=True, version="1.5"):
    preloaded_api = os.getenv('OPENAI_API_KEY')
    # sk-7KmzhCTvH5a8UqCC0JuCd8i3rrYBBH9imZXlNUl3mPAkSzbw
    if openai_api_key is None and preloaded_api is not None:
        openai_api_key = preloaded_api
    if openai_api_key is None and anthropic_api_key is None:
        raise Exception("No API key provided in query_model function")
    if openai_api_key is not None:
        openai.api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if anthropic_api_key is not None:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    if gemini_api_key is not None:
        os.environ["GEMINI_API_KEY"] = gemini_api_key
    for _ in range(tries):
        try:
            if model_str == "gpt-4o-mini" or model_str == "gpt4omini" or model_str == "gpt-4omini" or model_str == "gpt4o-mini":
                model_str = "gpt-4o-mini"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    if temp is None:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages
                        )
                    else:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages, temperature=temp
                        )
                else:
                    client = OpenAI(base_url="https://api.claudeshop.top/v1")
                    if temp is None:
                        completion = client.chat.completions.create(
                            model="gpt-4o-mini-2024-07-18", messages=messages, )
                    else:
                        completion = client.chat.completions.create(
                            model="gpt-4o-mini-2024-07-18", messages=messages, temperature=temp)
                answer = completion.choices[0].message.content

            elif model_str == "gemini-2.0-pro":
                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel(model_name="gemini-2.0-pro-exp-02-05", system_instruction=system_prompt)
                answer = model.generate_content(prompt).text
            elif model_str == "gemini-1.5-pro":
                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel(model_name="gemini-1.5-pro", system_instruction=system_prompt)
                answer = model.generate_content(prompt).text
            elif model_str == "o3-mini":
                model_str = "o3-mini"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  messages=messages)
                else:
                    client = OpenAI(base_url="https://api.claudeshop.top/v1")
                    completion = client.chat.completions.create(
                        model="o3-mini-2025-01-31", messages=messages)
                answer = completion.choices[0].message.content

            elif model_str == "claude-3.5-sonnet":
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                message = client.messages.create(
                    model="claude-3-5-sonnet-latest",
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}])
                answer = json.loads(message.to_json())["content"][0]["text"]
            elif model_str == "claude-3.7-sonnet":
                # model_str = "claude-3.7-sonnet"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    raise Exception("Please upgrade your OpenAI version to use DeepSeek client")
                else:
                    deepseek_client = OpenAI(
                        api_key=os.getenv('CLAUDE_API_KEY'),
                        base_url="https://api.claudeshop.top/v1"
                    )
                    if temp is None:
                        completion = deepseek_client.chat.completions.create(
                            model="claude-3.7-sonnet-20250219",
                            messages=messages)
                    else:
                        completion = deepseek_client.chat.completions.create(
                            model="claude-3.7-sonnet-20250219",
                            messages=messages,
                            temperature=temp)
                answer = completion.choices[0].message.content

            elif model_str == "gpt4o" or model_str == "gpt-4o":
                model_str = "gpt-4o"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    if temp is None:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages
                        )
                    else:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages, temperature=temp)
                else:
                    # client = OpenAI(base_url="https://api.claudeshop.top/v1")
                    client = OpenAI(base_url="https://api.ysaikeji.cn/v1")
                    if temp is None:
                        completion = client.chat.completions.create(
                            model="gpt-4o", messages=messages, )
                    else:
                        completion = client.chat.completions.create(
                            model="gpt-4o", messages=messages, temperature=temp)
                answer = completion.choices[0].message.content

            elif model_str == "gpt-5.4" or model_str == "gpt5.4":
                model_str = "gpt-5.4"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    if temp is None:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages
                        )
                    else:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages, temperature=temp)
                else:
                    # client = OpenAI(base_url="https://api.claudeshop.top/v1")
                    client = OpenAI(base_url="https://api.ysaikeji.cn/v1")

                    if temp is None:
                        completion = client.chat.completions.create(
                            model="gpt-5.4", messages=messages, )
                    else:
                        completion = client.chat.completions.create(
                            model="gpt-5.4", messages=messages, temperature=temp)
                answer = completion.choices[0].message.content

            elif model_str == "gpt-3.5-turbo":
                model_str = "gpt-3.5-turbo"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    if temp is None:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages
                        )
                    else:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages, temperature=temp)
                else:
                    client = OpenAI(base_url="https://api.claudeshop.top/v1")
                    if temp is None:
                        completion = client.chat.completions.create(
                            model="gpt-3.5-turbo", messages=messages, )
                    else:
                        completion = client.chat.completions.create(
                            model="gpt-3.5-turbo", messages=messages, temperature=temp)
                answer = completion.choices[0].message.content

            elif model_str == "deepseek-chat":
                model_str = "deepseek-chat"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    raise Exception("Please upgrade your OpenAI version to use DeepSeek client")
                else:
                    deepseek_client = OpenAI(
                        api_key=os.getenv('DEEPSEEK_API_KEY'),
                        base_url="https://api.deepseek.com/v1"
                    )
                    if temp is None:
                        completion = deepseek_client.chat.completions.create(
                            model="deepseek-chat",
                            messages=messages)
                    else:
                        completion = deepseek_client.chat.completions.create(
                            model="deepseek-chat",
                            messages=messages,
                            temperature=temp)
                answer = completion.choices[0].message.content

            elif model_str == "deepseek-chat-v3":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    raise Exception("Please upgrade your OpenAI version to use DeepSeek client")
                else:
                    deepseek_client = OpenAI(
                        api_key=os.getenv('QWEN_API_KEY'),
                        base_url="https://api.claudeshop.top/v1"
                    )
                    if temp is None:
                        completion = deepseek_client.chat.completions.create(
                            model="deepseek-chat-v3",
                            messages=messages)
                    else:
                        completion = deepseek_client.chat.completions.create(
                            model="deepseek-chat-v3",
                            messages=messages,
                            temperature=temp)
                answer = completion.choices[0].message.content

            elif model_str == "o1-mini":
                model_str = "o1-mini"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI(base_url="https://api.ysaikeji.cn/v1")
                    completion = client.chat.completions.create(
                        model="o1-mini", messages=messages)
                answer = completion.choices[0].message.content
            elif model_str == "o1":
                model_str = "o1"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model="o1-2024-12-17",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI(base_url="https://api.ysaikeji.cn/v1")
                    completion = client.chat.completions.create(
                        model="o1-2024-12-17", messages=messages)
                answer = completion.choices[0].message.content
            elif model_str == "o1-preview":
                model_str = "o1-preview"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI(base_url="https://api.claudeshop.top/v1")
                    completion = client.chat.completions.create(
                        model="o1-preview", messages=messages)
                answer = completion.choices[0].message.content

            elif model_str == "gpt-oss:20b":  # <<< 新增 gpt-oss 本地模型支持
                # ---------------------- 本地 Ollama ----------------------
                ollama_client = OpenAI(
                    base_url="http://localhost:11434/v1",  # Ollama 默认 API 地址
                    api_key="ollama"  # <<< 占位，不校验
                )
                completion = ollama_client.chat.completions.create(
                    model="gpt-oss:20b",  # <<< 这里固定为 gpt-oss:20b
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temp,
                )
                answer = completion.choices[0].message.content

            elif model_str == "qwen3-235b":
                model_str = "qwen3-235b"
                client = OpenAI(
                    api_key=os.getenv('OPENAI_API_KEY'),
                    base_url="https://api.claudeshop.com/v1"
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]

                completion = client.chat.completions.create(
                        model="qwen3-235b-a22b", messages=messages)
                answer = completion.choices[0].message.content

            try:
                if model_str in ["o1-preview", "o1-mini", "claude-3.5-sonnet", "o1", "o3-mini"]:
                    encoding = tiktoken.encoding_for_model("gpt-4o")
                elif model_str in ["gpt-4o","deepseek-chat", "gpt-oss:20b","gpt-5-mini","deepseek-chat-r1"]:
                    encoding = tiktoken.get_encoding("cl100k_base")
                else:
                    encoding = tiktoken.encoding_for_model(model_str)

                if model_str not in TOKENS_IN:
                    TOKENS_IN[model_str] = 0
                    TOKENS_OUT[model_str] = 0
                TOKENS_IN[model_str] += len(encoding.encode(system_prompt + prompt))
                TOKENS_OUT[model_str] += len(encoding.encode(answer))

                if print_cost:
                    print(f"Current experiment cost = ${curr_cost_est()}, ** Approximate values, may not reflect true cost")
                    print("\n=== Token Usage Summary ===")
                    print(f"{'Model':<20}{'Tokens In':<15}{'Tokens Out':<15}{'Total':<15}")
                    print("-" * 65)
                    for m in TOKENS_IN:
                        total = TOKENS_IN[m] + TOKENS_OUT[m]
                        print(f"{m:<20}{TOKENS_IN[m]:<15}{TOKENS_OUT[m]:<15}{total:<15}")
                    print("=" * 65 + "\n")

            except Exception as e:
                if print_cost:
                    print(f"Cost approximation has an error? {e}")
            time.sleep(timeout)
            return answer
        except Exception as e:
            print("Inference Exception:", e)
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")

TOKENS_OUT = dict()

encoding = tiktoken.get_encoding("cl100k_base")

def curr_cost_est():
    '''
    这是一个字典，记录了各模型的输入 token 成本，单位是 美元 / token。
    因为价格通常按“百万 token”为计价单位，所以这里都除以 1,000,000，表示单个 token 的价格。
    例如：
    GPT-4o 输入价格为 $2.50 / 1M tokens → 单 token 成本 = 0.0000025 USD
    GPT-4o-mini 输入价格为 $0.15 / 1M tokens → 单 token 成本 = 0.00000015 USD
    '''
    costmap_in = {
        "gpt-4o": 2.50 / 1000000,
        "gpt-5": 1.25 / 1000000,
        "gpt-3.5-turbo": 0.5 / 1000000,
        "gpt-4o-mini": 0.150 / 1000000,
        "o1-preview": 15.00 / 1000000,
        "o1-mini": 3.00 / 1000000,
        "claude-3-5-sonnet": 3.00 / 1000000,
        "claude-3-7-sonnet": 3.00 / 1000000,
        "deepseek-chat-chat": 1.00 / 1000000,
        "deepseek-chat-r1": 0.165 / 1000000,
        "deepseek-chat-v3": 0.081 / 1000000,
        "o1": 15.00 / 1000000,
        "o3-mini": 1.10 / 1000000,
    }
    costmap_out = {
        "gpt-4o": 10.00/ 1000000,
        "gpt-5": 10 / 1000000,
        "gpt-3.5-turbo": 1.0 / 1000000,
        "gpt-4o-mini": 0.6 / 1000000,
        "o1-preview": 60.00 / 1000000,
        "o1-mini": 12.00 / 1000000,
        "claude-3-5-sonnet": 12.00 / 1000000,
        "claude-3-7-sonnet": 15.00 / 1000000,
        "deepseek-chat-chat": 5.00 / 1000000,
        "deepseek-chat-r1": 0.66 / 1000000,
        "deepseek-chat-v1": 0.324 / 1000000,
        "o1": 60.00 / 1000000,
        "o3-mini": 4.40 / 1000000,
    }
    return sum([costmap_in[_]*TOKENS_IN[_] for _ in TOKENS_IN]) + sum([costmap_out[_]*TOKENS_OUT[_] for _ in TOKENS_OUT])


