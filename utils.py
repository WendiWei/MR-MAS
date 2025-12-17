import glob
import os, re
import shutil
import tempfile
import time
import tiktoken, openai
import subprocess, string
from openai import OpenAI
import google.generativeai as genai
from huggingface_hub import InferenceClient


def query_deepseekv3(prompt, system, api_key, attempt=0, temperature=0.0):
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat-chat",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            stream=False, temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Query qwen error: {e}")
        if attempt >= 10: return f"Your attempt to query deepseekv3 failed: {e}"
        return query_deepseekv3(prompt, system, attempt+1)


def query_qwen(prompt, system, api_key, attempt=0, temperature=0.0):
    try:
        client = InferenceClient(api_key=api_key)
        if system is not None:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}]
        else:
            messages = [
                {"role": "user", "content": prompt}]

        completion = client.chat.completions.create(
            model="Qwen/QwQ-32B",
            messages=messages,
            max_tokens=500,
            temperature=temperature
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Query qwen error: {e}")
        if attempt >= 10: return f"Your attempt to inference gemini failed: {e}"
        return query_qwen(prompt, system, attempt+1)


def query_gpt4omini(prompt, system, api_key, attempt=0, temperature=0.0):
    try:
        openai_api_key = api_key
        openai.api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        if system is not None:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}]
        else:
            messages = [
                {"role": "user", "content": prompt}]
        client = OpenAI(base_url="https://api.claudeshop.top/v1")
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, temperature=temperature).choices[0].message.content.strip()
        return response
    except Exception as e:
        print(f"Query 4o-mini error: {e}")
        if attempt >= 10: return f"Your attempt to inference gemini failed: {e}"
        return query_gpt4omini(prompt, system, attempt+1)



def query_gpt4o(prompt, system, api_key, attempt=0, temperature=0.0):
    try:
        openai_api_key = api_key
        openai.api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        if system is not None:
            messages = [
                {"role": "user", "content":system + prompt}]
        else:
            messages = [
                {"role": "user", "content": prompt}]
        client = OpenAI(base_url="https://api.claudeshop.top/v1")
        response = client.chat.completions.create(
            model="gpt-4o", messages=messages, temperature=temperature).choices[0].message.content.strip()
        return response
    except Exception as e:
        print(f"Query gpr-4o error: {e}")
        if attempt >= 10: return f"Your attempt to inference gemini failed: {e}"
        return query_gpt4o(prompt, system, attempt+1)



def query_gemini(prompt, system, api_key, attempt=0, temperature=0.0):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="gemini-1.5-pro", system_instruction=system)
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=temperature)).text.strip()
        time.sleep(1)
        return response
    except Exception as e:
        print(f"Gemini error: {e}")
        if attempt >= 10: return f"Your attempt to inference gemini failed: {e}"
        time.sleep(1)
        return query_gemini(prompt, system, attempt+1)



def query_gemini2p0(prompt, system, api_key, attempt=0, temperature=0.0,):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="gemini-2.0-flash", system_instruction=system)
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=temperature)).text.strip()
        time.sleep(1)
        return response
    except Exception as e:
        print(f"Gemini error: {e}")
        if attempt >= 10: return f"Your attempt to inference gemini failed: {e}"
        time.sleep(1)
        return query_gemini2p0(prompt, system, attempt+1)


def compile_latex(latex_code, output_path, compile=True, timeout=30):
    # 将原始代码中的 \documentclass{article} 替换为一个包含大量 常用 LaTeX 宏包 的版本（数学公式、算法环境、图表、颜色、链接等）。
    # 相当于在模板中强制引入一系列工具包，以确保文档能支持常见功能。
    # print(f"生成的内容为：{latex_code}")
    latex_code = latex_code.replace(
        r"\documentclass{article}",
        "\\documentclass{article}\n\\usepackage{amsmath}\n\\usepackage{amssymb}\n\\usepackage{array}\n\\usepackage{algorithm}\n\\usepackage{algorithmicx}\n\\usepackage{algpseudocode}\n\\usepackage{booktabs}\n\\usepackage{colortbl}\n\\usepackage{color}\n\\usepackage{enumitem}\n\\usepackage{fontawesome5}\n\\usepackage{float}\n\\usepackage{graphicx}\n\\usepackage{hyperref}\n\\usepackage{listings}\n\\usepackage{makecell}\n\\usepackage{multicol}\n\\usepackage{multirow}\n\\usepackage{pgffor}\n\\usepackage{pifont}\n\\usepackage{soul}\n\\usepackage{sidecap}\n\\usepackage{subcaption}\n\\usepackage{titletoc}\n\\usepackage[symbol]{footmisc}\n\\usepackage{url}\n\\usepackage{wrapfig}\n\\usepackage{xcolor}\n\\usepackage{xspace}")


    # Step 1: 在临时目录先验证 LaTeX 能否编译（不生成 PDF）
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_tex = os.path.join(tmpdir, "temp.tex")
        with open(temp_tex, "w", encoding="utf-8") as f:
            f.write(latex_code)

        try:
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "temp.tex"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                cwd=tmpdir
            )

        except subprocess.TimeoutExpired:
            msg = f"[CODE EXECUTION ERROR]: Compilation timed out after {timeout} seconds (validation step)"
            print(msg)
            return msg

        except subprocess.CalledProcessError as e:
            # 解码输出
            output = e.stdout.decode("utf-8", errors="ignore").splitlines()
            error_lines = []

            # 提取关键行
            for line in output:
                line = line.strip()
                if "File `" in line and "not found" in line:
                    error_lines.append(line)
                elif "undefined" in line and ("Reference" in line or "Citation" in line):
                    error_lines.append(line)
                elif "Overfull" in line or "Underfull" in line:
                    error_lines.append(line)
                elif line.startswith("!"):
                    error_lines.append(line)

            if not error_lines:
                # 如果没有捕获到关键行，返回前5行
                error_lines = output[:5]

            msg = "[CODE EXECUTION ERROR]: LaTeX validation failed, important info:\n" + "\n".join(error_lines)
            print(msg)
            return msg

    # Step 2: 验证通过后再写入目标目录
    print("该部分latex内容编译成功")
    dir_path = f"{output_path}"
    tex_file_path = os.path.join(dir_path, "temp.tex")
    # Write the LaTeX code to the .tex file in the specified directory
    with open(tex_file_path, "w",encoding="utf-8") as f:
        f.write(latex_code)
    print(f"保存的内容是：{latex_code}")
    if not compile:
        return f"Compilation successful"

    # Compiling the LaTeX code using pdflatex with non-interactive mode and timeout
    # 调用 pdflatex 编译生成 PDF
    # try:
    #     result = subprocess.run(
    #         ["pdflatex", "-interaction=nonstopmode", "temp.tex"],
    #         check=True,                   # Raises a CalledProcessError on non-zero exit codes
    #         stdout=subprocess.PIPE,        # Capture standard output
    #         stderr=subprocess.PIPE,        # Capture standard error
    #         timeout=timeout,               # Timeout for the process
    #         cwd=dir_path
    #     )
    #
    #     # If compilation is successful, return the success message
    #     print("pdflatex编译成功")
    #     return f"Compilation successful: {result.stdout.decode('utf-8')}"
    #
    # except subprocess.TimeoutExpired:
    #     # If the compilation takes too long, return a timeout message
    #     return "[CODE EXECUTION ERROR]: Compilation timed out after {} seconds".format(timeout)
    # except subprocess.CalledProcessError as e:
    #     # If there is an error during LaTeX compilation, return the error message
    #     return f"[CODE EXECUTION ERROR]: Compilation failed. There was an error in your latex."

def run_pdflatex(tex_file_path, timeout=30):
    """
    调用 pdflatex 编译单个 tex 文件,生成 PDF
    """
    dir_path = os.path.dirname(tex_file_path)
    try:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", os.path.basename(tex_file_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            cwd=dir_path
        )
        # 清理中间文件，但保留 PDF
        for ext in ["aux", "log", "out", "toc"]:
            for file in glob.glob(os.path.join(dir_path, f"*.{ext}")):
                os.remove(file)

        print(f" {os.path.basename(tex_file_path)} 编译成功")
        return f"Compilation successful: {result.stdout.decode('utf-8')}"
    except subprocess.TimeoutExpired:
        return f"[ERROR] Compilation timed out after {timeout} seconds: {tex_file_path}"
    except subprocess.CalledProcessError:
        return f"[ERROR] Compilation failed: {tex_file_path}"


# def sanitize_latex(text):
#     # 环境补全
#     envs = ["abstrate","itemize", "figure", "table"]
#     for env in envs:
#         if text.count(f"\\begin{{{env}}}") > text.count(f"\\end{{{env}}}"):
#             text += f"\n\\end{{{env}}}\n"
#
#     # 特殊符号转义
#     text = re.sub(r"(?<!\\)%", r"\\%", text)
#     text = re.sub(r"(?<!\\)_", r"\\_", text)
#     text = re.sub(r"(?<!\\)&", r"\\&", text)
#
#     # Unicode 特殊空格和不可见字符 → 普通空格
#     # 包括 U+202F（窄不间断空格）、U+200B（零宽空格）、U+00A0（不间断空格）
#     text = re.sub(r"[\u202F\u200B\u00A0\u2009]", " ", text)
#     # 数学符号 Unicode → LaTeX
#     symbol_map = {
#         "β": r"$\beta$", "α": r"$\alpha$", "γ": r"$\gamma$",
#         "≥": r"$\geq$", "≤": r"$\leq$", "±": r"$\pm$",
#         "°": r"$^\circ$", "×": r"$\times$", "∞": r"$\infty$",
#     }
#     for uni, latex in symbol_map.items():
#         text = text.replace(uni, latex)
#
#     # 自动把裸露的数j学表达式补进公式模式
#     text = re.sub(r"(?<!\$)(\d+\s*[<>=]\s*\d+)(?!\$)", r"$\1$", text)
#
#     return text

def sanitize_latex(text):
    """
    增强版 LaTeX 兜底函数：
    - 自动闭合未闭合环境（支持嵌套）
    - 智能转义特殊符号（% _ &，公式内不转义 &）
    - 替换 Unicode 数学符号为 LaTeX
    - 自动将裸露数学表达式补进 $...$
    - 处理特殊空格和不可见字符
    - 额外转义未知 LaTeX 控制序列，避免 Undefined control sequence 错误
    """

    import re

    # -----------------------------
    # 1 处理环境闭合，使用堆栈跟踪嵌套
    # -----------------------------
    env_stack = []
    pattern_begin = re.compile(r"\\begin\{(\w+)\}")
    pattern_end = re.compile(r"\\end\{(\w+)\}")

    lines = text.split("\n")
    for line in lines:
        # 检查 begin
        for m in pattern_begin.finditer(line):
            env_stack.append(m.group(1))
        # 检查 end
        for m in pattern_end.finditer(line):
            if m.group(1) in env_stack:
                env_stack.remove(m.group(1))

    # 自动闭合未闭合环境
    for env in reversed(env_stack):
        lines.append(f"\\end{{{env}}}")

    text = "\n".join(lines)

    # -----------------------------
    # 2 转义特殊符号 % _，除了已转义的
    # -----------------------------
    text = re.sub(r"(?<!\\)%", r"\\%", text)
    text = re.sub(r"(?<!\\)_", r"\\_", text)

    # -----------------------------
    # 3 智能处理 &
    # -----------------------------
    def replace_ampersand(match):
        start = match.start()
        before = text[:start]
        in_env = False
        for env in ["align", "tabular", "matrix"]:
            if before.count(f"\\begin{{{env}}}") > before.count(f"\\end{{{env}}}"):
                in_env = True
                break
        return "&" if in_env else "\\&"

    text = re.sub(r"(?<!\\)&", replace_ampersand, text)

    # -----------------------------
    # 4 替换 Unicode 空格和不可见字符
    # -----------------------------
    # text = re.sub(r"[\u202F\u200B\u00A0\u2009]", " ", text)
    text = re.sub(r"[\u2000-\u200B\u202F\u00A0]", " ", text)

    # -----------------------------
    # 5 替换数学符号
    # -----------------------------
    symbol_map = {
        "β": r"$\beta$", "α": r"$\alpha$", "γ": r"$\gamma$",
        "≥": r"$\geq$", "≤": r"$\leq$", "±": r"$\pm$",
        "°": r"$^\circ$", "×": r"$\times$", "∞": r"$\infty$",
        "⁻": r"$^{-}$",  # 避免 U+207B 造成 undefined control sequence
        "⁺": r"$^{+}$",  # 额外常见上标
    }

    for uni, latex_sym in symbol_map.items():
        text = text.replace(uni, latex_sym)

    # -----------------------------
    # 6 自动把裸数学表达式补进公式模式
    # -----------------------------
    text = re.sub(r"(?<!\$)(\d+\s*[<>=]\s*\d+)(?!\$)", r"$\1$", text)

    # -----------------------------
    # 7 转义潜在未知控制序列
    # -----------------------------
    text = re.sub(r"\\([A-Za-z]+)(?![A-Za-z*])", r"\\textbackslash{\1}", text)

    # 删除文本中的REPLACE
    text = re.sub(r"REPLACE", "", text)
    text = re.sub(r"```", "", text)

    # 移除 Markdown 或其他不必要的符号
    text = re.sub(r"\*\*(.*?)\*\*", r"\\textbf{\1}", text)  # 替换加粗的 Markdown
    # 处理奇怪的符号问题（去除非标准空格、标点等）
    text = re.sub(r"[^\x00-\x7F]+", "", text)  # 移除非 ASCII 字符
    text = re.sub(r"\s+", " ", text)  # 替换多余的空格

    return text


def count_tokens(messages, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    num_tokens = sum([len(enc.encode(message["content"])) for message in messages])
    return num_tokens

def remove_figures():
    """Remove a directory if it exists."""
    for _file in os.listdir("."):
        if "Figure_" in _file and ".png" in _file:
            os.remove(_file)

def remove_directory(dir_path):
    """Remove a directory if it exists."""
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        try:
            shutil.rmtree(dir_path)
            print(f"Directory {dir_path} removed successfully.")
        except Exception as e:
            print(f"Error removing directory {dir_path}: {e}")
    else:
        print(f"Directory {dir_path} does not exist or is not a directory.")


def save_to_file(location, filename, data):
    """Utility function to save data as plain text."""
    filepath = os.path.join(location, filename)
    try:
        with open(filepath, 'w', encoding="utf-8") as f:
            f.write(data)  # Write the raw string instead of using json.dump
        print(f"Data successfully saved to {filepath}")
    except Exception as e:
        print(f"Error saving file {filename}: {e}")


def clip_tokens(messages, model="gpt-4", max_tokens=100000):
    enc = tiktoken.encoding_for_model(model)
    total_tokens = sum([len(enc.encode(message["content"])) for message in messages])

    if total_tokens <= max_tokens:
        return messages  # No need to clip if under the limit

    # Start removing tokens from the beginning
    tokenized_messages = []
    for message in messages:
        tokenized_content = enc.encode(message["content"])
        tokenized_messages.append({"role": message["role"], "content": tokenized_content})

    # Flatten all tokens
    all_tokens = [token for message in tokenized_messages for token in message["content"]]

    # Remove tokens from the beginning
    clipped_tokens = all_tokens[total_tokens - max_tokens:]

    # Rebuild the clipped messages
    clipped_messages = []
    current_idx = 0
    for message in tokenized_messages:
        message_token_count = len(message["content"])
        if current_idx + message_token_count > len(clipped_tokens):
            clipped_message_content = clipped_tokens[current_idx:]
            clipped_message = enc.decode(clipped_message_content)
            clipped_messages.append({"role": message["role"], "content": clipped_message})
            break
        else:
            clipped_message_content = clipped_tokens[current_idx:current_idx + message_token_count]
            clipped_message = enc.decode(clipped_message_content)
            clipped_messages.append({"role": message["role"], "content": clipped_message})
            current_idx += message_token_count
    return clipped_messages



def extract_prompt(text, word):
    # 匹配以 ```word 这样形式开始，以 ``` 结束的内容
    code_block_pattern = rf"```{word}(.*?)```"  # 构造匹配模式
    code_blocks = re.findall(code_block_pattern, text, re.DOTALL)  # 提取所有匹配的代码块内容
    extracted_code = "\n".join(code_blocks).strip()  # 处理匹配结果
    return extracted_code  # 返回提取的文本

from typing import Dict, List

import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        out_doc = {
            "problem": doc["problem"],
            "solution": doc["solution"],
            "answer": remove_boxed(last_boxed_only_string(doc["solution"])),
        }
        return out_doc

    return dataset.map(_process_doc)


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    retval = 0
    indices = [pos for pos, char in enumerate(results[0]) if char == "$"]
    if len(indices) <= 1:
        answer = results[0]
    else:
        answer = results[0][indices[0] + 1 : indices[-1]]

    if is_equiv(answer, remove_boxed(last_boxed_only_string(doc["solution"]))):
        retval = 1

    results = {
        "exact_match": retval,
    }
    return results


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def clean_answer(s):
    s = s.replace("\\dfrac", "\\frac") # makes no difference but can lead to errors
    s = s.replace("x \\in", "")
    return s

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return clean_answer(s[len(left) : -1])


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"
    if string == "5.5":
        string = "\\frac{11}{2}"
    if "(x - 3)(x + 3)" in string:
        string = string.replace("(x - 3)(x + 3)", "(x+3)(x-3)")

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string
