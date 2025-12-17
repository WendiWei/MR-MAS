import random
from copy import copy
from copy import deepcopy
from common_imports import *
from abc import abstractmethod
import yaml

from tools import *
from inference import *
from pathlib import Path


from contextlib import contextmanager
import sys, os


os.environ["JOBLIB_VERBOSITY"] = "0"
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging
logging.getLogger('sklearn.model_selection').setLevel(logging.WARNING)


GLOBAL_REPAIR_ATTEMPTS = 10


class Command:
    def __init__(self):
        self.cmd_type = "OTHER"

    @abstractmethod
    def docstring(self) -> str:
        pass

    @abstractmethod
    def execute_command(self, *args) -> str:
        pass

    @abstractmethod
    def matches_command(self, cmd_str) -> bool:
        pass

    @abstractmethod
    def parse_command(self, cmd_str) -> tuple:
        pass


"""
@@@@@@@@@@@@@@@@@@
@@ CODING TOOLS @@
@@@@@@@@@@@@@@@@@@
"""

# 替换命令类，继承自 Command，用于整体替换代码
class Replace(Command):
    def __init__(self):
        super().__init__()
        self.cmd_type = "CODE-replace"  # 命令类型标记为代码替换

    def docstring(self) -> str:
        return (
            "============= REWRITE CODE EDITING TOOL =============\n"
            "You also have access to a code replacing tool. \n"
            "This tool allows you to entirely re-write/replace all of the current code and erase all existing code.\n"
            "You can use this tool via the following command: ```REPLACE\n<code here>\n```, where REPLACE is the word REPLACE and <code here> will be the new code that is replacing the entire set of old code. This tool is useful if you want to make very significant changes, such as entirely changing the model, or the learning process. Before changing the existing code to be your new code, your new code will be tested and if it returns an error it will not replace the existing code. Try limiting the use of rewriting and aim for editing the code more."
        )

    def execute_command(self, *args) -> str:
        # args[0] 是新的代码内容，直接返回
        args = args[0]
        return args[0]

    def matches_command(self, cmd_str) -> bool:
        # 判断命令字符串是否包含```REPLACE关键词
        if "```REPLACE" in cmd_str:
            return True
        return False

    def parse_command(self, *args) -> tuple:
        # 提取新代码并测试其可执行性
        new_code = extract_prompt(args[0], "REPLACE")
        code_exec = f"{args[1]}\n{new_code}"  # 添加datasetcode前缀
        code_ret = execute_r_code(code_exec)
        # 如果新代码执行出错，则不进行替换
        if "[CODE EXECUTION ERROR]" in code_ret:
            return False, (None, code_ret,)
        # 返回新代码按行分割的结果，以及执行结果
        return True, (new_code.split("\n"), code_ret)


# 编辑命令类，用于替换指定范围的代码行
class Edit(Command):
    def __init__(self):
        super().__init__()
        self.cmd_type = "CODE-edit"  # 命令类型标记为代码编辑

    def docstring(self) -> str:
        return (
            "============= CODE EDITING TOOL =============\n"
            "You also have access to a code editing tool. \n"
            "This tool allows you to replace lines indexed n through m (n:m) of the current code with as many lines of new code as you want to add. This removal is inclusive meaning that line n and m and everything between n and m is removed. This will be the primary way that you interact with code. \n"
            "You can edit code using the following command: ```EDIT N M\n<new lines to replace old lines>\n``` EDIT is the word EDIT, N is the first line index you want to replace and M the the last line index you want to replace (everything inbetween will also be removed), and <new lines to replace old lines> will be the new code that is replacing the old code. Before changing the existing code to be your new code, your new code will be tested and if it returns an error it will not replace the existing code. Your changes should significantly change the functionality of the code."
        )

    def execute_command(self, *args) -> str:
        # args 结构为：(N, M, current_code_lines, new_code_lines, dataset_code_prefix)
        try:
            args = args[0]
            current_code = args[2]  # 当前代码按行的列表
            lines_to_add = list(reversed(args[3]))  # 新代码按行反转，后插入用
            lines_to_replace = list(reversed(range(args[0], args[1]+1)))  # 要替换的行号（倒序处理）
            for _ln in lines_to_replace:
                current_code.pop(_ln)  # 删除原有的代码行
            for _line in lines_to_add:
                current_code.insert(args[0], _line)  # 插入新的代码行
            new_code = "\n".join(current_code)  # 拼接为完整代码
            code_exec = f"{args[4]}\n{new_code}"  # 加上dataset前缀代码
            code_ret = execute_r_code(code_exec)
            # 如果执行失败，返回错误信息
            if "CODE EXECUTION ERROR" in code_ret: return (False, None, code_ret)
            return (True, current_code, code_ret)
        except Exception as e:
            # 出现异常则返回错误信息
            return (False, None, str(e))

    def matches_command(self, cmd_str) -> bool:
        # 判断是否包含```EDIT关键字
        if "```EDIT" in cmd_str:
            return True
        return False

    def parse_command(self, *args) -> tuple:
        # 解析编辑命令字符串，提取起始行号、替换内容等
        cmd_str, codelines, datasetcode = args[0], args[1], args[2]
        success = True
        try:
            text = extract_prompt(cmd_str, "EDIT").split("\n")  # 提取并按行分割命令内容
            if len(text) == 0:
                return False, None
            lines_to_edit = text[0].split(" ")  # 获取起始行号
            if len(lines_to_edit) != 2:
                return False, None
            lines_to_edit = [int(_) for _ in lines_to_edit]  # 转换为整数
            if len(text[1:]) == 0:
                return False, None  # 检查新代码内容是否存在
            # 返回解析结果：起始行号、原始代码、新代码、数据前缀
            return success, (lines_to_edit[0], lines_to_edit[1], codelines, text[1:], datasetcode)
        except Exception as e:
            return False, (None, None, None, None, None)



def get_score(outlined_plan, code, code_return, REWARD_MODEL_LLM, attempts=3, openai_api_key=None):
    e = str()
    for _attempt in range(attempts):
        try:
            # todo: have a reward function here
            sys = (
                f"You are a statistical genetics and bioinformatics expert acting as a reward model.Your task is to evaluate how well a Mendelian Randomization (MR) analysis was implemented based on a given research plan, the code provided, and its output.\n"
                f"You must structure your score exactly in the following way: ```SCORE\n<score here>\n``` where SCORE is just the word score, <score here> is a floating point number between 0 and 1, where 1 means perfectly aligned with the plan, correct and insightful MR results, fully reproducible,where 0.5 means partially aligned or flawed but somewhat informative, and 0 means the analysis is missing key steps, incorrect, or produces uninterpretable results.\n"
                f"Evaluate the submission on the following criteria:\n"
                f"- Completeness and correctness of the MR analysis pipeline.Are the chosen MR methods appropriate and correctly applied?\n"
                f"- Whether the code logically follows the outlined plan and implements required statistical analyses.Does the code faithfully implement the outlined steps? Are all key functions or logic in the plan represented in the code?\n"
                f"- Whether the output contains meaningful MR results. Are the output results interpretable and statistically meaningful (e.g., causal estimates, standard errors, p-values)? Are relevant sensitivity analyses included and interpretable?\n"
                f"- Clarity and reproducibility of the approach. Is the code well-structured, readable, and documented? Can another researcher reproduce the analysis with the information provided?\n"
                f"- Robustness and Error Handling (Bonus).Does the code handle edge cases (e.g., missing data, allele mismatches) gracefully?"
                f"When the MR analysis pipeline is generally complete, the methods are correctly applied, and the results are interpretable and reasonably reproducible, but there may still be minor shortcomings in robustness, annotation, or presentation, a score of ≥0.85 should be given.\n"
                f"You must generate a brief justification after the score (max 100 words) to explain your decision.\n"
            )
            #     f"You are a professor agent who is serving as an expert reward model that can read a research plan, research code, and code output and are able to determine how well a model followed the plan, built the code, and got the proper output scored from 0 to 1 as a float.\n\n"
            #     f"You must structure your score exactly in the following way: ```SCORE\n<score here>\n``` where SCORE is just the word score, <score here> is a floating point number between 0 and 1 representing how well the model followed the plan, built the code, and got the proper output."
            #\
            # with open("prepare_mr_data.R", "w", encoding="utf-8") as file:
            #     file.write(code)
            # print(code)
            scoring = query_model(
                model_str=f"{REWARD_MODEL_LLM}",
                system_prompt=sys,
                openai_api_key=openai_api_key,
                prompt=(
                    f"Outlined in the following text is the research plan: {outlined_plan}\n\n"
                    f"The following text is the research code: \n{code}\n\n"
                    f"The following is the output from the code: {code_return}\n\n"
                    f"Please provide a detailed reflection on the strengths and weaknesses, and suggest improvements line-by-line if applicable."
                ),
                    temp=0.6  # 设定 temp=0.6 表示回答偏向中等随机性。
            )
            # print(scoring)
            performance = extract_prompt(text=scoring, word="SCORE")
            performance = float(eval(performance))
            print(performance)
            return performance, f"The performance of your submission is: {performance}", True
        except Exception as e:
            return None, str(e), False
    return 0, e
# 错误方法或工具识别模型
def error_identifier(error_log, ERROR_LLM, openai_api_key=None):
    identify_sys = (
        "You are an automated error analysis agent in a Mendelian Randomization system.\n"
        "You are given an R execution error log.Your task is to identify:\n"
        "1) Which Mendelian Randomization method (e.g., IVW, MR-Egger, CAUSE, etc.) or R package/tool (e.g., TwoSampleMR, ieugwasr, cause, etc.) caused the error.\n"
        "2) Which R function specifically triggered the error.\n"
        "Respond with two plain text values separated by a comma, in lowercase, "
        "no extra text or formatting.\n"
        "If you cannot determine a value, return 'unknown' for it."
    )

    model_resp = query_model(
        openai_api_key=openai_api_key,
        model_str=f"{ERROR_LLM}",
        system_prompt=identify_sys,
        prompt=f"Here is the error log:\n\n{error_log}",
        temp=0.0
    )
    # 解析返回值
    try:
        method_or_package, function_name = map(
            lambda x: x.strip().lower(), model_resp.strip().split(',')
        )
    except Exception:
        method_or_package, function_name = 'unknown', 'unknown'

    return method_or_package, function_name

# 错误修复模型
def code_repair(code, error, ctype, REPAIR_LLM, error_method_or_tool,method_or_tool_config, openai_api_key=None):
    if ctype == "replace":
        repair_sys = (
            "You are an automated code repair tool for R-based Mendelian Randomization.\n"
            "Your goal is to repair the given R code based on the provided error , using the relevant method or tool configuration.\n"
            "You must ensure that the same error does not occur again, and avoid introducing new errors, while preserving the original code logic as much as possible.\n"
            "Do not use tryCatch for error handling; instead, write straightforward code and let errors be printed directly.\n"
            "You are provided with the method or tool configuration below. Use this information to guide your fix.\n"
            "You must wrap the repaired code in:\n```R\n<code here>\n```\n"
            "Do not forget the opening ```R and the closing ```."
        )
        if method_or_tool_config == None:
            config_text = ""
        else:
            # 提取 CAUSE 部分配置（假设只有一个顶层键，比如 "CAUSE"）
            # method_key = list(method_or_tool_config.keys())[0]  # 获取 "CAUSE"
            # inner_config = method_or_tool_config[method_key]  # 获取其子内容

            # 输出格式化字符串 它会被 yaml.dump(...) 转成一段格式良好的 YAML 文本
            config_text = f"Here is the relevant method or tool config for **{error_method_or_tool}**:\n\n{yaml.dump(method_or_tool_config)},You can refer to this content to modify the code."

        model_resp = query_model(
            openai_api_key=openai_api_key,
            model_str=f"{REPAIR_LLM}",
            system_prompt=repair_sys,
            prompt=f"{config_text}\n\nHere is the error message:\n{error}\n\nHere is the code to be repaired:\n\n{code}",
            temp=0.8
        )
        return extract_prompt(model_resp, "R")

    elif ctype == "edit":
        repair_sys = (
            "You are an automated code repair tool for R-based Mendelian Randomization.\n"
            "Your goal is to repair the given R code based on the provided error and the method/tool configuration.\n"
            "You must ensure that the same error does not occur again, and avoid introducing new bugs.\n"
            "You are provided with the method or tool configuration to guide your fix.\n"
            "\n============= CODE EDITING TOOL =============\n"
            "You can edit the code using the following format:\n"
            "```EDIT N M\n<new code lines>\n```\n"
            "This command replaces lines N to M (inclusive) with the new lines you provide.\n"
            "Before applying the new code, it will be tested. If it produces an error, the change will be rejected.\n"
            "Only use the EDIT command format and nothing else."
        )

        if method_or_tool_config == None:
            config_text = ""
        else:
            # # 提取 CAUSE 部分配置（假设只有一个顶层键，比如 "CAUSE"）
            # method_key = list(method_or_tool_config.keys())[0]  # 获取 "CAUSE"
            # inner_config = method_or_tool_config[method_key]  # 获取其子内容

            # 输出格式化字符串 它会被 yaml.dump(...) 转成一段格式良好的 YAML 文本
            config_text = f"Here is the relevant method or tool config for **{error_method_or_tool}**:\n\n{yaml.dump(method_or_tool_config)},You can refer to this content to modify the code."

        model_resp = query_model(
            openai_api_key=openai_api_key,
            model_str=f"{REPAIR_LLM}",
            system_prompt=repair_sys,
            prompt=f"{config_text}\n\nHere is the error message:\n{error}\n\nHere is the code to be repaired:\n\n{code}",
            temp=0.2
        )
        return model_resp


class MLESolver:
    def __init__(self, dataset_code, openai_api_key=None, notes=None, max_steps=10, insights=None, plan=None, Original_exposure_data_path=None,Original_outcome_data_path=None, llm_str=None):
        self.supress_print = False
        if notes is None: self.notes = []
        else: self.notes = notes
        self.dataset_code = dataset_code
        if plan is None: self.plan = ""
        else: self.plan = plan

        self.method_or_tool_config = str()
        self.methods_list = ["ivw", "mr-egger", "weighted_median", "weighted_model", "mr-presso", "raps","mr-grip","mr-conmix","mrmix","gsmr","cause"]
        self.tools_list = ["twosamplemr", "cause", "mendelianrandomization"]

        self.llm_str = llm_str
        self.verbose = False
        self.max_codes = 1
        self.st_hist_len = 2
        self.min_gen_trials = 1
        self.code_lines = str()
        self.st_history = list()
        self.insights = insights

        self.Original_exposure_data_path = Original_exposure_data_path
        self.Original_outcome_data_path = Original_outcome_data_path

        self.code_reflect = str()
        self.max_steps = max_steps
        self.prev_code_ret = str()
        self.should_execute_code = True
        self.openai_api_key = openai_api_key


    def initial_solve(self):
        """
        Initialize the solver and get an initial set of code and a return
        该函数的作用是反复调用语言模型直到生成一段能被评分函数 get_score 接受的初始代码
        （使用 REPLACE 命令），并带有错误回顾机制来提升模型响应质量。
        @return: None
        """
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@ Initial CodeGen Commands @@
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        self.best_score = None
        self.commands = [Replace()] # 这通常用于注册、存储、或遍历多个可用命令工具。在一个命令调度系统或解释器中，它可以让你后续通过 self.commands 遍历、查找、执行不同的命令。
        self.model = f"{self.llm_str}"
        init_code, init_return, self.best_score = self.gen_initial_code()
        self.best_codes = [(copy(init_code), self.best_score, init_return) for _ in range(1)]

        self.code_lines = init_code
        self.model = f"{self.llm_str}"
        self.commands = [Replace(),Edit()]
        self.prev_working_code = copy(self.code_lines)

    @staticmethod
    def clean_text(text):
        text = text.replace("```\n", "```")  # 将"```\n"替换成 "```"
        text = text.replace("```R\n", "```REPLACE\n")  # 将"```R\n" 替换成 "```REPLACE\n"
        return text

    def gen_initial_code(self):
        num_attempts = 0  # 初始化尝试次数
        error_hist = list()  # 初始化错误历史记录列表

        while True:
            # 第一次尝试时，不需要任何错误提示
            if num_attempts == 0:
                err = str()
                err_hist = str()
            else:
                # 构造上一轮的错误提示，用于提醒模型避免重复错误
                err = f"The following was the previous command generated: {model_resp}. This was the error return {cmd_str}. You should make sure not to repeat this error and to solve the presented problem."
                error_hist.append(err)  # 添加到错误历史中

                # 仅保留最近 5 条错误记录
                if len(error_hist) == 5:
                    _ = error_hist.pop(0)

                # 拼接为一段多轮错误提示，指导模型不要重复同样的错误
                err = "\n".join(error_hist)
                err_hist = "The following is a history of your previous errors\n" + err + "\nDO NOT REPEAT THESE."
            else_prompt = "Do not use tryCatch for error handling; instead, write straightforward code and let errors be printed directly."
            # 向语言模型发送提示，要求使用 ```REPLACE 命令生成初始代码
            model_resp = query_model(
                openai_api_key=self.openai_api_key,
                model_str=self.model,
                system_prompt=self.system_prompt(),
                prompt=f"{err_hist}\n{else_prompt}\nYou should now use ```REPLACE to create initial code to solve the challenge. Now please enter the ```REPLACE command below:\n ",
                temp=1.0
            )

            # 清洗模型输出（去除多余字符、格式等）
            model_resp = self.clean_text(model_resp)

            # 尝试执行模型生成的命令，获取命令文本、代码、返回值、是否执行标志和得分
            cmd_str, code_lines, prev_code_ret, should_execute_code, score = self.process_command(model_resp)

            # 打印调试信息（尝试的命令和评分）
            if not self.supress_print:
                print(f"@@@ INIT ATTEMPT: Command Exec // Attempt {num_attempts}: ", str(cmd_str).replace("\n", " | "))
            if not self.supress_print:
                print(f"$$$ Score: {score}")

            # 如果得分不为 None，说明成功生成有效初始代码，退出循环
            if score is not None:
                break

            # 否则增加尝试次数，准备进行下一轮
            num_attempts += 1

        # 返回生成的代码、输出结果和评分
        return code_lines, prev_code_ret, score

    def solve(self):
        num_attempts = 0  # 尝试次数计数器
        best_pkg = None  # 最佳代码包（包含代码、输出等信息）
        top_score = None  # 当前最高分数
        self.prev_code_ret = None  # 上一次代码的返回值
        self.should_execute_code = False  # 是否应该执行代码的标志位

        while True:
            # 若命令列表只有两个元素，附加提示字符串；否则为空
            if len(self.commands) == 2:
                cmd_app_str = "You must output either the ```EDIT or ```REPLACE command immediately. "
            else:
                cmd_app_str = ""

            # 向模型发送请求，获取命令响应
            model_resp = query_model(
                openai_api_key=self.openai_api_key,
                model_str=self.model,
                system_prompt=self.system_prompt(),
                prompt=f"The following is your history:{self.history_str()}\n\n{cmd_app_str}Now please enter a command: ",
                temp=1.0
            )

            model_resp = self.clean_text(model_resp)  # 清洗模型响应内容
            self.code_lines = copy(random.choice(self.best_codes)[0])  # 从最优代码中随机选择一份作为基础

            # 解析命令并提取信息
            cmd_str, code_lines, prev_code_ret, should_execute_code, score = self.process_command(model_resp)
            # print(should_execute_code)
            # 将当前尝试结果加入历史记录
            self.st_history.append([model_resp, prev_code_ret, code_lines, cmd_str])
            if len(self.st_history) > self.st_hist_len:
                self.st_history.pop(0)  # 限制历史记录长度

            # 若评分非空，则更新最优代码包
            if score is not None:
                if top_score is None:
                    best_pkg = copy(code_lines), copy(prev_code_ret), copy(should_execute_code), copy(model_resp), copy(
                        cmd_str)
                    top_score = score
                elif score > top_score:
                    best_pkg = copy(code_lines), copy(prev_code_ret), copy(should_execute_code), copy(model_resp), copy(
                        cmd_str)
                    top_score = score

            # 打印调试信息（若未关闭打印）
            if not self.supress_print:
                print(f"@@@ Command Exec // Attempt {num_attempts}: ", str(cmd_str).replace("\n", " | "))
            if not self.supress_print:
                print(f"$$$ Score: {score}")

            # 评分大于 0.9 则跳出循环
            if type(score) is float:
                if score >= 0.9:
                    break
            # 达到尝试次数要求并有有效评分则跳出循环
            # if num_attempts >= self.min_gen_trials and top_score is not None:
            #     break
            num_attempts += 1

        # 恢复最优结果
        self.code_lines, self.prev_code_ret, self.should_execute_code, model_resp, cmd_str = best_pkg

        # 打印最终代码的输出结果（若未关闭打印）
        if not self.supress_print:
            print(prev_code_ret)

        # 若当前最佳评分高于最差保留评分，则更新最佳代码集合
        if top_score > self.best_codes[-1][1]:
            if len(self.best_codes) >= self.max_codes:
                self.best_codes.pop(-1)  # 移除得分最低的一项
                self.code_reflect = self.reflect_code()  # 重新生成代码反射结果

            # 添加当前最佳代码到列表
            self.best_codes.append((copy(self.code_lines), copy(top_score), self.prev_code_ret))
            # 按得分排序，确保最低分项在最后
            self.best_codes.sort(key=lambda x: x[1], reverse=True)

        return model_resp, cmd_str  # 返回模型响应和最终命令

    def reflect_code(self):
        """
        Provide a reflection on produced behavior for next execution
        @return: (str) language model-produced reflection
        """
        code_strs = ("$"*40 + "\n\n").join([self.generate_code_lines(_code[0]) + f"\nCode Return {_code[1]}" for _code in self.best_codes])
        code_strs = f"Please reflect on the following sets of code: {code_strs} and come up with generalizable insights that will help you improve your performance on this benchmark."
        syst = self.system_prompt(commands=False) + code_strs
        return query_model(prompt="Please reflect on ideas for how to improve your current code. Examine the provided code and think very specifically (with precise ideas) on how to improve performance, which methods to use, how to improve generalization on the test set with line-by-line examples below:\n", system_prompt=syst, model_str=f"{self.llm_str}", openai_api_key=self.openai_api_key)

    def process_command(self, model_resp):
        """
        从语言模型获取命令并在有效时执行
        编辑和替换代码的处理逻辑、错误修复机制、模型评分与验证流程等。
        @param model_resp: (str) 语言模型的输出字符串
        @return: (tuple) 返回包含以下内容的元组：
            - cmd_str: (str) 命令执行的返回信息
            - code_lines: (list) 修改后的代码行
            - prev_code_ret: (str) 执行后的代码输出
            - should_execute_code: (bool) 是否需要执行新代码
            - score: (float) 模型评分结果
        """

        # 获取先前执行结果和当前代码状态
        prev_code_ret = self.prev_code_ret
        should_execute_code = self.should_execute_code
        code_lines = copy(self.code_lines)

        # 移除绘图等可视化输出（防止干扰后续运行）
        remove_figures()

        # 遍历所有可用命令（例如 EDIT 或 REPLACE）
        for cmd in self.commands:
            if cmd.matches_command(model_resp):
                # 如果是 CODE-edit 类型命令
                if cmd.cmd_type == "CODE-edit":
                    score = None
                    failed = True
                    code_err = str()

                    # 最多尝试 GLOBAL_REPAIR_ATTEMPTS 次修复与执行
                    for _tries in range(GLOBAL_REPAIR_ATTEMPTS):
                        # 解析命令成功后尝试执行
                        success, args = cmd.parse_command(model_resp, copy(self.code_lines), self.dataset_code)
                        if success:
                            cmd_return = cmd.execute_command(args)  # 代码内容   只有在成功解析命令后，cmd.execute_command(args) 才是安全可调用的。
                            code_err = f"Return from executing code: {cmd_return[2]}"  # 代码的执行结果
                            # print(f"代码运行情况:{code_err}")
                            # 从报错信息中提取关键内容，减少输出内容
                            print(f"代码的执行结果是：{code_err}")
                            match = re.search(r"(Error.*?Execution halted)", code_err, re.S)
                            if match:
                                code_err = match.group(1).strip()
                                print(f"该代码编译错误，报错为{code_err}")
                            else:
                                print("未找到错误信息")
                            if cmd_return[0]:  # 执行成功
                                code_lines = copy(cmd_return[1])  # 创建 cmd_return[1] 的一个副本，并赋值给 code_lines，以便后续使用而不影响原始数据。
                                # 对新代码进行评分，并验证有效性
                                score, cmd_str, is_valid = get_score(
                                    self.plan, "\n".join(code_lines), cmd_return[2],
                                    openai_api_key=self.openai_api_key, REWARD_MODEL_LLM=self.llm_str
                                )
                                if is_valid:
                                    failed = False
                                    break
                                code_err += f"\nReturn from executing code {cmd_str}"

                        # 如果失败,调用错误识别模型 识别出错的方法或者工具
                        error_method_or_tool,function_name = error_identifier(code_err, ERROR_LLM=self.llm_str,openai_api_key=self.openai_api_key)
                        # 拼成字符串，再传入 normalize_error_result
                        raw_output = f"{error_method_or_tool},{function_name}"
                        error_method_or_tool, function_name = self.normalize_error_result(raw_output, code_err)
                        print(f"出错的方法或者工具是：{error_method_or_tool}")

                        # 然后去方法库或者工具库里面去匹配并返回给修复模型
                        if error_method_or_tool in self.methods_list:  # 方法名列表
                            method_configs = self.get_method_configs(error_method_or_tool)
                            self.method_or_tool_config = method_configs
                        elif error_method_or_tool in self.tools_list:  # 工具包名列表
                            tool_configs = self.get_tool_configs(error_method_or_tool,function_name)
                            self.method_or_tool_config = tool_configs
                        else:
                            print(f"Unknown error source: {error_method_or_tool}")
                            self.method_or_tool_config = None

                        # 如果失败，调用 code_repair 进行自动修复并重试
                        repaired_code = code_repair(
                            model_resp, code_err, REPAIR_LLM=self.llm_str,
                            ctype="edit", openai_api_key=self.openai_api_key,error_method_or_tool = error_method_or_tool,method_or_tool_config = self.method_or_tool_config
                        )
                        print(f"修复之后的代码为：{repaired_code}")
                        model_resp = repaired_code
                        if not self.supress_print:
                            print(f"     * Attempting repair // try {_tries}*")

                    if failed:
                        # 所有尝试都失败
                        cmd_str = f"Code editing FAILED due to the following error: {code_err}. Code was reverted back to original state before edits."
                        if not self.supress_print:
                            print("$$$$ CODE EDIT (failed)")
                    else:
                        # 成功编辑并获取返回值
                        cmd_str = "Code was successfully edited."
                        prev_code_ret = copy(cmd_return[2])
                        if not self.supress_print:
                            print("$$$$ CODE EDIT (success)")
                        should_execute_code = True

                    return cmd_str, code_lines, prev_code_ret, should_execute_code, score

                # 如果是 CODE-replace 类型命令
                elif cmd.cmd_type == "CODE-replace":
                    score = None
                    failed = True
                    code_err = str()

                    # 最多尝试 GLOBAL_REPAIR_ATTEMPTS 次修复与执行
                    for _tries in range(GLOBAL_REPAIR_ATTEMPTS):
                        # 执行代码，返回代码执行信息
                        print(f"大模型生成的代码为{model_resp}")
                        success, args = cmd.parse_command(model_resp, self.dataset_code)
                        code_err = f"Return from executing code: {args[1]}"
                        # print(f"代码运行情况:{code_err}")
                        if success:
                            print("该代码编译成功")
                            code_lines = copy(args[0])
                            # 对替换后的代码进行评分和验证
                            print("接下来对该代码进行评分")
                            score, cmd_str, is_valid = get_score(
                                self.plan, "\n".join(code_lines), args[1],
                                openai_api_key=self.openai_api_key, REWARD_MODEL_LLM=self.llm_str
                            )
                            print(f"该代码评分为{score}")
                            if is_valid:
                                failed = False
                                break
                            code_err += f"\nReturn from executing code {cmd_str}"
                        print(f"完整的报错信息是：{code_err}")
                        # 从报错信息中提取关键内容
                        match = re.search(r"(Error.*?Execution halted)", code_err, re.S)
                        if match:
                            code_err = match.group(1).strip()
                            print(f"该代码编译错误，报错为{code_err}")
                        else:
                            print("未找到错误信息")

                        # 调用错误识别模型 识别出错的方法或者工具
                        error_method_or_tool, function_name = error_identifier(code_err, ERROR_LLM=self.llm_str,openai_api_key=self.openai_api_key)
                        # 拼成字符串，再传入 normalize_error_result
                        raw_output = f"{error_method_or_tool},{function_name}"
                        error_method_or_tool, function_name = self.normalize_error_result(raw_output, code_err)

                        print(f"出错的方法或者工具是：{error_method_or_tool}")
                        print(f"出错的函数名是：{function_name}")

                        # 然后去方法库或者工具库里面去匹配并返回给修复模型
                        # if error_method_or_tool in self.methods_list and error_method_or_tool in self.tools_list:  # 方法名列表
                        #     method_configs = self.get_method_configs(error_method_or_tool)
                        #     tool_configs = self.get_tool_configs(error_method_or_tool)
                        #     self.method_or_tool_config = []
                        #     if method_configs:
                        #         self.method_or_tool_config.append(method_configs)
                        #     if tool_configs:
                        #         self.method_or_tool_config.append(tool_configs)
                        if error_method_or_tool in self.methods_list:
                            method_configs = self.get_method_configs(error_method_or_tool)
                            self.method_or_tool_config = method_configs
                        elif error_method_or_tool in self.tools_list:  # 工具包名列表
                            tool_configs = self.get_tool_configs(error_method_or_tool,function_name)
                            self.method_or_tool_config = tool_configs
                        else:
                            print(f"Unknown error source: {error_method_or_tool}")
                            self.method_or_tool_config = None

                        # 调用修复模型进行代码修复并包装为合法 REPLACE 命令格式
                        print(f"调用修复模型对该代码进行第{_tries}次修复")
                        repaired_code = code_repair(
                            extract_prompt(model_resp, "REPLACE"), code_err,
                            ctype="replace", openai_api_key=self.openai_api_key,error_method_or_tool = error_method_or_tool, method_or_tool_config = self.method_or_tool_config,REPAIR_LLM=self.llm_str,
                        )
                        # print(f"修复之后的代码为：{repaired_code}")
                        repaired_code = f"```REPLACE\n{repaired_code}\n```"
                        model_resp = repaired_code
                        if not self.supress_print:
                            print(f"     * Attempting repair // try {_tries}*")

                    if failed:
                        # 所有尝试都失败
                        cmd_str = f"Code replacement FAILED due to the following error: {code_err}.  Code was reverted back to original state before edits."

                        if not self.supress_print:
                            print("$$$$ CODE REPLACE (failed)")
                    else:
                        # 替换成功，更新代码与返回值

                        cmd_str = "Code was successfully replaced."
                        code_lines = copy(args[0])
                        prev_code_ret = copy(args[1])
                        print("代码替换成功")
                        if not self.supress_print:
                            print("$$$$ CODE REPLACE (success)")
                        should_execute_code = True

                    return cmd_str, code_lines, prev_code_ret, should_execute_code, score

        # 若无命令匹配，返回失败信息
        if not self.supress_print:
            print("$$$$ INVALID COMMAND (failed)")

        return "Command not supported, choose from existing commands", None, None, None, None

    def history_str(self):
        """
        Well-formatted history string
        @return: (str) history string
        """
        hist_str = ""
        for _hist in range(len(self.st_history)):
            hist_str += f"-------- History ({len(self.st_history)-_hist} steps ago) -----\n"
            hist_str += f"Because of the following response: {self.st_history[_hist][0]}\n" if len(self.st_history[_hist][0]) > 0 else ""
            hist_str += f"and the following COMMAND response output: {self.st_history[_hist][3]}\n"
            hist_str += f"With the following code used: {'#'*20}\n{self.st_history[_hist][2]}\n{'#'*20}\n\n"
            hist_str += f"The environment feedback and reflection was as follows: {self.st_history[_hist][1]}\n"
            hist_str += f"-------- End of history ({len(self.st_history)-_hist} steps ago) -------\n"
        return hist_str

    def get_method_configs(self,error_method_or_tool, base_path="root_library"):
        """只加载指定名称的方法配置"""
        method_configs = []
        config_path = os.path.join(base_path, "method_library", f"{error_method_or_tool}.yaml")
        if not os.path.exists(config_path):
            print(f"Warning: {error_method_or_tool}.yaml not found in tool_library.")
            return {}
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                method_config = yaml.load(f, Loader=yaml.FullLoader)
                method_configs.append(method_config)
        except FileNotFoundError:
            print(f"Error: {error_method_or_tool}.yaml not found in method_library.")
        return method_configs

    def get_tool_configs(self,error_method_or_tool,function_name, base_path="root_library"):
        """只加载指定名称的工具配置"""
        # tool_configs = []
        config_path = os.path.join(base_path, "tool_library", f"{error_method_or_tool}.yaml")
        if not os.path.exists(config_path):
            print(f"Warning: {error_method_or_tool}.yaml not found in tool_library.")
            return {}
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_content = yaml.load(f, Loader=yaml.FullLoader)
                # 根据 function_name 直接索引
                if function_name in yaml_content:
                    return yaml_content[function_name]

                # 如果找不到函数名，返回空字典
                return {}
                # tool_configs.append(tool_config)
        except FileNotFoundError:
            print(f"Error: {error_method_or_tool}.yaml not found in tool_library.")


    def normalize_error_result(self,model_output, error_log):
        """
        使用 LLM 结果，并结合规则进行校正。
        优先返回具体方法（如 ivw、mr-egger），其次才返回工具包（如 twosamplemr）。

        :param model_output: LLM 原始输出 "method_or_package,function_name"
        :param error_log: R 错误日志
        :return: 校正后的 "method_or_package,function_name"
        """
        # 常见 MR 方法（优先级最高）
        # 为了增强泛化性，稍微改写正则逻辑，使下划线和连字符可互换匹配。
        METHOD_ALIAS = {
            r"mr[_-]?ivw": "ivw",
            r"mr[_-]?egger": "mr-egger",
            r"mr[_-]?weighted[_-]?median": "weighted_median",
            r"mr[_-]?weighted[_-]?mode": "weighted_mode",
            r"mr[_-]?presso": "mr-presso",
            r"mr-raps": "raps",
            r"mr[_-]?grip": "mr-grip",
            r"mr[_-]?cml": "cml",
            r"mr[_-]?conmix": "mr-conmix",
            r"mr[_-]?mix": "mrmix",
            r"gsmr": "gsmr",
            r"cause": "cause",
        }

        # 常见 MR 工具/包（次优先级）
        PACKAGE_MAP = {
            r"mr_singlesnp|mr_leaveoneout|mr_heterogeneity|mr_pleiotropy_test": "twosamplemr",
            r"mr_scatter_plot|mr_funnel_plot|mr_leaveoneout_plot|mr_forest_plot": "twosamplemr",
            r"mr_raps|mr_grip": "twosamplemr",
            r"format_data|split_outcome|split_exposure|generate_odds_ratios|subset_on_method|combine_all_mrresults": "twosamplemr",
            r"twosamplemr": "twosamplemr",
            r"ieugwasr": "ieugwasr",
            r"cause": "cause",
            r"gsmr": "gsmr",
            r"mrmix": "mrmix",
        }
        # Step 1: 解析 LLM 输出
        parts = model_output.strip().lower().split(",")  # parts:['twosamplemr', 'get']
        if len(parts) != 2:
            method, func = "unknown", "unknown"
        else:
            method, func = parts[0].strip(), parts[1].strip()

        # Step 2: 优先匹配 MR 方法
        detected_method = None
        for pattern, mapped in METHOD_ALIAS.items():
            if re.search(pattern, error_log, re.IGNORECASE):
                detected_method = mapped
                break

        # Step 3: 如果没有方法，再匹配 MR 工具/包
        if not detected_method:
            for pattern, mapped in PACKAGE_MAP.items():
                if re.search(pattern, error_log, re.IGNORECASE) or re.search(pattern, func, re.IGNORECASE):
                    detected_method = mapped
                    break

        # Step 4: 函数名识别
        if not func or func == "unknown":
            m = re.search(r"error in\s+([a-zA-Z0-9_]+)", error_log, re.IGNORECASE)
            func = m.group(1).lower() if m else "unknown"

        # Step 5: 最终结果
        if not detected_method:
            detected_method = method if method != "unknown" else "unknown"

        return detected_method,func


    def system_prompt(self, commands=True):
        """
        Produce a system prompt for the mle-solver to solve ml problems
        @param commands: (bool) whether to use command prompt
        @return: (str) system prompt
        """
        # plan_path = "research_results/results/methods_tools.yaml"
        # # 加载方法和工具配置
        # method_configs = self.get_method_configs(plan_path)
        # tool_configs = self.get_tool_configs(plan_path)
        return (
            # ROLE DESCRIPTION
            f"{self.role_description()}.\n"
            # TASK INSTRUCTIONS
            f"The following are your task instructions: {self.phase_prompt()}\n"
            # LIT REVIEW INSIGHTS
            f"Provided below are some insights from a literature review summary:\n{self.insights}\n"
            # CODE INSIGHTS
            f"{self.code_reflect}"
            # NOTES
            f"The following are notes, instructions, and general tips for you: {self.notes}"
            # PLAN DESCRIPTION
            f"You are given a Mendelian Randomization (MR) analysis task described. The detailed experimental plan is provided below and should guide your code implementation: {self.plan}\n"
            # DATASET DESCRIPTION            
            f"{self.generate_dataset_descr_prompt()}"
            f"Note: The CAUSE method requires separate data preprocessing as specified in cause.yaml. The CAUSE method operates on the original raw data, not on the datasets processed during the general data preparation step.The original address of exposure data is:{self.Original_exposure_data_path}.The original address of exposure data is:{self.Original_outcome_data_path}\n"
            f"When using the cause method, ensure that instrumental variable screening is performed on the original data.Don't skip this step:LD pruning using PLINK.\n"
            f"If 'hm_rsid' is not available, check whether alternative columns such as 'rsid', 'variant_id', or 'rs_id' exist, and verify that the format meets the required standards. If none of these columns are found, print the available column names in the dataset so you can identify the correct column to use.\n"
            # Create Figures
            f"You should generate figures to showcase the analysis results, and this can be done using the TwoSampleMR package (twosample). If the user has not specified which figures to generate, you must create at least two figures to illustrate key results; if the user has specified particular figures, generate only those. For figure naming, use descriptive names based on the plot type, method type, and a sequence number to distinguish between single-method plots and multi-method comparison plots. For example, single-method plots can be named 'Scatter_IVW_1.png' or 'Funnel_MR-Egger_1.png', while multi-method comparison plots can be named 'Scatter_MultiMethod_1.png'. If multiple figures of the same type are generated, increment the sequence number, and avoid using generic names like 'Figure_1.png' unless no other context is available.\n"
            # Generate Table
            # f"After completing the analysis, generate a summary comparison table in CSV format with the following columns:exposure,outcome,method,nsnp,Estimate,se,pval"
            # f"Requirements:"
            # f"1. Each row corresponds to one MR method (e.g., IVW, MR-Egger, Weighted Median, Weighted Mode, MR-PRESSO, CAUSE) applied to the current exposure–outcome pair.\n"
            # f"2.exposure: name of the exposure trait.\n"
            # f"3.outcome: name of the outcome trait.\n"
            # f"4.method: MR method used.\n"
            # f"5.nsnp: number of SNPs used for the analysis.\n"
            # f"6.Estimate: effect estimate (beta).\n"
            # f"7.se: standard error of the estimate.\n"
            # f"8.pval: p-value of the causal effect,formatted in scientific notation.\n"
            # f"8.95% CI: 95% confidence interval of the effect estimate, formatted as [lower, upper].\n"
            # f"9.Ensure numeric values are precise and consistent (avoid rounding errors).\n"
            # f"10.Output strictly in CSV format without any extra text or explanation.\n"
            # transition
            f"Your goal is to solve the research plan as well as possible. \n"
            f"Before each experiment please include a print statement explaining exactly what the results are meant to show in great detail before printing the results out.\n"
            # COMMAND SET
            f"The following are commands you have access to: {self.command_descriptions()}\n. You should try to have a diversity of command responses if appropriate. Do not repeat the same commend too many times. Please consider looking through your history and not repeating commands too many times.\n" if commands else ""
            f"When generating R code, ensure comments are on separate lines and code statements are complete to avoid isolated symbols or variables."
        )
    #"You will receive a score after you write the code and should aim to maximize the score by following the plan instructions and writing high quality code.\n"

    def generate_code_lines(self, code):
        """
        Generate well-formatted code lines with line numbers
        @param code: (list) list of code line strings
        @return: (str) code lines formatted with line numbers
        """
        codestr = str()
        for _index in range(len(code)):
            codestr += f"{_index} |{code[_index]}\n"
        return codestr

    def feedback(self, code_return):
        """
        提供命令执行后的反馈信息（用于自我反思与改进）
        @param code_return: (str) 代码执行的输出结果
        @return: (str) 包含执行结果和反思内容的反馈字符串
        """

        if code_return is not None:
            # 将当前代码行拼接为字符串形式，供后续反思使用
            code_str = self.generate_code_lines(self.code_lines)

            # 情况 1：代码执行出现错误
            if "[CODE EXECUTION ERROR]" in code_return:
                if not self.supress_print:
                    print(f"@@@@ ERROR")  # 输出错误提示（不打印具体内容）
                # 构造反思提示，要求模型指出错误原因、出错代码行以及如何逐行修复
                reflect_prompt = f"This is your code: {code_str}\n\nYour code returned the following error {code_return}. Please provide a detailed reflection on why this error was returned, which lines in the code caused this error, and exactly (line by line) how you hope to fix this in the next update. This step is mostly meant to reflect in order to help your future self fix the error better. Do not provide entirely new code but provide suggestions on how to fix the bug using LINE EDITS."

            # 情况 2：代码执行成功并生成了 submission.csv 文件（表示提交成功）
            elif os.path.exists("submission.csv"):
                # 保存本次有效代码副本
                self.prev_working_code = copy(self.code_lines)
                # 使用评分函数评估当前代码的表现
                grade_return = get_score(
                    self.plan,
                    "\n".join(self.prev_working_code),
                    code_return,
                    openai_api_key=self.openai_api_key
                )[0]
                if not self.supress_print:
                    print(f"@@@@ SUBMISSION: model score {grade_return}", REWARD_MODEL_LLM=self.llm_str)
                # 构造反思提示，鼓励优化方法、调参等高级改进
                reflect_prompt = f"This is your code: {code_str}\n\nYour code successfully returned a submission csv. You may consider further improvements by incorporating more advanced variant selection methods, expanding sensitivity analyses, or applying multivariable MR models to enhance the robustness and accuracy of your findings. Please provide a detailed reflection on how to improve your analysis pipeline, specify which parts or steps of the code could be optimized, and explain step-by-step how you plan to enhance these in your next update. This reflection step is designed to help you consolidate experience and promote future experimental design and code refinement."

                # 清理当前目录下的所有 .csv 文件，防止后续干扰
                for file in os.listdir("."):
                    if file.endswith(".csv"):
                        os.system(f"rm {file}")

            # 情况 3：未出错但也未成功生成提交文件
            else:
                if not self.supress_print:
                    print("@@@@ No return")
                # 构造反思提示，要求分析为何未成功提交，并给出下一步计划
                reflect_prompt = f"This is your code: {code_str}\n\nYour code did not return an error, but also did not successfully submit a submission csv file. Please reflect on how you can improve your submission for the next cycle to submit a file and obtain a higher score."

        # 情况 4：代码未被执行（模型未修改代码）
        elif not self.should_execute_code:
            code_return = "No changes were made to the code."
            reflect_prompt = "Reflect on your future plans and next steps to improve the code."

        # 生成反思文本，供后续学习或评估参考
        reflection = self.reflection(reflect_prompt, code_str, code_return)

        # 返回代码执行输出及模型生成的反思内容
        return f"Code return: {code_return}\n\nReflection: {reflection}"

    def reflection(self, reflect_prompt, code_str, code_return):
        """
        Reflect on your future plans and next steps to improve the code
        @param reflect_prompt: (str) reflection prompt
        @param code_str: (str) code string
        @return: (str) reflection string
        """
        refl = query_model(prompt=reflect_prompt, system_prompt=self.system_prompt(commands=False), model_str=f"{self.llm_str}", openai_api_key=self.openai_api_key)
        return f"During the previous execution, the following code was run: \n\n{code_str}\n\nThis code returned the following: \n{code_return}\nThe following is your reflection from this feedback {refl}\n"

    def generate_dataset_descr_prompt(self):
        """
        Generate description prompt for kaggle dataset
        @param data_loader: (DataLoader) data loader
        @return: (str) description prompt
        """
        return f"\n- The following dataset code will be added to the beginning of your code always, so this does not need to be rewritten: {self.dataset_code}.DO NOT use the TwoSampleMR package for data searching or loading.DO NOT use web URLs or placeholder links to download data."

    def phase_prompt(self,):
        """
        Describe system role and general tips for mle-solver
        @return: (str) system role
        """
        phase_str = (
            "You are an expert in statistical genetics and bioinformatics at a leading research university, working on a Mendelian Randomization (MR) research project.\n",
            "Your goal is to write R code that obtains final results in MR analysis study. Be sure to integrate the provided plan, and ensure your code implements all the steps outlined in the plan. The data loading code will be added to the beginning of your code always, so this does not need to be rewritten. You should aim for simple code, easy to understand and execute. \n",
            # "Do not write different MR methods inside the same function.\n",
            "Do not include the function call `format_data()`. The input dataset is already formatted for TwoSampleMR.\n"
            "Do not update or reinstall existing R packages. You may load preinstalled packages if needed, but version upgrades are strictly prohibited. Do not include code that requires network access or API keys. Assume all GWAS data is preprocessed and stored locally.\n",
            "The selection of significant SNPs, LD pruning, IV selection, and data harmonization have already been completed. Do not repeat these steps. Use preprocessed datasets from 'research_results/processed_data/' and save results to 'research_results/results/'. Save figures to 'research_results/results/figures/'.\n",
            "Do not define new R functions (e.g., perform_ivw_mr). Write all analysis code directly in simple sequential blocks.\n"
            "Avoid using placeholders, empty objects, or unimplemented analysis steps in the code. Ensure that the input, analysis, and output for each method are fully executable.\n"
            "Avoid including commented-out code, pseudocode, or demonstration lines that are not actually executed. Make sure all methods have real, executable implementations with proper inputs, analysis steps, and outputs.\n"
            "Do not generate conditional statements that check whether a package is installed (e.g., if(\"PACKAGE\" %in% installed.packages())); only directly load the package and perform the analysis.\n"
            "Carefully review the plan to ensure that the code you write includes all MR methods specified in the plan.\n"
            "If 'hm_rsid' is not available, check whether alternative columns such as 'rsid', 'variant_id', or 'rs_id' exist, and verify that the format meets the required standards. If none of these columns are found, print the available column names in the dataset so you can identify the correct column to use.\n"
            "Do not generate code that requires long-running computations, repeated resampling, or processing of large-scale datasets. All analyses should be executed as quickly as possible and be suitable for a small number of SNPs or small-scale datasets.\n"
            "Important instructions for plotting:\n",
            "1. Use only the built-in TwoSampleMR visualization functions: mr_scatter_plot(), mr_forest_plot(),mr_funnel_plot(), and mr_leaveoneout_plot().\n",
            "2. Do NOT manually combine MR results into custom ggplot() calls.\n",
            "3. mr_scatter_plot() and mr_forest_plot() return lists of ggplot objects. Use a for-loop to save each plot individually with ggsave(). Do NOT pass the list directly to ggsave().\n",
            "4. Use the harmonized_data object directly; do NOT reload it with read.csv.\n",
            "5. Save all figures in 'research_results/results/figures/' with descriptive filenames.\n",
            "6. Ensure all plots are clearly labeled with titles, axes labels, and legends.\n",
            "7. Write simple code; each plotting block should be independent and readable.\n",
            "Follow these steps in each code block:\n"
            "1. Method Selection and Execution: Execute each MR method in an independent block using the correct input structure.\n"
            "2. Data Preprocessing and Checking: Verify the input data matches the method requirements.\n"
            "3. Perform Causal Analysis and Output Results: Print results to console and save to CSV.\n"
            "4. Error Handling: If any errors occur, print clear error messages for debugging.\n"
            "5. Reproducibility: Use set.seed() where needed to ensure reproducible results.\n"
        )
        # "Write smaller, independent code blocks for each MR method so that each block can be executed separately. Ensure each block prints its output and saves results correctly.\n",

        #  phase_str = (
       #      "You are an expert in statistical genetics and bioinformatics at a leading research university, working on a Mendelian Randomization (MR) research project.\n",
       #      "Your goal is to write R code that obtains final results in MR analysis study.Be sure to integrate the provided literature review and research plan, and ensure your code implements all the steps outlined in the plan.The data loading code will be added to the beginning of your code always, so this does not need to be rewritten.You should aim for simple code, not complex code. \n",
       #      "I would recommend writing smaller code so you do not run out of time but make sure to work on all points in the plan in the same code. You code should run every experiment outlined in the plan for a single code.\n",
       #      "Do not update or reinstall existing R packages. You may load preinstalled packages if needed, but version upgrades are strictly prohibited. If a package does not exist, you can download it\n",
       #      "Please do NOT include any code that calls `ieugwasr_auth()` or makes requests to the OpenGWAS API. Assume that all GWAS data is already preprocessed and provided locally, so there is no need for online authentication or data retrieval. Avoid generating any code that requires API keys or network access to prevent execution errors.\n",
       #      "the selection of significant SNPs, linkage disequilibrium (LD) pruning, instrumental variable (IV) selection, and data harmonization have already been completed in prior processing steps. Accordingly, these procedures should not be repeated at the current stage. Subsequent analyses should proceed directly using the preprocessed, quality-controlled, harmonized, and selected SNP and IV dataset.\n",
       #      "All raw GWAS data has already been stored in the directory: \"research_results/GWAS_data/\". All processed files, including harmonized data, have already been stored in:\"research_results/processed_data/\". You should locate and load data from the \"research_results/processed_data/\" directory. All figures and plots must be saved in:\"research_results/results/figures/\", and all MR analysis results must be saved in\"research_results/results/\".\n"
       #      "Please ensure that the following steps are followed when performing causal analysis:\n\n1. Method Selection and Execution: When conducting Mendelian Randomization analysis, first determine the analysis method to be used. Ensure that the correct input data is used for each method, and adjust parameters accordingly.\n\n2. Data Preprocessing and Checking: Before performing the analysis, check that the input data meets the requirements of the selected method, particularly the sample genotypes, effect sizes, standard errors, etc.\n\n3. Perform Causal Analysis and Output Results: When executing each method, ensure that the correct data structure for the respective method is used. After each method execution, make sure to print or output the results into the same file, ensuring the output data can be used to evaluate the effectiveness of each method.\n\n4. Error Handling and Reporting: If any errors occur (e.g., data issues or method incompatibility), ensure that error information is reported rather than silently handled. This will help track and resolve issues.\n\n5. Saving and Displaying Results: Save the results of each method to the appropriate file path. At the same time, provide appropriate visualizations, such as scatter plots or forest plots, to help compare the results of different methods.\n"
       # )
            # "You are a machine learning engineer and will be writing code for a Mendelian Randomization (MR) research project.\n"
            # "Your goal is to produce code that obtains final results for all experiments in the MR study. You should aim to write simple and straightforward code for data acquisition, preprocessing, and analysis—avoid complex logic. Be sure to integrate the provided literature review and research plan, and ensure your code implements all the steps outlined in the plan. The dataset loading code will always be added at the beginning of your script automatically, so you do not need to rewrite it. Do not write functions—just write loose executable code.\n"
            # "It is recommended that you keep your code modular and concise to save time, but make sure the entire MR analysis pipeline is covered in one script. Your code should be able to execute all MR experiments described in the plan in a single run.\n",
            # If your analysis requires models or methods, use what's already available from existing packages.
            # "You cannot pip install new libraries, but most MR-relevant Python libraries (such as pandas, numpy, scipy, etc.) are already available. If you want to use any models or tools in your code, please rely on existing packages.\n"
            # "Anything you print inside your code will be returned to you as input, so you will be able to see those outputs. It is recommended to use print statements at key steps to help with debugging and understanding the execution flow."

        #
        # phase_str = (
        #     "You are an algorithm engineer and you will be writing the code for a research project.\n"
        #     "Your goal is to produce code that obtains final results for a set of research experiments. You should aim for simple code to collect all results, not complex code. You should integrate the provided literature review and the plan to make sure you are implementing everything outlined in the plan. The dataset code will be added to the beginning of your code always, so this does not need to be rewritten. Make sure you do not write functions, only loose code.\n"
        #     "I would recommend writing smaller code so you do not run out of time but make sure to work on all points in the plan in the same code. You code should run every experiment outlined in the plan for a single code.\n",
        #     "You cannot pip install new libraries, but many machine learning libraries already work. If you wish to use a language model in your code, please use the following:\nAnything you decide to print inside your code will be provided to you as input, and you will be able to see that part of the code. Using print statements is useful for figuring out what is wrong and understanding your code better."
        # )
        return phase_str

    def role_description(self):
        """
        Provide role description
        @return: (str) role description
        """
        return "You are an expert in statistical genetics and bioinformatics at a leading research university, developing R code to perform advanced Mendelian Randomization analyses by integrating GWAS summary statistics, causal inference methods."
        # return "You are an expert machine learning engineer working at a top university to write code to solve machine learning research challenges using your machine learning expertise."

    @staticmethod
    def _common_code_errors():
        """
        Some general tips to avoid common code errors, also TF has many errors so we avoid this and ask to use pytorch
        @return: (str) common code errors
        """
        return (
            "Make sure to library everything that you are using.\n"
            "Reflect on the code before writing it to make sure there are no bugs or compilation issues.\n"
            "YOU MUST USE COMMANDS PROPERLY. Do not use the word COMMAND for the command that is incorrect. You must use an actual command (e.g. EDIT, REPLACE...) NOT THE WORD COMMAND. Do not make this mistake.\n"
            # "Under no circumstances should you use tensorflow or keras. Only use pytorch for scikitlearn for deep learning.\n"
        )

    def command_descriptions(self):
        """
        Provide command descriptions
        @return: (str) command descriptions
        """
        cmd_strings = "\n".join([_cmd.docstring() for _cmd in self.commands])
        return f"\nYou also have access to tools which can be interacted with using the following structure: ```COMMAND\n<command information here>\n```, where COMMAND is whichever command you want to run (e.g. EDIT, REPLACE...), <command information here> is information used for the command, such as code to run or a search query, and ``` are meant to encapsulate the command. ``` must be included as part of the command both at the beginning and at the end of the code. DO NOT FORGOT TO HAVE ``` AT THE TOP AND BOTTOM OF CODE. and this structure must be followed to execute a command correctly. YOU CAN ONLY EXECUTE A SINGLE COMMAND AT A TIME! Do not try to perform multiple commands EVER only one. {self._common_code_errors()}" + cmd_strings

    def run_code(self):
        """
        Actually execute the code that was generated
        @return: (str) code return
        """
        if self.prev_code_ret is not None:
            return self.prev_code_ret
        elif self.should_execute_code:
            return execute_r_code("\n".join(self.code_lines))
        return "Changes have not yet been made to the code."






# def get_method_configs(self,plan_file, base_path="root_library"):
    #     """加载研究计划文件中的所有方法配置"""
    #     method_configs = []
    #     try:
    #         with open(plan_file, 'r') as file:
    #             plan = yaml.load(file, Loader=yaml.FullLoader)
    #     except FileNotFoundError:
    #         print(f"Plan file {plan_file} not found.")
    #         return method_configs
    #
    #     for method in plan.get("methods", []):
    #         method_name = method.get("name")
    #         config_path = os.path.join(base_path, "method_library", f"{method_name}.yaml")
    #         try:
    #             with open(config_path, 'r') as f:
    #                 method_config = yaml.load(f, Loader=yaml.FullLoader)
    #                 method_configs.append(method_config)
    #         except FileNotFoundError:
    #             print(f"Error: {method_name}.yaml not found in method_library.")
    #
    #     return method_configs
    #
    # def get_tool_configs(self,plan_file, base_path="root_library"):
    #     """加载研究计划文件中的所有工具配置"""
    #     tool_configs = []
    #     try:
    #         with open(plan_file, 'r') as file:
    #             plan = yaml.load(file, Loader=yaml.FullLoader)
    #     except FileNotFoundError:
    #         print(f"Plan file {plan_file} not found.")
    #         return tool_configs
    #
    #     for tool in plan.get("tools", []):
    #         tool_name = tool.get("name")
    #         config_path = os.path.join(base_path, "tool_library", f"{tool_name}.yaml")
    #         try:
    #             with open(config_path, 'r') as f:
    #                 tool_config = yaml.load(f, Loader=yaml.FullLoader)
    #                 tool_configs.append(tool_config)
    #         except FileNotFoundError:
    #             print(f"Error: {tool_name}.yaml not found in tool_library.")
    #
    #     return tool_configs

# def code_repair(code, error, ctype, REPAIR_LLM, method_or_tool_config, openai_api_key=None):
#     if ctype == "replace":
#         repair_sys = (
#             "You are an automated code repair tool.\n"
#             "Your goal is to take in code and an error and repair the code to make sure the same error does not repeat itself, and also to remove any other potential errors from the code without affecting the code output.\n"
#             "Your output should match the original code as closely as possible.\n"
#             "You must wrap the code in the following ```R\n<code here>\n```\n"
#             "Do not forget the opening ```R and the closing ```."
#         )
#         model_resp = query_model(
#             openai_api_key=openai_api_key,
#             model_str=f"{REPAIR_LLM}",
#             system_prompt=repair_sys,
#             prompt=f"Provided here is the error: {error}\n\nProvided below is the code:\n\n{code}", temp=0.8)
#         return extract_prompt(model_resp, "R")
#     elif ctype == "edit":
#         repair_sys = (
#             "You are an automated code repair tool.\n"
#             "Your goal is to take in code and an error and repair the code to make sure the same error does not repeat itself, and also to remove any other potential errors from the code without affecting the code output.\n"
#             "Your output should match the original code as closely as possible.\n"
#
#             "============= CODE EDITING TOOL =============\n"
#             "You have access to a code editing tool. \n"
#             "This tool allows you to replace lines indexed n through m (n:m) of the current code with as many lines of new code as you want to add. This removal is inclusive meaning that line n and m and everything between n and m is removed. This will be the primary way that you interact with code. \n"
#             "You can edit code using the following command: ```EDIT N M\n<new lines to replace old lines>\n``` EDIT is the word EDIT, N is the first line index you want to replace and M the the last line index you want to replace (everything inbetween will also be removed), and <new lines to replace old lines> will be the new code that is replacing the old code. Before changing the existing code to be your new code, your new code will be tested and if it returns an error it will not replace the existing code.\n"
#             "Please use the code editing tool to fix this code."
#             "Do not forget the opening ```EDIT N M and the closing ```."
#             "Your output should look like the following\n\n```EDIT N M\n<new lines to replace old lines>\n```"
#         )
#         model_resp = query_model(
#             openai_api_key=openai_api_key,
#             model_str=f"{REPAIR_LLM}",
#             system_prompt=repair_sys,
#             prompt=f"Provided here is the error: {error}\n\nProvided below is the code:\n\n{code}", temp=0.2)
#         return model_resp