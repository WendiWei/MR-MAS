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
    def docstring(self):
        pass

    @abstractmethod
    def execute_command(self, *args):
        pass

    @abstractmethod
    def matches_command(self, cmd_str):
        pass

    @abstractmethod
    def parse_command(self, cmd_str):
        pass


"""
@@@@@@@@@@@@@@@@@@
@@ CODING TOOLS @@
@@@@@@@@@@@@@@@@@@
"""

class Replace(Command):
    def __init__(self):
        super().__init__()
        self.cmd_type = "CODE-replace"

    def docstring(self):
        return (
            "============= REWRITE CODE EDITING TOOL =============\n"
            "You also have access to a code replacing tool. \n"
            "This tool allows you to entirely re-write/replace all of the current code and erase all existing code.\n"
            "You can use this tool via the following command: ```REPLACE\n<code here>\n```, where REPLACE is the word REPLACE and <code here> will be the new code that is replacing the entire set of old code. This tool is useful if you want to make very significant changes, such as entirely changing the model, or the learning process. Before changing the existing code to be your new code, your new code will be tested and if it returns an error it will not replace the existing code. Try limiting the use of rewriting and aim for editing the code more."
        )

    def execute_command(self, *args):
        args = args[0]
        return args[0]

    def matches_command(self, cmd_str):
        if "```REPLACE" in cmd_str:
            return True
        return False

    def parse_command(self, *args):
        new_code = extract_prompt(args[0], "REPLACE")
        code_exec = f"{args[1]}\n{new_code}"
        code_ret = execute_r_code(code_exec)
        if "[CODE EXECUTION ERROR]" in code_ret:
            return False, (None, code_ret,)
        return True, (new_code.split("\n"), code_ret)


class Edit(Command):
    def __init__(self):
        super().__init__()
        self.cmd_type = "CODE-edit"

    def docstring(self):
        return (
            "============= CODE EDITING TOOL =============\n"
            "You also have access to a code editing tool. \n"
            "This tool allows you to replace lines indexed n through m (n:m) of the current code with as many lines of new code as you want to add. This removal is inclusive meaning that line n and m and everything between n and m is removed. This will be the primary way that you interact with code. \n"
            "You can edit code using the following command: ```EDIT N M\n<new lines to replace old lines>\n``` EDIT is the word EDIT, N is the first line index you want to replace and M the the last line index you want to replace (everything inbetween will also be removed), and <new lines to replace old lines> will be the new code that is replacing the old code. Before changing the existing code to be your new code, your new code will be tested and if it returns an error it will not replace the existing code. Your changes should significantly change the functionality of the code."
        )

    def execute_command(self, *args):
        try:
            args = args[0]
            current_code = args[2]
            lines_to_add = list(reversed(args[3]))
            lines_to_replace = list(reversed(range(args[0], args[1]+1)))
            for _ln in lines_to_replace:
                current_code.pop(_ln)
            for _line in lines_to_add:
                current_code.insert(args[0], _line)
            new_code = "\n".join(current_code)
            code_exec = f"{args[4]}\n{new_code}"
            code_ret = execute_r_code(code_exec)
            if "CODE EXECUTION ERROR" in code_ret: return (False, None, code_ret)
            return (True, current_code, code_ret)
        except Exception as e:
            return (False, None, str(e))

    def matches_command(self, cmd_str):
        if "```EDIT" in cmd_str:
            return True
        return False

    def parse_command(self, *args):
        cmd_str, codelines, datasetcode = args[0], args[1], args[2]
        success = True
        try:
            text = extract_prompt(cmd_str, "EDIT").split("\n")
            if len(text) == 0:
                return False, None
            lines_to_edit = text[0].split(" ")
            if len(lines_to_edit) != 2:
                return False, None
            lines_to_edit = [int(_) for _ in lines_to_edit]
            if len(text[1:]) == 0:
                return False, None
            return success, (lines_to_edit[0], lines_to_edit[1], codelines, text[1:], datasetcode)
        except Exception as e:
            return False, (None, None, None, None, None)

import re

def extract_score(scoring_text):
    """
    Extract score from a response formatted as:
    ```SCORE
    0.85
    ```
    """
    pattern = r"```SCORE\s*([\d.]+)\s*```"
    match = re.search(pattern, scoring_text, re.IGNORECASE)

    if not match:
        pattern = r"SCORE\s*[:\n]\s*([\d.]+)"
        match = re.search(pattern, scoring_text, re.IGNORECASE)

    if not match:
        raise ValueError(f"Could not extract SCORE from response:\n{scoring_text}")

    score = float(match.group(1))
    score = max(0.0, min(1.0, score))
    return score


def get_score(outlined_plan,code,code_return,REWARD_MODEL_LLM,attempts=3,openai_api_key=None):
    last_error = ""

    system_prompt = (
        "You are a statistical genetics and bioinformatics expert acting as a prompt-based reward scorer. "
        "Your task is to evaluate how well a Mendelian Randomization (MR) analysis was implemented based on "
        "a given research plan, the generated R code, and its execution output.\n\n"

        "You must evaluate the submission using the following weighted criteria:\n"
        "1. Completeness and correctness of the MR analysis pipeline (0.25). "
        "Are the selected MR methods appropriate and correctly applied, such as IVW, MR-Egger, weighted median, "
        "heterogeneity tests, pleiotropy tests, and leave-one-out analysis where applicable?\n"
        "2. Consistency between the outlined plan and the generated code (0.20). "
        "Does the code faithfully implement the planned exposure, outcome, data-processing steps, and MR methods?\n"
        "3. Meaningfulness and interpretability of the execution output (0.25). "
        "Does the output include interpretable causal estimates, standard errors, confidence intervals, p-values, "
        "and relevant sensitivity-analysis results?\n"
        "4. Clarity and reproducibility of the code (0.15). "
        "Is the code readable, logically organized, and sufficiently self-contained for another researcher to reproduce the analysis?\n"
        "5. Robustness and error handling (0.15). "
        "Does the code handle common MR data issues, such as missing values, weak instruments, allele mismatches, "
        "palindromic SNPs, failed package loading, or failed intermediate steps?\n\n"

        "Important scoring rules:\n"
        "- The score must be a floating point number between 0 and 1.\n"
        "- Use 1.0 only for a complete, correct, plan-aligned, reproducible, and interpretable MR analysis.\n"
        "- Use a score >= 0.85 when the MR pipeline is generally complete, correctly implemented, and produces interpretable results, "
        "even if there are minor limitations in annotation, robustness, or presentation.\n"
        "- Use around 0.5 for an analysis that is partially aligned with the plan or somewhat flawed but still informative.\n"
        "- Use <= 0.3 if the code fails to execute, produces only traceback/error messages, or lacks meaningful MR results.\n"
        "- Use <= 0.4 if the output does not include interpretable causal estimates, standard errors, p-values, or confidence intervals.\n"
        "- Use <= 0.6 if the main MR analysis is present but key sensitivity analyses are missing without justification.\n\n"

        "Return your answer strictly in the following format:\n"
        "```SCORE\n"
        "<score>\n"
        "```\n"
        "JUSTIFICATION: <brief justification within 100 words>\n"
        "REPAIR_HINTS: <brief suggestions for improving the code or output within 100 words>\n"
    )

    user_prompt = (
        "Outlined below is the MR research plan:\n"
        f"{outlined_plan}\n\n"

        "Generated R code:\n"
        f"{code}\n\n"

        "Execution output:\n"
        f"{code_return}\n\n"

        "Please evaluate whether the generated code correctly implements the research plan and whether the output is suitable "
        "for downstream MR interpretation and manuscript writing."
    )

    for attempt in range(attempts):
        try:
            scoring = query_model(
                model_str=f"{REWARD_MODEL_LLM}",
                system_prompt=system_prompt,
                openai_api_key=openai_api_key,
                prompt=user_prompt,
                temp=0.1
            )

            performance = extract_score(scoring)

            return (
                performance,
                scoring,
                True
            )

        except Exception as e:
            last_error = str(e)
            continue

    return (
        0.0,
        f"Reward scoring failed after {attempts} attempts. Last error: {last_error}",
        False
    )

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
    try:
        method_or_package, function_name = map(
            lambda x: x.strip().lower(), model_resp.strip().split(',')
        )
    except Exception:
        method_or_package, function_name = 'unknown', 'unknown'

    return method_or_package, function_name

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

            config_text = f"Here is the relevant method or tool config for **{error_method_or_tool}**:\n\n{yaml.dump(method_or_tool_config)},You can refer to this content to modify the code."

        model_resp = query_model(
            openai_api_key=openai_api_key,
            model_str=f"{REPAIR_LLM}",
            system_prompt=repair_sys,
            prompt=f"{config_text}\n\nHere is the error message:\n{error}\n\nHere is the code to be repaired:\n\n{code}",
            temp=0.2
        )
        return model_resp


class MRAnalysisEngine:
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
        self.commands = [Replace()]
        self.model = f"{self.llm_str}"
        init_code, init_return, self.best_score = self.gen_initial_code()
        self.best_codes = [(copy(init_code), self.best_score, init_return) for _ in range(1)]

        self.code_lines = init_code
        self.model = f"{self.llm_str}"
        self.commands = [Replace(),Edit()]
        self.prev_working_code = copy(self.code_lines)

    @staticmethod
    def clean_text(text):
        text = text.replace("```\n", "```")
        text = text.replace("```R\n", "```REPLACE\n")
        return text

    def gen_initial_code(self):
        num_attempts = 0
        error_hist = list()

        while True:
            if num_attempts == 0:
                err = str()
                err_hist = str()
            else:
                err = f"The following was the previous command generated: {model_resp}. This was the error return {cmd_str}. You should make sure not to repeat this error and to solve the presented problem."
                error_hist.append(err)

                if len(error_hist) == 5:
                    _ = error_hist.pop(0)

                err = "\n".join(error_hist)
                err_hist = "The following is a history of your previous errors\n" + err + "\nDO NOT REPEAT THESE."
            else_prompt = "Do not use tryCatch for error handling; instead, write straightforward code and let errors be printed directly."
            model_resp = query_model(
                openai_api_key=self.openai_api_key,
                model_str=self.model,
                system_prompt=self.system_prompt(),
                prompt=f"{err_hist}\n{else_prompt}\nYou should now use ```REPLACE to create initial code to solve the challenge. Now please enter the ```REPLACE command below:\n ",
                temp=1.0
            )

            model_resp = self.clean_text(model_resp)

            cmd_str, code_lines, prev_code_ret, should_execute_code, score = self.process_command(model_resp)

            if not self.supress_print:
                print(f"@@@ INIT ATTEMPT: Command Exec // Attempt {num_attempts}: ", str(cmd_str).replace("\n", " | "))
            if not self.supress_print:
                print(f"$$$ Score: {score}")

            if score is not None:
                break

            num_attempts += 1

        return code_lines, prev_code_ret, score

    def solve(self):
        num_attempts = 0
        best_pkg = None
        top_score = None
        self.prev_code_ret = None
        self.should_execute_code = False

        while True:
            if len(self.commands) == 2:
                cmd_app_str = "You must output either the ```EDIT or ```REPLACE command immediately. "
            else:
                cmd_app_str = ""

            model_resp = query_model(
                openai_api_key=self.openai_api_key,
                model_str=self.model,
                system_prompt=self.system_prompt(),
                prompt=f"The following is your history:{self.history_str()}\n\n{cmd_app_str}Now please enter a command: ",
                temp=1.0
            )

            model_resp = self.clean_text(model_resp)
            self.code_lines = copy(random.choice(self.best_codes)[0])

            cmd_str, code_lines, prev_code_ret, should_execute_code, score = self.process_command(model_resp)

            self.st_history.append([model_resp, prev_code_ret, code_lines, cmd_str])
            if len(self.st_history) > self.st_hist_len:
                self.st_history.pop(0)

            if score is not None:
                if top_score is None:
                    best_pkg = copy(code_lines), copy(prev_code_ret), copy(should_execute_code), copy(model_resp), copy(
                        cmd_str)
                    top_score = score
                elif score > top_score:
                    best_pkg = copy(code_lines), copy(prev_code_ret), copy(should_execute_code), copy(model_resp), copy(
                        cmd_str)
                    top_score = score

            if not self.supress_print:
                print(f"@@@ Command Exec // Attempt {num_attempts}: ", str(cmd_str).replace("\n", " | "))
            if not self.supress_print:
                print(f"$$$ Score: {score}")

            if type(score) is float:
                if score >= 0.8:
                    break

            num_attempts += 1

        self.code_lines, self.prev_code_ret, self.should_execute_code, model_resp, cmd_str = best_pkg

        if not self.supress_print:
            print(prev_code_ret)

        if top_score > self.best_codes[-1][1]:
            if len(self.best_codes) >= self.max_codes:
                self.best_codes.pop(-1)
                self.code_reflect = self.reflect_code()

            self.best_codes.append((copy(self.code_lines), copy(top_score), self.prev_code_ret))
            self.best_codes.sort(key=lambda x: x[1], reverse=True)

        return model_resp, cmd_str

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
        prev_code_ret = self.prev_code_ret
        should_execute_code = self.should_execute_code
        code_lines = copy(self.code_lines)

        remove_figures()

        for cmd in self.commands:
            if cmd.matches_command(model_resp):
                if cmd.cmd_type == "CODE-edit":
                    score = None
                    failed = True
                    code_err = str()

                    for _tries in range(GLOBAL_REPAIR_ATTEMPTS):
                        success, args = cmd.parse_command(model_resp, copy(self.code_lines), self.dataset_code)
                        if success:
                            cmd_return = cmd.execute_command(args)
                            code_err = f"Return from executing code: {cmd_return[2]}"
                            print(f"代码的执行结果是：{code_err}")
                            match = re.search(r"(Error.*?Execution halted)", code_err, re.S)
                            if match:
                                code_err = match.group(1).strip()
                                print(f"该代码编译错误，报错为{code_err}")
                            else:
                                print("未找到错误信息")
                            if cmd_return[0]:
                                code_lines = copy(cmd_return[1])
                                score, cmd_str, is_valid = get_score(
                                    self.plan, "\n".join(code_lines), cmd_return[2],
                                    openai_api_key=self.openai_api_key, REWARD_MODEL_LLM=self.llm_str
                                )
                                if is_valid:
                                    failed = False
                                    break
                                code_err += f"\nReturn from executing code {cmd_str}"

                        error_method_or_tool,function_name = error_identifier(code_err, ERROR_LLM=self.llm_str,openai_api_key=self.openai_api_key)
                        raw_output = f"{error_method_or_tool},{function_name}"
                        error_method_or_tool, function_name = self.normalize_error_result(raw_output, code_err)
                        print(f"出错的方法或者工具是：{error_method_or_tool}")

                        if error_method_or_tool in self.methods_list:
                            method_configs = self.get_method_configs(error_method_or_tool)
                            self.method_or_tool_config = method_configs
                        elif error_method_or_tool in self.tools_list:
                            tool_configs = self.get_tool_configs(error_method_or_tool,function_name)
                            self.method_or_tool_config = tool_configs
                        else:
                            print(f"Unknown error source: {error_method_or_tool}")
                            self.method_or_tool_config = None

                        repaired_code = code_repair(
                            model_resp, code_err, REPAIR_LLM=self.llm_str,
                            ctype="edit", openai_api_key=self.openai_api_key,error_method_or_tool = error_method_or_tool,method_or_tool_config = self.method_or_tool_config
                        )
                        print(f"修复之后的代码为：{repaired_code}")
                        model_resp = repaired_code
                        if not self.supress_print:
                            print(f"     * Attempting repair // try {_tries}*")

                    if failed:
                        cmd_str = f"Code editing FAILED due to the following error: {code_err}. Code was reverted back to original state before edits."
                        if not self.supress_print:
                            print("$$$$ CODE EDIT (failed)")
                    else:
                        cmd_str = "Code was successfully edited."
                        prev_code_ret = copy(cmd_return[2])
                        if not self.supress_print:
                            print("$$$$ CODE EDIT (success)")
                        should_execute_code = True

                    return cmd_str, code_lines, prev_code_ret, should_execute_code, score

                elif cmd.cmd_type == "CODE-replace":
                    score = None
                    failed = True
                    code_err = str()

                    for _tries in range(GLOBAL_REPAIR_ATTEMPTS):
                        print(f"大模型生成的代码为{model_resp}")
                        success, args = cmd.parse_command(model_resp, self.dataset_code)
                        code_err = f"Return from executing code: {args[1]}"
                        if success:
                            print("该代码编译成功")
                            code_lines = copy(args[0])
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
                        match = re.search(r"(Error.*?Execution halted)", code_err, re.S)
                        if match:
                            code_err = match.group(1).strip()
                            print(f"该代码编译错误，报错为{code_err}")
                        else:
                            print("未找到错误信息")

                        error_method_or_tool, function_name = error_identifier(code_err, ERROR_LLM=self.llm_str,openai_api_key=self.openai_api_key)
                        raw_output = f"{error_method_or_tool},{function_name}"
                        error_method_or_tool, function_name = self.normalize_error_result(raw_output, code_err)

                        print(f"出错的方法或者工具是：{error_method_or_tool}")
                        print(f"出错的函数名是：{function_name}")

                        if error_method_or_tool in self.methods_list:
                            method_configs = self.get_method_configs(error_method_or_tool)
                            self.method_or_tool_config = method_configs
                        elif error_method_or_tool in self.tools_list:
                            tool_configs = self.get_tool_configs(error_method_or_tool,function_name)
                            self.method_or_tool_config = tool_configs
                        else:
                            print(f"Unknown error source: {error_method_or_tool}")
                            self.method_or_tool_config = None

                        print(f"调用修复模型对该代码进行第{_tries}次修复")
                        repaired_code = code_repair(
                            extract_prompt(model_resp, "REPLACE"), code_err,
                            ctype="replace", openai_api_key=self.openai_api_key,error_method_or_tool = error_method_or_tool, method_or_tool_config = self.method_or_tool_config,REPAIR_LLM=self.llm_str,
                        )
                        repaired_code = f"```REPLACE\n{repaired_code}\n```"
                        model_resp = repaired_code
                        if not self.supress_print:
                            print(f"     * Attempting repair // try {_tries}*")

                    if failed:
                        cmd_str = f"Code replacement FAILED due to the following error: {code_err}.  Code was reverted back to original state before edits."

                        if not self.supress_print:
                            print("$$$$ CODE REPLACE (failed)")
                    else:

                        cmd_str = "Code was successfully replaced."
                        code_lines = copy(args[0])
                        prev_code_ret = copy(args[1])
                        print("代码替换成功")
                        if not self.supress_print:
                            print("$$$$ CODE REPLACE (success)")
                        should_execute_code = True

                    return cmd_str, code_lines, prev_code_ret, should_execute_code, score

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
                if function_name in yaml_content:
                    return yaml_content[function_name]

                return {}
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
        parts = model_output.strip().lower().split(",")  # parts:['twosamplemr', 'get']
        if len(parts) != 2:
            method, func = "unknown", "unknown"
        else:
            method, func = parts[0].strip(), parts[1].strip()

        detected_method = None
        for pattern, mapped in METHOD_ALIAS.items():
            if re.search(pattern, error_log, re.IGNORECASE):
                detected_method = mapped
                break

        if not detected_method:
            for pattern, mapped in PACKAGE_MAP.items():
                if re.search(pattern, error_log, re.IGNORECASE) or re.search(pattern, func, re.IGNORECASE):
                    detected_method = mapped
                    break

        if not func or func == "unknown":
            m = re.search(r"error in\s+([a-zA-Z0-9_]+)", error_log, re.IGNORECASE)
            func = m.group(1).lower() if m else "unknown"

        if not detected_method:
            detected_method = method if method != "unknown" else "unknown"

        return detected_method,func


    def system_prompt(self, commands=True):
        """
        Produce a system prompt for the mle-solver to solve ml problems
        @param commands: (bool) whether to use command prompt
        @return: (str) system prompt
        """
        return (
            f"{self.role_description()}.\n"
            f"The following are your task instructions: {self.phase_prompt()}\n"
            f"Provided below are some insights from a literature review summary:\n{self.insights}\n"
            f"{self.code_reflect}"
            f"The following are notes, instructions, and general tips for you: {self.notes}"
            f"You are given a Mendelian Randomization (MR) analysis task described. The detailed experimental plan is provided below and should guide your code implementation: {self.plan}\n"         
            f"{self.generate_dataset_descr_prompt()}"
            f"Note: The CAUSE method requires separate data preprocessing as specified in cause.yaml. The CAUSE method operates on the original raw data, not on the datasets processed during the general data preparation step.The original address of exposure data is:{self.Original_exposure_data_path}.The original address of exposure data is:{self.Original_outcome_data_path}\n"
            f"When using the cause method, ensure that instrumental variable screening is performed on the original data.Don't skip this step:LD pruning using PLINK.\n"
            f"If 'hm_rsid' is not available, check whether alternative columns such as 'rsid', 'variant_id', or 'rs_id' exist, and verify that the format meets the required standards. If none of these columns are found, print the available column names in the dataset so you can identify the correct column to use.\n"
            f"You should generate figures to showcase the analysis results, and this can be done using the TwoSampleMR package (twosample). If the user has not specified which figures to generate, you must create at least two figures to illustrate key results; if the user has specified particular figures, generate only those. For figure naming, use descriptive names based on the plot type, method type, and a sequence number to distinguish between single-method plots and multi-method comparison plots. For example, single-method plots can be named 'Scatter_IVW_1.png' or 'Funnel_MR-Egger_1.png', while multi-method comparison plots can be named 'Scatter_MultiMethod_1.png'. If multiple figures of the same type are generated, increment the sequence number, and avoid using generic names like 'Figure_1.png' unless no other context is available.\n"
            f"Your goal is to solve the research plan as well as possible. \n"
            f"Before each experiment please include a print statement explaining exactly what the results are meant to show in great detail before printing the results out.\n"
            f"The following are commands you have access to: {self.command_descriptions()}\n. You should try to have a diversity of command responses if appropriate. Do not repeat the same commend too many times. Please consider looking through your history and not repeating commands too many times.\n" if commands else ""
            f"When generating R code, ensure comments are on separate lines and code statements are complete to avoid isolated symbols or variables."
        )
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
            code_str = self.generate_code_lines(self.code_lines)
            if "[CODE EXECUTION ERROR]" in code_return:
                if not self.supress_print:
                    print(f"@@@@ ERROR")
                reflect_prompt = f"This is your code: {code_str}\n\nYour code returned the following error {code_return}. Please provide a detailed reflection on why this error was returned, which lines in the code caused this error, and exactly (line by line) how you hope to fix this in the next update. This step is mostly meant to reflect in order to help your future self fix the error better. Do not provide entirely new code but provide suggestions on how to fix the bug using LINE EDITS."

            elif os.path.exists("submission.csv"):
                self.prev_working_code = copy(self.code_lines)
                grade_return = get_score(
                    self.plan,
                    "\n".join(self.prev_working_code),
                    code_return,
                    openai_api_key=self.openai_api_key
                )[0]
                if not self.supress_print:
                    print(f"@@@@ SUBMISSION: model score {grade_return}", REWARD_MODEL_LLM=self.llm_str)
                reflect_prompt = f"This is your code: {code_str}\n\nYour code successfully returned a submission csv. You may consider further improvements by incorporating more advanced variant selection methods, expanding sensitivity analyses, or applying multivariable MR models to enhance the robustness and accuracy of your findings. Please provide a detailed reflection on how to improve your analysis pipeline, specify which parts or steps of the code could be optimized, and explain step-by-step how you plan to enhance these in your next update. This reflection step is designed to help you consolidate experience and promote future experimental design and code refinement."

                for file in os.listdir("."):
                    if file.endswith(".csv"):
                        os.system(f"rm {file}")

            else:
                if not self.supress_print:
                    print("@@@@ No return")
                reflect_prompt = f"This is your code: {code_str}\n\nYour code did not return an error, but also did not successfully submit a submission csv file. Please reflect on how you can improve your submission for the next cycle to submit a file and obtain a higher score."

        elif not self.should_execute_code:
            code_return = "No changes were made to the code."
            reflect_prompt = "Reflect on your future plans and next steps to improve the code."

        reflection = self.reflection(reflect_prompt, code_str, code_return)
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
        return phase_str

    def role_description(self):
        """
        Provide role description
        @return: (str) role description
        """
        return "You are an expert in statistical genetics and bioinformatics at a leading research university, developing R code to perform advanced Mendelian Randomization analyses by integrating GWAS summary statistics, causal inference methods."

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