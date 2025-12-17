import random
import string
from utils import *
# from tools import *
from PubmedSearch import *
from copy import copy
from inference import *
from pathlib import Path
from copy import deepcopy
from common_imports import *
from agents import get_score
from abc import abstractmethod

from contextlib import contextmanager
import sys, os


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


def execute_latex():
    return True


"""
@@@@@@@@@@@@@@@@@@
@@ SEARCH TOOLS @@
@@@@@@@@@@@@@@@@@@
"""


class PubMed(Command):
    def __init__(self):
        super().__init__()
        self.PubMed_eng = PubMedSearch(email="2418168449@qq.com")
        self.num_papers_per_search = 10
        self.cmd_type = "SEARCH-PubMed"

    def docstring(self) -> str:
        return (
            "============= PubMed SEARCH TOOL ============="
            "You also have access to biomedical paper from PubMed. "
            "To search for summaries of papers on PubMed you can use the following command: ```SUMMARY\n<search query>\n```\n where <search query> is a string that will be used as the search query to find papers with semantically similar content and SUMMARY is just the word SUMMARY.\n"
            "To get the full paper text for an PubMed paper, use the following command: ```FULL_TEXT\n<PubMed paper id>\n```\n where <PubMed paper id> is the ID of the PubMed paper (which can be found by using the SUMMARY command), and FULL_TEXT is just the word FULL_TEXT. Make sure to read the full text using the FULL_TEXT command before adding it to your list of relevant papers.\n"
            "When you read PubMed paper, pay special attention to the following aspects: the strategies used for instrumental variable (IV) selection, the MR analysis methods applied, and whether sensitivity analyses or pleiotropy assessments were included. These details are crucial for constructing, reproducing, or improving your own MR experimental design."
        )

    def execute_command(self, *args) -> str:
        # args[0] -> command
        # args[1] -> query
        if args[0] == "SUMMARY":
            return self.PubMed_eng.search_papers(args[1], self.num_papers_per_search)
        elif args[0] == "FULL_TEXT":
            pmcid = self.PubMed_eng.get_paper_pmcid(args[1])
            return self.PubMed_eng.retrieve_full_paper_text(pmcid)
        raise Exception("Invalid PubMed Search")

    def matches_command(self, cmd_str) -> bool:
        if "```SUMMARY" in cmd_str:
            return True
        elif "```FULL_TEXT" in cmd_str:
            return True
        return False

    def parse_command(self, *args) -> tuple:
        sum_text = extract_prompt(args[0], "SUMMARY").split("\n")
        full_text = extract_prompt(args[0], "FULL_TEXT").split("\n")
        if len(sum_text) == 0 and len(full_text) == 0: return False, None
        if len(sum_text) > 0: return True, ("SUMMARY", sum_text,)
        if len(full_text) > 0: return True, ("FULL_TEXT", sum_text,)


"""
@@@@@@@@@@@@@@@@@@@
@@ WRITING TOOLS @@
@@@@@@@@@@@@@@@@@@@
"""


class PaperReplace(Command):
    def __init__(self, save_loc):
        super().__init__()
        self.save_loc = save_loc
        self.cmd_type = "PAPER-replace"

    def docstring(self) -> str:
        return (
            "============= PAPER REPLACING TOOL =============\n"
            "You also have access to a paper replacing tool. \n"
            "This tool allows you to entirely re-write/replace all of the current latex and erase all existing latex.\n"
            "You can use this tool via the following command: ```REPLACE\n<latex here>\n```, where REPLACE is the word REPLACE and <latex here> will be the new latex that is replacing the entire set of old latex. This tool is useful if you want to make very significant changes, such as entirely changing the model, or the learning process. Before changing the existing latex to be your new latex, your new latex will be tested and if it returns an error it will not replace the existing latex. Try limiting the use of rewriting and aim for editing the latex more."
        )

    def execute_command(self, *args) -> str:
        # args[0] -> new latex
        args = args[0]
        return args[0]

    def matches_command(self, cmd_str) -> bool:
        if "```REPLACE" in cmd_str: return True
        return False

    def parse_command(self, *args) -> tuple:
        new_latex = extract_prompt(args[0], "REPLACE")
        # print(new_latex)
        latex_ret = compile_latex(new_latex, self.save_loc, compile=args[1])
        if "[CODE EXECUTION ERROR]" in latex_ret:
            return False, (None, latex_ret,)
        return True, (new_latex.split("\n"), latex_ret)


class PaperEdit(Command):
    def __init__(self, save_loc):
        super().__init__()
        self.save_loc = save_loc
        self.cmd_type = "PAPER-edit"

    def docstring(self) -> str:
        return (
            "============= PAPER EDITING TOOL =============\n"
            "You also have access to a paper editing tool. \n"
            "This tool allows you to replace lines indexed n through m (n:m) of the current latex with as many lines of new latex as you want to add. This removal is inclusive meaning that line n and m and everything between n and m is removed. This will be the primary way that you interact with latex. \n"
            "You can edit latex using the following command: ```EDIT N M\n<new lines to replace old lines>\n``` EDIT is the word EDIT, N is the first line index you want to replace and M the last line index you want to replace (everything inbetween will also be removed), and <new lines to replace old lines> will be the new latex that is replacing the old latex. DO NOT include any explanation, reasoning, or commentary inside or outside the command block. Only the EDIT command block is allowed as output.Before changing the existing latex to be your new latex, your new latex will be tested and if it returns an error it will not replace the existing latex. Your changes should significantly change the latex. You should write new paragraphs and update old ones. Try using the edit command often. Make sure to generate lots of text. You should also avoid editing lines 0 0, and should edit the main text of the paragraphs, such as editing lines in the middle of the text body."
        )

    def execute_command(self, *args) -> str:
        # args[0] -> N (int)
        # args[1] -> M (int)
        # args[2] -> old latex
        # args[3] -> new lines to replace
        try:
            args = args[0]
            current_latex = args[2]
            lines_to_add = list(reversed(args[3]))
            lines_to_replace = list(reversed(range(args[0], args[1] + 1)))
            for _ln in lines_to_replace:
                current_latex.pop(_ln)
            for _line in lines_to_add:
                current_latex.insert(args[0], _line)
            new_latex = "\n".join(current_latex)
            latex_exec = f"{new_latex}"
            latex_ret = compile_latex(latex_exec, self.save_loc, compile=args[4])
            if "error" in latex_ret.lower(): return (False, None, latex_ret)
            return (True, current_latex, latex_ret)
        except Exception as e:
            return (False, None, str(e))

    def matches_command(self, cmd_str) -> bool:
        if "```EDIT" in cmd_str: return True
        return False

    def parse_command(self, *args) -> tuple:
        cmd_str, latexlines = args[0], args[1]
        success = True
        try:
            text = extract_prompt(cmd_str, "EDIT").split("\n")
            if len(text) == 0: return False, (None, None, None, None)
            lines_to_edit = text[0].split(" ")
            if len(lines_to_edit) != 2: return False, (None, None, None, None)
            lines_to_edit = [int(_) for _ in lines_to_edit]
            if len(text[1:]) == 0: return False, (None, None, None, None)
            return success, (lines_to_edit[0], lines_to_edit[1], latexlines, text[1:])
        except Exception as e:
            return False, (None, None, None, None)


# Modified version of section tips from the AI scientist paper!
# Good work guys :) https://github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/perform_writeup.py
per_section_tips = {
    "abstract": """
    - TL;DR of the paper
- What are we trying to do and why is it relevant?
- Why is this hard?
- How do we solve it (i.e. our contribution!)
- How do we verify that we solved it (e.g. Experiments and results)
- This must only be a single paragraph, not more.
Please make sure the abstract reads smoothly and is well-motivated. This should be one continuous paragraph with no breaks between the lines.
""",
    "introduction": """
- Longer version of the Abstract, i.e. of the entire paper
- What are we trying to do and why is it relevant?
- Why is this hard?
- How do we solve it (i.e. our contribution!)
- How do we verify that we solved it (e.g. Experiments and results)
- New trend: specifically list your contributions as bullet points
- Extra space? Future work!
""",
    "related work": """
- Academic siblings of our work, i.e. alternative attempts in literature at trying to solve the same problem.
- Goal is to “Compare and contrast” - how does their approach differ in either assumptions or method? If their method is applicable to our Problem Setting I expect a comparison in the experimental section. If not, there needs to be a clear statement why a given method is not applicable.
- Note: Just describing what another paper is doing is not enough. We need to compare and contrast.
""",
    "methods": """
- What we do. Why we do it. All described using the general Formalism introduced in the Problem Setting and building on top of the concepts / foundations introduced in related work.
- Make sure you clearly report precise mathematical equations in the methods section and the precise methodology.
- Note: If our paper introduces a novel problem setting as part of its contributions, it's best to have a separate Section.
""",
    "experiments": """
- How do we test that our stuff works? Introduces a specific instantiation of the Problem Setting and specific implementation details of our Method for this Problem Setting.
- Do not imagine unknown hardware details.
- Includes a description of the datasets used (e.g., GWAS summary statistics, sequencing data), their sources (e.g., GWAS Catalog), evaluation metrics, important hyperparameters, and implementation details.
""",
    "results": """
- Shows the results of running Method on our problem described in experiments.
- Includes statements on hyperparameters and other potential issues of fairness.
- Only includes results that have actually been run and saved in the logs. Do not hallucinate results that don't exist.
- Make sure you clearly and numerically report experimental results in the results section.
- If results exist: compares to baselines and includes statistics and confidence intervals.
- If results exist: includes ablation studies to show that specific parts of the method are relevant.
- Discusses limitations of the method.
- Make sure to include all the results from the experiments, and include all relevant figures.
""",
    "conclusion": """
- Brief recap of the entire paper.
- To keep going with the analogy, you can think of future work as (potential) academic offspring.
"""
}

class PaperSolver:
    def __init__(self, llm_str, notes=None, max_steps=10, insights=None, plan=None, exp_code=None, exp_results=None,
                 lit_review=None, datasets_information=None, ref_papers=None, topic=None, openai_api_key=None,
                 compile_pdf=True, save_loc=None):
        self.supress_print = True
        if notes is None:
            self.notes = []
        else:
            self.notes = notes
        if plan is None:
            self.plan = ""
        else:
            self.plan = plan
        if exp_code is None:
            self.exp_code = ""
        else:
            self.exp_code = exp_code
        if exp_results is None:
            self.exp_results = ""
        else:
            self.exp_results = exp_results
        if lit_review is None:
            self.lit_review = ""
        else:
            self.lit_review = lit_review

        if datasets_information is None:
            self.datasets_information = ""
        else:
            self.datasets_information = datasets_information

        if insights is None:
            self.insights = ""
        else:
            self.insights = insights
        if ref_papers is None:
            self.ref_papers = ""
        else:
            self.ref_papers = ref_papers
        if topic is None:
            self.topic = ""
        else:
            self.topic = topic
        self.save_loc = save_loc
        self.compile_pdf = compile_pdf
        self.llm_str = llm_str
        self.notes = notes
        self.max_papers = 1
        self.st_hist_len = 10
        self.min_gen_trials = 2
        self.max_steps = max_steps
        self.paper_lines = str()
        self.prev_paper_ret = str()
        self.section_related_work = {}
        self.openai_api_key = openai_api_key

    def solve(self):
        num_attempts = 0  # 尝试次数计数器
        best_pkg = None  # 用于存储当前得分最高的方案（内容、输出等）
        top_score = None  # 当前最高得分
        self.prev_paper_ret = None  # 前一次命令执行的返回信息

        while True:
            # 从历史最佳报告中随机选取一篇作为当前尝试的起点
            self.paper_lines = copy(random.choice(self.best_report)[0])

            # 调用模型生成新的命令（对话式交互）
            model_resp = query_model(
                model_str=self.model,
                system_prompt=self.system_prompt(),
                prompt=f"\nNow please enter a command: ",
                temp=1.0,
                openai_api_key=self.openai_api_key
            )

            # 清洗模型输出（去除多余内容等）
            model_resp = self.clean_text(model_resp)

            # 处理模型生成的命令，返回命令内容、生成的论文行、原始返回、评分
            cmd_str, paper_lines, prev_paper_ret, score = self.process_command(model_resp)

            # 如果模型生成的命令得分有效，则更新当前最佳方案
            if score is not None:
                if top_score is None:
                    # 第一次赋值
                    best_pkg = copy(paper_lines), copy(prev_paper_ret), copy(model_resp), copy(cmd_str)
                    top_score = score
                elif score > top_score:
                    # 如果当前结果比之前的最高分更好，则更新
                    best_pkg = copy(paper_lines), copy(prev_paper_ret), copy(model_resp), copy(cmd_str)
                    top_score = score

            # 如果尝试次数超过最小要求且已有有效得分，则终止循环
            if num_attempts >= self.min_gen_trials and top_score is not None:
                break

            # 打印当前尝试的命令及得分（用于调试）
            if not self.supress_print:
                print(f"@@@ Command Exec // Attempt {num_attempts}: ", str(cmd_str).replace("\n", " | "))
            if not self.supress_print:
                print(f"$$$ Score: {score}")

            num_attempts += 1  # 增加尝试次数

        # 更新当前最佳方案为最终生成结果
        self.paper_lines, self.prev_paper_ret, model_resp, cmd_str = best_pkg

        # 如果当前最高得分超过最佳报告列表中的最低得分，则将其加入列表
        if top_score > self.best_report[-1][1]:
            # 如果最佳报告数量达到上限，则移除得分最低的那一个
            if len(self.best_report) >= self.max_papers:
                self.best_report.pop(-1)
            # 添加当前最佳方案到报告列表中
            self.best_report.append((copy(self.paper_lines), copy(top_score), self.prev_paper_ret))
            # 按照得分降序排序，以便未来移除最低分项
            self.best_report.sort(key=lambda x: x[1], reverse=True)

        # 返回最终的模型输出和命令字符串
        return model_resp, cmd_str

    def initial_solve(self):
        """
        Initialize the solver and get an initial set of papers and a return
        第一步： 用 PaperReplace 生成一份初始报告，并打分。
        第二步： 把这份报告作为当前最佳报告保存下来。
        第三步： 切换到 PaperEdit 命令，以便接下来对报告进行改进和优化。
        第四步： 保存一份副本，作为回退点。
        换句话说，它是论文自动生成流程的初始化阶段：
        生成一个初稿 → 评估 → 存档 → 进入优化循环的准备。
        @return: None
        """
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@ 初始论文生成 (PaperGen) 阶段命令 @@
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        # 初始化最佳得分为空
        self.best_score = None

        # 设置初始命令为 PaperReplace，用于替换生成报告
        self.commands = [PaperReplace(self.save_loc)]

        # 指定所使用的大模型
        self.model = f"{self.llm_str}"

        # 调用 gen_initial_report() 生成初始报告、返回值以及初始得分
        init_report, init_return, self.best_score = self.gen_initial_report()

        # 将初始报告、得分和返回结果保存为 best_report（作为当前最佳结果）
        # 把初始报告存入 best_report 里，作为当前的最佳结果。
        # best_report 是一个包含单个三元组的列表：[(报告内容, 得分, 附加返回值)]
        self.best_report = [(copy(init_report), self.best_score, init_return) for _ in range(1)]

        # 保存初始报告的内容到 paper_lines
        self.paper_lines = init_report

        # 再次指定大模型
        self.model = f"{self.llm_str}"

        # 将命令切换为 PaperEdit，用于后续编辑优化
        self.commands = [PaperEdit(self.save_loc)]  # , Replace()]

        # 保存当前报告作为 prev_working_report（便于后续对比和修改）
        # 把当前报告保存一份副本，留作“上一次的有效报告”。
        # 这样在后续优化时，如果新结果不如旧结果，可以回退。
        self.prev_working_report = copy(self.paper_lines)

        # PaperReplace 用于“第一次生成”，而 PaperEdit 用于“后续改进”。

    @staticmethod
    def clean_text(text):
        text = text.replace("```\n", "```")
        return text

    def gen_initial_report(self):
        """
        逐步生成一篇初始的研究论文草稿（LaTeX 形式），包括 scaffold（骨架）、摘要、引言、相关工作、方法、实验、结果、结论等部分。
        生成过程中会调用语言模型生成文本，并根据需要自动检索 PubMed 文献作为参考。
        """
        num_attempts = 0
        PubMed_eng = PubMedSearch(email="2418168449@qq.com")  # 初始化用于 PubMed 检索的对象
        section_scaffold = str()  # 初始化论文结构模板为空字符串
        # 初始化字典，
        self.used_pmids = {}

        # 定义需要生成的论文各部分内容，包括 scaffold 占位符和各章节标题
        for _section in ["scaffold", "abstract", "introduction", "related work", "methods",
                         "experiments", "results", "conclusion","references"]:
            section_complete = False

            # "introduction", "related work", "methods", "conclusion" 部分需要文献辅助，则先检索相关文献
            if _section in ["introduction", "related work", "methods", "conclusion"]:
                attempts = 0
                papers = str()
                first_attempt = True
                while len(papers) == 0:  # 如果检索不到文献就重试（最多5次）
                    att_str = str()
                    if attempts > 5:
                        break
                    # 第二次及之后尝试提示用更简单的搜索词
                    if not first_attempt:
                        att_str = "This is not your first attempt please try to come up with a simpler search query."

                    # 调用语言模型生成用于 PubMed 搜索的查询词
                    print("接下来调用语言模型生成用于 PubMed 搜索的查询词")
                    search_query = query_model(
                        model_str=f"{self.llm_str}",
                        prompt=f"Given the following research topic {self.topic} and research plan: \n\n{self.plan}\n\nPlease come up with a search query to find relevant papers on PubMed. Respond only with the search query and nothing else. Do not return the think process.This should be a a string that will be used to find papers with semantically similar content. {att_str}",
                        system_prompt=f"You are a research paper finder. You must find papers for the section {_section}. Query must be text nothing else.",
                        openai_api_key=self.openai_api_key
                    )
                    search_query = search_query.replace('"', '')  # 去除多余引号
                    print(f"生成的搜索词为：{search_query}")
                    print("接下来进行文献检索")
                    papers = PubMed_eng.search_papers(query=search_query)  # 检索前20篇相关文献
                    print("文献检索完成")
                    first_attempt = False
                    attempts += 1
                if len(papers) != 0:
                    self.section_related_work[_section] = papers  # 保存该部分对应的文献列表

            # 开始尝试生成该 section 的内容
            while not section_complete:
                section_scaffold_temp = copy(section_scaffold)  # 当前的 scaffold 副本
                if num_attempts == 0:
                    err = str()
                else:
                    # 如果之前有生成失败，则添加提示信息
                    err = f"The following was the previous command generated: {model_resp}. This was the error return {cmd_str}. You should make sure not to repeat this error and to solve the presented problem."

                if _section == "references":
                    references_list = []

                    # print("DEBUG: section_related_work =", self.section_related_work)
                    print("used_pmids =", self.used_pmids)

                    # 遍历每个章节实际引用的 PMID 列表
                    for sec, pmid_list in self.used_pmids.items():
                        papers = self.section_related_work.get(sec, [])
                        # 构建 pmid -> paper 映射
                        papers_dict = {paper.get("pmid"): paper for paper in papers}

                        for pmid in pmid_list:
                            paper = papers_dict.get(pmid)
                            if not paper:
                                continue  # 正文引用了但检索不到文献，跳过
                            title = paper.get("title", "No title")
                            authors = ", ".join(paper.get("authors", ["Unknown"]))
                            journal = paper.get("journal", "No journal")
                            year = paper.get("year", "No year")

                            # 拼接 LaTeX bibitem
                            references_list.append(
                                f"\\bibitem{{PubMed{pmid}}} {authors}. {title}. {journal}. {year}. (PubMed ID: {pmid})"
                            )

                    # 生成完整的 thebibliography 环境
                    if references_list:
                        new_text = "\\begin{thebibliography}{99}\n" + "\n".join(
                            references_list) + "\n\\end{thebibliography}"
                    else:
                        new_text = "\\begin{thebibliography}{99}\n\\end{thebibliography}"  # 没有引用文献也生成空环境

                    # 替换 scaffold 中占位符
                    section_scaffold_temp = section_scaffold_temp.replace("[REFERENCES HERE]", new_text)
                    section_scaffold = section_scaffold_temp
                    model_resp = '```REPLACE\n' + copy(section_scaffold_temp) + '\n```'

                else:
                    retries = 0
                    max_retries = 3
                    while retries < max_retries:
                        # MODIFIED: 给智能体提示上一次返回为空
                        empty_hint = ""
                        if retries > 0:
                            empty_hint = f"Previous output was empty. Please make sure to generate non-empty content for the section {_section}.\n"

                        # 针对 scaffold 部分与正文章节使用不同提示
                        if _section == "scaffold":
                            prompt = f"{err}\nNow please enter the ```REPLACE command to create the scaffold:\n "
                        else:
                            rp = str()
                            if _section in self.section_related_work:
                                # 引入相关文献信息用于参考
                                rp = f"Here are related papers you can cite: {self.section_related_work[_section]}. You can cite them just by putting the PubMed ID in parentheses, e.g. (PubMed 34662886)\n"
                            prompt = (f"{err}\n{rp}\n{empty_hint}\nPlease generate the text for the designated section wrapped exactly with triple quotes and the word REPLACE.The format MUST be:\n\n"
                                      f" ```REPLACE\n"
                                      f"<only the content of this section goes here>\n"
                                      f"```\n"
                                      f"make sure to only write the text for that section and nothing else Following the above format.. Do not include packages or section titles, just the section content:\n ")

                        # 调用语言模型生成章节内容
                        model_resp = query_model(
                            model_str=self.model,
                            system_prompt=self.system_prompt(section=_section),
                            prompt=f"{prompt}",
                            temp=0.8,
                            openai_api_key=self.openai_api_key
                        )
                        model_resp = self.clean_text(model_resp)  # 清洗返回结果
                        if model_resp and model_resp.strip():
                            break  # valid response
                        else:
                            retries += 1
                            print(
                                f"@@@ WARNING: model_resp empty, retry {_section} generation ({retries}/{max_retries})")

                    # 验证 scaffold 是否包含所有关键占位符
                    if _section == "scaffold":
                        for _sect in ["[ABSTRACT HERE]", "[INTRODUCTION HERE]", "[METHODS HERE]", "[RESULTS HERE]",
                                      "[CONCLUSION HERE]", "[REFERENCES HERE]"]:
                            if _sect not in model_resp:
                                cmd_str = "Error: scaffold section placeholders were not present (e.g. [ABSTRACT HERE])."
                                if not self.supress_print:
                                    print("@@@ INIT ATTEMPT:", cmd_str)
                                continue
                    elif _section != "scaffold":
                        # 智能体在生成正文时，会在正文后面附加参考的文献的pmid,所有可以提取正文中出现的pmid,为生成参考文献做准备（并不是所有检索到的文献都会作为写论文的参考）
                        pmids_in_text = re.findall(r'PubMed\s*(\d+)', model_resp)
                        self.used_pmids[_section] = pmids_in_text
                        # 提取生成的正文中的 REPLACE 内容，替换对应占位符
                        # new_text = extract_prompt(model_resp, "REPLACE")

                        def extract_replace_block(model_resp):
                            """
                            提取模型输出中的 REPLACE 块。
                            - 如果找到但内容为空，则兜底用原始输出
                            - 如果没找到，也兜底用原始输出
                            """
                            match = re.search(r"```REPLACE\s*(.*?)\s*```", model_resp, re.DOTALL)
                            if match:
                                content = match.group(1).strip()
                                if content:  # 找到并且非空
                                    return content
                                else:  # 找到但内容为空
                                    return model_resp.strip()
                            else:  # 没找到 REPLACE
                                return model_resp.strip()

                        # 使用新的后处理逻辑
                        new_text = extract_replace_block(model_resp)
                        # 新加：调用 sanitize_latex 进行 LaTeX 兜底修复
                        new_text = sanitize_latex(new_text)
                        section_scaffold_temp = section_scaffold_temp.replace(f"[{_section.upper()} HERE]", new_text)
                        model_resp = '```REPLACE\n' + copy(section_scaffold_temp) + '\n```'
                        # 检查是否错误地插入了 LaTeX 模板命令
                        # documentclass[unnumsec,webpdf,contemporary,large]
                        # if "documentclass{article}" in new_text or "usepackage{" in new_text:
                        if "documentclass[unnumsec,webpdf,contemporary,large]" in new_text or "usepackage{" in new_text:
                            cmd_str = "Error: You must not include packages or documentclass in the text! Your latex must only include the section text, equations, and tables."
                            if not self.supress_print:
                                print("@@@ INIT ATTEMPT:", cmd_str)
                            continue


                # 执行模型命令并返回编译状态与评分
                cmd_str, latex_lines, prev_latex_ret, score = self.process_command(model_resp, scoring=True)
                if not self.supress_print:
                    print(f"@@@ INIT ATTEMPT: Command Exec // Attempt {num_attempts}: ",
                          str(cmd_str).replace("\n", " | "))

                # 如果成功生成内容并评分非空，视为该部分已完成
                if score is not None:
                    section_complete = True
                    section_scaffold = "\n".join(latex_lines)
                    print(f"当前成功生成{_section}部分的内容，并且评分为{score}")
                num_attempts += 1

            # 更新当前生成好的论文内容
            self.paper_lines = section_scaffold.split("\n")
            if not self.supress_print:
                print("$" * 10, f"SCAFFOLD [{_section}] CREATED", "$" * 10)

        if not self.supress_print:
            print("$" * 10, "SCAFFOLD CREATED", "$" * 10)
        return latex_lines, prev_latex_ret, score  # 返回最终生成的 LaTeX 内容和评分

    def process_command(self, model_resp, scoring=True):
        """
        从语言模型接收命令并尝试执行有效的命令(replace,edit)
        验证生成的latex内容是否能编译成功并进行评分

        @param model_resp: (str) 语言模型的输出（命令文本）
        @param scoring: (bool) 是否执行评分（默认开启）
        @return: (tuple)
            - cmd_str: (str) 命令执行结果的返回信息
            - paper_lines: (list) 编辑后的论文文本行
            - prev_paper_ret: (str) 运行论文后的返回内容
            - score: (float) 命令对应论文的得分
        """
        cmd_str = None  # 存储命令执行后的结果信息（成功/失败说明）
        score = None  # 存储最终的论文评分（数值）
        prev_paper_ret = self.prev_paper_ret  # 保存上一次论文运行的返回值
        paper_lines = copy(self.paper_lines)  # 复制当前论文的行内容（避免直接修改原始数据）

        # 处理模型输出中可能包含的插图路径，将其替换为绝对路径（防止 LaTeX 找不到图片）
        if "\\includegraphics[width=\\textwidth]{Figure_1.png}" in model_resp or "\\includegraphics[width=\\textwidth]{Figure_2.png}" in model_resp:
            cwd = os.getcwd()  # 获取当前工作目录
            print(f"当前工作目录是：{cwd}")
            # fig1_path = os.path.abspath(os.path.join(cwd, "paper_research/Figure_1.png"))
            # fig2_path = os.path.abspath(os.path.join(cwd, "paper_research/Figure_2.png"))
            fig1_path = r"D:\pycharm_projects\AgentLaboratory\AgentLaboratory\paper_research\Figure_1.png"
            fig2_path = r"D:\pycharm_projects\AgentLaboratory\AgentLaboratory\paper_research\Figure_2.png"
            model_resp = model_resp.replace(
                "\\includegraphics[width=\\textwidth]{Figure_1.png}",
                f"\\includegraphics[width=\\textwidth]{fig1_path}"
            )
            model_resp = model_resp.replace(
                "\\includegraphics[width=\\textwidth]{Figure_2.png}",
                f"\\includegraphics[width=\\textwidth]{fig2_path}"
            )

        # 遍历当前支持的命令对象（比如 PAPER-edit / PAPER-replace）
        for cmd in self.commands:
            # 检查模型输出是否匹配该命令的模式
            if cmd.matches_command(model_resp):

                # ----------- 处理 PAPER-edit 命令 -----------
                if cmd.cmd_type == "PAPER-edit":
                    score = None
                    failed = True  # 默认设置执行失败
                    # 解析命令参数（返回是否成功，参数内容）
                    success, args = cmd.parse_command(model_resp, paper_lines)
                    # 初始错误信息，记录 latex 返回值
                    paper_err = f"Return from executing latex: {args[1]}"

                    if success:
                        # 执行编辑命令，返回 (是否成功, 新论文行, 返回值等)
                        args = cmd.execute_command((args[0], args[1], paper_lines, args[3], self.compile_pdf))
                        success = success and args[0]  # 确认执行是否真正成功

                        if success:
                            paper_lines = copy(args[1])  # 更新论文内容
                            if scoring:
                                # 如果启用评分，则调用评分函数
                                score, cmd_str, is_valid = get_score(self.plan, "\n".join(paper_lines),
                                                                     reward_model_llm=self.llm_str)
                            else:
                                # 不启用评分时，默认给一个成功分数
                                score, cmd_str, is_valid = 0.0, "Paper scored successfully", True
                            if is_valid:
                                failed = False  # 标记执行成功
                            # 将评分/错误信息拼接到错误日志中
                            paper_err += f"\nReturn from executing latex: {cmd_str}"
                        # 控制台输出执行状态
                        if not self.supress_print:
                            print("$$$$ PAPER EDIT (success)" if not failed else "$$$$ PAPER EDIT (failed)")

                    if failed:
                        # 如果失败，则恢复论文原状并记录失败原因
                        print("替换失败，恢复原始论文")
                        cmd_str = f"Paper edit FAILED due to the following error: {paper_err}.  Paper was reverted back to original state before edits."
                    else:
                        # 如果成功，更新结果与论文
                        cmd_str = "Paper was successfully edited."
                        paper_lines = copy(args[1])
                        prev_paper_ret = copy(args[2])

                # ----------- 处理 PAPER-replace 命令 -----------
                elif cmd.cmd_type == "PAPER-replace":
                    score = None
                    failed = True  # 默认设置执行失败
                    # 解析命令参数
                    success, args = cmd.parse_command(model_resp, self.compile_pdf)
                    # 初始错误信息
                    paper_err = f"Return from executing latex: {args[1]}"

                    if success:
                        paper_lines = copy(args[0])  # 替换整篇论文内容
                        if scoring:
                            # 评分
                            score, cmd_str, is_valid = get_score(self.plan, "\n".join(paper_lines),
                                                                 reward_model_llm=self.llm_str)
                        else:
                            score, cmd_str, is_valid = 0.0, "Paper scored successfully", True
                        if is_valid:
                            failed = False
                        # 拼接错误/结果日志
                        paper_err += f"\nReturn from executing code on real test set {cmd_str}"
                    if not self.supress_print:
                        print("$$$$ PAPER REPLACE (success)" if not failed else "$$$$ PAPER REPLACE (failed)")

                    if failed:
                        # 替换失败，恢复原始论文并报告
                        print("替换失败，恢复原始论文")
                        cmd_str = f"Paper replacement FAILED due to the following error: {paper_err}.  Paper was reverted back to original state before edits."
                    else:
                        # 替换成功，更新论文和返回值
                        cmd_str = "Paper was successfully replaced."
                        paper_lines = copy(args[0])
                        prev_paper_ret = copy(args[1])

        # 返回执行结果字符串、最终论文行内容、执行输出、论文得分
        return cmd_str, paper_lines, prev_paper_ret, score

    def generate_paper_lines(self, code):
        """
        Generate well-formatted code lines with line numbers
        @param code: (list) list of code line strings
        @return: (str) code lines formatted with line numbers
        """
        codestr = str()
        for _index in range(len(code)):
            codestr += f"{_index} |{code[_index]}\n"
        return codestr

    def system_prompt(self, commands=True, section=None):
        """
        Produce a system prompt for the paper-solver
        @param commands: (bool) whether to use command prompt
        @return: (str) system prompt
        """
        if section == "abstract":
            length = "This section should be ONLY 1 paragraph."
        else:
            length = "This section should be approximately 2-4 paragraphs and so your output should be several paragraphs of latex."
        methods_str = str()
#         if section == "results":
#             fig1_text = """\n\\begin{figure}[!t]
# 	\\centerline{\\includegraphics[width=\\columnwidth]{Figure_1.png}}
# 	\\caption{caption here}
# 	\\label{fig:fig1}
# \\end{figure}\n"""
#             fig2_text = """\n\\begin{figure}[!t]
# 	\\centerline{\\includegraphics[width=\\columnwidth]{Figure_2.png}}
# 	\\caption{caption here}
# 	\\label{fig:fig2}
# \\end{figure}\n"""
#             # fig1_text = """\n\\begin{figure}[!t]
#             # \\caption{<caption here>}
#             # \\centering
#             # \\includegraphics[width=\\textwidth]{Figure_1.png}
#             # \\label{fig:fig1}
#             # \\end{figure}\n"""
#             #
#             # fig2_text = """\n\\begin{figure}[h]
#             # \\caption{<caption here>}
#             # \\centering
#             # \\includegraphics[width=\\textwidth]{Figure_2.png}
#             # \\label{fig:fig1}
#             # \\end{figure}\n"""
#             fig_dir = "paper_research"
#             # fig1_path = os.path.join(fig_dir, "Figure_1.png")
#             # fig2_path = os.path.join(fig_dir, "Figure_2.png")
#             fig1_path = r"D:\pycharm_projects\AgentLaboratory\AgentLaboratory\paper_research\Figure_1.png"
#             fig2_path = r"D:\pycharm_projects\AgentLaboratory\AgentLaboratory\paper_research\Figure_2.png"
#             if os.path.exists(fig1_path) and os.path.exists(fig2_path):
#                 # print("Both figures found.")
#                 methods_str += f"You ABSOLUTELY must without fail also include Figure_1.png and Figure_2.png in your paper using {fig1_text} and {fig2_text} on a new line. Make sure to place these figures in separate locations."
#             elif os.path.exists(fig1_path):
#                 # print("One figures are missing.")
#                 methods_str += f"You ABSOLUTELY must without fail also include Figure_1.png in your paper using {fig1_text} on a new line.\n"
#             elif os.path.exists(fig2_path):
#                 # print("One figures are missing.")
#                 methods_str += f"You ABSOLUTELY must without fail also include Figure_2.png in your paper using {fig2_text} on a new line.\n"
        if section is not None and section == "scaffold":
            model_latex = """
\documentclass[unnumsec,webpdf,contemporary,large]{oup-authoring-template}
\graphicspath{{Fig/}}
\\theoremstyle{thmstyleone}
\\newtheorem{theorem}{Theorem}
\\newtheorem{proposition}[theorem]{Proposition}
\\theoremstyle{thmstyletwo}
\\newtheorem{example}{Example}
\\newtheorem{remark}{Remark}
\\theoremstyle{thmstylethree}
\\newtheorem{definition}{Definition}
\\begin{document}
\journaltitle{Journal Title Here}
\DOI{DOI HERE}
\copyrightyear{2022}
\pubyear{2019}
\\access{Advance Access Publication Date: Day Month Year}
\\appnotes{Paper}
\\firstpage{1}
\\title{title here}
\\authormark{Author Name et al.}
\\author{author here}
\date{}
\\abstract{abstract here}
\keywords{keywords here}
\maketitle
\section{Introduction}
\section{Related Work}
\section{Methods}
\section{Experiments}
\section{Results}
\section{Conclusion}
\end{document}
"""
            section_cmd = f"Your objective right now is to only build the scaffolding for the paper Refer to this format {model_latex}. You should not include any text in the body of the paper, but should have an empty scaffold for each of the sections.  Where the sections go, write [ABSTRACT HERE] for abstract, and write [INTRODUCTION HERE] for the introduction... etc. Your paper should have the following sections: 1. Abstract 2. Introduction, 3. Related Work 5. Methods, 5. Experiments 6. Results, 7. Conclusion and 8.References. Just create the scaffolding as compilable latex. Your title should start with Research Report: [title here] where title here is a title you choose. For author write Huazhong Agricultural University.You should generate 3-5 keywords"
        elif section is not None:
            if section == "introduction":
                bullet_instruction = (
                    "You may list the main contributions of this work using the `itemize` environment, "
                    "with each contribution as a separate `\\item`."
                )
            else:
                bullet_instruction = (
                    "Avoid using the `itemize` environment unless absolutely necessary; "
                    "write in normal paragraph format."
                )

            section_cmd = rf"""
            Your only goal is to generate LaTeX for the following {section}.
            DO NOT INCLUDE ANY PACKAGES OR SECTION COMMANDS.
            DO NOT INCLUDE A TITLE OR DATE, ONLY TEXT.
            You only have to generate text for this specific section.
            {length}
            {bullet_instruction}
            Ensure you are always writing fully compilable LaTeX code. 
            Common mistakes that should be fixed include:
            
            - LaTeX syntax errors (unenclosed math, unmatched braces, etc.).
            - Duplicate figure labels or references.
            - Escape all LaTeX special characters: & % _ # $ ^ ~ \.
            - Proper table and figure closure.
            - Do not hallucinate new citations or results not in the logs.
            - Use proper LaTeX math syntax: ALL mathematical expressions, numbers, percentages, and scientific notation MUST be in math mode \( ... \) or $...$. 
              Example: write "p < 0.05" as \(p < 0.05\) or $p < 0.05$.
            - Only use ASCII characters and LaTeX math commands; Greek letters must be in math mode: $\beta$, $\alpha$, $\gamma$.
            - To include a percentage sign %, ALWAYS use \%.
            - Write scientific notation in math mode using \times: \(1.2\times 10^{-8}\).
            - Avoid special or invisible Unicode characters; use normal spaces only.
            - Biological terms should use standard lowercase style with abbreviations in parentheses.
            - Each line of text or formula should not exceed 80 characters.
            - Formulas should be split using align or multline for multi-line equations.
            - Tables must fit within the page width; use fixed column widths like 5cm, or relative widths up to the page width.
            - Do not generate any LaTeX environments (abstract, figure, table, section); output only raw text content for this section.
            - Paragraphs should wrap naturally; avoid overly long continuous characters.
            - The style should be formal and concise, suitable for a bioinformatics journal.

            Here are some tips {per_section_tips[section]} {methods_str}.
            """

            # section_cmd = f"Your only goal is to generate latex for the following {section}. DO NOT INCLUDE ANY PACKAGES OR ANY SECTION COMMANDS. DO NOT INCLUDE A TITLE OR DATE ONLY TEXT. You only have to generate text for this specific section and do not have to output anything else. {length} I repeat DO NOT INCLUDE ANY PACKAGES OR ANY SECTION COMMANDS. DO NOT INCLUDE A TITLE OR DATE ONLY TEXT. Use proper LaTeX math syntax: ALL mathematical expressions, numbers, percentages, and scientific notation MUST be enclosed in `\( ... \)` or `$...$`. To include a percentage sign %, ALWAYS use a backslash \% or it will be treated as a comment.Write scientific notation correctly using `\times`, e.g., `\(1.2\times 10^{-8}\)`. Ensure biological terms use standard lowercase style with abbreviations in parentheses. Avoid using \\textless, use < instead inside math mode.Please format the content using the `itemize` environment with each contribution as a separate `\item`. The style should be formal and concise, suitable for a bioinformatics journal. Here are some tips {per_section_tips[section]}  {methods_str}.\n\n"
        else:
            section_cmd = ""
        paper_len = sum([i.strip(string.punctuation).isalpha() for i in ("".join(self.paper_lines)).split()])
        #paper_len2 = len(("".join(self.paper_lines)).split())
        if paper_len < 4000:
            paper_progress = f"The current length of the paper is {paper_len} words, you must increase this by {4000 - paper_len} words."
        else:
            paper_progress = ""
        if not self.supress_print: print(paper_progress)
        cmd_set = f"The following are commands you have access to: {self.command_descriptions()}\n." if commands else ""
        if len(self.ref_papers) == 0:
            ref_papers = ""
        else:
            refpapers = '\n'.join(self.ref_papers)
            ref_papers = f"Here is a reference paper that is high quality:\n{refpapers}\n\n\n"
        lit_review_str = str(self.lit_review)[:20000]
        return (
            f"{ref_papers}"
            # ROLE DESCRIPTION
            f"{self.role_description()}.\n"
            # TASK INSTRUCTIONS
            f"The following are your task instructions: {self.phase_prompt()}\n"
            # NOTES
            # f"The following are notes, instructions, and general tips for you: {self.notes}"
            # LIT REVIEW
            # f"The following literature review was provided for the paper:\n{lit_review_str}\n"
            # PLAN DESCRIPTION
            f"You are given a paper report writing task. The original research plan was described as follows: {self.plan}\n"
            # DATASET INFORMATION
            f"This is the detailed information of the exposure and outcome data you used: \n{self.datasets_information}\n"
            # EXPERIMENT CODE
            f"A team of research wrote the following code, following this plan: {self.exp_code}\n"
            # EXPERIMENT RESULTS
            f"After running this code, the following results were observed: {self.exp_results}\n Your results must ACCURATELY reflect the numbers presented here."
            # EXPERIMENT RESULT INSIGHTS
            f"Provided was an interpretation of the experimental results:\n{self.insights}\n"
            f"Your writing style should be boring and objective.\n"
            # transition
            f"Your goal is to write a research paper as well as possible. You will receive a score after you write the paper and should aim to maximize the score by writing a high quality research paper. The paper length should be 8 pages or 4000 words in total. It should be quite long and comprehensive. Remember, the paper MUST BE LONG. {paper_progress}\n"
            # COMMAND SET
            f"{cmd_set}\n"
            # PAPER
            f"Provided here is your current paper {self.generate_paper_lines(self.paper_lines)}"
            # optional section command
            f"{section_cmd}"
        )

    def command_descriptions(self):
        """
        提供系统可执行命令的说明和格式要求。
        @return: (str) command descriptions
        """
        cmd_strings = "\n".join([_cmd.docstring() for _cmd in self.commands])
        return f"\nYou also have access to tools which can be interacted with using the following structure: ```COMMAND\n<command information here>\n```, where COMMAND is whichever command you want to run (e.g. EDIT,...), <command information here> is information used for the command and ``` are meant to encapsulate the command. ``` must be included as part of the command both at the beginning and at the end of the command. DO NOT FORGOT TO HAVE ``` AT THE TOP AND BOTTOM OF COMMAND. and this structure must be followed to execute a command correctly. YOU CAN ONLY EXECUTE A SINGLE COMMAND AT A TIME! Do not try to perform multiple commands EVER only one." + cmd_strings

    def role_description(self):
        """
        提供角色设定描述。
        @return: (str) role description
        """
        return "You are a bioinformatics Academic Writing Specialist at a top university who has submitted their paper to a leading journal in the field called Bioinformatics. Your goal was to write a high-quality research article and receive favorable reviews so that it gets accepted for publication. Your manuscript should be approximately 8 pages and around 4000 words. It should ONLY CONTAIN the following eight sections, adhering to academic standards:1. Abstract, 2. Introduction, 3.Related Work, 4.Methods, 5.Experiments, 6. Results, 7.Conclusion and 8.References \n"
        # "You are a computer science PhD student at a top university who has submitted their paper to an ML conference called ICLR. Your goal was to write a research paper and get high scores from the reviewers so that it get accepted to the conference. Your paper should be approximately 8 pages and around 4000 words. Your article should ONLY CONTAIN EIGHT sections as follows: 1. Abstract 2. Introduction, 3. Background, 4. Related Work 5. Methods, 6. Experimental Setup 7. Results, and 8. Discussion.\n"

    def phase_prompt(self, ):
        """
        Describe system role and general tips for paper-solver
        描述系统当前在论文写作阶段的角色和目标。
        @return: (str) system role
        """
        phase_str = (
            "You are a Academic Writing Specialist who has submitted a paper to a bioinformatics journal called Bioinformatics. Your goal was to write a research paper and get high scores from the reviewers so that it get accepted to the journal.\n"
        )
        return phase_str
