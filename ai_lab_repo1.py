import ast

import PyPDF2
import threading
from app import *
from agents import *
from copy import copy
from pathlib import Path
from datetime import date
from common_imports import *
from mlesolver1 import MLESolver
import argparse, pickle, yaml
import json

GLOBAL_AGENTRXIV = None
DEFAULT_LLM_BACKBONE = "o1-mini"
RESEARCH_DIR_PATH = f"second_experiment/{DEFAULT_LLM_BACKBONE}"

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LaboratoryWorkflow:
    def __init__(self, research_topic, openai_api_key, max_steps=100, num_papers_lit_review=5,
                 agent_model_backbone=f"{DEFAULT_LLM_BACKBONE}", notes=list(), human_in_loop_flag=None,
                 compile_pdf=True, mlesolver_max_steps=3, papersolver_max_steps=3, paper_index=0,
                 parallelized=False, lab_dir=None,  agentRxiv=False, agentrxiv_papers=5,population="European"):
        """
        Initialize laboratory workflow
        @param research_topic: (str) description of research idea to explore
        @param max_steps: (int) max number of steps for each phase, i.e. compute tolerance budget
        @param num_papers_lit_review: (int) number of papers to include in the lit review
        @param agent_model_backbone: (str or dict) model backbone to use for agents
        @param notes: (list) notes for agent to follow during tasks
        """
        self.agentRxiv = agentRxiv
        self.max_prev_papers = 10
        self.parallelized = parallelized
        self.notes = notes
        self.lab_dir = lab_dir
        # lab_index = 0,
        # self.lab_index = lab_index
        self.max_steps = max_steps
        self.compile_pdf = compile_pdf
        self.paper_index = paper_index
        self.openai_api_key = openai_api_key

        # self.except_if_fail = except_if_fail
        # except_if_fail
        self.research_topic = research_topic
        self.population = population
        self.model_backbone = agent_model_backbone
        self.num_papers_lit_review = num_papers_lit_review

        self.print_cost = True
        self.review_override = True  # should review be overridden?
        self.review_ovrd_steps = 0  # review steps so far
        self.PubMed_paper_exp_time = 3
        self.reference_papers = list()

        self.exposure_file_path = None
        self.outcome_file_path = None
        self.exposure_trait = None
        self.outcome_trait = None
        ##########################################
        ####### COMPUTE BUDGET PARAMETERS ########
        ##########################################
        self.num_ref_papers = 1
        self.review_total_steps = 0  # num steps to take if overridden
        self.PubMed_num_summaries = 20
        self.num_agentrxiv_papers = agentrxiv_papers
        self.mlesolver_max_steps = mlesolver_max_steps
        self.papersolver_max_steps = papersolver_max_steps

        self.phases = [
            ("plan formulation", ["plan formulation"]),
            ("data preparation", ["data preparation"]),
            ("running experiments", ["running experiments"]),
            ("results interpretation", ["results interpretation"]),
            ("report writing", ["report writing"]),
            ("report refinement", ["report refinement"]),
        ]
        self.phase_status = dict()
        for phase, subtasks in self.phases:
            for subtask in subtasks:
                self.phase_status[subtask] = False

        self.phase_models = dict()
        if type(agent_model_backbone) == str:
            for phase, subtasks in self.phases:
                for subtask in subtasks:
                    self.phase_models[subtask] = agent_model_backbone
        elif type(agent_model_backbone) == dict:
            # todo: check if valid
            self.phase_models = agent_model_backbone

        self.human_in_loop_flag = human_in_loop_flag

        self.statistics_per_phase = {
            "literature review": {"time": 0.0, "steps": 0.0, },
            "plan formulation": {"time": 0.0, "steps": 0.0, },
            "data preparation": {"time": 0.0, "steps": 0.0, },
            "running experiments": {"time": 0.0, "steps": 0.0, },
            "results interpretation": {"time": 0.0, "steps": 0.0, },
            "report writing": {"time": 0.0, "steps": 0.0, },
            "report refinement": {"time": 0.0, "steps": 0.0, },
        }

        self.save = True
        self.verbose = True
        self.reviewers = ReviewersAgent(model=self.model_backbone, notes=self.notes, openai_api_key=self.openai_api_key)
        self.biological_engineer = BiologicalEngineer(model=self.model_backbone, notes=self.notes,
                                                      max_steps=self.max_steps,
                                                      openai_api_key=self.openai_api_key)
        self.statistical_geneticist = StatisticalGeneticist(model=self.model_backbone, notes=self.notes,
                                                            max_steps=self.max_steps,
                                                            openai_api_key=self.openai_api_key)
        self.writing_specialist = WritingSpecialist(model=self.model_backbone, notes=self.notes,
                                                    max_steps=self.max_steps,
                                                    openai_api_key=self.openai_api_key)
        self.algorithm_engineer = AlgorithmEngineer(model=self.model_backbone, notes=self.notes,
                                                    max_steps=self.max_steps,
                                                    openai_api_key=self.openai_api_key)
        self.data_analyst = DataAnalyst(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps,
                                        openai_api_key=self.openai_api_key)

    def set_model(self, model):
        self.set_agent_attr("model", model)
        self.reviewers.model = model

    def save_state(self, phase):
        """
        Save state for phase
        @param phase: (str) phase string
        @return: None
        """
        with open(f"state_saves/Paper{self.paper_index}.pkl", "wb") as f:
            pickle.dump(self, f)

    def set_agent_attr(self, attr, obj):
        """
        Set attribute for all agents
        @param attr: (str) agent attribute
        @param obj: (object) object attribute
        @return: None
        """
        setattr(self.biological_engineer, attr, obj)
        setattr(self.statistical_geneticist, attr, obj)
        setattr(self.writing_specialist, attr, obj)
        setattr(self.algorithm_engineer, attr, obj)
        setattr(self.data_analyst, attr, obj)

    def reset_agents(self):
        """
        Reset all agent states
        @return: None
        """
        pass

    def perform_research(self):
        """
        Loop through all research phases
        @return: None
        """
        for phase, subtasks in self.phases:
            phase_start_time = time.time()  # Start timing the phase
            if self.verbose:
                print(f"{'*' * 50}\nBeginning phase: {phase}\n{'*' * 50}")
            for subtask in subtasks:
                # if self.agentRxiv:
                #     if self.verbose:
                #       print(f"{'&' * 30}\n[Lab #{self.lab_index} Paper #{self.paper_index}] Beginning subtask: {subtask}\n{'&' * 30}")
                # else:
                #     if self.verbose:
                        #print(f"{'&' * 30}\nBeginning subtask: {subtask}\n{'&' * 30}")
                if type(self.phase_models) == dict:
                    if subtask in self.phase_models:
                        self.set_model(self.phase_models[subtask])
                    else:
                        self.set_model(f"{DEFAULT_LLM_BACKBONE}")
                # if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "literature review":
                #     repeat = True
                #     while repeat: repeat = self.literature_review()
                #     self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[
                    subtask]) and subtask == "plan formulation":
                    repeat = True
                    while repeat: repeat = self.plan_formulation()
                    self.phase_status[subtask] = True

                # # 如果当前子任务是 "plan formulation"，并且还没有被标记为完成
                # if subtask == "plan formulation" and not self.phase_status.get(subtask, False):
                #     # 重复执行 plan_formulation()，直到其返回 False
                #     while True:
                #         should_continue = self.plan_formulation()
                #         if not should_continue:
                #             break
                #
                #     # 标记该子任务为已完成
                #     self.phase_status[subtask] = True

                if (subtask not in self.phase_status or not self.phase_status[
                    subtask]) and subtask == "data preparation":
                    repeat = True
                    while repeat: repeat = self.data_preparation()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[
                    subtask]) and subtask == "running experiments":
                    repeat = True
                    while repeat: repeat = self.running_experiments()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[
                    subtask]) and subtask == "results interpretation":
                    repeat = True
                    while repeat: repeat = self.results_interpretation()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "report writing":
                    repeat = True
                    while repeat: repeat = self.report_writing()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[
                    subtask]) and subtask == "report refinement":
                    return_to_exp_phase = self.report_refinement()

                    if not return_to_exp_phase:
                        if self.save: self.save_state(subtask)
                        return
                pass

    def report_refinement(self):
        """
        Perform report refinement phase
        @return: (bool) whether to repeat the phase
        """
        pass
    def report_writing(self):
        """
        Perform report writing phase
        @return: (bool) whether to repeat the phase
        """
        # 从 algorithm_engineer.notes 中提取属于 "report writing" 阶段的笔记
        report_notes = [_note["note"] for _note in self.algorithm_engineer.notes if "report writing" in _note["phases"]]
        # 如果有笔记则拼接成字符串，否则为空
        report_notes = f"Notes for the task objective: {report_notes}\n" if len(report_notes) > 0 else ""

        # 导入 PaperSolver 并初始化参考文献列表
        # from papersolver import PaperSolver
        from paper_writing import PaperSolver

        self.reference_papers = []

        # 初始化 PaperSolver，用于生成论文报告
        solver = PaperSolver(
            notes=report_notes,  # 实验相关笔记
            max_steps=self.papersolver_max_steps,  # 最大迭代步数
            plan=self.biological_engineer.plan,  # 实验计划
            exp_code=self.biological_engineer.results_code,  # 实验代码
            exp_results=self.biological_engineer.exp_results,  # 实验结果
            insights=self.biological_engineer.interpretation,  # 结果解读
            lit_review=self.biological_engineer.lit_review,  # 文献综述
            datasets_information=self.biological_engineer.dataset_information,  # 数据集信息
            ref_papers=self.reference_papers,  # 参考文献（初始化为空）
            topic=research_topic,  # 研究主题
            openai_api_key=self.openai_api_key,  # OpenAI API key
            llm_str=self.model_backbone["report writing"],  # 使用的模型
            compile_pdf=compile_pdf,  # 是否编译 PDF
            save_loc=self.lab_dir  # 保存路径
        )

        # 执行初始解算
        solver.initial_solve()
        pass

    def results_interpretation(self):
        """
        Perform results interpretation phase
        @return: (bool) whether to repeat the phase
        """
        max_tries = self.max_steps
        dialogue = str()
        # iterate until max num tries to complete task is exhausted
        for _i in range(max_tries):
            # print(f"@@ Lab #{self.lab_index} Paper #{self.paper_index} @@")
            resp = self.statistical_geneticist.inference(self.research_topic, "results interpretation",
                                                         feedback=dialogue, step=_i)
            if self.verbose: print("statistical geneticist : ", resp, "\n~~~~~~~~~~~")
            dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the statistical geneticist : {dialogue}"
                if self.verbose: print("#" * 40, "\n", "statistical geneticist  Dialogue:", dialogue, "\n", "#" * 40)
            if "```INTERPRETATION" in resp:
                interpretation = extract_prompt(resp, "INTERPRETATION")
                if self.human_in_loop_flag["results interpretation"]:
                    retry = self.human_in_loop("results interpretation", interpretation)
                    if retry: return retry
                save_to_file(f"./{self.lab_dir}", "interpretation.txt", interpretation)
                self.set_agent_attr("interpretation", interpretation)
                # reset agent state
                self.reset_agents()
                self.statistics_per_phase["results interpretation"]["steps"] = _i
                return False
            resp = self.biological_engineer.inference(self.research_topic, "results interpretation", feedback=dialogue,
                                                      step=_i)
            if self.verbose: print("biological engineer: ", resp, "\n~~~~~~~~~~~")
            dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the biological engineer: {dialogue}"
                if self.verbose: print("#" * 40, "\n", "biological engineer:", dialogue, "#" * 40, "\n")
        raise Exception("Max tries during phase: Results Interpretation")

    def running_experiments(self):
        """
        执行“运行实验”阶段
        @return: (bool) 是否需要重复该阶段
        """
        # 获取与“running experiments”阶段相关的注释（notes）
        experiment_notes = [_note["note"] for _note in self.algorithm_engineer.notes if
                            "running experiments" in _note["phases"]]
        # 如果有相关注释，则整理为字符串形式
        experiment_notes = f"Notes for the task objective: {experiment_notes}\n" if len(experiment_notes) > 0 else ""
        cfg = ExperimentConfig()

        Original_exposure_data_path = cfg.get("data_preparation", "Original_exposure_data_path")
        Original_outcome_data_path = cfg.get("data_preparation", "Original_outcome_data_path")
        # 初始化 MLESolver（用于执行机器学习实验）
        solver = MLESolver(
            dataset_code=self.algorithm_engineer.dataset_code,  # 使用的数据集代码
            notes=experiment_notes,  # 上一步整理的实验注释
            insights=self.algorithm_engineer.lit_review_sum,  # 文献综述总结
            max_steps=self.mlesolver_max_steps,  # 最大实验迭代步数
            plan=self.algorithm_engineer.plan,  # 当前实验计划
            # Original_exposure_data_path=self.algorithm_engineer.Original_exposure_data_path,
            # Original_outcome_data_path=self.algorithm_engineer.Original_outcome_data_path,
            Original_exposure_data_path=Original_exposure_data_path,
            Original_outcome_data_path=Original_outcome_data_path,
            openai_api_key=self.openai_api_key,  # API 密钥
            llm_str=self.model_backbone["running experiments"]  # 使用的语言模型类型
        )

        # 执行初始求解步骤（初始化模型和实验流程）
        solver.initial_solve()
        pass

    def data_preparation(self):
        """
        Perform data preparation phase
        @return: (bool) whether to repeat the phase
        """
        global outcome_information, exposure_information
        max_tries = self.max_steps
        ae_feedback = str()
        ae_dialogue = str()
        da_feedback = str()
        ae_command = str()
        # GWAS Catalog 数据搜索
        GWAS_engine = GWASCatalogSearch()
        GWAS_trait_search = TraitMatcher()
        # GWAS Catalog 数据下载
        GWAS_download = GWASLoaderTool()
        cfg = ExperimentConfig()

        # iterate until max num tries to complete task is exhausted
        for _i in range(max_tries):
            # print(f"@@ Lab #{self.lab_index} Paper #{self.paper_index} @@")
            if ae_feedback != "":
                ae_feedback_in = "Feedback provided to the algorithm engineer: " + ae_feedback
            else:
                ae_feedback_in = ""
            resp = self.algorithm_engineer.inference(self.research_topic, "data preparation",
                                                     feedback=f"{ae_dialogue}\nFeedback from previous command: {da_feedback}\n{ae_command}{ae_feedback_in}",
                                                     step=_i)
            da_feedback = str()
            da_dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                da_dialogue = f"\nThe following is dialogue produced by the data analyst: {dialogue}\n"
                if self.verbose: print("#" * 40,
                                       f"\nThe following is dialogue produced by the data analyst: {dialogue}",
                                       "\n", "#" * 40)
            if "```SUBMIT_CODE" in resp:
                final_code = extract_prompt(resp, "SUBMIT_CODE")
                code_resp = execute_r_code(final_code, timeout=600)
                if self.verbose: print("!" * 100, "\n", f"CODE RESPONSE: {code_resp}")
                da_feedback += f"\nCode Response: {code_resp}\n"
                #combined_datasets_info = exposure_information + "\n\n" + outcome_information
                # print(combined_datasets_info)
                if "[CODE EXECUTION ERROR]" in code_resp:
                    da_feedback += "\nERROR: Final code had an error and could not be submitted! You must address and fix this error.\n"
                else:
                    if self.human_in_loop_flag["data preparation"]:
                        retry = self.human_in_loop("data preparation", final_code)
                        if retry: return retry
                    save_to_file(f"./{self.lab_dir}/code", "load_data.R", final_code)
                    self.set_agent_attr("dataset_code", final_code)
                    # self.set_agent_attr("dataset_information", combined_datasets_info)
                    self.set_agent_attr("Original_exposure_data_path", self.Original_exposure_data_path)
                    self.set_agent_attr("Original_outcome_data_path", self.Original_outcome_data_path)
                    # reset agent state
                    self.reset_agents()
                    self.statistics_per_phase["data preparation"]["steps"] = _i
                    return False

            if ae_feedback != "":
                ae_feedback_in = "Feedback from previous command: " + ae_feedback
            else:
                ae_feedback_in = ""
            resp = self.data_analyst.inference(
                self.research_topic, "data preparation",
                feedback=f"{da_dialogue}\n{ae_feedback_in}", step=_i)
            if self.verbose: print("data analyst: ", resp, "\n~~~~~~~~~~~")
            ae_feedback = str()
            ae_dialogue = str()
            ae_command = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                ae_dialogue = f"\nThe following is dialogue produced by the algorithm engineer: {dialogue}\n"
                if self.verbose: print("#" * 40,
                                       f"\nThe following is dialogue produced by the algorithm engineer: {dialogue}",
                                       "#" * 40, "\n")
            if "```R" in resp:
                code = extract_prompt(resp, "R")
                code = self.data_analyst.dataset_code + "\n" + code
                code_resp = execute_r_code(code, timeout=600)
                ae_command = f"Code produced by the ML agent:\n{code}"
                ae_feedback += f"\nCode Response: {code_resp}\n"
                if self.verbose: print("!" * 100, "\n", f"CODE RESPONSE: {code_resp}")
            if "```SEARCH_GWAS" in resp:
                # 判断是否下载好
                is_download = False
                hf_query = extract_prompt(resp, "SEARCH_GWAS")
                # 提取纯trait关键词，去掉_exposure或_outcome后缀
                if hf_query.endswith("_exposure"):
                    trait_keyword = hf_query[:-len("_exposure")]
                    data_type = "exposure"
                elif hf_query.endswith("_outcome"):
                    trait_keyword = hf_query[:-len("_outcome")]
                    data_type = "outcome"
                else:
                    trait_keyword = hf_query
                    data_type = "unknown"

                print(f"这是GWAS Catalog数据库的原始检索关键词：{trait_keyword}")
                # 根据原始搜索词返回5个搜索词
                hf_query_last_list = GWAS_trait_search.find_similar_traits(trait_keyword)
                # 问题：搜索词由智能体给出，主题词不变的情况下智能体给出的搜索词经常不一样，需要根据原始关键词进行关键词优化
                for hf_query_last in hf_query_last_list:
                    print(f"这是GWAS Catalog数据库的最终检索关键词：{hf_query_last}")
                    if data_type == "exposure":
                        filter_results = GWAS_engine._filter_and_sort_data(hf_query_last, data_type, population)
                        # 如果 filter_results 为空，说明没有检索到相关的研究，需要换下一个检索词进行检索
                        if filter_results is None or filter_results.empty:
                            continue
                        self.exposure_trait = hf_query_last
                        result_strs, down_list = GWAS_engine.results_str(filter_results,population)

                        # 修改：检索返回不止一个研究，依次判断是否公开数据集
                        # down_list 是包含下载链接的列表，如果 down_list 为空, 说明该检索词检索发现没有可下载的数据集，则跳过当前检索词，考虑下一个检索词
                        if len(down_list) == 0:
                            continue
                        hf_res = "\n".join(result_strs)
                        total_num = len(result_strs)
                        print(f"共检索到{total_num}个研究")
                        # print(f"共检索到{total_num}个研究，这是GWAS Catalog数据库中暴露数据的检索结果：{hf_res}")
                        exposure_information = hf_res.strip().split('-------------------------')[0]
                        # file_path 是文件下载之后的存储路径
                        self.Original_exposure_data_path = GWAS_download.run(down_list[0])
                        # 提取暴露数据的样本量sample_size.用正则匹配 Sample description 行
                        match = re.search(r"Sample description:\s*([\d,]+)", exposure_information)
                        # 用正则匹配 Sample description 行
                        if match:
                            sample_size_str = match.group(1)
                            # 去掉逗号，转成整数
                            sample_size = int(sample_size_str.replace(",", ""))
                            print("样本量:", sample_size)
                        else:
                            print("未找到样本量")

                        cfg.set(
                            "data_preparation",
                            "exposure_data_information",
                            json.dumps(exposure_information, indent=4)  # 保持字典结构
                        )

                        # 将原始数据的保存路径传下来（实验阶段）
                        # Original_exposure_data_path = file_path
                        # self.set_agent_attr("Original_exposure_data_path", file_path)
                        if not self.Original_exposure_data_path:
                            continue
                        is_download = True
                        cfg.set("data_preparation", "search_query_of_exposure", hf_query_last)
                        cfg.set("data_preparation", "original_exposure_data_path", self.Original_exposure_data_path)
                        print(f"当前下载的是暴露变量 [{hf_query_last}] 的数据，已保存至 [{self.Original_exposure_data_path}]")
                        # 对数据进行列名映射
                        # 对暴露数据筛选显著工具变量
                        exposure_data,used_thresh  = filter_significant_mutations(self.Original_exposure_data_path)
                        print(f"筛选显著变异，实际使用阈值: {used_thresh}")
                        exposure_data_path= "research_results/processed_data/exposure_data.csv"
                        exposure_data.to_csv(exposure_data_path, index=False)

                        self.exposure_file_path = exposure_data_path
                        print(f"已对暴露数据筛选完显著snps，已保存至 [{self.exposure_file_path}]")
                        # snps为0怎么办
                        # 生成snps列表
                        possible_cols = ["rsid", "hm_rsid", "variant_id"]
                        col_name = next((c for c in possible_cols if c in exposure_data.columns), None)

                        if col_name is None:
                            raise ValueError("data中未找到hm_rsid、rsid或variant_id字段")
                        self.exposure_data_rsid = exposure_data[[col_name]].drop_duplicates().rename(columns={col_name: "hm_rsid"})
                        self.exposure_data_rsid_path= "research_results/processed_data/exposure_data_rsid.csv"
                        self.exposure_data_rsid.to_csv(self.exposure_data_rsid_path, index=False)
                        ae_command = f"Exposure GWAS search command produced by the ML agent:\n{hf_query}"
                        ae_feedback += (
                            f"Exposure GWAS Catalog dataset of {hf_query_last} has been successfully downloaded.The exposure data has been saved at the following path:{self.Original_exposure_data_path}.\n"
                            f"Significant variants were selected using a threshold of {used_thresh},the storage path is: {self.exposure_file_path}.And generate a list included only the rsid,which is stored in the{self.exposure_data_rsid_path} \n"
                            f" Do not apply p-value filtering again — these SNPs are already genome-wide significant.\n"
                            f"Next, generate a search term to search the outcome data.\n"
                            # "Please use the LD_PRUNE command to perform LD pruning. You can set specific values for r2 and window_kb based on the basic characteristics of the dataset.\n"
                            # "```LD_PRUNE\n{'r2': 0.1, 'window_kb': 500}\n```\n"
                        )
                    elif data_type == "outcome":
                        filter_results = GWAS_engine._filter_and_sort_data(hf_query_last, data_type, population)
                        # 如果 filter_results 为空，说明没有检索到相关的研究，需要换下一个检索词进行检索
                        if filter_results is None or filter_results.empty:
                            continue
                        self.outcome_trait = hf_query_last
                        result_strs, down_list = GWAS_engine.results_str(filter_results,population)
                        # 修改：检索返回不止一个研究，依次判断是否公开数据集
                        # down_list 是包含下载链接的列表，如果 down_list 为空, 说明该检索词检索发现没有可下载的数据集，则跳过当前检索词，考虑下一个检索词
                        if len(down_list) == 0:
                            continue
                        hf_res = "\n".join(result_strs)
                        total_num = len(result_strs)
                        print(f"共检索到{total_num}个研究")

                        # print(f"共检索到{total_num}个研究，这是GWAS Catalog数据库中结局数据的检索结果：{hf_res}")
                        exposure_outcome_intersection_snps_path = "research_results/processed_data/exposure_outcome_intersection_snps.csv"
                        last_outcome_data_path = "research_results/processed_data/last_outcome_data.csv"
                        is_first_write = True if not os.path.exists(exposure_outcome_intersection_snps_path) else False
                        for num, url in enumerate(down_list):
                            outcome_information = hf_res.strip().split('-------------------------')[num]
                            # print(f"取第{num + 1}个研究的数据集，验证性状是否符合用户要求")
                            print(f"第{num + 1}个研究信息是：{outcome_information}")
                            self.outcome_data_path = GWAS_download.run(url)
                            self.Original_outcome_data_path = self.outcome_data_path
                            is_download = True
                            cfg.set("data_preparation", "search_query_of_outcome", hf_query_last)
                            cfg.set("data_preparation", "original_outcome_data_path", self.outcome_data_path)
                            cfg.set(
                                "data_preparation",
                                "outcome_data_information",
                                json.dumps(outcome_information, indent=4)  # 保持字典结构
                            )
                            print(f"当前下载的是结局变量 [{hf_query_last}] 的数据，已保存至 [{self.outcome_data_path}]")

                            try:
                                data = pd.read_csv(self.outcome_data_path, sep='\t', compression='infer', low_memory=False)
                            except Exception as e:
                                print(f"读取失败: {e}")
                                continue

                            possible_cols = ["hm_rsid", "rsid", "variant_id"]
                            col_name = next((c for c in possible_cols if c in data.columns), None)

                            if col_name is None:
                                raise ValueError("data中未找到hm_rsid、rsid或variant_id字段")

                            temp_filter_data = data[[col_name]].drop_duplicates().rename(columns={col_name: "hm_rsid"})
                            #temp_exposure_data_rsid = self.exposure_data_rsid[['rsid']].drop_duplicates()
                            len_exposure_data_rsid = len(self.exposure_data_rsid)
                            self.exposure_data_rsid_temp = pd.read_csv(self.exposure_data_rsid_path)
                            self.exposure_data_rsid_temp.columns = ['hm_rsid']
                            # self.exposure_data_rsid = self.exposure_data_rsid.rename(columns={'rsid': 'hm_rsid'})
                            df_common = pd.merge(temp_filter_data, self.exposure_data_rsid_temp, how='inner',on='hm_rsid')
                            # filtered_rows  (8896,1)
                            num_rows = df_common.shape[0]
                            print(f"本次取交集有 {num_rows} 行数据，来源文件: {self.outcome_data_path}")

                            data = data.rename(columns={col_name: "hm_rsid"})
                            filtered_rows = pd.merge(data, df_common, on='hm_rsid', how='inner')
                            # filtered_rows  (8919,24)
                            if df_common.empty:
                                continue

                            # 添加标记列（本地文件地址）
                            # df_common['source_file'] = self.outcome_data_path
                            # 写入结果，追加方式
                            df_common.to_csv(
                                exposure_outcome_intersection_snps_path,
                                mode='a',
                                header=is_first_write,
                                index=False
                            )
                            filtered_rows.to_csv(
                                last_outcome_data_path,
                                mode='a',
                                header=is_first_write,
                                index=False
                            )
                            is_first_write = False  # 写入一次后不再写表头
                            num_rows = df_common.shape[0]
                            print(f"追加后交集有 {num_rows} 行数据")
                            if num_rows < 1000 and num_rows < len_exposure_data_rsid:
                                continue
                            else:
                                break
                        ae_command = f"Outcome GWAS search command produced by the ML agent:\n{hf_query}"
                        ae_feedback += (f"Outcome GWAS Catalog dataset of {hf_query_last} has been successfully downloaded.The outcome data has been saved at the following path: {self.outcome_data_path}. "
                                        f"Has taken the intersection of the snps list of the exposed data and the snps list of the ending data, and the intersection snps list is saved in the{exposure_outcome_intersection_snps_path}.\n"
                                        f"Next you need to perform LD pruning on the intersection snps list,Please use the LD_PRUNE command to perform LD pruning. You can set specific values for r2 and window_kb based on the basic characteristics of the dataset.\n"
                                        )
                    else:
                        print("无法确定数据类型，无法继续。")

                    break
                # 返回的5个搜索词都失效，则让智能体换一个原始搜索词
                if not is_download:
                    ae_command = f"The previous search query generated by the ML agent was: {hf_query}. Unfortunately, it returned no relevant results. Please generate a new, more effective search query that is likely to yield relevant information. \n"
                    ae_feedback += f"No relevant dataset download,lease generate a new, more effective search query that is likely to yield relevant information.\n"



            if "```LD_PRUNE" in resp:
                # 从resp里提取剪枝参数JSON
                # prune_params_str = extract_prompt(resp, "LD_PRUNE")
                # print("prune_params_str:", prune_params_str)
                # prune_params = ast.literal_eval(prune_params_str)
                # print("prune_params:", prune_params)
                # 执行LD剪枝
                # if 'exposure_file_path' in locals():
                exposure_outcome_intersection_snps_path = "research_results/processed_data/exposure_outcome_intersection_snps.csv"
                snps_data = pd.read_csv(exposure_outcome_intersection_snps_path)
                snps_list = snps_data['hm_rsid']
                if len(snps_list) != 0:
                    # print(f"开始对snps列表进行LD剪枝，参数：{prune_params}")
                    print(f"开始对snps列表进行LD剪枝")

                    # 如果 prune_params 字典中有键 'r2'，就使用它的值；否则，使用默认值 0.1。
                    agent = LDPrunerAgent(
                        gwas_file=exposure_outcome_intersection_snps_path,
                        # pval_threshold=5e-8,
                        output_dir="./research_results/ld_processed_data",
                        r2 = 0.001,
                        window_kb = 10000
                        # r2=prune_params.get('r2', 0.001),
                        # window_kb=prune_params.get('window_kb', 10000)
                    )
                    result = agent.command_run_auto({
                        "gwas_file": exposure_outcome_intersection_snps_path,
                        "min_snps_threshold": 10,
                        "max_attempts": 4,
                        "step": 10,
                        "r2": 0.001,
                        "window_kb": 10000
                        # "r2": prune_params.get('r2'),
                        # "window_kb": prune_params.get('window_kb')
                    })
                    # print("剪枝结果文件:", result["pruned_gwas_file"])
                    print("使用的r2参数:", result["r2_used"])
                    print("使用的window_kb:", result["window_kb_used"])
                    print("剪枝后SNP数量:", result["snp_count"])

                    # 读取剪枝后的 SNP 列表，确保列名为 'rsid'
                    ld_pruned_snps = pd.read_csv("research_results/ld_processed_data/ld_pruned.prune.in",
                                                 names=['rsid'])

                    # 加载暴露数据和结局数据
                    exposure_df = pd.read_csv("research_results/processed_data/exposure_data.csv")
                    outcome_df = pd.read_csv("research_results/processed_data/last_outcome_data.csv")

                    # 优先顺序：rsid > hm_rsid > variant_id
                    column_candidates = ['rsid', 'hm_rsid', 'variant_id']

                    # 标准化列名,自动识别主列名并标准化为 'rsid'
                    exposure_df = standardize_rsid_column(exposure_df, column_candidates)
                    outcome_df = standardize_rsid_column(outcome_df, column_candidates)

                    # 进行 merge（内连接）
                    filtered_exposure_data = pd.merge(exposure_df, ld_pruned_snps, on="rsid", how="inner")
                    filtered_outcome_data = pd.merge(outcome_df, ld_pruned_snps, on="rsid", how="inner")

                    # 这里新增一步，列名映射，生成TwoSampleMR需要的标准列名
                    def map_columns_for_exposure(df):
                        return pd.DataFrame({
                            'SNP': df['rsid'],
                            'id.exposure': self.exposure_trait,  # 这里可以根据具体暴露变量名替换
                            'exposure': self.exposure_trait,
                            'beta.exposure': df['beta'],  # 请确保df中有对应列
                            'se.exposure': df['standard_error'],  # 请确保df中有对应列
                            'effect_allele.exposure': df['effect_allele'],
                            'other_allele.exposure': df['other_allele'],
                            'eaf.exposure': df.get('effect_allele_frequency'),  # 如果没有可用get防止报错
                            'pval.exposure': df['p_value']
                        })

                    def map_columns_for_outcome(df):
                        return pd.DataFrame({
                            'SNP': df['rsid'],
                            'id.outcome': self.outcome_trait,  # 这里可以根据具体结局变量名替换
                            'outcome': self.outcome_trait,
                            'beta.outcome': df['beta'],
                            'se.outcome': df['standard_error'],
                            'effect_allele.outcome': df['effect_allele'],
                            'other_allele.outcome': df['other_allele'],
                            'eaf.outcome': df.get('effect_allele_frequency'),
                            'pval.outcome': df['p_value']
                        })

                    # 映射列名
                    filtered_exposure_data_mapped = map_columns_for_exposure(filtered_exposure_data)
                    filtered_outcome_data_mapped = map_columns_for_outcome(filtered_outcome_data)

                    # 对暴露数据计算F-statistic,F值通常要 > 10 进一步筛选
                    # 这里的N是样本量，也就是原始 GWAS 研究中对该性状测量的人数，而不是SNP的数量
                    N = sample_size
                    exposure_df_all, exposure_df_filtered = add_r2_fstat_filter(filtered_exposure_data_mapped, N)
                    print("\n=== 筛选后 SNP（F > 10） ===")
                    print(exposure_df_filtered[["SNP", "beta.exposure", "se.exposure", "eaf.exposure", "R2", "F_stat"]])
                    # 保存筛选后的snp列表
                    exposure_data_rsid = exposure_df_filtered[["SNP"]]
                    exposure_data_rsid_path = "research_results/processed_data/exposure_data_filtered_rsid.csv"
                    exposure_data_rsid.to_csv(exposure_data_rsid_path, index=False)

                    filtered_outcome_data = pd.merge(filtered_outcome_data_mapped, exposure_data_rsid, how='inner',
                                                     on='SNP')
                    # 保存文件，供后续R读取
                    filtered_exposure_data_path = "research_results/processed_data/filtered_exposure_data.csv"
                    filtered_outcome_data_path = "research_results/processed_data/filtered_outcome_data.csv"
                    cfg.set("data_preparation", "filtered_exposure_data_path", filtered_exposure_data_path)
                    cfg.set("data_preparation", "filtered_outcome_data_path", filtered_outcome_data_path)
                    if len(exposure_data_rsid) >= 10:
                        exposure_df_filtered.to_csv(filtered_exposure_data_path, index=False)
                        filtered_outcome_data.to_csv(filtered_outcome_data_path, index=False)
                    else:
                        print("警告: 筛选F>10的SNP之后，SNP 数量不足 10，保存原始映射数据")
                        filtered_exposure_data_mapped.to_csv(filtered_exposure_data_path, index=False)
                        filtered_outcome_data_mapped.to_csv(filtered_outcome_data_path, index=False)

                    ae_command = f"LD pruning command produced by the AE agent"
                    ae_feedback += (
                        f"The intersection snps list has been successfully pruned\n"
                        f"After LD pruning, the pruned GWAS file is stored at: {result['pruned_gwas_file']}\n"
                        f"Pruning parameters used - r²: {result['r2_used']}, window size: {result['window_kb_used']} kb\n"
                        f"SNPs remaining after pruning: {result['snp_count']}\n"
                        f"Column name mapping has been completed, and the standard column names required by TwoSampleMR have been generated.\n"
                        f"The filtering of exposure data and outcome data has been completed based on the pruning results, and the filtered exposure data is saved in {filtered_exposure_data_path}, and the filtered outcome data is saved in {filtered_outcome_data_path}\n"
                    )
                    if not result["completed"]:
                        ae_feedback += f"Warning: LD pruning was not completed successfully. Reason: {result['warning']}\n"
                        print("警告:", result["warning"])
                else:
                    print("没有找到暴露数据文件路径，无法进行LD剪枝。")

        raise Exception("Max tries during phase: Data Preparation")

    def plan_formulation(self):
        """
        Perform plan formulation phase
        @return: (bool) whether to repeat the phase
        """
        max_tries = self.max_steps
        dialogue = str()
        # iterate until max num tries to complete task is exhausted
        for _i in range(max_tries):
            #print(f"@@ Lab #{self.lab_index} Paper #{self.paper_index} @@")
            resp = self.statistical_geneticist.inference(self.research_topic, "plan formulation", feedback=dialogue,
                                                         step=_i)
            if self.verbose: print("statistical geneticist : ", resp, "\n~~~~~~~~~~~")
            dialogue = str()

            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the statistical geneticist : {dialogue}"
                if self.verbose: print("#" * 40, "\n", "statistical geneticist  Dialogue:", dialogue, "\n", "#" * 40)

            # if "```METHOD" in resp:
            #     methods_tools = extract_prompt(resp, "METHOD")
            #     save_dir = "research_results/results"
            #     os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
            #     save_path = os.path.join(save_dir, "methods_tools.yaml")
            #     with open(save_path, "w", encoding="utf-8") as f:
            #         f.write(methods_tools)
            #     print(f"methods_tools.yaml 文件已保存到 {save_path}。")

            if "```PLAN" in resp:
                plan = extract_prompt(resp, "PLAN")
                # print (plan)
                if self.human_in_loop_flag["plan formulation"]:
                    retry = self.human_in_loop("plan formulation", plan)
                    if retry: return retry
                save_to_file(f"./{RESEARCH_DIR_PATH}", "plan.txt", plan)
                self.set_agent_attr("plan", plan)
                # reset agent state
                self.reset_agents()
                self.statistics_per_phase["plan formulation"]["steps"] = _i
                return False

            resp = self.biological_engineer.inference(self.research_topic, "plan formulation", feedback=dialogue,
                                                      step=_i)
            if self.verbose: print("biological engineer: ", resp, "\n~~~~~~~~~~~")
            dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the biological engineer: {dialogue}"
                if self.verbose: print("#" * 40, "\n", "biological_engineer Dialogue:", dialogue, "#" * 40, "\n")
        if self.except_if_fail:
            raise Exception("Max tries during phase: Plan Formulation")
        else:
            plan = "No plan specified."
            if self.human_in_loop_flag["plan formulation"]:
                retry = self.human_in_loop("plan formulation", plan)
                if retry: return retry
            self.set_agent_attr("plan", plan)
            # reset agent state
            self.reset_agents()
            return False

    def literature_review(self):
        """
        执行文献综述阶段
        @return: 是否需要重复此阶段
        """
        # 创建 PubMed 检索器实例
        pmb_eng = PubMedSearch(email="2418168449@qq.com")

        # 最大尝试次数（允许该阶段执行多次以充分完成）
        max_tries = self.max_steps

        # 获取博士代理人针对“literature review”阶段的初始回复
        resp = self.biological_engineer.inference(self.research_topic, "literature review", step=0, temp=0.4)
        if self.verbose: print(resp, "\n~~~~~~~~~~~")

        # 在未达到最大尝试次数前持续迭代执行文献综述逻辑
        for _i in range(max_tries):
            # print(f"@@ Lab #{self.lab_index} Paper #{self.paper_index} @@")
            feedback = str()

            # 处理 SUMMARY 指令：从 PubMed 搜索摘要
            if "```SUMMARY" in resp:
                query = extract_prompt(resp, "SUMMARY")
                papers = pmb_eng.search_papers(query, max_results=self.PubMed_num_summaries)
                # 如果使用了 agentRxiv，额外检索其数据库
                if self.agentRxiv:
                    if GLOBAL_AGENTRXIV.num_papers() > 0:
                        papers += GLOBAL_AGENTRXIV.search_agentrxiv(query, self.num_agentrxiv_papers)
                feedback = f"You requested PubMed papers related to the query {query}, here was the response\n{papers}"

            # 处理 FULL_TEXT 指令：获取全文内容
            elif "```FULL_TEXT" in resp:
                query = extract_prompt(resp, "FULL_TEXT")
                # 如果是 agentRxiv 数据，调用其接口
                if self.agentRxiv and "AgentRxiv" in query:
                    full_text = GLOBAL_AGENTRXIV.retrieve_full_text(query)
                else:
                    # 否则从 PubMed 检索 PMC ID 并获取全文
                    pmcid = pmb_eng.get_paper_pmcid(query)
                    if pmcid == None:
                        continue
                    else:
                        full_text = pmb_eng.retrieve_full_paper_text(pmcid)
                # 设置过期时间限制，防止文献在上下文中停留过久
                PubMed_paper = f"```EXPIRATION {self.PubMed_paper_exp_time}\n" + full_text + "```"
                feedback = PubMed_paper

            # 处理 ADD_PAPER 指令：将文章加入综述
            elif "```ADD_PAPER" in resp:
                query = extract_prompt(resp, "ADD_PAPER")
                if self.agentRxiv and "AgentRxiv" in query:
                    feedback, text = self.biological_engineer.add_review(query, pmb_eng, agentrxiv=True,
                                                                         GLOBAL_AGENTRXIV=GLOBAL_AGENTRXIV)
                else:
                    feedback, text = self.biological_engineer.add_review(query, pmb_eng)
                # 如果参考文献数量未超过设定上限，则加入参考文献列表
                if len(self.reference_papers) < self.num_ref_papers:
                    self.reference_papers.append(text)
                    print(self.reference_papers)

            # 检查是否完成文献综述任务
            if len(self.biological_engineer.lit_review) >= self.num_papers_lit_review:
                # 格式化整理最终综述内容
                lit_review_sum = self.biological_engineer.format_review()
                print(f"最终文献综述内容为：{lit_review_sum}")
                # 若设定为人工参与阶段，征询用户是否满意
                if self.human_in_loop_flag["literature review"]:
                    retry = self.human_in_loop("literature review", lit_review_sum)
                    # 若不满意，则清空综述并重新开始
                    if retry:
                        self.biological_engineer.lit_review = []
                        return retry
                # 否则直接返回完成的文献综述并进入下一阶段
                if self.verbose: print(self.biological_engineer.lit_review_sum)
                self.set_agent_attr("lit_review_sum", lit_review_sum)
                self.reset_agents()
                self.statistics_per_phase["literature review"]["steps"] = _i
                return False

            # 若未完成综述，则继续与博士代理人进行交互
            resp = self.biological_engineer.inference(self.research_topic, "literature review", feedback=feedback,
                                                      step=_i + 1,
                                                      temp=0.4)
            if self.verbose: print(resp, "\n~~~~~~~~~~~")

        # 超过最大尝试次数仍未完成，抛出异常或强制完成
        if self.except_if_fail:
            raise Exception("Max tries during phase: Literature Review")
        else:
            if len(self.biological_engineer.lit_review) >= self.num_papers_lit_review:
                lit_review_sum = self.biological_engineer.format_review()
                if self.human_in_loop_flag["literature review"]:
                    retry = self.human_in_loop("literature review", lit_review_sum)
                    if retry:
                        self.biological_engineer.lit_review = []
                        return retry
                if self.verbose: print(self.biological_engineer.lit_review_sum)
                self.set_agent_attr("lit_review_sum", lit_review_sum)
                self.reset_agents()
                self.statistics_per_phase["literature review"]["steps"] = _i
                return False

    def human_in_loop(self, phase, phase_prod):
        """
        Get human feedback for phase output
        @param phase: (str) current phase
        @param phase_prod: (str) current phase result
        @return: (bool) whether to repeat the loop
        """
        print("\n\n\n\n\n")
        print(f"Presented is the result of the phase [{phase}]: {phase_prod}")
        y_or_no = None
        # repeat until a valid answer is provided
        while y_or_no not in ["y", "n"]:
            y_or_no = input("\n\n\nAre you happy with the presented content? Respond Y or N: ").strip().lower()
            # if person is happy with feedback, move on to next stage
            if y_or_no == "y":
                pass
            # if not ask for feedback and repeat
            elif y_or_no == "n":
                # ask the human for feedback
                notes_for_agent = input(
                    "Please provide notes for the agent so that they can try again and improve performance: ")
                # reset agent state
                self.reset_agents()
                # add suggestions to the notes
                self.notes.append({
                    "phases": [phase],
                    "note": notes_for_agent})
                return True
            else:
                print("Invalid response, type Y or N")
        return False


def parse_arguments():
    parser = argparse.ArgumentParser(description="MR Research Workflow")

    parser.add_argument(
        '--copilot-mode',
        type=str,
        default="False",
        help='Enable human interaction mode.'
    )

    parser.add_argument(
        '--deepseek-chat-api-key',
        type=str,
        help='Provide the DeepSeek API key.'
        # help = 'sk-034a6a6132d24a1ba18a168c678f4023'
    )

    parser.add_argument(
        '--load-existing',
        type=str,
        default="False",
        help='Do not load existing state; start a new workflow.'
    )

    parser.add_argument(
        '--load-existing-path',
        type=str,
        help='Path to load existing state; start a new workflow, e.g. state_saves/results_interpretation.pkl'
    )

    parser.add_argument(
        '--research-topic',
        type=str,
        help='Specify the research topic.'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        help='Provide the OpenAI API key.'
    )

    parser.add_argument(
        '--compile-latex',
        type=str,
        default="True",
        help='Compile latex into pdf during paper writing phase. Disable if you can not install pdflatex.'
    )

    parser.add_argument(
        '--llm-backend',
        type=str,
        default="deepseek-chat-chat",
        help='Backend LLM to use for agents in Agent Laboratory.'
    )

    parser.add_argument(
        '--language',
        type=str,
        default="English",
        help='Language to operate Agent Laboratory in.'
    )

    parser.add_argument(
        '--num-papers-lit-review',
        type=str,
        default="5",
        help='Total number of papers to summarize in literature review stage'
    )

    parser.add_argument(
        '--mlesolver-max-steps',
        type=str,
        default="3",
        help='Total number of mle-solver steps'
    )

    parser.add_argument(
        '--papersolver-max-steps',
        type=str,
        default="5",
        help='Total number of paper-solver steps'
    )

    return parser.parse_args()


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()

    # 获取指定的 LLM 后端，如果没有指定则使用默认值
    # llm_backend = args.llm_backend
    # llm_backend = "deepseek-chat-r1"
    llm_backend = DEFAULT_LLM_BACKBONE
    # 判断是否为人类协助模式，如果用户指定了 "true" 则启用人类模式
    human_mode = args.copilot_mode.lower() == "true"

    # 判断是否需要编译 LaTeX 文档
    compile_pdf = args.compile_latex.lower() == "False"

    # 判断是否需要加载现有的工作状态
    load_existing = args.load_existing.lower() == "true"
    # except_if_fail = False
    research_topic = "Your goal is to investigate the causal relationship between Systolic Blood Pressure and Coronary Artery Disease, using Mendelian Randomization based on GWAS summary statistics."

    # research_topic = input("请输入您的孟德尔随机化分析任务目标：\nYour goal is to investigate the causal relationship between Systolic Blood Pressure and Coronary Artery Disease, using Mendelian Randomization based on GWAS summary statistics.\n")
    mr_populations = [
        "european",
        "east asian",
        "south asian",
        "white british",
        "african",
        "african american",
        "hispanic",
        "native american",
        "middle eastern",
        "central asian",
        "oceanian",
    ]
    population = "european"

    # population = input(f"种群列表是：{mr_populations}，请输入您要分析的种群：\n")


    try:
        num_papers_lit_review = int(args.num_papers_lit_review.lower())
    except Exception:
        raise Exception("args.num_papers_lit_review must be a valid integer!")

    # 处理 papersolver 的最大步数参数，确保它是一个有效的整数
    try:
        papersolver_max_steps = int(args.papersolver_max_steps.lower())
    except Exception:
        raise Exception("args.papersolver_max_steps must be a valid integer!")

    # 处理 mlesolver 的最大步数参数，确保它是一个有效的整数
    try:
        mlesolver_max_steps = int(args.mlesolver_max_steps.lower())
    except Exception:
        raise Exception("args.papersolver_max_steps must be a valid integer!")

    # 获取 OpenAI API 密钥和 DeepSeek API 密钥
    api_key = os.getenv('OPENAI_API_KEY') or args.api_key
    # print("OpenAI API Key:", api_key)
    deepseek_api_key = os.getenv('DEEPSEEK_API_KEY') or args.deepseek_api_key

    # 如果用户提供了 API 密钥且环境变量没有设置，则手动设置环境变量
    if args.api_key is not None and os.getenv('OPENAI_API_KEY') is None:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.deepseek_api_key is not None and os.getenv('DEEPSEEK_API_KEY') is None:
        os.environ["DEEPSEEK_API_KEY"] = args.deepseek_api_key

    # 如果没有提供 API 密钥，则抛出错误
    if not api_key and not deepseek_api_key:
        raise ValueError(
            "API key must be provided via --api-key / -deepseek-chat-api-key or the OPENAI_API_KEY / DEEPSEEK_API_KEY environment variable.")
        # LLM 任务笔记，定义每个阶段需要执行的操作和相关的提示信息
    task_notes_LLM = [
        {"phases": ["plan formulation"],
        "note": f"You should design a plan for ONE experiment focused on testing the causal effect.DO NOT PLAN FOR TOO LONG. Submit your experiment plan early to allow feedback and iteration."},
        {"phases": ["data preparation"],
         "note": f"Do NOT conduct any causal analysis during this phase."},
        {"phases": ["running experiments"],
         "note": f"Do not include the function call `format_data()`. The input dataset is already formatted for TwoSampleMR.\nYou can use any MR package.\nProvide detailed result logs and summaries after execution.\nGenerate clear and well-labeled figures to interpret results."},
        {"phases": ["results interpretation"],
         "note": f"Summarize whether there is statistically significant causal evidence.Comment on consistency between methods and presence of potential pleiotropy or bias."},
        {"phases": ["report writing"],
         "note": f"Your report should clearly state the causal question, methodology , data sources, and main findings.\nInclude visualizations and statistical outputs that support your interpretation.End with conclusions and potential follow-up studies or validations."},
    ]

    if args.language != "English":
        task_notes_LLM.append(
            {"phases": ["literature review", "plan formulation", "data preparation", "running experiments",
                        "results interpretation", "report writing", "report refinement"],
             "note": f"You should always write in the following language to converse and to write the report {args.language}"},
        )


    human_in_loop = {
        "literature review": human_mode,
        "plan formulation": human_mode,
        "data preparation": human_mode,
        "running experiments": human_mode,
        "results interpretation": human_mode,
        "report writing": human_mode,
        "report refinement": human_mode,
    }

    agent_models = {
        "literature review": llm_backend,
        "plan formulation": llm_backend,
        "data preparation": llm_backend,
        "running experiments": llm_backend,
        "report writing": llm_backend,
        "results interpretation": llm_backend,
        "paper refinement": llm_backend,
    }

    # remove previous files
    # 移除之前生成的图像或中间结果
    remove_figures()


    # 创建用于保存中间状态的目录（如果不存在）
    if not os.path.exists("state_saves"):
        os.mkdir(os.path.join(".", "state_saves"))

    # 初始化计时字符串
    time_str = str()
    # 获取当前时间（用于后续统计耗时）
    time_now = time.time()

    # 构造当前实验的研究目录路径
    # lab_direct = f"{RESEARCH_DIR_PATH}/o1-mini"
    remove_directory(RESEARCH_DIR_PATH)  # 删除研究目录
    remove_directory("./research_results/processed_data")  # 删除研究目录

    # 创建研究目录及其子目录 src 和 tex
    os.mkdir(os.path.join(".", RESEARCH_DIR_PATH))
    os.mkdir(os.path.join("./research_results", "processed_data"))
    # os.mkdir(os.path.join(f"./{RESEARCH_DIR_PATH}", "data"))
    os.mkdir(os.path.join(f"./{RESEARCH_DIR_PATH}", "code"))
    os.mkdir(os.path.join(f"./{RESEARCH_DIR_PATH}", "paper"))

    # 初始化 LaboratoryWorkflow 实例，配置实验参数
    lab = LaboratoryWorkflow(
        research_topic=research_topic,  # 研究主题
        notes=task_notes_LLM,  # 任务笔记（来自大模型）
        agent_model_backbone=agent_models,  # 智能体模型
        human_in_loop_flag=human_in_loop,  # 是否启用人类参与
        openai_api_key=api_key,  # OpenAI API 密钥
        compile_pdf=compile_pdf,  # 是否编译为 PDF
        num_papers_lit_review=num_papers_lit_review,  # 文献综述中使用的论文数量
        papersolver_max_steps=papersolver_max_steps,  # 解题器最大步数（论文层面）
        mlesolver_max_steps=mlesolver_max_steps,  # 解题器最大步数（机器学习层面）
        paper_index=0,  # 当前论文索引
        # except_if_fail=except_if_fail,  # 是否在失败时抛出异常
        agentRxiv=False,  # 此处不使用 agentRxiv 模式
        # lab_index=lab_index,  # 实验室索引
        lab_dir=f"./{RESEARCH_DIR_PATH}",  # 当前实验目录路径
        population=population
    )

    # 执行研究工作流程
    lab.perform_research()
    print(research_topic)
    # 记录当前研究流程的耗时
    time_str += str(time.time() - time_now) + " | "

    # 将耗时信息写入文件
    with open(f"agent_times.txt", "w") as f:
        f.write(time_str)

    # 更新时间记录起点，用于下一个论文流程计时
    time_now = time.time()

