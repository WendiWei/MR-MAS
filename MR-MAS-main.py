import ast
import PyPDF2
import threading
from app import *
from agents import *
from copy import copy
from pathlib import Path
from datetime import date
from common_imports import *
from mr_code_engine import MRAnalysisEngine
import argparse, pickle, yaml
import json

DEFAULT_LLM_BACKBONE = "gpt-4o"
RESEARCH_DIR_PATH = f"forth_experiment/{DEFAULT_LLM_BACKBONE}"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MRMASWorkflow:
    def __init__(self, research_topic, openai_api_key, max_steps=100, num_papers_lit_review=5,
                 agent_model_backbone=f"{DEFAULT_LLM_BACKBONE}", notes=list(), human_in_loop_flag=None,
                 compile_pdf=True, mrcode_max_steps=3, manuscriptengine_max_steps=3, paper_index=0,
                 parallelized=False, lab_dir=None, population="European"):
        self.max_prev_papers = 10
        self.parallelized = parallelized
        self.notes = notes
        self.lab_dir = lab_dir
        self.max_steps = max_steps
        self.compile_pdf = compile_pdf
        self.paper_index = paper_index
        self.openai_api_key = openai_api_key
        self.research_topic = research_topic
        self.population = population
        self.model_backbone = agent_model_backbone
        self.num_papers_lit_review = num_papers_lit_review
        self.print_cost = True
        self.review_override = True
        self.review_ovrd_steps = 0
        self.PubMed_paper_exp_time = 3
        self.reference_papers = list()
        self.exposure_file_path = None
        self.outcome_file_path = None
        self.exposure_trait = None
        self.outcome_trait = None
        self.num_ref_papers = 1
        self.review_total_steps = 0
        self.PubMed_num_summaries = 20
        self.mrcode_max_steps = mrcode_max_steps
        self.manuscriptengine_max_steps = manuscriptengine_max_steps
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
        self.biological_engineer.reset()
        self.statistical_geneticist.reset()
        self.writing_specialist.reset()
        self.algorithm_engineer.reset()
        self.data_analyst.reset()

    def perform_research(self):
        """
        Loop through all research phases
        @return: None
        """
        for phase, subtasks in self.phases:
            phase_start_time = time.time()
            if self.verbose:
                print(f"{'*' * 50}\nBeginning phase: {phase}\n{'*' * 50}")
            for subtask in subtasks:
                if type(self.phase_models) == dict:
                    if subtask in self.phase_models:
                        self.set_model(self.phase_models[subtask])
                    else:
                        self.set_model(f"{DEFAULT_LLM_BACKBONE}")
                if (subtask not in self.phase_status or not self.phase_status[
                    subtask]) and subtask == "plan formulation":
                    repeat = True
                    while repeat: repeat = self.plan_formulation()
                    self.phase_status[subtask] = True
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

                    self.set_agent_attr("second_round", return_to_exp_phase)
                    self.set_agent_attr("prev_report", copy(self.biological_engineer.report))
                    self.set_agent_attr("prev_exp_results", copy(self.biological_engineer.exp_results))
                    self.set_agent_attr("prev_results_code", copy(self.biological_engineer.results_code))
                    self.set_agent_attr("prev_interpretation", copy(self.biological_engineer.interpretation))

                    self.phase_status["plan formulation"] = False
                    self.phase_status["data preparation"] = False
                    self.phase_status["running experiments"] = False
                    self.phase_status["results interpretation"] = False
                    self.phase_status["report writing"] = False
                    self.phase_status["report refinement"] = False
                    self.perform_research()
                if self.save: self.save_state(subtask)
                phase_end_time = time.time()
                phase_duration = phase_end_time - phase_start_time
                print(f"Phase '{subtask}' completed in {phase_duration:.2f} seconds.")
                self.statistics_per_phase[subtask]["time"] = phase_duration

    def report_refinement(self):
        """
        Perform report refinement phase
        @return: (bool) whether to repeat the phase
        """
        reviews = self.reviewers.inference(self.biological_engineer.plan, self.biological_engineer.report)
        print("Reviews:", reviews)
        if self.human_in_loop_flag["report refinement"]:
            print(f"Provided are reviews from a set of three reviewers: {reviews}")
            input(
                "Would you like to be completed with the project or should the agents go back and improve their experimental results?\n (y) for go back (n) for complete project: ")
        else:
            review_prompt = f"Provided are reviews from a set of three reviewers: {reviews}. Would you like to be completed with the project or do you want to go back to the planning phase and improve your experiments?\n Type y and nothing else to go back, type n and nothing else for complete project."
            self.biological_engineer.phases.append("report refinement")
            if self.review_override:
                if self.review_total_steps == self.review_ovrd_steps:
                    response = "n"
                else:
                    response = "y"
                    self.review_ovrd_steps += 1
            else:
                response = self.biological_engineer.inference(
                    research_topic=self.research_topic, phase="report refinement", feedback=review_prompt, step=0)
            if len(response) == 0:
                raise Exception("Model did not respond")
            response = response.lower().strip()[0]
            if response == "n":
                if self.verbose: print("*" * 40, "\n", "REVIEW COMPLETE", "\n", "*" * 40)
                return False
            elif response == "y":
                self.set_agent_attr("reviewer_response",
                                    f"Provided are reviews from a set of three reviewers: {reviews}.")
                return True
            else:
                raise Exception("Model did not respond")

    def report_writing(self):
        """
        Perform report writing phase
        @return: (bool) whether to repeat the phase
        """
        report_notes = [_note["note"] for _note in self.algorithm_engineer.notes if "report writing" in _note["phases"]]
        report_notes = f"Notes for the task objective: {report_notes}\n" if len(report_notes) > 0 else ""
        from manuscript_engine import ManuscriptEngine
        self.reference_papers = []

        solver = ManuscriptEngine(
            notes=report_notes,
            max_steps=self.manuscriptengine_max_steps,
            plan=self.biological_engineer.plan,
            exp_code=self.biological_engineer.results_code,
            exp_results=self.biological_engineer.exp_results,
            insights=self.biological_engineer.interpretation,
            lit_review=self.biological_engineer.lit_review,
            datasets_information=self.biological_engineer.dataset_information,
            ref_papers=self.reference_papers,
            topic=research_topic,
            openai_api_key=self.openai_api_key,
            llm_str=self.model_backbone["report writing"],
            compile_pdf=compile_pdf,
            save_loc=self.lab_dir
        )
        solver.initial_solve()

        if solver.best_score < 0.9:
            for _ in range(self.manuscriptengine_max_steps):
                solver.solve()

        report = "\n".join(solver.best_report[0][0])
        score = solver.best_report[0][1]

        match = re.search(r'\\title\{([^}]*)\}', report)
        if match:
            report_title = match.group(1).replace(" ", "_")
        else:
            report_title = "\n".join([str(random.randint(0, 10)) for _ in range(10)])
        if self.verbose:
            print(f"Report writing completed, reward function score: {score}")

        if self.human_in_loop_flag["report writing"]:
            retry = self.human_in_loop("report writing", report)
            if retry:
                return retry

        self.set_agent_attr("report", report)

        readme = self.writing_specialist.generate_readme()
        save_to_file(f"./{self.lab_dir}", "readme.md", readme)
        save_to_file(f"./{self.lab_dir}/paper", "paper.txt", report)
        self.reset_agents()
        return False

    def results_interpretation(self):
        """
        Perform results interpretation phase
        @return: (bool) whether to repeat the phase
        """
        max_tries = self.max_steps
        dialogue = str()
        for _i in range(max_tries):
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
        experiment_notes = [_note["note"] for _note in self.algorithm_engineer.notes if
                            "running experiments" in _note["phases"]]
        experiment_notes = f"Notes for the task objective: {experiment_notes}\n" if len(experiment_notes) > 0 else ""
        cfg = ExperimentConfig()
        Original_exposure_data_path = cfg.get("data_preparation", "Original_exposure_data_path")
        Original_outcome_data_path = cfg.get("data_preparation", "Original_outcome_data_path")
        solver = MRAnalysisEngine(
            dataset_code=self.algorithm_engineer.dataset_code,
            notes=experiment_notes,
            insights=self.algorithm_engineer.lit_review_sum,
            max_steps=self.mrcode_max_steps,
            plan=self.algorithm_engineer.plan,
            Original_exposure_data_path=Original_exposure_data_path,
            Original_outcome_data_path=Original_outcome_data_path,
            openai_api_key=self.openai_api_key,
            llm_str=self.model_backbone["running experiments"]
        )
        solver.initial_solve()
        if solver.best_score < 0.8:
            print("代码评分小于0.9，接下来进行代码优化")
            for _ in range(mrcode_max_steps - 1):
                solver.solve()

        code = "\n".join(solver.best_codes[0][0])
        score = solver.best_codes[0][1]
        exp_results = solver.best_codes[0][2]
        if self.verbose:
            print(f"Running experiments completed, reward function score: {score}")

        if self.human_in_loop_flag["running experiments"]:
            retry = self.human_in_loop("data preparation", code)
            if retry:
                return retry

        save_to_file(f"./{self.lab_dir}/code", "run_experiments.R", code)
        save_to_file(f"./{self.lab_dir}/code", "experiment_output.log", exp_results)
        self.set_agent_attr("results_code", code)
        self.set_agent_attr("exp_results", exp_results)
        self.reset_agents()
        return False

    def data_preparation(self):
        global outcome_information, exposure_information
        max_tries = self.max_steps
        ae_feedback = str()
        ae_dialogue = str()
        da_feedback = str()
        ae_command = str()
        GWAS_engine = GWASCatalogSearch()
        GWAS_trait_search = TraitMatcher()
        GWAS_download = GWASLoaderTool()
        cfg = ExperimentConfig()

        for _i in range(max_tries):
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
                if "[CODE EXECUTION ERROR]" in code_resp:
                    da_feedback += "\nERROR: Final code had an error and could not be submitted! You must address and fix this error.\n"
                else:
                    if self.human_in_loop_flag["data preparation"]:
                        retry = self.human_in_loop("data preparation", final_code)
                        if retry: return retry
                    save_to_file(f"./{self.lab_dir}/code", "load_data.R", final_code)
                    self.set_agent_attr("dataset_code", final_code)
                    self.set_agent_attr("Original_exposure_data_path", self.Original_exposure_data_path)
                    self.set_agent_attr("Original_outcome_data_path", self.Original_outcome_data_path)
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
                is_download = False
                hf_query = extract_prompt(resp, "SEARCH_GWAS")
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
                hf_query_last_list = GWAS_trait_search.find_similar_traits(trait_keyword)
                for hf_query_last in hf_query_last_list:
                    print(f"这是GWAS Catalog数据库的最终检索关键词：{hf_query_last}")
                    if data_type == "exposure":
                        filter_results = GWAS_engine._filter_and_sort_data(hf_query_last, data_type, population)
                        if filter_results is None or filter_results.empty:
                            continue
                        self.exposure_trait = hf_query_last
                        result_strs, down_list = GWAS_engine.results_str(filter_results,population)
                        if len(down_list) == 0:
                            continue
                        hf_res = "\n".join(result_strs)
                        total_num = len(result_strs)
                        print(f"共检索到{total_num}个研究")
                        exposure_information = hf_res.strip().split('-------------------------')[0]
                        self.Original_exposure_data_path = GWAS_download.run(down_list[0])
                        match = re.search(r"Sample description:\s*([\d,]+)", exposure_information)
                        if match:
                            sample_size_str = match.group(1)
                            sample_size = int(sample_size_str.replace(",", ""))
                            print("样本量:", sample_size)
                        else:
                            print("未找到样本量")

                        cfg.set(
                            "data_preparation",
                            "exposure_data_information",
                            json.dumps(exposure_information, indent=4)
                        )

                        if not self.Original_exposure_data_path:
                            continue
                        is_download = True
                        cfg.set("data_preparation", "search_query_of_exposure", hf_query_last)
                        cfg.set("data_preparation", "original_exposure_data_path", self.Original_exposure_data_path)
                        print(f"当前下载的是暴露变量 [{hf_query_last}] 的数据，已保存至 [{self.Original_exposure_data_path}]")
                        exposure_data,used_thresh  = filter_significant_mutations(self.Original_exposure_data_path)
                        print(f"筛选显著变异，实际使用阈值: {used_thresh}")
                        exposure_data_path= "research_results/processed_data/exposure_data.csv"
                        exposure_data.to_csv(exposure_data_path, index=False)
                        self.exposure_file_path = exposure_data_path
                        print(f"已对暴露数据筛选完显著snps，已保存至 [{self.exposure_file_path}]")
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
                        )
                    elif data_type == "outcome":
                        filter_results = GWAS_engine._filter_and_sort_data(hf_query_last, data_type, population)
                        if filter_results is None or filter_results.empty:
                            continue
                        self.outcome_trait = hf_query_last
                        result_strs, down_list = GWAS_engine.results_str(filter_results,population)
                        if len(down_list) == 0:
                            continue
                        hf_res = "\n".join(result_strs)
                        total_num = len(result_strs)
                        print(f"共检索到{total_num}个研究")
                        exposure_outcome_intersection_snps_path = "research_results/processed_data/exposure_outcome_intersection_snps.csv"
                        last_outcome_data_path = "research_results/processed_data/last_outcome_data.csv"
                        is_first_write = True if not os.path.exists(exposure_outcome_intersection_snps_path) else False
                        for num, url in enumerate(down_list):
                            outcome_information = hf_res.strip().split('-------------------------')[num]
                            print(f"第{num + 1}个研究信息是：{outcome_information}")
                            self.outcome_data_path = GWAS_download.run(url)
                            self.Original_outcome_data_path = self.outcome_data_path
                            is_download = True
                            cfg.set("data_preparation", "search_query_of_outcome", hf_query_last)
                            cfg.set("data_preparation", "original_outcome_data_path", self.outcome_data_path)
                            cfg.set(
                                "data_preparation",
                                "outcome_data_information",
                                json.dumps(outcome_information, indent=4)
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
                            len_exposure_data_rsid = len(self.exposure_data_rsid)
                            self.exposure_data_rsid_temp = pd.read_csv(self.exposure_data_rsid_path)
                            self.exposure_data_rsid_temp.columns = ['hm_rsid']
                            df_common = pd.merge(temp_filter_data, self.exposure_data_rsid_temp, how='inner',on='hm_rsid')
                            num_rows = df_common.shape[0]
                            print(f"本次取交集有 {num_rows} 行数据，来源文件: {self.outcome_data_path}")

                            data = data.rename(columns={col_name: "hm_rsid"})
                            filtered_rows = pd.merge(data, df_common, on='hm_rsid', how='inner')
                            if df_common.empty:
                                continue

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
                            is_first_write = False
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
                if not is_download:
                    ae_command = f"The previous search query generated by the ML agent was: {hf_query}. Unfortunately, it returned no relevant results. Please generate a new, more effective search query that is likely to yield relevant information. \n"
                    ae_feedback += f"No relevant dataset download,lease generate a new, more effective search query that is likely to yield relevant information.\n"



            if "```LD_PRUNE" in resp:
                exposure_outcome_intersection_snps_path = "research_results/processed_data/exposure_outcome_intersection_snps.csv"
                snps_data = pd.read_csv(exposure_outcome_intersection_snps_path)
                snps_list = snps_data['hm_rsid']
                if len(snps_list) != 0:
                    print(f"开始对snps列表进行LD剪枝")
                    agent = LDPrunerAgent(
                        gwas_file=exposure_outcome_intersection_snps_path,
                        output_dir="./research_results/ld_processed_data",
                        r2 = 0.001,
                        window_kb = 10000
                    )
                    result = agent.command_run_auto({
                        "gwas_file": exposure_outcome_intersection_snps_path,
                        "min_snps_threshold": 10,
                        "max_attempts": 4,
                        "step": 10,
                        "r2": 0.001,
                        "window_kb": 10000
                    })
                    print("使用的r2参数:", result["r2_used"])
                    print("使用的window_kb:", result["window_kb_used"])
                    print("剪枝后SNP数量:", result["snp_count"])
                    ld_pruned_snps = pd.read_csv("research_results/ld_processed_data/ld_pruned.prune.in",
                                                 names=['rsid'])

                    exposure_df = pd.read_csv("research_results/processed_data/exposure_data.csv")
                    outcome_df = pd.read_csv("research_results/processed_data/last_outcome_data.csv")
                    column_candidates = ['rsid', 'hm_rsid', 'variant_id']
                    exposure_df = standardize_rsid_column(exposure_df, column_candidates)
                    outcome_df = standardize_rsid_column(outcome_df, column_candidates)
                    filtered_exposure_data = pd.merge(exposure_df, ld_pruned_snps, on="rsid", how="inner")
                    filtered_outcome_data = pd.merge(outcome_df, ld_pruned_snps, on="rsid", how="inner")

                    def map_columns_for_exposure(df):
                        return pd.DataFrame({
                            'SNP': df['rsid'],
                            'id.exposure': self.exposure_trait,
                            'exposure': self.exposure_trait,
                            'beta.exposure': df['beta'],
                            'se.exposure': df['standard_error'],
                            'effect_allele.exposure': df['effect_allele'],
                            'other_allele.exposure': df['other_allele'],
                            'eaf.exposure': df.get('effect_allele_frequency'),
                            'pval.exposure': df['p_value']
                        })

                    def map_columns_for_outcome(df):
                        return pd.DataFrame({
                            'SNP': df['rsid'],
                            'id.outcome': self.outcome_trait,
                            'outcome': self.outcome_trait,
                            'beta.outcome': df['beta'],
                            'se.outcome': df['standard_error'],
                            'effect_allele.outcome': df['effect_allele'],
                            'other_allele.outcome': df['other_allele'],
                            'eaf.outcome': df.get('effect_allele_frequency'),
                            'pval.outcome': df['p_value']
                        })

                    filtered_exposure_data_mapped = map_columns_for_exposure(filtered_exposure_data)
                    filtered_outcome_data_mapped = map_columns_for_outcome(filtered_outcome_data)
                    N = sample_size
                    exposure_df_all, exposure_df_filtered = add_r2_fstat_filter(filtered_exposure_data_mapped, N)
                    print("\n=== 筛选后 SNP（F > 10） ===")
                    print(exposure_df_filtered[["SNP", "beta.exposure", "se.exposure", "eaf.exposure", "R2", "F_stat"]])
                    exposure_data_rsid = exposure_df_filtered[["SNP"]]
                    exposure_data_rsid_path = "research_results/processed_data/exposure_data_filtered_rsid.csv"
                    exposure_data_rsid.to_csv(exposure_data_rsid_path, index=False)

                    filtered_outcome_data = pd.merge(filtered_outcome_data_mapped, exposure_data_rsid, how='inner',
                                                     on='SNP')
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
        max_tries = self.max_steps
        dialogue = str()
        for _i in range(max_tries):
            resp = self.statistical_geneticist.inference(self.research_topic, "plan formulation", feedback=dialogue,
                                                         step=_i)
            if self.verbose: print("statistical geneticist : ", resp, "\n~~~~~~~~~~~")
            dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the statistical geneticist : {dialogue}"
                if self.verbose: print("#" * 40, "\n", "statistical geneticist  Dialogue:", dialogue, "\n", "#" * 40)

            if "```PLAN" in resp:
                plan = extract_prompt(resp, "PLAN")
                if self.human_in_loop_flag["plan formulation"]:
                    retry = self.human_in_loop("plan formulation", plan)
                    if retry: return retry
                save_to_file(f"./{RESEARCH_DIR_PATH}", "plan.txt", plan)
                self.set_agent_attr("plan", plan)
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

    def human_in_loop(self, phase, phase_prod):
        print("\n\n\n\n\n")
        print(f"Presented is the result of the phase [{phase}]: {phase_prod}")
        y_or_no = None
        while y_or_no not in ["y", "n"]:
            y_or_no = input("\n\n\nAre you happy with the presented content? Respond Y or N: ").strip().lower()
            if y_or_no == "y":
                pass
            elif y_or_no == "n":
                notes_for_agent = input(
                    "Please provide notes for the agent so that they can try again and improve performance: ")
                self.reset_agents()
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
        help='Backend LLM to use for agents in MR-MAS.'
    )
    parser.add_argument(
        '--language',
        type=str,
        default="English",
        help='Language to operate MR-MAS in.'
    )
    parser.add_argument(
        '--num-papers-lit-review',
        type=str,
        default="5",
        help='Total number of papers to summarize'
    )

    parser.add_argument(
        '--mrcode-max-steps',
        type=str,
        default="3",
        help='Total number of MRAnalysis steps'
    )

    parser.add_argument(
        '--manuscriptengine-max-steps',
        type=str,
        default="5",
        help='Total number of ManuscriptEngine steps'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    llm_backend = DEFAULT_LLM_BACKBONE
    human_mode = args.copilot_mode.lower() == "true"
    compile_pdf = args.compile_latex.lower() == "False"
    load_existing = args.load_existing.lower() == "true"
    research_topic = input("请输入您的孟德尔随机化分析任务目标：\nYour goal is to investigate the causal relationship between Systolic Blood Pressure and Coronary Artery Disease, using Mendelian Randomization based on GWAS summary statistics.\n")
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
    try:
        num_papers_lit_review = int(args.num_papers_lit_review.lower())
    except Exception:
        raise Exception("args.num_papers_lit_review must be a valid integer!")
    try:
        manuscriptengine_max_steps = int(args.manuscriptengine_max_steps.lower())
    except Exception:
        raise Exception("args.manuscriptengine_max_steps must be a valid integer!")
    try:
        mrcode_max_steps = int(args.mrcode_max_steps.lower())
    except Exception:
        raise Exception("args.manuscriptengine_max_steps must be a valid integer!")

    api_key = os.getenv('OPENAI_API_KEY') or args.api_key
    deepseek_api_key = os.getenv('DEEPSEEK_API_KEY') or args.deepseek_api_key
    if args.api_key is not None and os.getenv('OPENAI_API_KEY') is None:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.deepseek_chat_api_key is not None and os.getenv('DEEPSEEK_API_KEY') is None:
        os.environ["DEEPSEEK_API_KEY"] = args.deepseek_api_key

    if not api_key and not deepseek_api_key:
        raise ValueError(
            "API key must be provided via --api-key / -deepseek-chat-api-key or the OPENAI_API_KEY / DEEPSEEK_API_KEY environment variable.")
    task_notes_LLM = [
        {"phases": ["plan formulation"],
        "note": f"You should design a plan for ONE experiment focused on testing the causal effect.DO NOT PLAN FOR TOO LONG. Submit your experiment plan early to allow feedback and iteration.Perform Mendelian Randomization (MR) analysis on the given exposure–outcome pair only using the following methods: IVW, MR-Egger, Weighted Median, Weighted Mode, MR-PRESSO."},
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
            {"phases": ["plan formulation", "data preparation", "running experiments",
                        "results interpretation", "report writing", "report refinement"],
             "note": f"You should always write in the following language to converse and to write the report {args.language}"},
        )

    human_in_loop = {
        "plan formulation": human_mode,
        "data preparation": human_mode,
        "running experiments": human_mode,
        "results interpretation": human_mode,
        "report writing": human_mode,
        "report refinement": human_mode,
    }
    agent_models = {
        "plan formulation": llm_backend,
        "data preparation": llm_backend,
        "running experiments": llm_backend,
        "report writing": llm_backend,
        "results interpretation": llm_backend,
        "paper refinement": llm_backend,
    }

    remove_figures()

    if not os.path.exists("state_saves"):
        os.mkdir(os.path.join(".", "state_saves"))

    time_str = str()
    time_now = time.time()


    remove_directory(RESEARCH_DIR_PATH)
    remove_directory("./research_results/processed_data")

    os.mkdir(os.path.join(".", RESEARCH_DIR_PATH))
    os.mkdir(os.path.join("./research_results", "processed_data"))
    os.mkdir(os.path.join(f"./{RESEARCH_DIR_PATH}", "code"))
    os.mkdir(os.path.join(f"./{RESEARCH_DIR_PATH}", "paper"))

    lab = MRMASWorkflow(
        research_topic=research_topic,
        notes=task_notes_LLM,
        agent_model_backbone=agent_models,
        human_in_loop_flag=human_in_loop,
        openai_api_key=api_key,
        compile_pdf=compile_pdf,
        num_papers_lit_review=num_papers_lit_review,
        manuscriptengine_max_steps=manuscriptengine_max_steps,
        mrcode_max_steps=mrcode_max_steps,
        paper_index=0,
        lab_dir=f"./{RESEARCH_DIR_PATH}",
        population=population
    )
    lab.perform_research()
    print(research_topic)
    time_str += str(time.time() - time_now) + " | "
    with open(f"agent_times.txt", "w") as f:
        f.write(time_str)
    time_now = time.time()

