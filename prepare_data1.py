'''
无检查性状这一步
'''
from tools import *
import ast


class BaseAgent:
    def __init__(self, model="gpt-4o-mini", notes=None, max_steps=100, openai_api_key=None):
        if notes is None:
            self.notes = []
        else:
            self.notes = notes

        with open("experiment_research/plan1.txt", "r", encoding="utf-8") as f:
            plan = f.read()

        self.max_steps = max_steps
        self.model = model
        self.phases = []
        self.plan = plan
        self.report = str()
        self.history = list()
        self.prev_comm = str()
        self.prev_report = str()
        # self.exp_results = exp_results
        self.dataset_code = str()
        self.dataset_information = str()

        # self.results_code = results_code
        self.lit_review_sum = str()
        self.interpretation = str()
        self.prev_exp_results = str()
        self.reviewer_response = str()
        self.prev_results_code = str()
        self.prev_interpretation = str()
        self.openai_api_key = openai_api_key

        self.second_round = False
        self.max_hist_len = 15

    def set_model_backbone(self, model):
        self.model = model

    @staticmethod
    def clean_text(text):
        """
        Fix minor corrections
        :return: (str) corrected text
        """
        text = text.replace("```\n", "```")
        return text

    def override_inference(self, query, temp=0.0):
        sys_prompt = f"""You are {self.role_description()}"""
        model_resp = query_model(model_str=self.model, system_prompt=sys_prompt, prompt=query, temp=temp,
                                 openai_api_key=self.openai_api_key)
        return model_resp

    def inference(self, research_topic, phase, step, feedback="", temp=None):
        sys_prompt = f"""You are {self.role_description()} \nTask instructions: {self.phase_prompt(phase)}\n{self.command_descriptions(phase)}"""
        context = self.context(phase)
        history_str = "\n".join([_[1] for _ in self.history])
        phase_notes = [_note for _note in self.notes if phase in _note["phases"]]
        notes_str = f"Notes for the task objective: {phase_notes}\n" if len(phase_notes) > 0 else ""
        complete_str = str()
        if step / (self.max_steps - 1) > 0.7: complete_str = "You must finish this task and submit as soon as possible!"
        prompt = (
            f"""{context}\n{'~' * 10}\nHistory: {history_str}\n{'~' * 10}\n"""
            f"Current Step #{step}, Phase: {phase}\n{complete_str}\n"
            f"[Objective] Your goal is to perform research on the following topic: {research_topic}\n"
            f"Feedback: {feedback}\nNotes: {notes_str}\nYour previous command was: {self.prev_comm}. Make sure your new output is very different.\nPlease produce a single command below:\n")
        model_resp = query_model(model_str=self.model, system_prompt=sys_prompt, prompt=prompt, temp=temp,
                                 openai_api_key=self.openai_api_key)
        print("^" * 50, phase, "^" * 50)
        model_resp = self.clean_text(model_resp)
        self.prev_comm = model_resp
        steps_exp = None
        if feedback is not None and "```EXPIRATION" in feedback:
            steps_exp = int(feedback.split("\n")[0].replace("```EXPIRATION ", ""))
            feedback = extract_prompt(feedback, "EXPIRATION")
        self.history.append(
            (steps_exp, f"Step #{step}, Phase: {phase}, Feedback: {feedback}, Your response: {model_resp}"))
        # remove histories that have expiration dates
        for _i in reversed(range(len(self.history))):
            if self.history[_i][0] is not None:
                self.history[_i] = (self.history[_i][0] - 1, self.history[_i][1])
                if self.history[_i][0] < 0:
                    self.history.pop(_i)
        if len(self.history) >= self.max_hist_len:
            self.history.pop(0)
        return model_resp

    def reset(self):
        self.history.clear()  # Clear the deque
        self.prev_comm = ""

    def context(self, phase):
        raise NotImplementedError("Subclasses should implement this method.")

    def phase_prompt(self, phase):
        raise NotImplementedError("Subclasses should implement this method.")

    def role_description(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def command_descriptions(self, phase):
        raise NotImplementedError("Subclasses should implement this method.")

    def example_command(self, phase):
        raise NotImplementedError("Subclasses should implement this method.")


class DataAnalyst(BaseAgent):
    def __init__(self, model="gpt4omini", notes=None, max_steps=100, openai_api_key=None):
        super().__init__(model, notes, max_steps, openai_api_key)
        self.phases = [
            "data preparation",
        ]

    def context(self, phase):
        sr_str = str()
        if self.second_round:
            sr_str = (
                f"The following are results from the previous experiments\n",
                f"Previous Experiment code: {self.prev_results_code}\n"
                f"Previous Results: {self.prev_exp_results}\n"
                f"Previous Interpretation of results: {self.prev_interpretation}\n"
                f"Previous Report: {self.prev_report}\n"
                f"{self.reviewer_response}\n\n\n"
            )
        if phase == "data preparation":
            return (
                sr_str,
                f"Current Literature Review: {self.lit_review_sum}\nPlan: {self.plan}",
                f"Current Plan: {self.plan}")
        #elif phase == "running experiments":
        #    return (
        #        sr_str,
        #        f"Current Literature Review: {self.lit_review_sum}\n"
        #        f"Current Plan: {self.plan}\n"
        #        f"Current Dataset code: {self.dataset_code}\n"
        #    )
        return ""

    def example_command(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        return ()

    def command_descriptions(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        if phase == "data preparation":
            return (
                "You can produce code using the following command: ```R\ncode here\n```\n where code here is the actual code you will execute in a R environment, and R is just the word R. Try to incorporate some print functions. Do not use any classes or functions. If your code returns any errors, they will be provided to you, and you are also able to see print statements. You will receive all print statement results from the code. Make sure function variables are created inside the function or passed as a function parameter.Please submit the code promptly once all data preparation steps required for Mendelian Randomization analysis are completed. Do not proceed with any subsequent MR analysis steps.\n"  # Try to avoid creating functions. 
                "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where dialogue here is the actual dialogue you will send, and DIALOGUE is just the word DIALOGUE.\n"
                "You can generate a phenotype keyword that best matches the trait description in the research objective using the following command: ```SEARCH_GWAS\ntrait keyword here\n``` where trait keyword here is your generated term，which should align closely with the efo_trait field in the GWAS Catalog to ensure highly relevant search results,and SEARCH_GWAS is the word SEARCH_GWAS.The command will return metadata and file paths for datasets related to the specified trait. When generating keywords for GWAS search, please add the suffix “_exposure” for exposure variables and “_outcome” for outcome variables to the keywords according to the user's analytical objectives, for example “body mass index_exposure” or ‘type 2 diabetes_outcome’.Do not download data again—datasets have already been fetched. Your task is to write code that loads data from the returned file paths, using only the datasets from the GWAS Catalog.\n"
                "You can perform LD pruning using the following command: ```LD_PRUNE\n{'r2': value, 'window_kb': value}\n``` where LD_PRUNE is the word LD_PRUNE, and the JSON object includes the pruning parameters to be used. The keys are 'r2'(default: 0.001) for the squared correlation threshold and 'window_kb'(default: 10000) for the window size in kilobases. This command instructs the agent to perform LD pruning with the specified parameters using PLINK. \n"
                "You MUST use a GWAS Catalog dataset in your code. DO NOT CREATE A MAIN FUNCTION. Try to make the code very simple.\n"
                "You can only use a SINGLE command per inference turn. Do not use more than one command per inference. If you use multiple commands, then only one of them will be executed, NOT BOTH.\n"
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. R, DIALOGUE, SEARCH_GWAS).\n"
            )

    def phase_prompt(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        if phase == "data preparation":
            phase_str = (
                 "You are a data analyst working under the guidance of a algorithm engineer. You will collaborate by writing R code through dialogue.\n",
                "Your current task is to write executable R code that prepares and processes GWAS data for use in a Mendelian Randomization (MR) experiment. You must follow the provided literature review and experimental plan.\n",
                "All steps must be implemented via R code that performs all necessary data preparation steps from start to finish. Do NOT complete any step manually in advance.\n",
                "Strictly Perform the following steps in order:\n"
                "Execution Steps (Strict Order):\n"
                "1. Generating a search query for the exposure dataset.\n"
                "2. Generating a search query for the outcome dataset.\n"
                "3. Perform LD pruning on the exposure dataset with the following parameters: r2 (default: 0.001): squared correlation threshold. window_kb (default: 10000): window size in kilobases\n"
                "4. Load the exposure data from: research_results/processed_data/filtered_exposure_data.csv\n"
                "5. Load the outcome data from: research_results/processed_data/filtered_outcome_data.csv\n"
                "6. (Optional) Performing additional instrumental variable (IV) selection procedures.\n"
                "7. Performing data harmonization: Use the harmonise_data() function from the TwoSampleMR package to align alleles and remove ambiguous SNPs (e.g., A/T or C/G).\n"
                "Output Paths:\n"
                "All raw GWAS data files must be saved in: \"research_results/raw_gwas_data/\"\n"
                "All processed output files must be saved in: \"research_results/processed_data/\"\n"
                "Code Requirements:\n"
                "R code only; fully executable and clearly structured.\n"
                "Include comments; use print() for intermediate results.\n"
                "Do not create functions or classes; variables must be local or passed as parameters.\n"
                "Avoid multiple versions of similar files; process data in memory as much as possible.\n"
                "If columns are missing, do not regenerate queries; rename columns to match expected formats.\n"
                "Prohibited Actions:\n"
                "Do not perform causal inference methods.\n"
                "Do not update or reinstall R packages.\n"
                "Do not use TwoSampleMR for data searching or loading.\n"
                "Do not use web URLs or placeholder links to download data.\n"
                "Do not create or write to other folders unless absolutely necessary.\n"
                "Do not generate duplicate search terms.\n"
                "Example Code:",
                "```r",
                "library(TwoSampleMR)",
                "library(dplyr)",
                "library(readr)",
                "# Load preprocessed exposure and outcome datasets",
                "exposure_data <- read_csv(\"research_results/processed_data/filtered_exposure_data.csv\")",
                "outcome_data <- read_csv(\"research_results/processed_data/filtered_outcome_data.csv\")",
                "# Harmonize exposure and outcome data, removing ambiguous SNPs (A/T or C/G)",
                "harmonized_data <- harmonise_data(",
                "  exposure_dat = exposure_data,",
                "  outcome_dat = outcome_data,",
                "  action = 2",
                ")",
                "# Save harmonized results",
                "write.csv(harmonized_data, \"research_results/processed_data/harmonized_data.csv\", row.names = FALSE)",
                "cat(\"Data harmonization completed and saved to processed_data/harmonized_data.csv\\n\")",
                "```"
            )
        return phase_str

    def role_description(self):
        return "A data analyst working at a leading research institute, specializing in biological and genomic data interpretation."


class AlgorithmEngineer(BaseAgent):
    def __init__(self, model="gpt4omini", notes=None, max_steps=100, openai_api_key=None):
        super().__init__(model, notes, max_steps, openai_api_key)
        self.phases = [
            "data preparation",
            "running experiments",
        ]

    def context(self, phase):
        sr_str = str()
        if self.second_round:
            sr_str = (
                f"The following are results from the previous experiments\n",
                f"Previous Experiment code: {self.prev_results_code}\n"
                f"Previous Results: {self.prev_exp_results}\n"
                f"Previous Interpretation of results: {self.prev_interpretation}\n"
                f"Previous Report: {self.prev_report}\n"
                f"{self.reviewer_response}\n\n\n"
            )
        if phase == "data preparation":
            return (
                sr_str,
                f"Current Literature Review: {self.lit_review_sum}\nPlan: {self.plan}",
                f"Current Plan: {self.plan}")
        elif phase == "running experiments":
           return (
               sr_str,
               f"Current Literature Review: {self.lit_review_sum}\n"
               f"Current Plan: {self.plan}\n"
               f"Current Dataset code: {self.dataset_code}\n"
               f"Storage path for original exposure data:{self.Original_exposure_data_path}\n"
               f"Storage path for original outcome data:{self.Original_outcome_data_path}\n"
           )
        return ""

    def example_command(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        return ()

    def command_descriptions(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        if phase == "data preparation":
            return (
                "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where 'dialogue here' is the actual dialogue you will send and DIALOGUE is just the word DIALOGUE.\n"
                "When you and the data analyst have finalized your dataset preparation code and are ready to submit the final code, please use the following command: ```SUBMIT_CODE\ncode here\n```\n where 'code here' is the finalized code you will send and SUBMIT_CODE is just the word SUBMIT_CODE. Do not use any classes or functions.  If your code returns any errors, they will be provided to you, and you are also able to see print statements.  Make sure function variables are created inside the function or passed as a function parameter. DO NOT CREATE A MAIN FUNCTION.\n"
                "Make sure to submit code in a reasonable amount of time. Do not make the code too complex, try to make it simple. Do not take too long to submit code. \n"
                "You can only use a single command per inference turn. Do not use more than one command per inference. If you use multiple commands, then only one of them will be executed, not both.\n"
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. SUBMIT_CODE, DIALOGUE).\n")
        elif phase == "running experiments":
            return(
                "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where 'dialogue here' is the actual dialogue you will send and DIALOGUE is just the word DIALOGUE.\n"
                "When you and the biological engineer have finalized your code and are ready to submit the final code, please use the following command: ```SUBMIT_CODE\ncode here\n```\n where 'code here' is the finalized code you will send and SUBMIT_CODE is just the word SUBMIT_CODE. Do not use any classes or functions.  If your code returns any errors, they will be provided to you, and you are also able to see print statements.  Make sure function variables are created inside the function or passed as a function parameter. DO NOT CREATE A MAIN FUNCTION.\n"
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. SUBMIT_CODE, DIALOGUE).\n"
            )
        return ""

    def phase_prompt(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        elif phase == "data preparation":
            phase_str = (
                "You are a algorithm engineer directing a data analyst, where the data analyst will be writing the code, and you can interact with them through dialogue.\n"
                "Your goal is to Help the data analyst write simple R code to prepare data for the experiment, integrating the provided plan.\n"
                "If all data preparation tasks have been successfully completed, your task is complete.Avoid repetition or unnecessary looping. \n"
                "Strictly Perform the following steps in order:\n"
                "1. Generating a search query for the exposure dataset.\n"
                "2. Generating a search query for the outcome dataset.\n"
                "3. Perform LD pruning on the exposure dataset with the following parameters: r2 (default: 0.001): squared correlation threshold. window_kb (default: 10000): window size in kilobases\n"
                "4. Load the exposure data from: research_results/processed_data/filtered_exposure_data.csv\n"
                "5. Load the outcome data from: research_results/processed_data/filtered_outcome_data.csv\n"
                "6. (Optional) Performing additional instrumental variable (IV) selection procedures.\n"
                "7. Performing data harmonization: Use the harmonise_data() function from the TwoSampleMR package to align alleles and remove ambiguous SNPs (e.g., A/T or C/G).\n"
                 )
        elif phase == "running experiments":
            phase_str = (
                "You are an Algorithm Engineer focused on optimizing R code, and you are directed by a Biological Engineer. You can interact with them through dialogue.\n"
                "You do not generate initial code, but improve existing code based on feedback from the Biological Engineer.Be sure to integrate the provided plan\n" 
                "The Biological Engineer checks the code, the outputs of each MR method, and the generated plots for correctness and completeness, and provides suggestions for improvements.\n"
                "You modify the code according to this feedback, addressing method calls, logic errors, or plotting issues.\n"
                "You collaborate through dialogue, forming an iterative optimization loop until the analysis results and visualizations are accurate and complete.\n"
            )
        return phase_str

    def role_description(self):
        return "An algorithm engineer dedicated to implementing Mendelian Randomization algorithms and processing large-scale genomic and GWAS datasets, working at a premier research institute."


def human_in_loop(phase, phase_prod):
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
            # reset_agents()
            # add suggestions to the notes
            notes.append({
                "phases": [phase],
                "note": notes_for_agent})
            return True
        else:
            print("Invalid response, type Y or N")
    return False



if __name__ == '__main__':
    # 获取与“running experiments”阶段相关的注释
    task_notes = {
        "plan formulation": [
            "You should design a plan for ONE experiment focused on testing the causal effect.",
            "DO NOT PLAN FOR TOO LONG. Submit your experiment plan early to allow feedback and iteration.",
            "Perform Mendelian Randomization (MR) analysis on the given exposure–outcome pair using the following methods: IVW, MR-Egger, Weighted Median, Weighted Mode, MR-PRESSO, and CAUSE."
        ],
        "data preparation": [
            "Do NOT conduct any causal analysis during this phase."
        ],
        "running experiments": [
            "You can use any MR package",
            "You must perform Mendelian Randomization (MR) analysis on the given exposure–outcome pair using the following methods: IVW, MR-Egger, Weighted Median, Weighted Mode, MR-PRESSO.",
            # only using CAUSE method. #using the following methods: IVW, MR-Egger, Weighted Median, Weighted Mode, MR-PRESSO, and CAUSE.
            "Provide detailed result logs and summaries after execution.",
            "Generate clear and well-labeled figures to interpret results,including Scatter plot for multi-method comparison (IVW, MR-Egger, Weighted Median, Weighted Mode), forest plot, leave-one-out analysis plot, and funnel plot ."
        ],
        "results interpretation": [
            "Summarize whether there is statistically significant causal evidence.",
            "Comment on consistency between methods and presence of potential pleiotropy or bias."
        ],
        "report writing": [
            "Your report should clearly state the causal question, methodology, data sources, and main findings.",
            "Include visualizations and statistical outputs that support your interpretation.",
            "End with conclusions and potential follow-up studies or validations."
        ]
    }
    # 将 task_notes 转换为列表字典形式，直接赋值给 notes
    notes = list()
    phase = "data preparation"
    for text in task_notes.get(phase,[]):
        if text is not None:
            notes.append({"phases": [phase], "content": text})
    population = "European"
    openai_api_key = os.getenv('OPENAI_API_KEY')
    research_topic = "Your goal is to investigate the causal relationship between Systolic Blood Pressure and Stroke (Any), using Mendelian Randomization based on GWAS summary statistics."
    # low density lipoprotein cholesterol and stroke
    model_backbone = "gpt-4o"
    max_steps = 100
    verbose = True
    human_in_loop_flag = True
    lab_dir = "successrate"
    global outcome_information, exposure_information
    algorithm_engineer = AlgorithmEngineer(model=model_backbone, notes=notes, max_steps=max_steps,
                           openai_api_key=openai_api_key)
    data_analyst = DataAnalyst(model=model_backbone, notes=notes, max_steps=max_steps,
                               openai_api_key=openai_api_key)
    max_tries = 100
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
        # print(f"@@ Lab #{lab_index} Paper #{paper_index} @@")
        if ae_feedback != "":
            ae_feedback_in = "Feedback provided to the algorithm engineer: " + ae_feedback
        else:
            ae_feedback_in = ""
        resp = algorithm_engineer.inference(research_topic, "data preparation",
                                                 feedback=f"{ae_dialogue}\nFeedback from previous command: {da_feedback}\n{ae_command}{ae_feedback_in}",
                                                 step=_i)
        da_feedback = str()
        da_dialogue = str()
        if "```DIALOGUE" in resp:
            dialogue = extract_prompt(resp, "DIALOGUE")
            da_dialogue = f"\nThe following is dialogue produced by the data analyst: {dialogue}\n"
            if verbose: print("#" * 40, f"\nThe following is dialogue produced by the data analyst: {dialogue}",
                                   "\n", "#" * 40)
        if "```SUBMIT_CODE" in resp:
            final_code = extract_prompt(resp, "SUBMIT_CODE")
            code_resp = execute_r_code(final_code, timeout=600)
            if verbose: print("!" * 100, "\n", f"CODE RESPONSE: {code_resp}")
            da_feedback += f"\nCode Response: {code_resp}\n"
            combined_datasets_info = exposure_information + "\n\n" + outcome_information
            # print(combined_datasets_info)
            if "[CODE EXECUTION ERROR]" in code_resp:
                da_feedback += "\nERROR: Final code had an error and could not be submitted! You must address and fix this error.\n"
            else:

                if human_in_loop_flag:
                    retry = human_in_loop("data preparation", final_code)
                    if retry: break
                save_to_file(f"./{lab_dir}", "load_data.R", final_code)
                # set_agent_attr("dataset_code", final_code)
                # set_agent_attr("dataset_information", combined_datasets_info)
                # set_agent_attr("Original_exposure_data_path", Original_exposure_data_path)
                # set_agent_attr("Original_outcome_data_path", Original_outcome_data_path)
                # # reset agent state
                # reset_agents()
                # statistics_per_phase["data preparation"]["steps"] = _i
                break

        if ae_feedback != "":
            ae_feedback_in = "Feedback from previous command: " + ae_feedback
        else:
            ae_feedback_in = ""
        resp = data_analyst.inference(
            research_topic, "data preparation",
            feedback=f"{da_dialogue}\n{ae_feedback_in}", step=_i)
        if verbose: print("data analyst: ", resp, "\n~~~~~~~~~~~")
        ae_feedback = str()
        ae_dialogue = str()
        ae_command = str()
        if "```DIALOGUE" in resp:
            dialogue = extract_prompt(resp, "DIALOGUE")
            ae_dialogue = f"\nThe following is dialogue produced by the algorithm engineer: {dialogue}\n"
            if verbose: print("#" * 40, f"\nThe following is dialogue produced by the algorithm engineer: {dialogue}",
                                   "#" * 40, "\n")
        if "```R" in resp:
            code = extract_prompt(resp, "R")
            code = data_analyst.dataset_code + "\n" + code
            code_resp = execute_r_code(code, timeout=600)
            ae_command = f"Code produced by the DA agent:\n{code}"
            ae_feedback += f"\nCode Response: {code_resp}\n"
            if verbose: print("!" * 100, "\n", f"CODE RESPONSE: {code_resp}")
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
                # 问题：检索到了研究，但是研究不一定公开了数据集
                # 修改：对研究进行排序之前，优先选择公开了数据集的研究，然后再排序
                # filter_results = GWAS_engine._filter_and_sort_data(hf_query_last,population)
                # # 如果 filter_results 为空，说明没有检索到相关的研究，需要换下一个检索词进行检索
                # if filter_results is None or filter_results.empty:
                #     continue
                # result_strs, down_list = GWAS_engine.results_str(filter_results)
                # print(down_list)
                # # 修改：检索返回不止一个研究，依次判断是否公开数据集
                # # down_list 是包含下载链接的列表，如果 down_list 为空, 说明该检索词检索发现没有可下载的数据集，则跳过当前检索词，考虑下一个检索词
                # if len(down_list) == 0:
                #     continue
                # hf_res = "\n".join(result_strs)
                # print(f"这是GWAS Catalog数据库的检索结果：{hf_res}")
                # # file_path 是文件下载之后的存储路径
                # file_path = GWAS_download.run(down_list[0])
                # if not file_path:
                #     continue
                # print(file_path)
                # is_download = True

                if data_type == "exposure":
                    filter_results = GWAS_engine._filter_and_sort_data(hf_query_last, data_type, population)
                    # 如果 filter_results 为空，说明没有检索到相关的研究，需要换下一个检索词进行检索
                    if filter_results is None or filter_results.empty:
                        continue
                    exposure_trait = hf_query_last
                    result_strs, down_list = GWAS_engine.results_str(filter_results,population)
                    # 修改：检索返回不止一个研究，依次判断是否公开数据集
                    # down_list 是包含下载链接的列表，如果 down_list 为空, 说明该检索词检索发现没有可下载的数据集，则跳过当前检索词，考虑下一个检索词
                    if len(down_list) == 0:
                        continue
                    hf_res = "\n".join(result_strs)
                    num = 0
                    total_num = len(result_strs)
                    # isvilid = False
                    print(f"共检索到{total_num}个研究，这是GWAS Catalog数据库中暴露数据的检索结果：{hf_res}")
                    exposure_information = hf_res.strip().split('-------------------------')[num]
                    Original_exposure_data_path = GWAS_download.run(down_list[num])
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
                    # set_agent_attr("Original_exposure_data_path", file_path)
                    if not Original_exposure_data_path:
                        continue
                    # print(file_path)
                    is_download = True
                    cfg.set("data_preparation", "search_query_of_exposure", hf_query_last)
                    cfg.set("data_preparation", "original_exposure_data_path", Original_exposure_data_path)
                    print(f"当前下载的是暴露变量 [{hf_query_last}] 的数据，已保存至 [{Original_exposure_data_path}]")
                    # 对数据进行列名映射
                    # 对暴露数据筛选显著工具变量
                    exposure_data, used_thresh = filter_significant_mutations(Original_exposure_data_path)
                    cfg.set("data_preparation", "threshold_used_for_selecting_significant_variables", used_thresh)

                    print(f"筛选显著变异，实际使用阈值: {used_thresh}")
                    exposure_data_path = "research_results/processed_data/exposure_data.csv"
                    exposure_data.to_csv(exposure_data_path, index=False)

                    exposure_file_path = exposure_data_path
                    print(f"已对暴露数据筛选完显著snps，已保存至 [{exposure_file_path}]")
                    # snps为0怎么办
                    # 生成snps列表
                    possible_cols = ["rsid", "hm_rsid", "variant_id"]
                    col_name = next((c for c in possible_cols if c in exposure_data.columns), None)

                    if col_name is None:
                        raise ValueError("data中未找到hm_rsid、rsid或variant_id字段")
                    exposure_data_rsid = exposure_data[[col_name]].drop_duplicates().rename(
                        columns={col_name: "hm_rsid"})
                    exposure_data_rsid_path = "research_results/processed_data/exposure_data_rsid.csv"
                    exposure_data_rsid.to_csv(exposure_data_rsid_path, index=False)
                    ae_command = f"Exposure GWAS search command produced by the ML agent:\n{hf_query}"
                    ae_feedback += (
                        f"Exposure GWAS Catalog dataset of {hf_query_last} has been successfully downloaded.The exposure data has been saved at the following path:{Original_exposure_data_path}.\n"
                        f"Significant variants were selected using a threshold of {used_thresh},the storage path is: {exposure_file_path}.And generate a list included only the rsid,which is stored in the{exposure_data_rsid_path} \n"
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
                    outcome_trait = hf_query_last
                    result_strs, down_list = GWAS_engine.results_str(filter_results,population)
                    if len(down_list) == 0:
                        continue
                    hf_res = "\n".join(result_strs)
                    num = 0
                    total_num = len(result_strs)
                    #isvilid = False
                    print(f"共检索到{total_num}个研究，这是GWAS Catalog数据库中结局数据的检索结果：{hf_res}")
                    # outcome_information = hf_res.strip().split('-------------------------')[0]
                    # file_path 是文件下载之后的存储路径
                    # 遍历下载，获取可用的数据行
                    filter_data = None
                    # SNP
                    exposure_outcome_intersection_snps_path = "research_results/processed_data/exposure_outcome_intersection_snps.csv"
                    last_outcome_data_path = "research_results/processed_data/last_outcome_data.csv"
                    is_first_write = True if not os.path.exists(exposure_outcome_intersection_snps_path) else False
                    for num,url in enumerate(down_list):
                        outcome_information = hf_res.strip().split('-------------------------')[num]
                        #print(f"取第{num + 1}个研究的数据集，验证性状是否符合用户要求")
                        print(f"第{num + 1}个研究信息是：{outcome_information}")
                        # 验证性状是否符合要求
                        #user_trait = hf_query_last
                        # 使用正则匹配 Disease/Trait 后面的内容，直到遇到下一个字段
                        #match = re.search(r"Disease/Trait:\s*(.*?)\s+Population:", outcome_information)
                        #if match:
                            #gwas_trait = match.group(1)
                        #else:
                            #print("Not found")
                        # openai_api_key = os.getenv('OPENAI_API_KEY')
                        #decision, explanation = phenotype_matcher(user_trait, gwas_trait, openai_api_key=openai_api_key)
                        #if decision == 'yes':
                            #print(f"经检查发现该数据集性状符合用户要求，{explanation}")
                            # file_path 是文件下载之后的存储路径
                        outcome_data_path = GWAS_download.run(url)
                        Original_outcome_data_path = outcome_data_path
                        is_download = True
                        cfg.set("data_preparation", "search_query_of_outcome", hf_query_last)
                        cfg.set("data_preparation", "original_outcome_data_path", outcome_data_path)
                        cfg.set(
                            "data_preparation",
                            "outcome_data_information",
                            json.dumps(outcome_information, indent=4)  # 保持字典结构
                        )
                        print(f"当前下载的是结局变量 [{hf_query_last}] 的数据，已保存至 [{outcome_data_path}]")
                        # else:
                        #     print(f"经检查发现该数据集性状不符合用户要求，原因是：{explanation}")
                        #     print("选取下一个数据集")
                        # # if not outcome_data_path:
                        #     continue

                        try:
                            data = pd.read_csv(outcome_data_path, sep='\t', compression='infer', low_memory=False)
                        except Exception as e:
                            print(f"读取失败: {e}")
                            continue

                        possible_cols = ["hm_rsid", "rsid", "variant_id"]
                        col_name = next((c for c in possible_cols if c in data.columns), None)
                        if col_name is None:
                            raise ValueError("data中未找到hm_rsid、rsid或variant_id字段")

                        temp_filter_data = data[[col_name]].drop_duplicates().rename(columns={col_name: "hm_rsid"})
                        # temp_exposure_data_rsid = exposure_data_rsid[['rsid']].drop_duplicates()
                        len_exposure_data_rsid = len(exposure_data_rsid)
                        exposure_data_rsid_temp = pd.read_csv(exposure_data_rsid_path)
                        exposure_data_rsid_temp.columns = ['hm_rsid']
                        # exposure_data_rsid = exposure_data_rsid.rename(columns={'rsid': 'hm_rsid'})
                        df_common = pd.merge(temp_filter_data, exposure_data_rsid_temp, how='inner', on='hm_rsid')
                        # filtered_rows  (8896,1)
                        num_rows = df_common.shape[0]
                        print(f"本次取交集有 {num_rows} 行数据，来源文件: {outcome_data_path}")
                        data = data.rename(columns={col_name: "hm_rsid"})
                        filtered_rows = pd.merge(data, df_common, on='hm_rsid', how='inner')
                        # filtered_rows  (8919,24)
                        if df_common.empty:
                            continue
                        # 添加标记列（本地文件地址）
                        # df_common['source_file'] = outcome_data_path
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
                        # if num_rows < 1000 and num_rows < len_exposure_data_rsid:
                        if num_rows < len_exposure_data_rsid:
                            continue
                        else:
                            break
                    ae_command = f"Outcome GWAS search command produced by the ML agent:\n{hf_query}"
                    ae_feedback += (
                        f"Outcome GWAS Catalog dataset of {hf_query_last} has been successfully downloaded.The outcome data has been saved at the following path: {outcome_data_path}. "
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
            prune_params_str = extract_prompt(resp, "LD_PRUNE")
            # print("prune_params_str:", prune_params_str)
            prune_params = ast.literal_eval(prune_params_str)
            # print("prune_params:", prune_params)
            # 执行LD剪枝
            # if 'exposure_file_path' in locals():
            exposure_outcome_intersection_snps_path = "research_results/processed_data/exposure_outcome_intersection_snps.csv"
            snps_data = pd.read_csv(exposure_outcome_intersection_snps_path)
            snps_list = snps_data['hm_rsid']
            if len(snps_list) != 0:
                print(f"开始对snps列表进行LD剪枝，参数：{prune_params}")
                # 如果 prune_params 字典中有键 'r2'，就使用它的值；否则，使用默认值 0.1。
                LDagent = LDPrunerAgent(
                    gwas_file=exposure_outcome_intersection_snps_path,
                    # pval_threshold=5e-8,
                    output_dir="./research_results/ld_processed_data",
                    r2=prune_params.get('r2', 0.001),
                    window_kb=prune_params.get('window_kb', 10000)
                )
                result = LDagent.command_run_auto({
                    "gwas_file": exposure_outcome_intersection_snps_path,
                    "min_snps_threshold": 10,
                    "max_attempts": 10,
                    "step": 10,
                    "r2": prune_params.get('r2'),
                    "window_kb": prune_params.get('window_kb')
                })
                # print("剪枝结果文件:", result["pruned_gwas_file"])
                print("使用的r2参数:", result["r2_used"])
                print("使用的window_kb:", result["window_kb_used"])
                print("剪枝后SNP数量:", result["snp_count"])
                cfg.set("data_preparation", "ld_prune",
                        {"r2": result["r2_used"], "window_kb": result["window_kb_used"]})
                cfg.set("data_preparation", "snp_count", result["snp_count"])

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
                        'id.exposure': exposure_trait,  # 这里可以根据具体暴露变量名替换
                        'exposure': exposure_trait,
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
                        'id.outcome': outcome_trait,  # 这里可以根据具体结局变量名替换
                        'outcome': outcome_trait,
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

                filtered_outcome_data = pd.merge(filtered_outcome_data_mapped, exposure_data_rsid, how='inner', on='SNP')

                # 保存文件，供后续R读取
                filtered_exposure_data_path = "research_results/processed_data/filtered_exposure_data.csv"
                filtered_outcome_data_path = "research_results/processed_data/filtered_outcome_data.csv"
                cfg.set("data_preparation", "filtered_exposure_data_path", filtered_exposure_data_path)
                cfg.set("data_preparation", "filtered_outcome_data_path", filtered_outcome_data_path)
                if len(exposure_data_rsid)>=10:
                    exposure_df_filtered.to_csv(filtered_exposure_data_path, index=False)
                    filtered_outcome_data.to_csv(filtered_outcome_data_path, index=False)
                else:
                    print("警告: 筛选F>10的SNP之后，SNP 数量不足 10，保存原始映射数据")
                    filtered_exposure_data_mapped.to_csv(filtered_exposure_data_path, index=False)
                    filtered_outcome_data_mapped.to_csv(filtered_outcome_data_path, index=False)

                # filtered_exposure_data = pd.merge(exposure_df, ld_pruned_snps, on="rsid", how="inner")
                # filtered_exposure_data_path = "research_results/processed_data/filtered_exposure_data.csv"
                # filtered_exposure_data.to_csv(filtered_exposure_data_path, index=False)
                #
                # filtered_outcome_data = pd.merge(outcome_df, ld_pruned_snps, on="rsid", how="inner")
                # filtered_outcome_data_path = "research_results/processed_data/filtered_outcome_data.csv"
                # filtered_outcome_data.to_csv(filtered_outcome_data_path, index=False)

                # 现在 filtered_exposure 和 filtered_outcome 就是筛选后的数据
                # 调用合并函数，将剪枝后的暴露数据和已准备好的结局数据合并
                # merger = GWASMerger()
                # last_merge_data,last_merge_data_path = merger.merge(filtered_exposure_data, filtered_outcome_data)
                # print(f"合并之后的数据为：{last_merge_data}")

                ae_command = f"LD pruning command produced by the ML agent:\n{prune_params_str}"
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

    # raise Exception("Max tries during phase: Data Preparation")
    # 循环结束后处理 plan
    if final_code is None:
        raise Exception("Max tries during phase: Plan Formulation")