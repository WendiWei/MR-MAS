from utils import *
from tools import *
from inference import *
import random, string


class ReviewersAgent:
    def __init__(self, model="gpt-4o-mini", notes=None, openai_api_key=None):
        if notes is None: self.notes = []
        else: self.notes = notes
        self.model = model
        self.openai_api_key = openai_api_key

    def inference(self, plan, report):
        reviewer_1 = "You are a harsh but fair reviewer and expect good experiments that lead to insights for the research topic."
        review_1 = get_score(outlined_plan=plan, latex=report, reward_model_llm=self.model, reviewer_type=reviewer_1, openai_api_key=self.openai_api_key)

        reviewer_2 = "You are a harsh and critical but fair reviewer who is looking for an idea that would be impactful in the field."
        review_2 = get_score(outlined_plan=plan, latex=report, reward_model_llm=self.model, reviewer_type=reviewer_2, openai_api_key=self.openai_api_key)

        reviewer_3 = "You are a harsh but fair open-minded reviewer that is looking for novel ideas that have not been proposed before."
        review_3 = get_score(outlined_plan=plan, latex=report, reward_model_llm=self.model, reviewer_type=reviewer_3, openai_api_key=self.openai_api_key)

        return f"Reviewer #1:\n{review_1}, \nReviewer #2:\n{review_2}, \nReviewer #3:\n{review_3}"


class BaseAgent:
    def __init__(self, model="gpt-4o-mini", notes=None, max_steps=100, openai_api_key=None):
        if notes is None: self.notes = []
        else: self.notes = notes
        self.max_steps = max_steps
        self.model = model
        self.phases = []
        self.plan = str()
        self.report = str()
        self.history = list()
        self.prev_comm = str()
        self.prev_report = str()
        self.exp_results = str()
        self.dataset_code = str()
        self.dataset_information = str()

        self.Original_exposure_data_path = str(),
        self.Original_outcome_data_path = str(),
        self.results_code = str()
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
        model_resp = query_model(model_str=self.model, system_prompt=sys_prompt, prompt=query, temp=temp, openai_api_key=self.openai_api_key)
        return model_resp

    def inference(self, research_topic, phase, step, feedback="", temp=None):
        sys_prompt = f"""You are {self.role_description()} \nTask instructions: {self.phase_prompt(phase)}\n{self.command_descriptions(phase)}"""
        context = self.context(phase)
        history_str = "\n".join([_[1] for _ in self.history])
        phase_notes = [_note for _note in self.notes if phase in _note["phases"]]
        notes_str = f"Notes for the task objective: {phase_notes}\n" if len(phase_notes) > 0 else ""
        complete_str = str()
        if step/(self.max_steps-1) > 0.7: complete_str = "You must finish this task and submit as soon as possible!"
        prompt = (
            f"""{context}\n{'~' * 10}\nHistory: {history_str}\n{'~' * 10}\n"""
            f"Current Step #{step}, Phase: {phase}\n{complete_str}\n"
            f"[Objective] Your goal is to perform research on the following topic: {research_topic}\n"
            f"Feedback: {feedback}\nNotes: {notes_str}\nYour previous command was: {self.prev_comm}. Make sure your new output is very different.\nPlease produce a single command below:\n")
        model_resp = query_model(model_str=self.model, system_prompt=sys_prompt, prompt=prompt, temp=temp, openai_api_key=self.openai_api_key)
        print("^"*50, phase, "^"*50)
        model_resp = self.clean_text(model_resp)
        self.prev_comm = model_resp
        steps_exp = None
        if feedback is not None and "```EXPIRATION" in feedback:
            steps_exp = int(feedback.split("\n")[0].replace("```EXPIRATION ", ""))
            feedback = extract_prompt(feedback, "EXPIRATION")
        self.history.append((steps_exp, f"Step #{step}, Phase: {phase}, Feedback: {feedback}, Your response: {model_resp}"))
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


class WritingSpecialist(BaseAgent):
    def __init__(self, model="gpt4omini", notes=None, max_steps=100, openai_api_key=None):
        super().__init__(model, notes, max_steps, openai_api_key)
        self.phases = ["report writing"]

    def generate_readme(self):
        sys_prompt = f"""You are {self.role_description()} \n Here is the written paper \n{self.report}. Task instructions: Your goal is to integrate all of the knowledge, code, reports, and notes provided to you and generate a readme.md for a github repository."""
        history_str = "\n".join([_[1] for _ in self.history])
        prompt = (
            f"""History: {history_str}\n{'~' * 10}\n"""
            f"Please produce the readme below in markdown:\n")
        model_resp = query_model(model_str=self.model, system_prompt=sys_prompt, prompt=prompt, openai_api_key=self.openai_api_key)
        return model_resp.replace("```markdown", "")

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
        if phase == "report writing":
           return (
               sr_str,
               f"Current Literature Review: {self.lit_review_sum}\n"
               f"Current Plan: {self.plan}\n"
               f"Current Dataset code: {self.dataset_code}\n"
               f"Current Experiment code: {self.results_code}\n"
               f"Current Results: {self.exp_results}\n"
               f"Current Interpretation of results: {self.interpretation}\n"
           )
        return ""

    def example_command(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        return (
            "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where dialogue here is the actual dialogue you will send and DIALOGUE is just the word DIALOGUE.\n"
            "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\n<Insert command here>\n``` where COMMAND is the specific command you want to run (e.g. REPORT, DIALOGUE).\n")

    def command_descriptions(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        return (
            "When you believe a good paper has been arrived at between you and the biological engineer you can use the following command to end the dialogue and submit the plan ```LATEX\nreport here\n```\n where report here is the actual report written in compilable latex to be transmitted and LATEX is just the word LATEX.\n"
            "Your report should include numbers, relevant metrics to the experiment and measures of significance. You must propagate this information accurately. You must also submit the report promptly. Do not delay too long.\n"
            "You must be incredibly detailed about what you did for the experiment and all of the findings.\n"
            )

    def phase_prompt(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        phase_str = (
            "You are directing a biological engineer to help them write a paper in latex based on results from an experiment, and you interact with them through dialogue.\n"
            "Your goal is to write a report in latex for an experiment. You should read through the code, read through the interpretation, and look at the results to understand what occurred. You should then discuss with the biological engineer how they can write up the results and give their feedback to improve their thoughts.\n"
        )
        return phase_str

    def role_description(self):
        return "An experienced academic researcher at a top university, guiding researchers in scientific writing and publication."


class StatisticalGeneticist(BaseAgent):
    def __init__(self, model="gpt4omini", notes=None, max_steps=100, openai_api_key=None):
        super().__init__(model, notes, max_steps, openai_api_key)
        self.phases = ["plan formulation", "results interpretation"]

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
        if phase == "plan formulation":
            return (
                sr_str,
                f"Current Literature Review: {self.lit_review_sum}",
            )
        elif phase == "results interpretation":
            return (
                sr_str,
                f"Current Literature Review: {self.lit_review_sum}\n"
                f"Current Plan: {self.plan}\n"
                f"Current Dataset code: {self.dataset_code}\n"
                f"Current Datasets information: {self.dataset_information}\n"
                f"Current Experiment code: {self.results_code}\n"
                f"Current Results: {self.exp_results}"
            )
        return ""

    def example_command(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        return ()

    def command_descriptions(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        if phase == "plan formulation":
            return (
                "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where dialogue here is the actual dialogue you will send and DIALOGUE is just the word DIALOGUE.\n"
                "When you believe a good plan has been arrived at between you and the biological engineer you can use the following command to  to end the dialogue and submit the plan ```PLAN\nplan here\n```\n where plan here is the actual plan to be transmitted and PLAN is just the word PLAN. The plan should provide a clear execution outline and cover the entire automated Mendelian Randomization analysis process.\n"
                "You can only use a SINGLE command per inference turn. Do not use more than one command per inference. If you use multiple commands, then only one of them will be executed, NOT BOTH.\n"
                "Make sure not to produce too much dialogue and to submit an plan in reasonable time."
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. PLAN, DIALOGUE).\n"
            )
        elif phase == "results interpretation":
            return (
                "When you believe a good interpretation has been arrived at between you and the biological engineer you can use the following command to end the dialogue and submit the plan ```INTERPRETATION\ninterpretation here\n```\n where interpretation here is the actual interpretation to be transmitted and INTERPRETATION is just the word INTERPRETATION. Please provide an INTERPRETATION in a reasonable amount of time.\n"
                "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where dialogue here is the actual dialogue you will send and DIALOGUE is just the word DIALOGUE.\n"
                "You must submit the interpretation during this phase in a reasonable amount of time. Do not delay the submission."
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. INTERPRETATION, DIALOGUE).\n"
            )

    def phase_prompt(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        if phase == "plan formulation":
            phase_str = (
                "You are directing a biological engineer to help them come up with a good plan, and you interact with them through dialogue.\n"
                "Your goal is to produce a comprehensive, end-to-end foundational experimental plan for Mendelian Randomization (MR) analyses on the given topic.\n"
                "The plan should include the following core components:\n"
                "1. Data Acquisition: Generate search queries sequentially for the exposure and outcome traits.Obtain GWAS summary statistics for the exposure and outcome traits from the GWAS Catalog database.Data acquisition for this part of the program content is only required to generate search terms. DO NOT use the TwoSampleMR package for data searching or loading.DO NOT use web URLs or placeholder links to download data.\n"
                "2. Instrumental Variable Selection: Select significant variants from the exposure dataset.Perform LD pruning on both the filtered exposure and outcome datasets.Optionally, apply novel methods to further refine the selection of instrumental variables.Harmonize alleles between datasets.\n"
                "3. Mendelian randomization Analysis: Use standard Mendelian Randomization methods, optionally apply novel methods, and emphasize comparisons between different approaches.\n"
                "4. Optional Sensitivity Analyses: In sensitivity analysis, the primary goal is to assess the robustness of causal inference using different methods, ensuring that the results are not influenced by potential biases. \n"
                "5. Results Output and Visualization: Output results as summary tables and generate basic plots."
                "The data requirements may differ for different Mendelian Randomization methods, so the plan should clearly specify how each method should handle the data.\n"
                "Your plan should clearly state:What processing steps will be applied.What MR methods will be used.How to perform sensitivity analysis.What kind of plots should be generated and how to generate them.What tools or R packages are required.\n"
                 )
        elif phase == "results interpretation":
            phase_str = (
                "You are directing a biological engineer to help them come up with an interpretation for results from an experiment, and you interact with them through dialogue.\n"
                "Your goal is to interpret results from experiments that were previously run. You should read through the code and look at the results to understand what occurred. You should then discuss with the biological engineer how they can interpret the results and give their feedback to improve their thoughts. You should integrate the provided literature review, code, and plans to come up with an exciting interpretation that could make a compelling paper. Your plans should provide a clear outline that can be used to write an academic paper.\n"
                "Your interpretation must include the effect estimates (e.g., causal β coefficients), standard errors, confidence intervals, and p-values for each MR method (such as IVW, MR-Egger, and Weighted Median), and accurately communicate and interpret these values. You should also consider whether the results are consistent across methods and whether sensitivity analyses (such as the MR-Egger intercept test, heterogeneity test, leave-one-out analysis, and MR-PRESSO) support the robustness of the findings. You must also complete this in a reasonable amount of time and then submit your results.\n"
            )
        return phase_str

    def role_description(self):
        return "An experienced statistical geneticist specializing in Mendelian Randomization and genomic causal analysis."


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
                "You can generate a phenotype keyword that best matches the trait description in the research objective using the following command: ```SEARCH_GWAS\ntrait keyword here\n``` where trait keyword here is your generated term，which should align closely with the efo_trait field in the GWAS Catalog to ensure highly relevant search results,and SEARCH_GWAS is the word SEARCH_GWAS.Please note that the generated search terms should not use underscores to connect compound words (e.g., systolic_blood_pressure); instead, they should use spaces (e.g., systolic blood pressure).The command will return metadata and file paths for datasets related to the specified trait. When generating keywords for GWAS search, please add the suffix “_exposure” for exposure variables and “_outcome” for outcome variables to the keywords according to the user's analytical objectives, for example “body mass index_exposure” or ‘type 2 diabetes_outcome’.Do not download data again—datasets have already been fetched. Your task is to write code that loads data from the returned file paths, using only the datasets from the GWAS Catalog.\n"
                "You can perform LD pruning using the following command: ```LD_PRUNE\n{'r2': value, 'window_kb': value}\n``` where LD_PRUNE is the word LD_PRUNE, and the JSON object includes the pruning parameters to be used. The keys are 'r2' for the squared correlation threshold and 'window_kb' for the window size in kilobases. This command instructs the agent to perform LD pruning with the specified parameters using PLINK. \n"
                "You MUST use a GWAS Catalog dataset in your code. DO NOT CREATE A MAIN FUNCTION. Try to make the code very simple.\n"
                "You can only use a SINGLE command per inference turn. Do not use more than one command per inference. If you use multiple commands, then only one of them will be executed, NOT BOTH.\n"
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. R, DIALOGUE, SEARCH_GWAS).\n"
            )

    def phase_prompt(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        if phase == "data preparation":
            phase_str = (
                "You are a data analyst working under the guidance of a algorithm engineer. You will collaborate with algorithm engineerby to write R code through dialogue.\n",
                "Your current task is to write executable R code that prepares and processes GWAS data for use in a Mendelian Randomization (MR) experiment. You must follow the provided plan.\n",
                # "All steps must be implemented via R code that performs all necessary data preparation steps from start to finish. Do NOT complete any step manually in advance.\n",
                "Strictly Perform the following steps in order:\n"
                "Execution Steps (Strict Order):\n"
                "1. Generating a search query for the exposure dataset.\n"
                "2. Generating a search query for the outcome dataset.\n"
                "First, generate the search terms for the exposure variables, and then generate the search terms for the outcome variables.After generating the search term, do not write code to retrieve the data yourself; instead, generate the SEARCH_GWAS command.\n"
                "3. Perform LD pruning on the exposure dataset with the following parameters: r2 (default: 0.001): squared correlation threshold. window_kb (default: 10000): window size in kilobases\n"
                "4. Load the exposure data from: research_results/processed_data/filtered_exposure_data.csv\n"
                "5. Load the outcome data from: research_results/processed_data/filtered_outcome_data.csv\n"
                "6. (Optional) Performing additional instrumental variable (IV) selection procedures.\n"
                "7. Performing data harmonization: Use the harmonise_data() function from the TwoSampleMR package to align alleles and remove ambiguous SNPs (e.g., A/T or C/G).\n"
                "After performing the harmonization operation, the task is complete. Do not perform any other operations.\n"
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
                "3. Load the exposure data from: research_results/processed_data/filtered_exposure_data.csv\n"
                "4. Load the outcome data from: research_results/processed_data/filtered_outcome_data.csv\n"
                "5. (Optional) Performing additional instrumental variable (IV) selection procedures.\n"
                "6. Performing data harmonization: Use the harmonise_data() function from the TwoSampleMR package to align alleles and remove ambiguous SNPs (e.g., A/T or C/G).\n"
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


class BiologicalEngineer(BaseAgent):
    def __init__(self, model="gpt4omini", notes=None, max_steps=100, openai_api_key=None):
        super().__init__(model, notes, max_steps, openai_api_key)
        self.phases = [
            "literature review",
            "plan formulation",
            "running experiments",
            "results interpretation",
            "report writing",
            "report refinement",
        ]
        self.lit_review = []

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
        if phase == "plan formulation":
            return (
                sr_str,
                f"Current Literature Review: {self.lit_review_sum}",)
        elif phase == "data preparation":
            return (
                sr_str,
                f"Current Literature Review: {self.lit_review_sum}\n"
                f"Current Plan: {self.plan}"
            )
        elif phase == "running experiments":
            return (
                sr_str,
                f"Current Literature Review: {self.lit_review_sum}\n"
                f"Current Plan: {self.plan}\n"
                f"Current Dataset code: {self.dataset_code}\n"
                f"Current Datasets information: {self.dataset_information}\n"
            )
        elif phase == "results interpretation":
            return (
                sr_str,
                f"Current Literature Review: {self.lit_review_sum}\n"
                f"Current Plan: {self.plan}\n"
                f"Current Dataset code: {self.dataset_code}\n"
                f"Current Datasets information: {self.dataset_information}\n"
                f"Current Experiment code: {self.results_code}\n"
                f"Current Results: {self.exp_results}"
            )
        elif phase == "report refinement":
            return (
                sr_str,
                f"Current Literature Review: {self.lit_review_sum}\n"
                f"Current Plan: {self.plan}\n"
                f"Current Dataset code: {self.dataset_code}\n"
                f"Current Datasets information: {self.dataset_information}\n"
                f"Current Experiment code: {self.results_code}\n"
                f"Current Results: {self.exp_results}\n"
                f"Current Interpretation of results: {self.interpretation}"
            )
        elif phase == "literature review":
            return sr_str
        else:
            return ""

    def requirements_txt(self):
        sys_prompt = f"""You are {self.role_description()} \nTask instructions: Your goal is to integrate all of the knowledge, code, reports, and notes provided to you and generate a requirements.txt for a github repository for all of the code."""
        history_str = "\n".join([_[1] for _ in self.history])
        prompt = (
            f"""History: {history_str}\n{'~' * 10}\n"""
            f"Please produce the requirements.txt below in markdown:\n")
        model_resp = query_model(model_str=self.model, system_prompt=sys_prompt, prompt=prompt, openai_api_key=self.openai_api_key)
        return model_resp

    def example_command(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        return ()

    def command_descriptions(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        if phase == "literature review":
            return (
                "To collect paper summaries, use the following command: ```SUMMARY\nSEARCH QUERY\n```\n where SEARCH QUERY is a string that will be used to find papers with semantically similar content and SUMMARY is just the word SUMMARY. Never use the same search term more than once.You must generate a very short queries (2–4 keywords) used to find semantically similar papers. Use standard biomedical terms instead of vague or general words.Avoid vague words like “risk”, “health”, or “disease”.Each search query must be unique — do not repeat any previously used combinations.Do not use the same search term more than once. Make sure your search queries are very short.\n"
                "To get the full paper text for an Pubmed paper, use the following command: ```FULL_TEXT\nPubMed paper ID\n```\n where PubMed paper ID is the ID of the PubMed paper (which can be found by using the SUMMARY command), and FULL_TEXT is just the word FULL_TEXT. Make sure to read the full text using the FULL_TEXT command before adding it to your list of relevant papers.\n"
                "After reading the full text, you must make a decision : If the paper is (1) relevant to the current research topic, (2) introduces a novel MR method, or (3) is considered a foundational or widely used MR method paper (e.g., IVW, MR-Egger, Weighted Median, MR-PRESSO, CAUSE),then you must add it using the following command: ```ADD_PAPER\nPubMed_paper_ID\nPAPER_SUMMARY\n```\nwhere PubMed_paper_ID is the ID of the PubMed paper, PAPER_SUMMARY is a brief summary of the paper that includes key information such as the main objective of the research, a summary of the methods used including algorithms, statistical models, and experimental designs, details about any computational tools, software packages, or scripts mentioned in the paper, along with installation links or instructions and guidance on how to use them for experiment replication. It also includes key experimental results and conclusions, as well as links to any provided code or scripts. The summary should focus not only on the paper’s content but also on the technical details that can be directly applied to the subsequent experimental phase, and ADD_PAPER is just the word ADD_PAPER. You can only add one paper at a time. \n"
                "Make sure to use ADD_PAPER when you see a relevant paper.You only need to add the same paper once, don't add it repeatedly. DO NOT use SUMMARY too many times."
                "If the full text of a paper is not retrieved, then skip that paper.\n"
                "You can only use a single command per inference turn. Do not use more than one command per inference. If you use multiple commands, then only one of them will be executed, not both.\n"
                "Make sure to extensively discuss the experimental results in your summary.\n"
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. ADD_PAPER, FULL_TEXT, SUMMARY). Do not use the word COMMAND make sure to use the actual command, e.g. your command should look exactly like this: ```ADD_PAPER\ntext\n``` (where the command could be from ADD_PAPER, FULL_TEXT, SUMMARY)\n"
            )
        # You must generate a short search query (2–4 keywords) that combines one exposure, one outcome, and optionally one method.To conduct an effective literature review for a Mendelian Randomization (MR) study, alternate your search strategy across rounds. In odd-numbered rounds, generate short queries (2–4 keywords) combining one exposure, one outcome, and optionally a method term like “MR”. In even-numbered rounds, focus on MR methodology, using short phrases related to specific methods (e.g., “MR Egger”, “CAUSE”).that like:exposure + outcome、exposure + MR、outcome + MR、exposure + outcome + MR、specific methods of MR.
        # You must combine at least one exposure (e.g., BMI, LDL-C, smoking) and one outcome (e.g., coronary artery disease, Type 2 Diabetes).Optionally, you may include a method keyword such as “Mendelian Randomization”, “MR-Egger”, “CAUSE”, or “MR”.
        elif phase == "plan formulation":
            return (
                "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where 'dialogue here' is the actual dialogue you will send and DIALOGUE is just the word DIALOGUE.\n"
                "You can only use a single command per inference turn. Do not use more than one command per inference. If you use multiple commands, then only one of them will be executed, not both.\n"
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. DIALOGUE).\n"
            )
        elif phase == "data preparation":
            return (
                "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where 'dialogue here' is the actual dialogue you will send and DIALOGUE is just the word DIALOGUE.\n"
                "When you and the algorithm engineer have finalized your dataset preparation code and are ready to submit the final code, please use the following command: ```SUBMIT_CODE\ncode here\n```\n where 'code here' is the finalized code you will send and SUBMIT_CODE is just the word SUBMIT_CODE. Do not use any classes or functions. The submitted code must have a HuggingFace dataset import and must use an external HuggingFace dataset. If your code returns any errors, they will be provided to you, and you are also able to see print statements.  Make sure function variables are created inside the function or passed as a function parameter. DO NOT CREATE A MAIN FUNCTION.\n"
                "Make sure to submit code in a reasonable amount of time. Do not make the code too complex, try to make it simple. Do not take too long to submit code. Submit the code early. You should submit the code ASAP.\n"
                "You can only use a single command per inference turn. Do not use more than one command per inference. If you use multiple commands, then only one of them will be executed, not both.\n"
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. SUBMIT_CODE, DIALOGUE).\n")
        elif phase == "running experiments":
            return (
                "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where 'dialogue here' is the actual dialogue you will send and DIALOGUE is just the word DIALOGUE.\n"
                "Make sure not to produce too much dialogue and to submit an plan in reasonable time."
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. DIALOGUE).\n"
            )
        elif phase == "results interpretation":
            return (
                "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where 'dialogue here' is the actual dialogue you will send and DIALOGUE is just the word DIALOGUE.\n"
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. DIALOGUE).\n"
            )
        elif phase == "report writing":
           return (
               "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where 'dialogue here' is the actual dialogue you will send and DIALOGUE is just the word DIALOGUE.\n"
               "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. DIALOGUE).\n")
        elif phase == "report refinement":
            return ""
        return ""

    def phase_prompt(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")

        if phase == "literature review":
            phase_str = (
                "Your goal is to perform a literature review for the presented task and add papers to the literature review.\n"
                "You have access to PubMed and can perform two search operations: (1) finding many different paper summaries from a search query and (2) getting a single full paper text for an PubMed paper.Once you have retrieved a list of papers using the SUMMARY command, immediately follow up by reading the full text of any paper that appears relevant using the FULL_TEXT command.\n"
                "You must not only identify papers directly relevant to the research topic, but also collect methodological and classic MR papers.These include papers that introduce or compare MR methods such as IVW, MR-Egger, CAUSE, MR-PRESSO, etc.After reading the full text, always decide whether to add the paper. A paper that is relevant, **introduces a novel MR method, or is widely cited as foundational must be added.This ensures that your review captures both application-specific and theoretical developments in Mendelian Randomization.\n"
            )
            # "your goal is not only to retrieve papers specifically relevant to the research topic, but also to collect foundational and methodological papers that introduce or compare core Mendelian Randomization techniques.If you find a paper that (1) is directly related to the research topic, (2) introduces a novel MR method, or (3) is widely cited as a classic in MR methodology (e.g., IVW, MR-Egger, Weighted Median, MR-PRESSO), you should add it to the literature review using the ADD_PAPER command after reading its full text.This ensures the literature review includes not only applied studies but also the theoretical underpinnings of Mendelian Randomization.Prioritize meaningful and concise summaries, and avoid adding irrelevant or duplicated content.\n"
            rev_papers = "Papers in your review so far: " + " ".join([_paper["PubMed_id"] for _paper in self.lit_review])
            phase_str += rev_papers if len(self.lit_review) > 0 else ""
        elif phase == "plan formulation":
            phase_str = (
                "You are a biological engineer being directed by a statistical geneticist who will help you come up with a good plan, and you interact with them through dialogue.\n" 
                "Your goal is to produce plans that would make good experiments for the given topic.You should aim for a very simple experiment that showcases your plan, not a complex one.\n" 
                "Your plans should provide a clear outline for how to achieve the task,including What processing steps will be applied and how to realize.What methods will be used and how to realize.What tools or R packages are required (e.g., the TwoSampleMR package).\n"
                "Your current task is to construct a complete, end-to-end basic MR experiment plan\n"
                # "You are a biological engineer being directed by a statistical geneticist  who will help you come up with a good plan, and you interact with them through dialogue.\n"
                # "Your goal is to produce plans that would make good experiments for the given topic. You should aim for a very simple experiment that showcases your plan, not a complex one. You should integrate the provided literature review and come up with plans on how to expand and build on these works for the given topic. Your plans should provide a clear outline for how to achieve the task,  including the Mendelian randomization methods to be used and implemented, the datasets to be identified and analyzed, and the specific details of the experiment. \n"
                # Your idea should be very innovative and unlike anything seen before.
            )
        elif phase == "running experiments":
            phase_str = (
                "You are a biological engineer directing an algorithm engineer, where the algorithm engineer will writing the code, and you can interact with them through dialogue.\n"
                "Your goal is to write R code that obtains final results in MR analysis study. Be sure to integrate the provided plan, and ensure your code implements all the steps outlined in the plan. The data loading code will be added to the beginning of your code always, so this does not need to be rewritten. You should aim for simple code, not complex monolithic code. The code must be as simple as possible, easy to understand and execute.\n"
            )
        elif phase == "results interpretation":
            phase_str = (
                "You are a Biological Engineer directing an Algorithm Engineer, where the Algorithm Engineer will optimize the R code for Mendelian Randomization based on your feedback, and you can interact with them through dialogue.\n"
                "You are a biological engineer being directed by a statistical geneticist  who will help you come up with an interpretation for results from an experiment, and you interact with them through dialogue.\n"
                "Your goal is to interpret results from experiments that were previously run. You should read through the code and look at the results to understand what occurred. You should then discuss with the statistical geneticist  your interpretation and use their feedback to improve your thoughts. You should integrate the provided literature review, code, and plans to come up with an exciting interpretation that could make a compelling paper. Your plans should provide a clear outline that can be used to write an academic paper.\n"
                "Your interpretation must include effect estimates (e.g., causal β coefficients), standard errors, confidence intervals, and p-values for each MR method used (e.g., IVW, MR-Egger, Weighted Median), and accurately communicate and interpret these values. You should also consider whether the results are consistent across methods and whether sensitivity analyses (e.g., MR-Egger intercept, heterogeneity test, leave-one-out analysis, MR-PRESSO) support the robustness of the findings.\n"
                "You must submit the interpretation during this phase in a reasonable amount of time. Do not delay the submission."
            )
        elif phase == "report writing":
           phase_str = (
               "You are a biological engineer being directed by a writing specialist who will help you write a report based on results from an experiment, and you interact with them through dialogue.\n"
               "Your goal is to write a report for an experiment entirely in latex. You should read through the code, read through the interpretation, and look at the results to understand what occurred. You should then discuss with the writing specialist how you can write up the results and receive their feedback to improve your thoughts.\n"
               "Your report should include numbers, relevant metrics to the experiment and measures of significance  in latex. You must propagate this information accurately.\n"
               "You must be incredibly detailed about what you did for the experiment and all of the findings.\n"
           )
        elif phase == "report refinement":
            phase_str = (
                "You are a biological engineer who has submitted their paper to a bioinformatics journal called Bioinformatics. Your goal was to write a research paper and get high scores from the reviewers so that it get accepted to the journal.\n"
            )
        else:
            phase_str = ""
        return phase_str

    def role_description(self):
        return "A bioinformatics engineer dedicated to genomic data analysis and Mendelian Randomization studies, working at a premier research institute."

    def add_review(self, review, arx_eng, agentrxiv=False, GLOBAL_AGENTRXIV=None):
        try:
            if agentrxiv:
                # 如果使用的是 AgentRxiv 数据源：
                # 从 review 文本中提取 PubMed ID（第一行）和摘要（其余部分）
                PubMed_id = review.split("\n")[0]
                review_text = "\n".join(review.split("\n")[1:])
                # 调用 AgentRxiv 全局对象获取该论文的全文
                full_text = GLOBAL_AGENTRXIV.retrieve_full_text(PubMed_id, )
            else:
                # 如果使用的是默认的 arx_eng 引擎：
                # 从 review 中提取 PubMed ID 和摘要，按第一行和其余部分分开
                PubMed_id, review_text = review.strip().split("\n", 1)
                # 调用 arx_eng 引擎获取该论文全文
                pmcid = arx_eng.get_paper_pmcid(PubMed_id)
                full_text = arx_eng.retrieve_full_paper_text(pmcid)

            # 构造文献综述条目，包括 PubMed ID、全文和用户提供的摘要
            review_entry = {
                "PubMed_id": PubMed_id,
                "full_text": full_text,
                "summary": review_text,
            }

            # 将该条目添加到文献综述列表中
            self.lit_review.append(review_entry)
            print(self.lit_review)

            # 返回成功消息和全文内容
            return f"Successfully added paper {PubMed_id}", full_text

        except Exception as e:
            # 如果过程中出现异常（如格式错误或无效 ID），返回错误消息
            return (
                f"Error trying to add review -- bad formatting, try again: {str(e)}. "
                "Your provided PubMed ID might not be valid. Make sure it references a real paper, "
                "which can be found using the SUMMARY command.",
                ""
            )

    def format_review(self):
        return "Provided here is a literature review on this topic:\n" + "\n".join(
            f"PubMed ID: {_l['PubMed_id']}, Summary: {_l['summary']}"
            for _l in self.lit_review)
