from inference import *
from utils import *


class BaseAgent:
    def __init__(self, model="gpt-4o-mini", notes=None, max_steps=100, openai_api_key=None):
        if notes is None:
            self.notes = []
        else:
            self.notes = notes
        with open("experiment_research/Considered_causal/DBP_CAD/run_experiments.R", "r", encoding="utf-8") as f:
            results_code = f.read()
        with open("experiment_research/Considered_causal/DBP_CAD/plan.txt", "r", encoding="utf-8") as f:
            plan = f.read()
        with open("experiment_research/Considered_causal/DBP_CAD/experiment_output.log", "r", encoding="utf-8") as f:
            exp_results = f.read()
        self.max_steps = max_steps
        self.model = model
        self.phases = []
        self.plan = plan
        self.report = str()
        self.history = list()
        self.prev_comm = str()
        self.prev_report = str()
        self.exp_results = exp_results
        self.dataset_code = str()
        self.dataset_information = str()

        self.results_code = results_code
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


class StatisticalGeneticist(BaseAgent):
    def __init__(self, model="gpt4omini", notes=None, max_steps=100, openai_api_key=None):
        super().__init__(model, notes, max_steps, openai_api_key)
        self.phases = ["results interpretation"]

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

        if phase == "results interpretation":
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
        if phase == "results interpretation":
            return (
                "When you believe a good interpretation has been arrived at between you and the biological engineer you can use the following command to end the dialogue and submit the interpretation ```INTERPRETATION\ninterpretation here\n```\n where interpretation here is the actual interpretation to be transmitted and INTERPRETATION is just the word INTERPRETATION. Please provide an INTERPRETATION in a reasonable amount of time.\n"
                "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where dialogue here is the actual dialogue you will send and DIALOGUE is just the word DIALOGUE.\n"
                "You must submit the interpretation during this phase in a reasonable amount of time. Do not delay the submission."
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. INTERPRETATION, DIALOGUE).\n"

            )

    def phase_prompt(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")

        if phase == "results interpretation":
            phase_str = (
                "You are directing a biological engineer to help them come up with an interpretation for results from an experiment, and you interact with them through dialogue.\n"
                "Your goal is to interpret results from experiments that were previously run. You should read through the code and look at the results to understand what occurred. You should then discuss with the biological engineer how they can interpret the results and give their feedback to improve their thoughts. You should integrate the provided literature review, code, and plans to come up with an exciting interpretation that could make a compelling paper. Your plans should provide a clear outline that can be used to write an academic paper.\n"
                "Your interpretation must include the effect estimates (e.g., causal β coefficients), standard errors, confidence intervals, and p-values for each MR method (such as IVW, MR-Egger, and Weighted Median), and accurately communicate and interpret these values. You should also consider whether the results are consistent across methods and whether sensitivity analyses (such as the MR-Egger intercept test, heterogeneity test, leave-one-out analysis, and MR-PRESSO) support the robustness of the findings. You must also complete this in a reasonable amount of time and then submit your results.\n"
            )
        return phase_str

    def role_description(self):
        return "An experienced statistical geneticist specializing in Mendelian Randomization and genomic causal analysis."


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

        if phase == "results interpretation":
            return (
                sr_str,
                f"Current Literature Review: {self.lit_review_sum}\n"
                f"Current Plan: {self.plan}\n"
                f"Current Dataset code: {self.dataset_code}\n"
                f"Current Datasets information: {self.dataset_information}\n"
                f"Current Experiment code: {self.results_code}\n"
                f"Current Results: {self.exp_results}"
            )

        else:
            return ""

    def requirements_txt(self):
        sys_prompt = f"""You are {self.role_description()} \nTask instructions: Your goal is to integrate all of the knowledge, code, reports, and notes provided to you and generate a requirements.txt for a github repository for all of the code."""
        history_str = "\n".join([_[1] for _ in self.history])
        prompt = (
            f"""History: {history_str}\n{'~' * 10}\n"""
            f"Please produce the requirements.txt below in markdown:\n")
        model_resp = query_model(model_str=self.model, system_prompt=sys_prompt, prompt=prompt,
                                 openai_api_key=self.openai_api_key)
        return model_resp

    def example_command(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        return ()

    def command_descriptions(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        if phase == "results interpretation":
            return (
                "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where 'dialogue here' is the actual dialogue you will send and DIALOGUE is just the word DIALOGUE.\n"
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. DIALOGUE).\n"
            )
        return ""

    def phase_prompt(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")

        if phase == "results interpretation":
            phase_str = (
                "You are a Biological Engineer being directed by a statistical geneticist  who will help you come up with an interpretation for results from an experiment, and you interact with them through dialogue.\n"
                "Your goal is to interpret results from experiments that were previously run. You should read through the code and look at the results to understand what occurred. You should then discuss with the statistical geneticist your interpretation and use their feedback to improve your thoughts. You should integrate the provided literature review, code, and plans to come up with an exciting interpretation that could make a compelling paper. Your plans should provide a clear outline that can be used to write an academic paper.\n"
                "Your interpretation must include effect estimates (e.g., causal β coefficients), standard errors, confidence intervals, and p-values for each MR method used (e.g., IVW, MR-Egger, Weighted Median), and accurately communicate and interpret these values. You should also consider whether the results are consistent across methods and whether sensitivity analyses (e.g., MR-Egger intercept, heterogeneity test, leave-one-out analysis, MR-PRESSO) support the robustness of the findings.\n"
                "You must submit the interpretation during this phase in a reasonable amount of time. Do not delay the submission."
            )
        else:
            phase_str = ""
        return phase_str

    def role_description(self):
        return "A bioinformatics engineer dedicated to genomic data analysis and Mendelian Randomization studies, working at a premier research institute."


    def format_review(self):
        return "Provided here is a literature review on this topic:\n" + "\n".join(
            f"PubMed ID: {_l['PubMed_id']}, Summary: {_l['summary']}"
            for _l in self.lit_review)


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
            "Generate clear and well-labeled figures to interpret results."
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
    phase = "results interpretation"
    for text in task_notes.get(phase,[]):
        if text is not None:
            notes.append({"phases": [phase], "content": text})

    max_tries = 100
    lab_index = 1
    model_backbone = "gpt-4o"
    lab_dir = "experiment_research"

    max_steps = 100
    verbose = True
    human_in_loop_flag = True
    openai_api_key = os.getenv('OPENAI_API_KEY')
    research_topic = "Your goal is to investigate the causal relationship between Dystolic Blood Pressure and Coronary Artery Disease, using Mendelian Randomization based on GWAS summary statistics."

    statistical_geneticist = StatisticalGeneticist(model=model_backbone, notes=notes, max_steps=max_steps,
                           openai_api_key=openai_api_key)
    biological_engineer = BiologicalEngineer(model=model_backbone, notes=notes, max_steps=max_steps,
                               openai_api_key=openai_api_key)
    dialogue = str()
    # iterate until max num tries to complete task is exhausted
    for _i in range(max_tries):
        # print(f"@@ Lab #{lab_index} Paper #{paper_index} @@")
        resp = statistical_geneticist.inference(research_topic, "results interpretation", feedback=dialogue, step=_i)
        if verbose: print("statistical geneticist: ", resp, "\n~~~~~~~~~~~")
        dialogue = str()
        if "```DIALOGUE" in resp:
            dialogue = extract_prompt(resp, "DIALOGUE")
            dialogue = f"The following is dialogue produced by the statistical geneticist: {dialogue}"
            if verbose: print("#" * 40, "\n", "statistical geneticist Dialogue:", dialogue, "\n", "#" * 40)
        if "```INTERPRETATION" in resp:
            interpretation = extract_prompt(resp, "INTERPRETATION")
            if human_in_loop_flag:
                retry = human_in_loop("results interpretation", interpretation)
                if retry: break
            save_to_file(f"./{lab_dir}", "interpretation.txt", interpretation)
            break
        resp = biological_engineer.inference(research_topic, "results interpretation", feedback=dialogue, step=_i)
        if verbose: print("biological engineer: ", resp, "\n~~~~~~~~~~~")
        dialogue = str()
        if "```DIALOGUE" in resp:
            dialogue = extract_prompt(resp, "DIALOGUE")
            dialogue = f"The following is dialogue produced by the biological engineer: {dialogue}"
            if verbose: print("#" * 40, "\n", "biological engineer Dialogue:", dialogue, "#" * 40, "\n")
    # 循环结束后处理 plan
    if interpretation is None:
        raise Exception("Max tries during phase: Plan Formulation")
