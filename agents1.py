from utils import *
from tools import *
from inference import *
import random, string


def extract_json_between_markers(llm_output):
    # Regular expression pattern to find JSON content between ```json and ```
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues
            try:
                # Remove invalid control characters
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue  # Try next match

    return None  # No valid JSON found

def get_score(outlined_plan, latex, reward_model_llm, reviewer_type=None, attempts=3, openai_api_key=None):
    e = str()
    for _attempt in range(attempts):
        try:
            # todo: have a reward function here
            # ====================== 1. 构造评审模板 ======================
            # 这里定义了 AI 评审的格式，要求模型先输出 THOUGHT，再输出一个 JSON
            # JSON 中包含论文的多维度评价（原创性、质量、清晰度、重要性等）
            # 这些格式要求确保输出可以被机器自动解析
            template_instructions = """
            Respond in the following format:

            THOUGHT:
            <THOUGHT>

            REVIEW JSON:
            ```json
            <JSON>
            ```

            In <THOUGHT>, first briefly discuss your intuitions and reasoning for the evaluation.
            Detail your high-level arguments, necessary choices and desired outcomes of the review.
            Do not make generic comments here, but be specific to your current paper.
            Treat this as the note-taking phase of your review.

            In <JSON>, provide the review in JSON format with the following fields in the order:
            - "Summary": A summary of the paper content and its contributions.
            - "Strengths": A list of strengths of the paper.
            - "Weaknesses": A list of weaknesses of the paper.
            - "Originality": A rating from 1 to 4 (low, medium, high, very high).
            - "Quality": A rating from 1 to 4 (low, medium, high, very high).
            - "Clarity": A rating from 1 to 4 (low, medium, high, very high).
            - "Significance": A rating from 1 to 4 (low, medium, high, very high).
            - "Questions": A set of clarifying questions to be answered by the paper authors.
            - "Limitations": A set of limitations and potential negative societal impacts of the work.
            - "Ethical Concerns": A boolean value indicating whether there are ethical concerns.
            - "Soundness": A rating from 1 to 4 (poor, fair, good, excellent).
            - "Presentation": A rating from 1 to 4 (poor, fair, good, excellent).
            - "Contribution": A rating from 1 to 4 (poor, fair, good, excellent).
            - "Overall": A rating from 1 to 10 (very strong reject to award quality).
            - "Confidence": A rating from 1 to 5 (low, medium, high, very high, absolute).
            - "Decision": A decision that has to be one of the following: Accept, Reject.

            For the "Decision" field, don't use Weak Accept, Borderline Accept, Borderline Reject, or Strong Reject. Instead, only use Accept or Reject.
            This JSON will be automatically parsed, so ensure the format is precise.The JSON must include ALL of the following fields exactly as named:
            "Summary", "Strengths", "Weaknesses", "Originality", "Quality", "Clarity","Significance", "Questions", "Limitations", "Ethical Concerns", 
            "Soundness", "Presentation", "Contribution", "Overall", "Confidence", "Decision".

            Do not add, rename, or remove any field.
            If a field has no content, use an empty string "" or default numeric value.
            """
            # NeurIPS 会议的详细评审表（解释每个评分维度的含义和评分标准）
            # 这是 prompt 的一部分，用来确保模型给出专业、结构化的评审
            neurips_form = ("""
                ## Review Form
                Below is a description of the questions you will be asked on the review form for each paper and some guidelines on what to consider when answering these questions.
                When writing your review, please keep in mind that after decisions have been made, reviews and meta-reviews of accepted papers and opted-in rejected papers will be made public. 

                1. Summary: Briefly summarize the paper and its contributions. This is not the place to critique the paper; the authors should generally agree with a well-written summary.
                  - Strengths and Weaknesses: Please provide a thorough assessment of the strengths and weaknesses of the paper, touching on each of the following dimensions:
                  - Originality: Are the tasks or methods new? Is the work a novel combination of well-known techniques? (This can be valuable!) Is it clear how this work differs from previous contributions? Is related work adequately cited
                  - Quality: Is the submission technically sound? Are claims well supported (e.g., by theoretical analysis or experimental results)? Are the methods used appropriate? Is this a complete piece of work or work in progress? Are the authors careful and honest about evaluating both the strengths and weaknesses of their work
                  - Clarity: Is the submission clearly written? Is it well organized? (If not, please make constructive suggestions for improving its clarity.) Does it adequately inform the reader? (Note that a superbly written paper provides enough information for an expert reader to reproduce its results.)
                  - Significance: Are the results important? Are others (researchers or practitioners) likely to use the ideas or build on them? Does the submission address a difficult task in a better way than previous work? Does it advance the state of the art in a demonstrable way? Does it provide unique data, unique conclusions about existing data, or a unique theoretical or experimental approach?

                2. Questions: Please list up and carefully describe any questions and suggestions for the authors. Think of the things where a response from the author can change your opinion, clarify a confusion or address a limitation. This can be very important for a productive rebuttal and discussion phase with the authors.  

                3. Limitations: Have the authors adequately addressed the limitations and potential negative societal impact of their work? If not, please include constructive suggestions for improvement.
                In general, authors should be rewarded rather than punished for being up front about the limitations of their work and any potential negative societal impact. You are encouraged to think through whether any critical points are missing and provide these as feedback for the authors.

                4. Ethical concerns: If there are ethical issues with this paper, please flag the paper for an ethics review. For guidance on when this is appropriate, please review the NeurIPS ethics guidelines.

                5. Soundness: Please assign the paper a numerical rating on the following scale to indicate the soundness of the technical claims, experimental and research methodology and on whether the central claims of the paper are adequately supported with evidence.
                  4: excellent
                  3: good
                  2: fair
                  1: poor

                6. Presentation: Please assign the paper a numerical rating on the following scale to indicate the quality of the presentation. This should take into account the writing style and clarity, as well as contextualization relative to prior work.
                  4: excellent
                  3: good
                  2: fair
                  1: poor

                7. Contribution: Please assign the paper a numerical rating on the following scale to indicate the quality of the overall contribution this paper makes to the research area being studied. Are the questions being asked important? Does the paper bring a significant originality of ideas and/or execution? Are the results valuable to share with the broader NeurIPS community.
                  4: excellent
                  3: good
                  2: fair
                  1: poor

                8. Overall: Please provide an "overall score" for this submission. Choices: 
                  10: Award quality: Technically flawless paper with groundbreaking impact on one or more areas of AI, with exceptionally strong evaluation, reproducibility, and resources, and no unaddressed ethical considerations.
                  9: Very Strong Accept: Technically flawless paper with groundbreaking impact on at least one area of AI and excellent impact on multiple areas of AI, with flawless evaluation, resources, and reproducibility, and no unaddressed ethical considerations.
                  8: Strong Accept: Technically strong paper, with novel ideas, excellent impact on at least one area of AI or high-to-excellent impact on multiple areas of AI, with excellent evaluation, resources, and reproducibility, and no unaddressed ethical considerations.
                  7: Accept: Technically solid paper, with high impact on at least one sub-area of AI or moderate-to-high impact on more than one area of AI, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.
                  6: Weak Accept: Technically solid, moderate-to-high impact paper, with no major concerns with respect to evaluation, resources, reproducibility, ethical considerations.
                  5: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
                  4: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
                  3: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations.
                  2: Strong Reject: For instance, a paper with major technical flaws, and/or poor evaluation, limited impact, poor reproducibility and mostly unaddressed ethical considerations.
                  1: Very Strong Reject: For instance, a paper with trivial results or unaddressed ethical considerations

                9. Confidence:  Please provide a "confidence score" for your assessment of this submission to indicate how confident you are in your evaluation. Choices:
                  5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.
                  4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
                  3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
                  2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
                  1: Your assessment is an educated guess. The submission is not in your area or the submission was difficult to understand. Math/other details were not carefully checked.

                  You must make sure that all sections are properly created: abstract, introduction, methods, results, and discussion. Points must be reduced from your scores if any of these are missing.
                """ + template_instructions)
            # 如果没有指定 reviewer_type，就设为空字符串
            if reviewer_type is None: reviewer_type = ""

            # 系统提示，告诉模型：你是 AI 研究员，正在审稿，要批判性和谨慎
            sys = (
                      "You are an AI researcher who is reviewing a paper that was submitted to a prestigious ML venue. "
                      f"Be critical and cautious in your decision. {reviewer_type}\n"
                  ) + neurips_form

            # ====================== 2. 调用语言模型生成评审 ======================
            scoring = query_model(
                model_str=f"{reward_model_llm}",  # 评审使用的 LLM
                system_prompt=sys,  # 系统提示，带上评审表
                openai_api_key=openai_api_key,
                prompt=(  # 输入 prompt，包括研究计划和 LaTeX 论文内容
                    f"Outlined in the following text is the research plan that the machine learning engineer was tasked with building: {outlined_plan}\n\n"
                    f"The following text is the research latex that the model produced: \n{latex}\n\n"), temp=0.0)  # 设置温度为 0，保证模型输出稳定、确定
            # 从模型输出中提取 JSON（用于后续评分）
            review_json = extract_json_between_markers(scoring)

            # ====================== 3. 提取 JSON 中的各项分数并归一化 ======================
            overall = int(review_json["Overall"]) / 10          # 总体评分（1-10）
            soundness = int(review_json["Soundness"]) / 4       # 技术可靠性（1-4）
            confidence = int(review_json["Confidence"]) / 5     # 审稿人信心（1-5）
            contribution = int(review_json["Contribution"]) / 4 # 贡献度（1-4）
            presentation = int(review_json["Presentation"]) / 4 # 展示质量（1-4）
            clarity = int(review_json["Clarity"]) / 4           # 清晰度（1-4）
            originality = int(review_json["Originality"]) / 4   # 原创性（1-4）
            quality = int(review_json["Quality"]) / 4           # 技术质量（1-4）
            significance = int(review_json["Significance"]) / 4 # 重要性（1-4）

            # ====================== 4. 定义各个指标的权重 ======================
            clarity_weight = 0.1
            quality_weight = 0.1
            overall_weight = 1.0
            soundness_weight = 0.1
            confidence_weight = 0.1
            originality_weight = 0.1
            significance_weight = 0.1
            contribution_weight = 0.4
            presentation_weight = 0.2

            # 最大加权和，用于归一化
            max_score = (
                clarity_weight + quality_weight + overall_weight + soundness_weight + confidence_weight + originality_weight + significance_weight + contribution_weight + presentation_weight)

            # ====================== 5. 计算最终 performance 分数 ======================
            performance = ((
                                   soundness_weight * soundness +
                                   presentation_weight * presentation +
                                   confidence_weight * confidence +
                                   contribution_weight * contribution +
                                   overall_weight * overall +
                                   originality_weight * originality +
                                   significance * significance_weight +
                                   clarity_weight * clarity +
                                   quality_weight * quality) / max_score) * 10  # 归一化后乘以 10
            # 返回 performance 分数 + 模型原始评分文本
            return performance, f"The performance of your submission is: {performance}" + scoring, True
        except Exception as e:
            # 如果出错，打印错误并返回失败结果
            print("评分模型原始输出:\n", scoring)
            print(f"评分出错了,错误信息是：{e}")
            return None, str(e), False
    # 如果所有 attempts 都失败，返回 0 分
    return 0, e


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


class ProfessorAgent(BaseAgent):
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
        #sr_str = str()
        #if self.second_round:
        #    sr_str = (
        #        f"The following are results from the previous experiments\n",
        #        f"Previous Experiment code: {self.prev_results_code}\n"
        #        f"Previous Results: {self.prev_exp_results}\n"
        #        f"Previous Interpretation of results: {self.prev_interpretation}\n"
        #        f"Previous Report: {self.prev_report}\n"
        #        f"{self.reviewer_response}\n\n\n"
        #    )
        #if phase == "report writing":
        #    return (
        #        sr_str,
        #        f"Current Literature Review: {self.lit_review_sum}\n"
        #        f"Current Plan: {self.plan}\n"
        #        f"Current Dataset code: {self.dataset_code}\n"
        #        f"Current Experiment code: {self.results_code}\n"
        #        f"Current Results: {self.exp_results}\n"
        #        f"Current Interpretation of results: {self.interpretation}\n"
        #    )
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
            "When you believe a good report has been arrived at between you and the PhD student you can use the following command to end the dialogue and submit the plan ```LATEX\nreport here\n```\n where report here is the actual report written in compilable latex to be transmitted and LATEX is just the word LATEX.\n"
            "Your report should include numbers, relevant metrics to the experiment and measures of significance. You must propagate this information accurately. You must also submit the report promptly. Do not delay too long.\n"
            "You must be incredibly detailed about what you did for the experiment and all of the findings.\n"
            )

    def phase_prompt(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        phase_str = (
            "You are directing a PhD student to help them write a report in latex based on results from an experiment, and you interact with them through dialogue.\n"
            "Your goal is to write a report in latex for an experiment. You should read through the code, read through the interpretation, and look at the results to understand what occurred. You should then discuss with the PhD student how they can write up the results and give their feedback to improve their thoughts.\n"
        )
        return phase_str

    def role_description(self):
        return "a computer science professor at a top university."


class PostdocAgent(BaseAgent):
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
                # "Before submitting the research plan, you must generate a YAML file that summarizes the Mendelian Randomization (MR) methods and analysis tools used in the plan use the following command:```METHOD\nmethod and tool here\n```\n where method and tool here is should be replaced with the actual MR methods and the corresponding tools or software packages used to implement them, as specified in your plan.And METHOD is just the word METHOD.\n"
                "When you believe a good plan has been arrived at between you and the PhD student you can use the following command to  to end the dialogue and submit the plan ```PLAN\nplan here\n```\n where plan here is the actual plan to be transmitted and PLAN is just the word PLAN. The plan should provide a clear execution outline and cover the entire automated MR analysis process, including the Mendelian randomization methods to be used and implemented, the datasets of GWAS Catalog to be identified and analyzed, and the specific details of the experiment.Only datasets in the GWAS Catalog can be used.\n"
                "You can only use a SINGLE command per inference turn. Do not use more than one command per inference. If you use multiple commands, then only one of them will be executed, NOT BOTH.\n"
                "Make sure not to produce too much dialogue and to submit an plan and a YAML file in reasonable time."
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. PLAN, DIALOGUE).\n"
            )
        elif phase == "results interpretation":
            return (
                "When you believe a good interpretation has been arrived at between you and the PhD student you can use the following command to end the dialogue and submit the plan ```INTERPRETATION\ninterpretation here\n```\n where interpretation here is the actual interpretation to be transmitted and INTERPRETATION is just the word INTERPRETATION. Please provide an INTERPRETATION in a reasonable amount of time.\n"
                "You can produce dialogue using the following command: ```DIALOGUE\ndialogue here\n```\n where dialogue here is the actual dialogue you will send and DIALOGUE is just the word DIALOGUE.\n"
                "You must submit the interpretation during this phase in a reasonable amount of time. Do not delay the submission."
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. INTERPRETATION, DIALOGUE).\n"
            )

    def phase_prompt(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        if phase == "plan formulation":
            phase_str = (
                "You are directing a PhD student to help them come up with a good plan, and you interact with them through dialogue.\n"
                "Your goal is to produce plans that would make good experiments for the given topic.\n"#  and generate YAML file to Describe the Mendelian randomization methods to be used and the tools to implement them.
                "You should incorporate relevant literature review, but avoid overcomplicating the experimental design. Focus on drafting a clear and foundational experiment plan that includes the following core components:\n"
                "1. Data Acquisition:Generate search queries sequentially for the exposure and outcome traits.Obtain GWAS summary statistics for the exposure and outcome traits from the GWAS Catalog database.Data acquisition for this part of the program content is only required to generate search terms. DO NOT use the TwoSampleMR package for data searching or loading.DO NOT use web URLs or placeholder links to download data.\n"
                "2. Instrumental variable Selection:Select significant variants from the exposure dataset.Perform LD pruning on both the filtered exposure and outcome datasets.Optionally, apply novel methods to further refine the selection of instrumental variables.Harmonize alleles between datasets (e.g., align effect alleles across exposure and outcome datasets).\n"
                "3. MR Analysis:Use standard Mendelian Randomization methods such as Inverse-Variance Weighted (IVW), MR-Egger, and Weighted Median.Optionally apply novel MR methods, such as CAUSE.Highlight comparisons between different methods.\n"
                "4. Optional Sensitivity Analyses.In sensitivity analysis, the primary goal is to assess the robustness of causal inference using different methods, ensuring that the results are not influenced by potential biases. Common sensitivity analysis methods include MR-Egger regression, which detects horizontal pleiotropy; weighted median, a robust method against horizontal pleiotropy; MR-PRESSO, used to exclude outlier instrumental variables; heterogeneity analysis, which evaluates SNP effect consistency via IVW; and comparing traditional methods with novel approaches to check result consistency. These methods help ensure the reliability and robustness of causal inference results.\n"
                "5. Results Output and Visualization:Output results as summary tables and generate basic plots."
                "Generate a scatter plot to show the differences in causal effect estimates between exposure and outcome using different Mendelian Randomization methods. The X-axis represents the effect sizes of exposure-related SNPs, and the Y-axis represents the effect sizes of outcome-related SNPs. The estimates from each method should be distinguished using different colors or markers. Ensure that the estimates from each method and the variations between them are displayed, helping to assess the consistency and differences across the methods.\n"
                "Generate a forest plot to show the causal effect estimates and standard errors for each SNP between exposure and outcome. The X-axis represents the estimated causal effect, and the Y-axis lists each SNP. Different colors or line styles should be used to distinguish different MR methods. This plot helps compare the causal effect estimates and precision across different methods for each SNP.\n" 
                "Generate a comparison plot of causal effect estimates to compare the causal effect estimates between exposure and outcome using different Mendelian Randomization methods. The X-axis represents the different analysis methods, and the Y-axis represents the estimated causal effect. The results from each method can be displayed using bar plots or dot plots, with the estimates and standard errors indicated for each method. This plot helps visually display the differences in causal effect estimates across methods, evaluating their validity and consistency. \n"
                "The data requirements may differ for different Mendelian Randomization methods, so the plan should clearly specify how each method should handle the data.\n"
                "Your plan should clearly state:What processing steps will be applied.What MR methods will be used.How to perform sensitivity analysis.What kind of plots should be generated and how to generate them.What tools or packages are required (e.g., the TwoSampleMR package).\n"
                "Your current task is to construct a complete and end-to-end basic MR experiment, which can serve as a foundation for future extensions.\n"
                # "After completing your research plan, you must generate a methods_tools.yaml file that includes the Mendelian Randomization (MR) methods and the corresponding tools or software packages used to implement them.Structure the YAML content in two sections:1.methods: a list of MR methods you selected, each with a name and a short description of what the method does.tools: a list of tools or software packages used to implement those methods, each with a name and a short description of the tool’s purpose.Format your response using the following template:\n\n```METHOD\nmethods:\n\nname: \"...\" \ndescription: \"...\" \n\nname: \"...\" \ndescription: \"...\" \n\ntools:\n\nname: \"...\" \ndescription: \"...\" \n\nname: \"...\" \ndescription: \"...\" \n``` .Replace the ... with actual MR method names and tools you used in your plan.Method names and tool names must be written in lowercase and use abbreviations (for example: ivw, mr-egger, cause, weighted_median, mr-presso).\n"

                # "The plan outlined in the data loading step not only requires loading the data but also processing the loaded data to ensure it can be directly used for subsequent Mendelian Randomization analysis. You should aim for a very simple experiment that showcases your plan, not a complex one. You should integrate the provided literature review and come up with plans on how to expand and build on these works for the given topic. Your plans should provide a clear outline for how to achieve the task,  including the Mendelian randomization methods to be used and implemented, the datasets to be identified and analyzed, and the specific details of the experiment. Your idea should be very innovative and unlike anything seen before.\n"
            )
        elif phase == "results interpretation":
            phase_str = (
                "You are directing a PhD student to help them come up with an interpretation for results from an experiment, and you interact with them through dialogue.\n"
                "Your goal is to interpret results from experiments that were previously run. You should read through the code and look at the results to understand what occurred. You should then discuss with the PhD student how they can interpret the results and give their feedback to improve their thoughts. You should integrate the provided literature review, code, and plans to come up with an exciting interpretation that could make a compelling paper. Your plans should provide a clear outline that can be used to write an academic paper.\n"
                "Your interpretation must include the effect estimates (e.g., causal β coefficients), standard errors, confidence intervals, and p-values for each MR method (such as IVW, MR-Egger, and Weighted Median), and accurately communicate and interpret these values. You should also consider whether the results are consistent across methods and whether sensitivity analyses (such as the MR-Egger intercept test, heterogeneity test, leave-one-out analysis, and MR-PRESSO) support the robustness of the findings. You must also complete this in a reasonable amount of time and then submit your results.\n"
            )
        return phase_str

    def role_description(self):
        return "a computer science postdoctoral student at a top university."


class MLEngineerAgent(BaseAgent):
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
                "You can generate a phenotype keyword that best matches the trait description in the research objective using the following command: ```SEARCH_GWAS\ntrait keyword here\n``` where trait keyword here is your generated term，which should align closely with the efo_trait field in the GWAS Catalog to ensure highly relevant search results,and SEARCH_GWAS is the word SEARCH_GWAS.The command will return metadata and file paths for datasets related to the specified trait. When generating keywords for GWAS search, please add the suffix “_exposure” for exposure variables and “_outcome” for outcome variables to the keywords according to the user's analytical objectives, for example “body mass index_exposure” or ‘type 2 diabetes_outcome’.Do not download data again—datasets have already been fetched. Your task is to write code that loads data from the returned file paths, using only the external datasets from the GWAS Catalog.\n"
                "You can perform LD pruning using the following command: ```LD_PRUNE\n{'r2': value, 'window_kb': value}\n``` where LD_PRUNE is the word LD_PRUNE, and the JSON object includes the pruning parameters to be used. The keys are 'r2' for the squared correlation threshold and 'window_kb' for the window size in kilobases. This command instructs the agent to perform LD pruning with the specified parameters using PLINK. \n"
                "You MUST use a GWAS Catalog dataset in your code. DO NOT CREATE A MAIN FUNCTION. Try to make the code very simple.\n"
                "You can only use a SINGLE command per inference turn. Do not use more than one command per inference. If you use multiple commands, then only one of them will be executed, NOT BOTH.\n"
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. R, DIALOGUE, SEARCH_GWAS).\n")

                # "You also have access the GWAS Catalog database. You can search the datasets repository using the following command: ```SEARCH_GWAS\nsearch query here\n``` where search query here is the query term you will use to search for datasets in the GWAS Catalog (specifically, a trait keyword), and SEARCH_GWAS is the word SEARCH_GWAS. The search results will return detailed information about datasets related to the query term (trait). But you don't need to download the data, as the dataset has already been downloaded using a data retrieval tool. You only need to write code to load the data based on the returned file path.The search result is already related to the query term(trait). You must use the already downloaded GWAS Catalog dataset.Your code must use the external datasets provided by the GWAS Catalog.\n"
                # detailed information about datasets related to the query term (trait), including the Study ID, disease/trait, publication date, sample size, species information, PMID, journal name, and a direct download link to the summary statistics file (e.g., https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST003001-GCST004000/GCST003044/harmonised/26192919-GCST003044-EFO_0000384-Build37.f.tsv.gz). You can use this link along with an appropriate data loading method (e.g., pandas) to read the  complete data.
                # Just load the complete data. No need to process or analyze it.
    def phase_prompt(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        if phase == "data preparation":
            phase_str = (
                "You are a machine learning engineer working under the guidance of a software engineer. You will collaborate by writing R code through dialogue.\n",
                "Your current task is to write executable R code that prepares and processes GWAS data for use in a Mendelian Randomization (MR) experiment. You must follow the provided literature review and experimental plan.\n",
                "All steps must be implemented via R code that performs all necessary data preparation steps from start to finish. Do NOT complete any step manually in advance.\n",
                "Strictly Perform the following steps in order:\n"
                "1. Generating a search query for the exposure dataset.\n"
                "2. Generating a search query for the outcome dataset.\n"
                "You must prioritize generating the exposure query. Execute only one search at a time.This part doesn't need to be in the code. Do NOT issue both queries simultaneously. Do NOT regenerate the exposure or outcome search queries under any circumstances.\n"
                "3. Load the exposure data from: research_results/processed_data/filtered_exposure_data.csv\n"
                "4. Load the outcome data from: research_results/processed_data/filtered_outcome_data.csv\n"
                "5. (Optional) Performing additional instrumental variable (IV) selection procedures.\n"
                "6. Performing data harmonization: Use the harmonise_data() function from the TwoSampleMR package to align alleles and remove ambiguous SNPs (e.g., A/T or C/G).\n"
                "You do NOT need to write code for downloading data, filtering significant variants, and performing LD clumping.\n"
                "All R code must be clearly structured, fully executable, and include comments that explain each operation.\n"
                "DO NOT perform any causal inference methods.DO NOT update or reinstall any R packages.DO NOT write any Python code — all code must be written in R.DO NOT use the TwoSampleMR package for data searching or loading.DO NOT use web URLs or placeholder links to download data.DO NOT create or write to any other folders. Avoid intermediate files unless absolutely necessary. Process data in memory as much as possible. DO NOT generate multiple versions of similar files.\n"
                "If the data is missing required columns, DO NOT regenerate search queries. Instead, check the column names for formatting issues and rename them to match expected formats before continuing.\n"
                "All raw GWAS data files must be saved in: \"research_results/raw_gwas_data/\"\n"
                "All processed output files must be saved in: \"research_results/processed_data/\"\n"
                "**CODE EXAMPLE:**",
                "```r",
                "library(TwoSampleMR)",
                "library(dplyr)",
                "library(readr)",
                "",
                "# Load preprocessed exposure and outcome datasets",
                "exposure_data <- read_csv(\"research_results/processed_data/filtered_exposure_data.csv\")",
                "outcome_data <- read_csv(\"research_results/processed_data/filtered_outcome_data.csv\")",
                "",
                "# Harmonize exposure and outcome data, removing ambiguous SNPs (A/T or C/G)",
                "harmonized_data <- harmonise_data(",
                "  exposure_dat = exposure_data,",
                "  outcome_dat = outcome_data,",
                "  action = 2",
                ")",
                "",
                "# Save harmonized results",
                "write.csv(harmonized_data, \"research_results/processed_data/harmonized_data.csv\", row.names = FALSE)",
                "",
                "cat(\"Data harmonization completed and saved to processed_data/harmonized_data.csv\\n\")",
                "```"
            )
        return phase_str

    def role_description(self):
        return "a machine learning engineer working at a top university."


class SWEngineerAgent(BaseAgent):
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
                "When you and the ML engineer have finalized your dataset preparation code and are ready to submit the final code, please use the following command: ```SUBMIT_CODE\ncode here\n```\n where 'code here' is the finalized code you will send and SUBMIT_CODE is just the word SUBMIT_CODE. Do not use any classes or functions.  If your code returns any errors, they will be provided to you, and you are also able to see print statements.  Make sure function variables are created inside the function or passed as a function parameter. DO NOT CREATE A MAIN FUNCTION.\n"
                "Make sure to submit code in a reasonable amount of time. Do not make the code too complex, try to make it simple. Do not take too long to submit code. Once data harmonization is completed successfully, you MUST immediately stop the dialogue and submit your final code using the SUBMIT_CODE command. Do NOT continue generating further dialogue or repeat tasks.\n"
                "You can only use a single command per inference turn. Do not use more than one command per inference. If you use multiple commands, then only one of them will be executed, not both.\n"
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. SUBMIT_CODE, DIALOGUE).\n")
        return ""

    def phase_prompt(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        elif phase == "data preparation":
            phase_str = (
                "You are a software engineer directing a machine learning engineer, where the machine learning engineer will be writing the code, and you can interact with them through dialogue.\n"
                "Your goal is to help the ML engineer write R code that prepares the data for the provided experiment. You should aim for very simple code to prepare the data, not complex code. You should integrate the provided literature review and the plan and come up with code to prepare data for this experiment.\n"
                "If all data preparation tasks have been successfully completed, your task is complete.Avoid repetition or unnecessary looping. \n"
                "Strictly Perform the following steps in order:\n"
                "1. Generating a search query for the exposure dataset.\n"
                "2. Generating a search query for the outcome dataset.\n"
                "You must prioritize generating the exposure query. Execute only one search at a time.This part doesn't need to be in the code. Do NOT issue both queries simultaneously. Do NOT regenerate the exposure or outcome search queries under any circumstances.\n"
                "3. Load the exposure data from: research_results/processed_data/filtered_exposure_data.csv\n"
                "4. Load the outcome data from: research_results/processed_data/filtered_outcome_data.csv\n"
                "5. (Optional) Performing additional instrumental variable (IV) selection procedures.\n"
                "6. Performing data harmonization: Use the harmonise_data() function from the TwoSampleMR package to align alleles and remove ambiguous SNPs (e.g., A/T or C/G).\n"
            )
        return phase_str

    def role_description(self):
        return "a software engineer working at a top university."


class PhDStudentAgent(BaseAgent):
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
                "When you and the ML engineer have finalized your dataset preparation code and are ready to submit the final code, please use the following command: ```SUBMIT_CODE\ncode here\n```\n where 'code here' is the finalized code you will send and SUBMIT_CODE is just the word SUBMIT_CODE. Do not use any classes or functions. The submitted code must have a HuggingFace dataset import and must use an external HuggingFace dataset. If your code returns any errors, they will be provided to you, and you are also able to see print statements.  Make sure function variables are created inside the function or passed as a function parameter. DO NOT CREATE A MAIN FUNCTION.\n"
                "Make sure to submit code in a reasonable amount of time. Do not make the code too complex, try to make it simple. Do not take too long to submit code. Submit the code early. You should submit the code ASAP.\n"
                "You can only use a single command per inference turn. Do not use more than one command per inference. If you use multiple commands, then only one of them will be executed, not both.\n"
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. SUBMIT_CODE, DIALOGUE).\n")
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
                "You are a PhD student working under the guidance of a postdoctoral researcher, and you will collaboratively design a Mendelian Randomization (MR) experiment plan through dialogue.\n" #  and design a  YAML file to Describe the Mendelian randomization methods to be used and the tools to implement them
                "Your goal is to produce plans that would make good experiments for the given topic.\n" #  and design a YAML file to Describe the Mendelian randomization methods to be used and the tools to implement them.
                "You should incorporate relevant literature review, but avoid overcomplicating the experimental design. Focus on drafting a clear and foundational experiment plan that includes the following core components:\n"
                "1. Data Acquisition: Generate search queries sequentially for the exposure and outcome traits.Obtain GWAS summary statistics for the exposure and outcome traits from the GWAS Catalog database.\n"
                "2. Instrumental variable Selection:Select significant variants from the exposure dataset.Perform LD pruning on both the filtered exposure and outcome datasets.Optionally, apply novel methods to further refine the selection of instrumental variables.Harmonize alleles between datasets (e.g., align effect alleles across exposure and outcome datasets).\n"
                "3. MR Analysis:Use standard Mendelian Randomization methods such as Inverse-Variance Weighted (IVW), MR-Egger, and Weighted Median.Optionally apply novel MR methods, such as CAUSE.Highlight comparisons between different methods.\n"
                "4. Optional Sensitivity Analyses.\n"
                "5. Results Output and Visualization:Output results as summary tables and generate basic plots (e.g., scatter plots, forest plots, funnel plots).\n"
                "Your plan should clearly state:What processing steps will be applied and how to realize.What MR methods will be used and how to realize.What tools or packages are required (e.g., the TwoSampleMR package).\n"
                "Your current task is to construct a complete and end-to-end basic MR experiment, which can serve as a foundation for future extensions.\n"
                # "You must design a methods_tools.yaml file that includes the Mendelian Randomization (MR) methods and the corresponding tools or software packages used to implement them.Structure the YAML content in two sections:1.methods: a list of MR methods you selected, each with a name and a short description of what the method does.tools: a list of tools or software packages used to implement those methods, each with a name and a short description of the tool’s purpose.Format your response using the following template:\n\n```METHOD\nmethods:\n\nname: \"...\" \ndescription: \"...\" \n\nname: \"...\" \ndescription: \"...\" \n\ntools:\n\nname: \"...\" \ndescription: \"...\" \n\nname: \"...\" \ndescription: \"...\" \n``` .Replace the ... with actual MR method names and tools you used in your plan.Method names and tool names must be written in lowercase and use abbreviations (for example: ivw, mr-egger, cause, weighted_median, mr-presso).\n"
                # "You are a PhD student being directed by a postdoc who will help you come up with a good plan, and you interact with them through dialogue.\n"
                # "Your goal is to produce plans that would make good experiments for the given topic. You should aim for a very simple experiment that showcases your plan, not a complex one. You should integrate the provided literature review and come up with plans on how to expand and build on these works for the given topic. Your plans should provide a clear outline for how to achieve the task,  including the Mendelian randomization methods to be used and implemented, the datasets to be identified and analyzed, and the specific details of the experiment. \n"
                # Your idea should be very innovative and unlike anything seen before.
            )
        elif phase == "results interpretation":
            phase_str = (
                "You are a PhD student being directed by a postdoc who will help you come up with an interpretation for results from an experiment, and you interact with them through dialogue.\n"
                "Your goal is to interpret results from experiments that were previously run. You should read through the code and look at the results to understand what occurred. You should then discuss with the postdoc your interpretation and use their feedback to improve your thoughts. You should integrate the provided literature review, code, and plans to come up with an exciting interpretation that could make a compelling paper. Your plans should provide a clear outline that can be used to write an academic paper.\n"
                "Your interpretation must include effect estimates (e.g., causal β coefficients), standard errors, confidence intervals, and p-values for each MR method used (e.g., IVW, MR-Egger, Weighted Median), and accurately communicate and interpret these values. You should also consider whether the results are consistent across methods and whether sensitivity analyses (e.g., MR-Egger intercept, heterogeneity test, leave-one-out analysis, MR-PRESSO) support the robustness of the findings.\n"
                "You must submit the interpretation during this phase in a reasonable amount of time. Do not delay the submission."
            )
        elif phase == "report writing":
           phase_str = (
               "You are a PhD student being directed by a professor who will help you write a report based on results from an experiment, and you interact with them through dialogue.\n"
               "Your goal is to write a report for an experiment entirely in latex. You should read through the code, read through the interpretation, and look at the results to understand what occurred. You should then discuss with the professor how you can write up the results and receive their feedback to improve your thoughts.\n"
               "Your report should include numbers, relevant metrics to the experiment and measures of significance  in latex. You must propagate this information accurately.\n"
               "You must be incredibly detailed about what you did for the experiment and all of the findings.\n"
           )
        elif phase == "report refinement":
            phase_str = (
                "You are a PhD student who has submitted their paper to a bioinformatics journal called Bioinformatics. Your goal was to write a research paper and get high scores from the reviewers so that it get accepted to the journal.\n"
            )
        else:
            phase_str = ""
        return phase_str

    def role_description(self):
        return "a computer science PhD student at a top university."

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

    # "You are a machine learning engineer working under the guidance of a software engineer. You will collaborate by writing code through dialogue.",
    # "Your current task is to write executable R code that prepares and processes GWAS data for use in a Mendelian Randomization (MR) experiment.",
    # "You must follow the provided literature review and experimental plan to implement a complete and reusable data preparation pipeline.",
    #
    # "**IMPORTANT RESTRICTIONS:**",
    # "Do NOT perform any causal inference methods (e.g., IVW, MR-Egger, weighted median, etc.).",
    # "Do NOT calculate causal effect estimates, z-scores, or p-values for MR outcomes.",
    # "Do NOT save any MR results or output files named like mr_ivw_results.csv or mr_egger_results.csv.",
    # "Your task ends strictly after data harmonization is completed. MR analysis will be executed in a later experimental phase.",
    # "Do NOT update or reinstall any R package versions. Use only the packages already installed in the environment.",
    # "Only use the TwoSampleMR package to harmonize the exposure and outcome datasets—specifically to align alleles and remove ambiguous SNPs (e.g., A/T or C/G). Do not use TwoSampleMR for clumping, instrument selection, or any other downstream MR operations.",
    #
    # "**ABSOLUTELY DO NOT REPEAT ANY STEPS:**",
    # "- Perform clumping (LD pruning) or instrument selection **only once** as defined in the experimental plan. " ,
    # "- Do NOT repeat clumping, pruning, or other filtering steps unless explicitly instructed.",
    # "- Avoid reprocessing datasets that were already harmonized or pruned.",
    #
    # "**YOUR RESPONSIBILITIES INCLUDE:**",
    # "Write and execute R code that prepares data for MR experiments.",
    # "Your first step MUST be generating a search query for the exposure, followed by the outcome — one at a time.",
    # "First, generate one search query for the exposure trait, execute it, and ensure the corresponding dataset is downloaded.",
    # "Next, generate one search query for the outcome trait, execute it, and ensure the corresponding dataset is downloaded.",
    # "Only proceed to data loading and harmonization after BOTH exposure and outcome files have been confirmed to exist.",
    # "Use the TwoSampleMR package in R to harmonize exposure and outcome data, ensuring allele alignment and removing ambiguous SNPs (A/T, C/G).",
    # "Do NOT perform any regression or causal estimation tasks.",
    #
    # "**DATA HANDLING RULES:**",
    # "Do NOT use TwoSampleMR to search or load data—if the required data files are missing, generate a search query instead.",
    # "All original GWAS files must be saved in the research_results/GWAS_data/ folder.",
    # "All processed output must be saved in the research_results/processed_data/ folder.",
    # "Do not create new folders or store files elsewhere.",
    # "When processing GWAS exposure and outcome data, follow the principles of efficiency and reusability while strictly limiting file generation. Only download datasets that are directly relevant to the experiment. Avoid creating intermediate temporary files unless absolutely necessary. All intermediate processing should be performed in memory whenever possible, and only the essential final results should be written to disk. Redundant outputs or saving multiple similar versions of files should be avoided.",
    #
    # "**NOTES:**",
    # "The datasets do not contain a 'trait' column, but they are already filtered using your search queries.",
    # "Only after confirming both datasets are present may you proceed to load and harmonize the data using R.",
    # "Do not re-filter p-values or apply additional significance thresholds unless explicitly instructed.",
    # "You should integrate the provided literature and plan when writing the data preparation code.",
    #
    # "**SUMMARY:**",
    # "Your job is to write executable R code that fully prepares input data for MR analysis — NOT to perform the MR itself.",
    # "Do not go beyond harmonization. Do not perform IVW, MR-Egger, or any other causal inference methods.",
    # "Do NOT write any Python code. All scripts must be written in R.",
    #
    # "**CODE EXAMPLE:**",
    # "```r",
    # "# Load TwoSampleMR package",
    # "library(TwoSampleMR)",
    # "library(dplyr)"
    # "library(readr)"
    # "",
    # "# Read exposure data"
    # "exposure_data <- read_csv(\"D:/pycharm_projects/AgentLaboratory/AgentLaboratory/research_results/processed_data/filtered_exposure_data.csv\")",
    # "# Read outcome data",
    # "outcome_data <- read_csv(\"D:/pycharm_projects/AgentLaboratory/AgentLaboratory/research_results/processed_data/filtered_outcome_data.csv\")",
    #
    # # "outcome_file <- \"research_results/GWAS_data/filtered_outcome_data.csv\"",
    #
    # "# Harmonize datasets, removing ambiguous SNPs (A/T and C/G)",
    # "harmonized_data <- harmonise_data(",
    # "  exposure_dat = exposure_data,",
    # "  outcome_dat = outcome_data,",
    # "  action = 2  # Remove ambiguous SNPs",
    # ")",
    # "",
    # "# Save harmonized data",
    # "write.csv(harmonized_data, \"research_results/processed_data/harmonized_data.csv\", row.names = FALSE)",
    # "",
    # "cat(\"Data harmonization completed and saved to research_results/processed_data/harmonized_data.csv\\n\")",
    # "```"

#     "You are a machine learning engineer working under the guidance of a software engineer who will help you write the code, and you can interact through dialogue.\n"
#     "You must use data from the GWAS Catalog database. Your goal is to write code to prepare and process the data for the provided experiment.\n"
#     "Before writing code, you must first generate at least two search queries in sequence: one describing the exposure (e.g., target phenotype, disease, or gene expression levels as biological traits), and one describing the outcome. These queries ensure that you can collect the exposure and outcome datasets required for Mendelian Randomization analysis. All datasets related to exposure and outcome are downloaded using a data retrieval tool based on the search terms.\n"
#     "You should generate one search term at a time, multiple times if necessary, to ensure all data required for the experiment can be found, but you should not generate multiple search terms all at once.\n"
#     "Before checking whether the files exist or starting the data loading process, make sure that all queries have been successfully executed. Only proceed to the next step once all queries have been successfully generated and executed.\n"
#     #"You only need to write code to load the data based on the returned file paths. Your goal is to write simple code for data preparation, not complex logic.\n"
#     "After loading the data, harmonization of the data was performed to ensure allelic concordance so that it can be directly used for subsequent Mendelian Randomization analysis.\n"
#     # "Make sure to apply LD pruning parameters only to the exposure data and keep the outcome data unchanged before merging.\n"
#     # "When processing the data, please ensure that the following essential columns are always retained (if they exist): chromosome (chromosome information), base_pair_location (genomic location), variant_id (variant identifier), beta (effect size), standard_error (standard error), p_value (p-value), effect_allele (effect allele), other_allele (other allele), and effect_allele_frequency (effect allele frequency). Make sure these columns are not removed or lost during any data processing or filtering steps. If any of these columns are missing, adjust the processing workflow accordingly and ensure that the processed data contains all the specified columns.\n"
#     "Please focus on writing simple code. You should integrate the provided literature review and the plan to write code that prepares data for this experiment.\n"
#     "Although the dataset does not contain a column named 'trait', all entries are already related to the search terms, so there is no need to further filter the dataset to include only variants associated with a specific trait or disease.\n"
#     "The downloaded data is all stored in the research_results/GWAS_data folder. The processed data should be saved in the research_results/processed_data folder. Do not create other folders or save files elsewhere.\n"
#     "Do not perform any causal inference methods such as MR-Egger or IVW.\n"
#     "Avoid generating intermediate temporary files unless necessary. Keep all intermediate data in memory.\n"
#     "When processing GWAS exposure and outcome data, please prioritize efficiency, reusability, and minimal file generation. Only download and handle the most relevant datasets, and avoid creating intermediate temporary files unless absolutely necessary. All intermediate data should be processed in memory whenever possible, with only the final necessary results written to disk. Avoid redundant outputs or saving multiple versions of similar files.\n"
#     # "Merging(Harmonization) exposure and outcome data on 'rsid'.End data preparation phase after merging and randomize to experimental phase \n"

# After pruning, merge the two datasets for further analysis.
# You can use a URL along with an appropriate data loading method (e.g., pandas) to read the  complete data. Use the provided URL to download the complete dataset, which is already related to the trait of interest, so no further SNP filtering is needed.Make sure to download the complete dataset from the GWAS Catalog database.Just load the complete data. No need to process or analyze it.Ensure that the dataset is correctly downloaded and loaded into your environment for use in the experiment.
#                 "Before loading the data, you must first generate a search query using a trait keyword that describes the target phenotype or disease. This ensures the dataset aligns with the experimental goal. Do not check for file existence or begin loading until the query has been generated and executed.\n"
#                 "Before loading the data, you must first generate at least two search queries using trait keywords: one describing the exposure (e.g., the target phenotype or disease), and the other describing the outcome. This ensures that you collect the exposure and outcome datasets required for Mendelian Randomization analysis. Before checking if the files exist or beginning the data loading process, make sure both queries have been executed. Only proceed with the next steps once both queries have been successfully generated and executed.Process the loaded exposure and outcome data so that they can be directly used for subsequent Mendelian Randomization analysis.\n"
