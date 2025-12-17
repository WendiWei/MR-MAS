from mlesolver1 import *
task_notes = {
    "plan_formulation": [
        "You should design a plan for ONE experiment focused on testing the causal effect.",
        "DO NOT PLAN FOR TOO LONG. Submit your experiment plan early to allow feedback and iteration.",
        "Perform Mendelian Randomization (MR) analysis on the given exposure–outcome pair only using the following methods: IVW, MR-Egger, Weighted Median, Weighted Mode, MR-PRESSO."
    ],
    "data_preparation": [
        "Do NOT conduct any causal analysis during this phase."
    ],
    "running_experiments": [
        "You can use any MR package",
        "You must perform Mendelian Randomization (MR) analysis on the given exposure–outcome pair only using cause method",# only using CAUSE method. #using the following methods: IVW, MR-Egger, Weighted Median, Weighted Mode, MR-PRESSO, and CAUSE.
        "Provide detailed result logs and summaries after execution.",
        # "Generate clear and well-labeled figures to interpret results,including Scatter plot for multi-method comparison (IVW, MR-Egger, Weighted Median, Weighted Mode), forest plot, leave-one-out analysis plot, and funnel plot ."
    ],
    "results_interpretation": [
        "Summarize whether there is statistically significant causal evidence.",
        "Comment on consistency between methods and presence of potential pleiotropy or bias."
    ],
    "report_writing": [
        "Your report should clearly state the causal question, methodology, data sources, and main findings.",
        "Include visualizations and statistical outputs that support your interpretation.",
        "End with conclusions and potential follow-up studies or validations."
    ]
}

# 获取与“running experiments”阶段相关的注释
experiment_notes = "\n".join(task_notes["running_experiments"]) if "running_experiments" in task_notes else ""
# 如果有相关注释，则整理为字符串形式
with open("./experiment_research/Considered_causal/DBP_Stroke/load_data.R", "r", encoding="utf-8") as f:
    dataset_code = f.read()
with open("./experiment_research/Considered_causal/DBP_Stroke/plan1.txt", "r", encoding="utf-8") as f:
    plan = f.read()
    # plan = None
lit_review_sum=None
cfg = ExperimentConfig()
mlesolver_max_steps = 2
# research_topic="Your goal is to investigate the causal relationship between BMI and Type 2 Diabetes, using Mendelian Randomization based on GWAS summary statistics."
research_topic = "Your goal is to investigate the causal relationship between diastolic blood pressure and Stroke, using cause method."
openai_api_key = os.getenv('OPENAI_API_KEY')
DEFAULT_LLM_BACKBONE = "gpt-4o"
compile_pdf = False
Population = "European"
lab_dir = "experiment_research/Considered_causal/DBP_Stroke/CAUSE"
# 初始化 MLESolver（用于执行机器学习实验）
# ld_params = cfg.get("data_preparation", "Original_exposure_data_path")
Original_exposure_data_path = cfg.get("data_preparation", "Original_exposure_data_path")
Original_outcome_data_path = cfg.get("data_preparation", "Original_outcome_data_path")
# Original_exposure_data_path = "D:\\pycharm_projects\\AgentLaboratory\\AgentLaboratory\\research_results\\raw_gwas_data\\GCST90310294.h.tsv.gz"
# Original_outcome_data_path = "D:\\pycharm_projects\\AgentLaboratory\\AgentLaboratory\\research_results\\raw_gwas_data\\GCST90132315.h.tsv.gz"
solver = MLESolver(
    dataset_code=dataset_code,  # 使用的数据 集代码
    notes=experiment_notes,  # 上一步整理的实验注  释
    insights=lit_review_sum,  # 文献综述总结
    max_steps=mlesolver_max_steps,  # 最大实验迭代步数
    plan=plan,  # 当前实验计划
    Original_exposure_data_path=Original_exposure_data_path,
    Original_outcome_data_path=Original_outcome_data_path,
    openai_api_key=openai_api_key,  # API 密钥
    llm_str=DEFAULT_LLM_BACKBONE  # 使用的语言模型类型
)

if __name__ == "__main__":
    # 执行初始求解步骤（初始化模型和实验流程）
    solver.initial_solve()
    print("初始代码生成成功")
    # 运行多轮优化实验（从第2步到最大步数）
    if solver.best_score < 0.9:
        print("代码评分小于0.9，接下来进行代码优化")
        for _ in range(mlesolver_max_steps - 1):
            solver.solve()
    # print("代码优化成功")
    # 获取得分最高的一组代码字符串（格式为多行代码拼接成字符串）
    code = "\n".join(solver.best_codes[0][0])
    # 提取# 提取实验结果日志
    exp_results = solver.best_codes[0][2]

    # 将最终实验代码保存至实验文件夹中的 src 目录下
    save_to_file(f"./{lab_dir}", "run_experiments.R", code)
    # 将实验输出日志保存到文件中
    save_to_file(f"./{lab_dir}", "experiment_output.log", exp_results)
