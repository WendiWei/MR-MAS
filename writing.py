# 导入 PaperSolver 并初始化参考文献列表
from paper_writing import PaperSolver
from utils import *
import os
import time
reference_papers = []
report_notes = None
papersolver_max_steps = 1
plan = None
start = time.time()
# with open("./results/plan.txt", "r", encoding="utf-8") as f:
#     plan = f.read()
# 读取 result_code.R 文件内容，并赋值给 result_code 变量
with open("./results/result_code.R", "r", encoding="utf-8") as f:
    results_code = f.read()

with open("./results/experiment_output.log", "r", encoding="utf-8") as f:
    exp_results = f.read()

with open("./results/interpretation.txt", "r", encoding="utf-8") as f:
    interpretation = f.read()
lit_review=None
sdataset_information =None
research_topic="Your goal is to investigate the causal relationship between BMI and Type 2 Diabetes, using Mendelian Randomization based on GWAS summary statistics."
openai_api_key = os.getenv('OPENAI_API_KEY')
DEFAULT_LLM_BACKBONE = "gpt-4o"
compile_pdf = False
lab_dir = "paper_research"
# 初始化 PaperSolver，用于生成论文报告
solver = PaperSolver(
    notes=report_notes,  # 实验相关笔记
    max_steps=papersolver_max_steps,  # 最大迭代步数
    plan=plan,  # 实验计划
    exp_code=results_code,  # 实验代码
    exp_results=exp_results,  # 实验结果
    insights=interpretation,  # 结果解读
    lit_review=lit_review,  # 文献综述
    datasets_information=sdataset_information,  # 数据集信息
    ref_papers=reference_papers,  # 参考文献（初始化为空）
    topic=research_topic,  # 研究主题
    openai_api_key=openai_api_key,  # OpenAI API key
    llm_str=DEFAULT_LLM_BACKBONE,  # 使用的模型
    compile_pdf=compile_pdf,  # 是否编译 PDF
    save_loc=lab_dir  # 保存路径
)

# 执行初始解算
solver.initial_solve()
print(f"初始最高分为：{solver.best_score}")
# 如果初始得分小于 0.5，则进行多次优化迭代

for _ in range(papersolver_max_steps):
    solver.solve()

# 获取最优报告的内容和得分
report = "\n".join(solver.best_report[0][0])
# score = solver.best_report[0][1]

# 保存报告到文本文件
# save_to_file(f"./{lab_dir}", "report.txt", report)
# tex_file_path = f"./{lab_dir}/report.txt"
# # tex_file_path = "./paper_research/report.txt"
# run_pdflatex(tex_file_path)

end = time.time()
print(f"程序运行时间: {end - start:.4f} 秒")
