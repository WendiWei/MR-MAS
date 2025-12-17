import subprocess

import pandas as pd

# from test_code.deepseek_chat import deepseek_client
from tools import LDPrunerAgent, ExperimentConfig
from utils import save_to_file

# # 模拟 hf_res 原始字符串内容
# hf_res1 = """
# Study ID: GCST90018926
# Paper title: A cross-population atlas of genetic associations for 220 human phenotypes.
# Disease/Trait: Type 2 diabetes
# Population: European
# Publication Day: 2021-09-30
# Downloads Path: https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90018001-GCST90019000/GCST90018926/harmonised/34594039-GCST90018926-EFO_0001360.h.tsv.gz
# Sample description: 38,841 European ancestry cases, 451,248 European ancestry controls, 45,383 East Asian ancestry cases, 132,032 East Asian ancestry controls
# PMID: 34594039
# Journal: Nat Genet
# -------------------------
# Study ID: GCST90043638
# Paper title: A generalized linear mixed model association tool for biobank-scale data.
# Disease/Trait: Type 2 diabetes (PheCode 250.2)
# Population: European
# Publication Day: 2021-11-04
# Downloads Path: https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90043001-GCST90044000/GCST90043638/harmonised/34737426-GCST90043638-EFO_0001360.h.tsv.gz
# Sample description: 714 European ancestry cases, 455,634 European ancestry controls
# PMID: 34737426
# Journal: Nat Genet
# -------------------------
# """
#
# hf_res2 = """
# Study ID: GCST90429775
# Paper title: Characterising the genetic architecture of changes in adiposity during adulthood using electronic health records.
# Disease/Trait: Body mass index
# Population: European
# Publication Day: 2024-07-10
# Downloads Path: https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90429001-GCST90430000/GCST90429775/harmonised/GCST90429775.h.tsv.gz
# Sample description: 161,564 White British ancestry individuals
# PMID: 38987242
# Journal: Nat Commun
# -------------------------
# Study ID: GCST90018947
# Paper title: A cross-population atlas of genetic associations for 220 human phenotypes.
# Disease/Trait: Body mass index
# Population: European
# Publication Day: 2021-09-30
# Downloads Path: https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90018001-GCST90019000/GCST90018947/harmonised/34594039-GCST90018947-EFO_0004340.h.tsv.gz
# Sample description: 359,983 European ancestry individuals, 163,835 East Asian ancestry individuals
# PMID: 34594039
# Journal: Nat Genet
# -------------------------
# """
# # 提取第一条记录
# exposure_information = hf_res1.strip().split('-------------------------')[0]
# outcome_information = hf_res2.strip().split('-------------------------')[0]
#
# combined_info = exposure_information + "\n\n" + outcome_information
#
# print(combined_info)


# 读取 result_code.R 文件内容，并赋值给 result_code 变量
# with open("./results/plan.txt", "r", encoding="utf-8") as f:
#     plan = f.read()
#
# # 打印前100个字符，确认读取成功
# print(plan[:100])


# import os
# openai_api_key=os.getenv('OPENAI_API_KEY')
# print(openai_api_key)


# lab_dir = "paper_research"
# report = "hello"
# save_to_file(f"./{lab_dir}", "report.txt", report)


# import os
# def run_pdflatex(tex_file_path, timeout=30):
#     """
#     调用 pdflatex 编译单个 tex 文件,生成 PDF
#     """
#     dir_path = os.path.dirname(tex_file_path)
#     try:
#         result = subprocess.run(
#             ["pdflatex", "-interaction=nonstopmode", os.path.basename(tex_file_path)],
#             check=True,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             timeout=timeout,
#             cwd=dir_path
#         )
#         print(f" {os.path.basename(tex_file_path)} 编译成功")
#         return f"Compilation successful: {result.stdout.decode('utf-8')}"
#     except subprocess.TimeoutExpired:
#         return f"[ERROR] Compilation timed out after {timeout} seconds: {tex_file_path}"
#     except subprocess.CalledProcessError:
#         return f"[ERROR] Compilation failed: {tex_file_path}"
#
#
# tex_file_path = "./paper_research/report.txt"
# run_pdflatex(tex_file_path)


# import re
# # 初始化字典
# used_pmids = {}
# with open("./paper_research/report.txt", "r", encoding="utf-8") as f:
#     model_resp = f.read()
# _section = "Introduction"
# # model_resp = ""
# pmids_in_text = re.findall(r'PubMed\s*(\d+)', model_resp)
# used_pmids[_section] = pmids_in_text
#
# # 打印调试信息
# print(f"Section: {_section}")
# print(f"Found PubMed IDs: {used_pmids[_section]}")


# import re
#
# # 常见 MR 方法（优先级最高）
# METHOD_ALIAS = {
#     r"mr_ivw": "ivw",
#     r"mr_egger": "mr-egger",
#     r"mr_weighted_median": "weighted_median",
#     r"mr_weighted_mode": "weighted_mode",
#     r"mr_presso": "mr-presso",
#     r"mr_raps": "raps",
#     r"mr_contaminationmixture": "contamination_mixture",
#     r"mr_cml": "cml",
#     r"mr_mix": "mrmix",  # 有时函数写成 mr_mix
#     r"gsmr": "gsmr",
#     r"cause": "cause",
# }
#
# # 常见 MR 工具/包（次优先级）
# PACKAGE_MAP = {
#     r"harmonise_data|ld_clump|mr_leaveoneout|mr_scatter_plot|mr_funnel_plot": "twosamplemr",
#     r"twosamplemr": "twosamplemr",
#     r"ieugwasr": "ieugwasr",
#     r"cause": "cause",
#     r"gsmr": "gsmr",
#     r"mrmix": "mrmix",
#     r"plink": "plink",
#     r"ldsc": "ldsc",
# }
#
#
# def normalize_error_result(model_output, error_log):
#     """
#     使用 LLM 结果，并结合规则进行校正。
#     优先返回具体方法（如 ivw、mr-egger），其次才返回工具包（如 twosamplemr）。
#
#     :param model_output: LLM 原始输出 "method_or_package,function_name"
#     :param error_log: R 错误日志
#     :return: 校正后的 "method_or_package,function_name"
#     """
#     # Step 1: 解析 LLM 输出
#     parts = model_output.strip().lower().split(",")     # parts:['twosamplemr', 'get']
#     if len(parts) != 2:
#         method, func = "unknown", "unknown"
#     else:
#         method, func = parts[0].strip(), parts[1].strip()
#
#     # Step 2: 优先匹配 MR 方法
#     detected_method = None
#     for pattern, mapped in METHOD_ALIAS.items():
#         if re.search(pattern, error_log, re.IGNORECASE):
#             detected_method = mapped
#             break
#
#     # Step 3: 如果没有方法，再匹配 MR 工具/包
#     if not detected_method:
#         for pattern, mapped in PACKAGE_MAP.items():
#             if re.search(pattern, error_log, re.IGNORECASE) or re.search(pattern, func, re.IGNORECASE):
#                 detected_method = mapped
#                 break
#
#     # Step 4: 函数名识别
#     if not func or func == "unknown":
#         m = re.search(r"error in\s+([a-zA-Z0-9_]+)", error_log, re.IGNORECASE)
#         func = m.group(1).lower() if m else "unknown"
#
#     # Step 5: 最终结果
#     if not detected_method:
#         detected_method = method if method != "unknown" else "unknown"
#
#     return detected_method, func
#
#
# error_log = "Error in get(meth) : object 'mr_egger' not found Calls: mr ... -> get"
# raw_output = "twosamplemr,get"  # LLM 可能返回这个
#
# error_method_or_tool,function_name = normalize_error_result(raw_output, error_log)
# print(error_method_or_tool)
# print(function_name)


# import re
#
# log_text = """[CODE EXECUTION ERROR] Warning message: package 'readr' was built under R version 4.4.3 TwoSampleMR version 0.6.22 [>] New authentication requirements: https://mrcieu.github.io/ieugwasr/articles/guide.html#authentication. [>] Major upgrades to our servers completed to improve service and stability. [>] We need your help to shape our emerging roadmap! Please take 2 minutes to give us feedback - https://forms.office.com/e/eSr7EFAfCG Rows: 8 Columns: 9 ── Column specification ──────────────────────────────────────────────────────── Delimiter: "," chr (5): SNP, id.exposure, exposure, effect_allele.exposure, other_allele.ex... dbl (4): beta.exposure, se.exposure, eaf.exposure, pval.exposure ℹ Use spec() to retrieve the full column specification for this data. ℹ Specify the column types or set show_col_types = FALSE to quiet this message. Rows: 16 Columns: 9 ── Column specification ──────────────────────────────────────────────────────── Delimiter: "," chr (5): SNP, id.outcome, outcome, effect_allele.outcome, other_allele.outcome dbl (4): beta.outcome, se.outcome, eaf.outcome, pval.outcome ℹ Use spec() to retrieve the full column specification for this data. ℹ Specify the column types or set show_col_types = FALSE to quiet this message. Harmonising systolic blood pressure (systolic blood pressure) and coronary artery disease (coronary artery disease) Attaching package: 'MendelianRandomization' The following objects are masked from 'package:TwoSampleMR': mr_ivw, mr_median Warning message: package 'MendelianRandomization' was built under R version 4.4.3 Rows: 8 Columns: 9 ── Column specification ──────────────────────────────────────────────────────── Delimiter: "," chr (5): SNP, id.exposure, exposure, effect_allele.exposure, other_allele.ex... dbl (4): beta.exposure, se.exposure, eaf.exposure, pval.exposure ℹ Use spec() to retrieve the full column specification for this data. ℹ Specify the column types or set show_col_types = FALSE to quiet this message. Rows: 16 Columns: 9 ── Column specification ──────────────────────────────────────────────────────── Delimiter: "," chr (5): SNP, id.outcome, outcome, effect_allele.outcome, other_allele.outcome dbl (4): beta.outcome, se.outcome, eaf.outcome, pval.outcome ℹ Use spec() to retrieve the full column specification for this data. ℹ Specify the column types or set show_col_types = FALSE to quiet this message. Harmonising systolic blood pressure (systolic blood pressure) and coronary artery disease (coronary artery disease) Analysing 'systolic blood pressure' on 'coronary artery disease' Analysing 'systolic blood pressure' on 'coronary artery disease' Analysing 'systolic blood pressure' on 'coronary artery disease' Analysing 'systolic blood pressure' on 'coronary artery disease' Rows: 7550174 Columns: 14 ── Column specification ──────────────────────────────────────────────────────── Delimiter: "\t" chr (6): effect_allele, other_allele, rsid, rs_id, hm_coordinate_conversion,... dbl (8): chromosome, base_pair_location, beta, standard_error, effect_allele... ℹ Use spec() to retrieve the full column specification for this data. ℹ Specify the column types or set show_col_types = FALSE to quiet this message. Rows: 20220308 Columns: 25 ── Column specification ──────────────────────────────────────────────────────── Delimiter: "\t" chr (6): effect_allele, other_allele, markername, direction, hm_coordinate_... dbl (18): chromosome, base_pair_location, beta, standard_error, effect_allel... lgl (1): variant_id ℹ Use spec() to retrieve the full column specification for this data. ℹ Specify the column types or set show_col_types = FALSE to quiet this message. Error in cause_method(raw_exposure_data, raw_outcome_data) : could not find function "cause_method" Execution halted"""
#
# # 匹配以 Error 开头到 Execution halted 之间的内容
# match = re.search(r"(Error.*?Execution halted)", log_text, re.S)
#
# if match:
#     code_err = match.group(1).strip()
#     print(f"该代码编译错误，报错为{code_err}")
# else:
#     print("未找到错误信息")


# import pandas as pd
# def filter_significant_mutations(file_path, pvalue_threshold=5e-8, sep='\t', min_variants=10):
#     """
#     从 .gz 压缩文件中筛选显著变异，并根据阈值动态调整。
#
#     参数：
#     - file_path: 输入文件路径（支持 .gz），需包含 'p_value' 和 'variant_id' 两列
#     - pvalue_threshold: 初始显著性阈值，默认 5e-8
#     - sep: 文件分隔符，默认制表符 '\t'
#     - min_variants: 最少显著位点数，默认 20
#
#     返回：
#     - significant_df: 筛选出的显著变异 DataFrame
#     - used_threshold: 实际使用的 p 值阈值
#     """
#     try:
#         df = pd.read_csv(file_path, sep=sep, compression='infer')
#     except Exception as e:
#         print(f"读取文件失败：{e}")
#         return None, None
#
#     required_columns = {'p_value', 'variant_id'}
#     if not required_columns.issubset(df.columns):
#         print(f"文件中缺少以下必要列：{required_columns - set(df.columns)}")
#         return None, None
#
#     # 定义一系列备用阈值
#     thresholds = [pvalue_threshold, 1e-7, 1e-6, 1e-5]
#
#     for thresh in thresholds:
#         significant_df = df[df['p_value'] < thresh]
#         len_snps = len(significant_df)
#         if len(significant_df) >= min_variants:
#             print(f"使用阈值 {thresh} 共筛选出 {len(significant_df)} 个显著变异")
#             return significant_df, thresh
#
#     # 如果所有阈值都不足，则返回最后一个阈值的结果
#     print(f"即使使用最大阈值 {thresholds[-1]}，共筛选出 {len(significant_df)} 个显著变异")
#     return significant_df, thresholds[-1]
#
# Original_exposure_data_path = "research_results/raw_gwas_data/34737426-GCST90043687-EFO_0004318.h.tsv.gz"
# df_sig, used_thresh = filter_significant_mutations(Original_exposure_data_path)
# print(f"实际使用阈值: {used_thresh}")



# cfg = ExperimentConfig()
# # prune_params_str = extract_prompt(resp, "LD_PRUNE")
# # print("prune_params_str:", prune_params_str)
# prune_params = {'r2': 0.01, 'window_kb': 500}
# # print("prune_params:", prune_params)
# # 执行LD剪枝
# # if 'exposure_file_path' in locals():
# exposure_outcome_intersection_snps_path = "research_results/processed_data/exposure_outcome_intersection_snps.csv"
# snps_data = pd.read_csv(exposure_outcome_intersection_snps_path)
# snps_list = snps_data['hm_rsid']
# if len(snps_list) != 0:
#     print(f"开始对snps列表进行LD剪枝，参数：{prune_params}")
#     # 如果 prune_params 字典中有键 'r2'，就使用它的值；否则，使用默认值 0.1。
#     LDagent = LDPrunerAgent(
#         gwas_file=exposure_outcome_intersection_snps_path,
#         # pval_threshold=5e-8,
#         output_dir="./research_results/ld_processed_data",
#         r2=prune_params.get('r2', 0.001),
#         window_kb=prune_params.get('window_kb', 10000)
#     )
#     result = LDagent.command_run_auto({
#         "gwas_file": exposure_outcome_intersection_snps_path,
#         "min_snps_threshold": 10,
#         "max_attempts": 10,
#         "step": 10,
#         "r2": prune_params.get('r2'),
#         "window_kb": prune_params.get('window_kb')
#     })
#     # print("剪枝结果文件:", result["pruned_gwas_file"])
#     print("使用的r2参数:", result["r2_used"])
#     print("使用的window_kb:", result["window_kb_used"])
#     print("剪枝后SNP数量:", result["snp_count"])
#     cfg.set("data_preparation", "ld_prune", {"r2": result["r2_used"], "window_kb": result["window_kb_used"]})
#
# ld_params = cfg.get("data_preparation", "ld_prune")
# print("LD pruning parameters:", ld_params)






# import configparser
# import json
#
# cfg = configparser.ConfigParser()
#
# # exposure_data_information 字符串
# exposure_info_str = """Study ID: GCST90310295
# Paper title: Genome-wide analysis in over 1 million individuals of European ancestry yields improved polygenic risk scores for blood pressure traits.
# Disease/Trait: Diastolic blood pressure
# Population: European
# Publication Day: 2024-04-30
# Downloads Path: https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90310001-GCST90311000/GCST90310295/harmonised/GCST90310295.h.tsv.gz
# Sample description: 1,028,980 European ancestry individuals
# PMID: 38689001
# Journal: Nat Genet
# """
#
# # 将字符串按行拆分，再按第一个冒号拆分为键值对
# exposure_information = {}
# for line in exposure_info_str.strip().split("\n"):
#     if ": " in line:
#         key, value = line.split(": ", 1)
#         exposure_information[key.strip()] = value.strip()
#
# cfg.add_section("data_preparation")
# cfg.set(
#     "data_preparation",
#     "exposure_data_information",
#     json.dumps(exposure_information, indent=4)  # 保持字典结构
# )
#
# # 保存到文件
# with open("config.ini", "w") as f:
#     cfg.write(f)



# from pandasgwas.get_variants import get_variants_by_gene_name
# import pandas as pd
#
# gene_name = "APOE"
# variants = get_variants_by_gene_name(gene_name)
#
# # 确保是列表
# if not isinstance(variants, list):
#     variants = [variants]
#
# records = []
# for v in variants:
#     # SNP 基本信息
#     rsid = v.rsid
#     pvalue = v.pvalue
#     trait = v.trait
#     pmid = v.pmid
#
#     # locations 是 DataFrame，可能有多行
#     if hasattr(v, "locations") and not v.locations.empty:
#         for _, loc in v.locations.iterrows():
#             chromosome = loc.get("chromosome")
#             position = loc.get("position")
#
#             # genomic_contexts 是 DataFrame，也可能有多行
#             if hasattr(v, "genomic_contexts") and not v.genomic_contexts.empty:
#                 for _, ctx in v.genomic_contexts.iterrows():
#                     ensembl_ids = ctx.get("ensembl_gene_ids")
#                     entrez_ids = ctx.get("entrez_gene_ids")
#
#                     records.append({
#                         "rsid": rsid,
#                         "chromosome": chromosome,
#                         "position": position,
#                         "trait": trait,
#                         "pvalue": pvalue,
#                         "pmid": pmid,
#                         "ensembl_gene_ids": ensembl_ids,
#                         "entrez_gene_ids": entrez_ids
#                     })
#             else:
#                 # 没有 genomic_contexts
#                 records.append({
#                     "rsid": rsid,
#                     "chromosome": chromosome,
#                     "position": position,
#                     "trait": trait,
#                     "pvalue": pvalue,
#                     "pmid": pmid,
#                     "ensembl_gene_ids": None,
#                     "entrez_gene_ids": None
#                 })
#     else:
#         # 没有 locations
#         records.append({
#             "rsid": rsid,
#             "chromosome": None,
#             "position": None,
#             "trait": trait,
#             "pvalue": pvalue,
#             "pmid": pmid,
#             "ensembl_gene_ids": None,
#             "entrez_gene_ids": None
#         })
#
# df_snps = pd.DataFrame(records)
# print(df_snps.head())





# import ieugwaspy.query as query
#
# # 指定基因名
# gene_name = 'KIAA0319'
#
# # 使用 phewas 函数查询与基因相关的 GWAS 数据
# result = query.phewas(gene_name)
#
# # 打印查询结果
# print(result)




# import os
# import ieugwaspy
# import ieugwaspy.query as query
# import pandas as pd
#
# def setup_jwt():
#     """
#     配置 JWT，首次运行需要手动输入 Token，会生成 .ieugwaspy.json
#     """
#     jwt_path = os.path.expanduser("~/.ieugwaspy.json")
#     if not os.path.exists(jwt_path):
#         print("JWT 配置文件不存在，请输入你的 OpenGWAS JWT Token：")
#         ieugwaspy.get_jwt()  # 会提示输入 Token 并生成配置文件
#     else:
#         print("JWT 已配置，可直接使用 OpenGWAS API。")
#
# def fetch_gwas_by_gene(gene_name):
#     """
#     输入基因名，返回 MR-ready GWAS DataFrame
#     """
#     try:
#         # phewas 返回的是 dict
#         result = query.phewas(gene_name)
#
#         if not result or 'data' not in result:
#             print(f"未找到基因 {gene_name} 的 GWAS 数据")
#             return pd.DataFrame()
#
#         # 'data' 键包含 GWAS 条目列表
#         df = pd.DataFrame(result['data'])
#         if df.empty:
#             print(f"基因 {gene_name} 的 GWAS 数据为空")
#             return pd.DataFrame()
#
#         # 选择 MR 相关列，如果不存在则补 None
#         mr_cols = ['rsid', 'trait', 'p', 'beta', 'se', 'ea', 'nea', 'n']
#         for col in mr_cols:
#             if col not in df.columns:
#                 df[col] = None
#
#         df_mr = df[mr_cols]
#         return df_mr
#
#     except Exception as e:
#         print(f"获取 GWAS 数据失败：{e}")
#         return pd.DataFrame()
#
# # 示例
# gene = "FOXA1"
# df_mr_ready = fetch_gwas_by_gene(gene)
# if not df_mr_ready.empty:
#     print(df_mr_ready.head())
# else:
#     print("未获取到有效 GWAS 数据。")





# import pandas as pd
#
#
# def add_r2_fstat_filter(df, N, f_threshold = 10):
#     """
#     给 exposure 数据框添加 R² 和 F 统计量，并过滤掉弱工具变量 (F <= f_threshold)
#
#     参数:
#         df : pd.DataFrame
#             必须包含以下列：
#             - beta.exposure
#             - se.exposure
#             - eaf.exposure
#         N : int
#             样本量
#         f_threshold : float
#             F 统计量的阈值 (默认 10)
#
#     返回:
#         df_all : 带 R2 和 F_stat 的完整表格
#         df_filtered : 过滤掉弱工具 (F <= f_threshold) 的子集
#     """
#
#     def calc_r2(beta, se, eaf, n):
#         return (2 * eaf * (1 - eaf) * beta ** 2) / (
#                 2 * eaf * (1 - eaf) * beta ** 2 + 2 * eaf * (1 - eaf) * n * se ** 2
#         )
#
#     def calc_f(r2, n):
#         return (r2 * (n - 2)) / (1 - r2)
#
#     df_all = df.copy()
#     df_all["R2"] = df_all.apply(
#         lambda row: calc_r2(row["beta.exposure"], row["se.exposure"], row["eaf.exposure"], N),
#         axis=1
#     )
#     df_all["F_stat"] = df_all.apply(
#         lambda row: calc_f(row["R2"], N),
#         axis=1
#     )
#
#     # 过滤掉弱工具
#     # 把 F ≤ 10 的 SNP 自动剔除
#     df_filtered = df_all[df_all["F_stat"] > f_threshold].reset_index(drop=True)
#
#     return df_all, df_filtered
#
#
# exposure_df = pd.read_csv("research_results/processed_data/filtered_exposure_data.csv")
# N = 50
# # outcome_df = pd.read_csv("research_results/processed_data/filtered_outcome_data.csv")
# # df_out = add_r2_fstat_filter(exposure_df, N)
# df_all, df_filtered = add_r2_fstat_filter(exposure_df, N)
# exposure_data_rsid = df_filtered[["SNP"]]
# exposure_data_rsid_path = "research_results/processed_data/exposure_data_filtered_rsid.csv"
# exposure_data_rsid.to_csv(exposure_data_rsid_path, index=False)
#
# print("\n=== 筛选后 SNP（F > 10） ===")
# print(df_filtered[["SNP", "beta.exposure", "se.exposure", "eaf.exposure", "R2", "F_stat"]])





# 提取暴露数据的样本量
# import re
# import json
#
# # 原始数据（你提供的 exposure_information）
# exposure_information = {
#     "exposure_data_information": "\"\n        Study ID: GCST90013975\\n\n        Paper title: Computationally efficient whole-genome regression for quantitative and binary traits.\\n\n        Disease/Trait: Body fat percentage (UKB data field 23099)\\n\n        Population: European\\n\n        Publication Day: 2021-05-20\\n\n        Downloads Path: https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90013001-GCST90014000/GCST90013975/harmonised/34017140-GCST90013975-EFO_0007800.h.tsv.gz\\n\n        Sample description: 401,772 British ancestry individuals\\n\n        PMID: 34017140\\n\n        Journal: Nat Genet\\n\""
# }
#
# # 提取字符串
# info_str = exposure_information["exposure_data_information"]
#
# # 用正则匹配 Sample description 行
# match = re.search(r"Sample description:\s*([\d,]+)", info_str)
#
# if match:
#     sample_size_str = match.group(1)
#     # 去掉逗号，转成整数
#     sample_size = int(sample_size_str.replace(",", ""))
#     print("样本量:", sample_size)
# else:
#     print("未找到样本量")





# 提取种族
# import re
# from inference import *
#
# text = "Sample description: 121,940 European ancestry cases, 1,254,131 European ancestry controls\n"
# # # text = "Sample description: 38,841 European ancestry cases, 451,248 European ancestry controls, 45,383 East Asian ancestry cases, 132,032 East Asian ancestry controls\n"
# # text = "Sample description: 38,841 European ancestry cases, 451,248 European ancestry controls, 45,383 East Asian ancestry cases, 132,032 East Asian ancestry controls\n"
# # text = "Sample description: 7,050 European ancestry men, 1,324 African American men\n"
# # text = "Sample description: up to 104,666 European ancestry male individuals, up to 132,115 European ancestry female individuals, 370 African American male individuals, 517 African American female individuals, 512 Hispanic male individuals, 764 Hispanic female individuals\n"
# # text = "517 African American individuals"
# #text = "Sample description: 176,408 White British ancestry individuals\n"
# mr_populations = [
#     "european",
#     "east asian",
#     "south asian",
#     "white british",
#     "african",
#     "african american",
#     "hispanic",
#     "native american",
#     "middle eastern",
#     "central asian",
#     "oceanian",
#     "multi-ancestry",
#     "admixed",
#     "unknown"
# ]
# # 构造正则 pattern，注意按长度降序排序，避免部分匹配冲突
# mr_populations_sorted = sorted(mr_populations, key=len, reverse=True)
# pattern = r"\b(" + "|".join(re.escape(p) for p in mr_populations_sorted) + r")\b"
#
# found = re.findall(pattern, text.lower())
# found = list(dict.fromkeys(found))  # 去重，保持顺序
#
# print(found)      # ['white british']
# print(len(found)) # 1



# 种群识别大模型
# def population_identifier(sample_text, POP_LLM, openai_api_key=None):
#     identify_sys = (
#         "You are an automated population extraction agent in a Mendelian Randomization (MR) system.\n"
#         "You are given a free-text sample description or population metadata.\n"
#         "Your task is to identify which population/ethnic groups are mentioned.\n\n"
#         "Output format:\n"
#         "- Return a comma-separated list of populations (e.g., european, east asian, african american, hispanic).\n"
#         "- All lowercase.\n"
#         "- Use concise standardized labels (e.g., 'european', not 'british ancestry'; 'african american', not 'african american male').\n"
#         "- If multiple populations exist, list them all separated by commas.\n"
#         "- If you cannot determine, return 'unknown'.\n"
#     )
#
#     model_resp = query_model(
#         openai_api_key=openai_api_key,
#         model_str=f"{POP_LLM}",
#         system_prompt=identify_sys,
#         prompt=f"Here is the sample description:\n\n{sample_text}",
#         temp=0.0
#     )
#
#     # 尝试解析输出，转成标准形式
#     try:
#         populations = [p.strip().lower() for p in model_resp.split(",") if p.strip()]
#         populations = list(dict.fromkeys(populations))  # 去重并保持顺序
#     except Exception:
#         populations = ["unknown"]
#
#     return populations
#
# openai_api_key = os.getenv('OPENAI_API_KEY')
# print(population_identifier(text, "gpt-4o", openai_api_key=openai_api_key))
# 匹配 ancestry 前面的族群名称，转小写
# 匹配族群名，直到遇到 ancestry、male、female 或 individuals，但不把这些词抓进结果
# matches = re.findall(r'([A-Za-z ]+?)(?=\s+(?:ancestry|male|female|individuals))', text)
# print(type(matches))
#
# # 转小写、去掉首尾空格、去重并保持顺序
# unique_ethnicities = list(dict.fromkeys([m.strip().lower() for m in matches]))
# print(unique_ethnicities)
#
# # print(unique_ethnicities)


# import requests
#
# base_url = "https://www.ebi.ac.uk/gwas/rest/api/studies"
# page = 0
# all_populations = set()
#
# while True:
#     resp = requests.get(f"{base_url}?page={page}&size=100")
#     data = resp.json()
#     studies = data.get("_embedded", {}).get("studies", [])
#     if not studies:
#         break
#     for st in studies:
#         ancestries = st.get("ancestries", [])
#         for anc in ancestries:
#             # anc 是字典，取 ancestry 字段
#             ancestry_name = anc.get("ancestry")
#             if ancestry_name:
#                 all_populations.add(ancestry_name.lower().strip())
#                 print(ancestry_name)
#     page += 1
#
# print(all_populations)



# from inference import *
# # 可以用一个大模型来自动判断所选 GWAS 数据集的表型是否适合你的 MR 分析目标
# def phenotype_matcher(user_trait, gwas_trait_description, LLM_model="gpt-4o", openai_api_key=None):
#     """
#     使用大模型判断 GWAS 数据集表型是否适合用户指定的性状，
#     返回两个独立值：decision 和 explanation。
#     """
#     system_prompt = (
#         "You are an expert in human genetics and Mendelian Randomization studies.\n"
#         "You are given a user's target trait and a GWAS dataset trait description.\n"
#         "Judge whether this GWAS dataset is suitable to study the user's target trait.\n"
#         "Rules:\n"
#         "1. If the GWAS dataset only covers a subset or specific subtype of the trait, respond 'no'.\n"
#         "2. Otherwise, respond 'yes' if suitable.\n"
#         "3. Provide a brief explanation (max 50 words).\n"
#         "Output format: decision: <yes/no>, explanation: <brief explanation>"
#     )
#
#     prompt = (
#         f"User target trait: {user_trait}\n"
#         f"GWAS dataset trait description: {gwas_trait_description}\n"
#         "Return your judgment in the specified format."
#     )
#
#     # 调用大模型
#     response = query_model(
#         openai_api_key=openai_api_key,
#         model_str=LLM_model,
#         system_prompt=system_prompt,
#         prompt=prompt,
#         temp=0.0
#     )
#
#     # 尝试解析为 decision 和 explanation
#     try:
#         parts = response.split("explanation:")
#         decision = parts[0].replace("decision:", "").strip().replace(",", "").lower()
#         explanation = parts[1].strip()
#     except Exception:
#         decision = "unknown"
#         explanation = response.strip()
#
#     return decision, explanation
#
#
# user_trait = "'high density lipoprotein cholesterol measurement'"
# gwas_trait = "HDL cholesterol levels (UKB data field 30760)"
# openai_api_key = os.getenv('OPENAI_API_KEY')
# decision, explanation = phenotype_matcher(user_trait, gwas_trait, openai_api_key=openai_api_key)
#
# print(decision)
# print("\\\\\\\\\\\\\\\\\\\\\\\\\\")# "no"
# print(explanation) # "The GWAS dataset covers only postmenopausal breast cancer, not the full trait."


# import requests
#
# # 设置API密钥和请求URL
# API_KEY = "sk-DQ4r68PsB9E8tVhI718162703dF1416d9aCeC61758759aDa"  # 替换为你的laozhang.ai API密钥
# API_URL = "https://api.laozhang.ai/v1/chat/completions"
#
# # 构建请求
# headers = {
#     "Content-Type": "application/json",
#     "Authorization": f"Bearer {API_KEY}"
# }
#
# payload = {
#     "model": "claude-3-5-sonnet",
#     "messages": [
#         {"role": "system", "content": "你是Claude 3.5 Sonnet，一个由Anthropic开发的AI助手。请用中文回答问题。"},
#         {"role": "user", "content": "请分析一下你与GPT-4o相比有哪些优势和不足？"}
#     ],
#     "temperature": 0.7,
#     "max_tokens": 2000
# }
#
# # 发送请求
# response = requests.post(API_URL, headers=headers, json=payload)
# response_data = response.json()
#
# # 处理响应
# if "choices" in response_data:
#     message = response_data["choices"][0]["message"]["content"]
#     print(message)
# else:
#     print("错误:", response_data)










# import os
# import openai
#
# # 设置 API Key
# openai.api_key = os.getenv("OPENAI_API_KEY")  # 或直接写入字符串
# print(os.getenv("OPENAI_API_KEY") )
# # 构建对话消息
# messages = [
#     {"role": "system", "content": "你是 GPT-5-mini，一个高效、快速的语言模型助手。"},
#     {"role": "user", "content": "请帮我总结一下GPT-4o和GPT-5的区别。"}
# ]
#
# # 调用 GPT-5-mini
# response = openai.chat.completions.create(
#     model="gpt-5-mini",
#     messages=messages,
#     temperature=0.7,
#     max_tokens=1000
# )
#
# # 获取模型回复
# reply = response.choices[0].message.content
# print(reply)


import os
from openai import OpenAI

# def call_qwen3(prompt, system_prompt="", version="latest"):
#     model_str = "deepseek-chat-r1"
#     client = OpenAI(
#         api_key=os.getenv("QWEN_API_KEY"),
#         base_url="https://api.claudeshop.top/v1"  # ClaudeShop 的 OpenAI兼容接口
#     )
#
#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": prompt}
#     ]
#
#     completion = client.chat.completions.create(
#         model=model_str,
#         messages=messages
#     )
#
#     return completion.choices[0].message.content
# def claude(prompt, system_prompt="", version="latest"):
#     model_str = "claude-3-7-sonnet-20250219"
#     client = OpenAI(
#         api_key=os.getenv("CLAUDE_API_KEY"),
#         base_url="https://api.claudeshop.top/v1"  # ClaudeShop 的 OpenAI兼容接口
#     )
#
#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": prompt}
#     ]
#
#     completion = client.chat.completions.create(
#         model=model_str,
#         messages=messages
#     )
#
#     return completion.choices[0].message.content
# def GPT(prompt, system_prompt="", version="latest"):
#     model_str = "gpt-5"
#     client = OpenAI(
#         api_key=os.getenv("OPENAI_API_KEY"),
#         base_url="https://api.claudeshop.top/v1"  # ClaudeShop 的 OpenAI兼容接口
#     )
#
#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": prompt}
#     ]
#
#     completion = client.chat.completions.create(
#         model=model_str,
#         messages=messages
#     )
#
#     return completion.choices[0].message.content
# # 示例调用
# if __name__ == "__main__":
#     result = GPT("帮我写一首关于秋天的诗")
#     print(result)
if __name__ == '__main__':
    score = 0.9
    score = float(score)
    if type(score) is float:
        if score >= 0.5:
            print('success')
    else:
        print('false')