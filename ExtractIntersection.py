import pandas as pd
import subprocess
from pathlib import Path

# ========== 配置 ==========
trait = "BMI"  # 自定义性状名
A_path = "D:\研二上\data\TwoSampleMR\BMI\IV_BMI_filtered_F10.csv"
B_path = "D:\pycharm_projects\AgentLaboratory\AgentLaboratory\experiment_research\Supported_by_literature\BMI_CAD\data\exposure_data_filtered_rsid.csv"
out_file_path = "D:/MR_project/compare_snps/BMI_results_summary.csv"

plink_prefix = r"D:\pycharm_projects\AgentLaboratory\AgentLaboratory\data\1000G_EUR\1000G_phase3_common_norel"  # PLINK .bed/.bim/.fam 文件前缀
r2_thresh = 0.8
window_kb = 1000
locus_window = 250000


# ========== 辅助函数 ==========
def compute_metrics(matched_set, A_size, B_size):
    TP = len(matched_set)
    precision = TP / B_size if B_size > 0 else None
    recall = TP / A_size if A_size > 0 else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall and (precision + recall) != 0 else None
    return {"TP": TP, "Precision": precision, "Recall": recall, "F1": f1}


def locus_match(A_pos, B_pos, window=locus_window):
    matched = []
    for idx, row in A_pos.iterrows():
        chr_i, pos_i, snp = row['chr'], row['pos'], row['SNP']
        subsetB = B_pos[(B_pos['chr'] == chr_i) & (B_pos['pos'] >= pos_i - window) & (B_pos['pos'] <= pos_i + window)]
        if len(subsetB) > 0:
            matched.append(snp)
    return list(set(matched))


def get_proxy_plink(snp_list, plink_prefix, r2_thresh=0.8, window_kb=1000):
    snp_file = Path("temp_snps.txt")
    snp_file.write_text("\n".join(snp_list))

    out_prefix = "temp_ld"

    cmd = [
        "plink",
        "--bfile", plink_prefix,
        "--r2",
        "--ld-snp-list", str(snp_file),
        "--ld-window", "99999",
        "--ld-window-kb", str(window_kb),
        "--ld-window-r2", str(r2_thresh),
        "--out", out_prefix
    ]

    subprocess.run(cmd, check=True)

    ld_file = Path(out_prefix + ".ld")
    if not ld_file.exists():
        return pd.DataFrame(columns=["SNP_A", "SNP_B"])

    ld_data = pd.read_csv(ld_file, delim_whitespace=True, usecols=["SNP_A", "SNP_B"])
    return ld_data


# ========== 主逻辑 ==========
# 读入 SNP 集合
A = pd.read_csv(A_path)
B = pd.read_csv(B_path)
A_snps = A['SNP'].unique()
B_snps = B['SNP'].unique()

# -------- 精确匹配 --------
exact_inter = list(set(A_snps) & set(B_snps))
metrics_exact = compute_metrics(exact_inter, len(A_snps), len(B_snps))

# -------- LD proxy (PLINK) --------
ld_df = get_proxy_plink(A_snps.tolist(), plink_prefix, r2_thresh, window_kb)
matched_via_proxy = ld_df[ld_df['SNP_B'].isin(B_snps)]['SNP_A'].unique().tolist()
metrics_proxy = compute_metrics(matched_via_proxy, len(A_snps), len(B_snps))

# -------- Locus ±250kb 匹配 --------
bim_file = pd.read_csv(plink_prefix + ".bim", sep="\t", header=None,
                       names=["chr", "SNP", "cm", "pos", "A1", "A2"])
A_pos = bim_file[bim_file['SNP'].isin(A_snps)][['chr', 'pos', 'SNP']]
B_pos = bim_file[bim_file['SNP'].isin(B_snps)][['chr', 'pos', 'SNP']]
locus_matches = locus_match(A_pos, B_pos, window=locus_window)
metrics_locus = compute_metrics(locus_matches, len(A_snps), len(B_snps))

# -------- 保存结果 --------
results = []
for metric_name, metrics in zip(["Exact", "Proxy_r2>=0.8", "Locus_250kb"],
                                [metrics_exact, metrics_proxy, metrics_locus]):
    results.append({"Trait": trait, "Metric": metric_name, **metrics})

results_df = pd.DataFrame(results)
results_df.to_csv(out_file_path, index=False)
print(f"结果已保存到: {out_file_path}")

# A_path = "D:\研二上\data\TwoSampleMR\BMI\IV_BMI_filtered_F10.csv"
# B_path = "D:\pycharm_projects\AgentLaboratory\AgentLaboratory\experiment_research\Supported_by_literature\BMI_CAD\data\exposure_data_filtered_rsid.csv"
# A_data = pd.read_csv(A_path)
# B_data = pd.read_csv(B_path)
# # 提取 SNP 列
# A_snp_list = A_data["SNP"].tolist()
# B_snp_list = B_data["SNP"].tolist()
# print(A_snp_list)
# print(B_snp_list)
# # 求交集
# intersection = set(A_snp_list) & set(B_snp_list)
#
# # 交集数量
# # print(intersection)
# print(f"交集数量: {len(intersection)}")