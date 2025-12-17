import json
from asyncio import timeout
from inference import *
from utils import *

import arxiv
import io, sys
import traceback
import matplotlib
import numpy as np
import multiprocessing
from pypdf import PdfReader
from tqdm import tqdm
from psutil._common import bytes2human
from datasets import load_dataset_builder, load_dataset
from semanticscholar import SemanticScholar
from sklearn.metrics.pairwise import linear_kernel
from pandasgwas import get_studies
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from requests.adapters import HTTPAdapter, Retry
import pandas as pd
import itertools
import subprocess
import time
import os
from Bio import Entrez
import requests
from bs4 import BeautifulSoup

import json
import os


class ExperimentConfig:
    """
    通用的参数存储类 ExperimentConfig，支持跨阶段保存和读取参数
    """

    def __init__(self, file_path="experiment_config.json"):
        self.file_path = file_path
        self.config = {}
        if os.path.exists(file_path):
            self.load()

    def set(self, phase, key, value):
        """为某个阶段设置参数"""
        if phase not in self.config:
            self.config[phase] = {}
        self.config[phase][key] = value
        self.save()

    def get(self, phase, key=None, default=None):
        """获取某个阶段的参数"""
        if phase not in self.config:
            return default
        if key is None:
            return self.config[phase]
        return self.config[phase].get(key, default)

    def save(self):
        """保存到文件"""
        with open(self.file_path, "w") as f:
            json.dump(self.config, f, indent=4)

    def load(self):
        """从文件加载"""
        with open(self.file_path, "r") as f:
            self.config = json.load(f)


class HFDataSearch:
    def __init__(self, like_thr=3, dwn_thr=50) -> None:
        """
        Class for finding relevant huggingface datasets
        :param like_thr:
        :param dwn_thr:
        """
        self.dwn_thr = dwn_thr
        self.like_thr = like_thr
        self.ds = load_dataset("nkasmanoff/huggingface-datasets")["train"]

        # Initialize lists to collect filtered data
        filtered_indices = []
        filtered_descriptions = []
        filtered_likes = []
        filtered_downloads = []

        # Iterate over the dataset and filter based on criteria
        for idx, item in enumerate(self.ds):
            # Get likes and downloads, handling None values
            likes = int(item['likes']) if item['likes'] is not None else 0
            downloads = int(item['downloads']) if item['downloads'] is not None else 0

            # Check if likes and downloads meet the thresholds
            if likes >= self.like_thr and downloads >= self.dwn_thr:
                # Check if the description is a non-empty string
                description = item['description']
                if isinstance(description, str) and description.strip():
                    # Collect the data
                    filtered_indices.append(idx)
                    filtered_descriptions.append(description)
                    filtered_likes.append(likes)
                    filtered_downloads.append(downloads)

        # Check if any datasets meet all criteria
        if not filtered_indices:
            print("No datasets meet the specified criteria.")
            self.ds = []
            self.descriptions = []
            self.likes_norm = []
            self.downloads_norm = []
            self.description_vectors = None
            return  # Exit the constructor

        # Filter the datasets using the collected indices
        self.ds = self.ds.select(filtered_indices)

        # Update descriptions, likes, and downloads
        self.descriptions = filtered_descriptions
        self.likes = np.array(filtered_likes)
        self.downloads = np.array(filtered_downloads)

        # Normalize likes and downloads
        self.likes_norm = self._normalize(self.likes)
        self.downloads_norm = self._normalize(self.downloads)

        # Vectorize the descriptions
        self.vectorizer = TfidfVectorizer()
        self.description_vectors = self.vectorizer.fit_transform(self.descriptions)

    def _normalize(self, arr):
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val == 0:
            return np.zeros_like(arr, dtype=float)
        return (arr - min_val) / (max_val - min_val)

    def retrieve_ds(self, query, N=10, sim_w=1.0, like_w=0.0, dwn_w=0.0):
        """
        Retrieves the top N datasets matching the query, weighted by likes and downloads.
        :param query: The search query string.
        :param N: The number of results to return.
        :param sim_w: Weight for cosine similarity.
        :param like_w: Weight for likes.
        :param dwn_w: Weight for downloads.
        :return: List of top N dataset items.
        """
        if not self.ds or self.description_vectors is None:
            print("No datasets available to search.")
            return []

        query_vector = self.vectorizer.transform([query])
        cosine_similarities = linear_kernel(query_vector, self.description_vectors).flatten()
        # Normalize cosine similarities
        cosine_similarities_norm = self._normalize(cosine_similarities)
        # Compute final scores
        final_scores = (
                sim_w * cosine_similarities_norm +
                like_w * self.likes_norm +
                dwn_w * self.downloads_norm
        )
        # Get top N indices
        top_indices = final_scores.argsort()[-N:][::-1]
        # Convert indices to Python ints
        top_indices = [int(i) for i in top_indices]
        top_datasets = [self.ds[i] for i in top_indices]
        # check if dataset has a test & train set
        has_test_set = list()
        has_train_set = list()
        ds_size_info = list()
        for i in top_indices:
            try:
                dbuilder = load_dataset_builder(self.ds[i]["id"], trust_remote_code=True).info
            except Exception as e:
                has_test_set.append(False)
                has_train_set.append(False)
                ds_size_info.append((None, None, None, None))
                continue

            if dbuilder.splits is None:
                has_test_set.append(False)
                has_train_set.append(False)
                ds_size_info.append((None, None, None, None))
                continue
            # Print number of examples for
            has_test, has_train = "test" in dbuilder.splits, "train" in dbuilder.splits
            has_test_set.append(has_test)
            has_train_set.append(has_train)
            test_dwn_size, test_elem_size = None, None
            train_dwn_size, train_elem_size = None, None
            if has_test:
                test_dwn_size = bytes2human(dbuilder.splits["test"].num_bytes)
                test_elem_size = dbuilder.splits["test"].num_examples
            if has_train:
                train_dwn_size = bytes2human(dbuilder.splits["train"].num_bytes)
                train_elem_size = dbuilder.splits["train"].num_examples
            ds_size_info.append((test_dwn_size, test_elem_size, train_dwn_size, train_elem_size))
        for _i in range(len(top_datasets)):
            top_datasets[_i]["has_test_set"] = has_test_set[_i]
            top_datasets[_i]["has_train_set"] = has_train_set[_i]
            top_datasets[_i]["test_download_size"] = ds_size_info[_i][0]
            top_datasets[_i]["test_element_size"] = ds_size_info[_i][1]
            top_datasets[_i]["train_download_size"] = ds_size_info[_i][2]
            top_datasets[_i]["train_element_size"] = ds_size_info[_i][3]
        return top_datasets

    def results_str(self, results):
        """
        Provide results as list of results in human-readable format.
        :param results: (list(dict)) list of results from search
        :return: (list(str)) list of results in human-readable format
        """
        result_strs = list()
        for result in results:
            res_str = f"Dataset ID: {result['id']}\n"
            res_str += f"Description: {result['description']}\n"
            res_str += f"Likes: {result['likes']}\n"
            res_str += f"Downloads: {result['downloads']}\n"
            res_str += f"Has Testing Set: {result['has_test_set']}\n"
            res_str += f"Has Training Set: {result['has_train_set']}\n"
            res_str += f"Test Download Size: {result['test_download_size']}\n"
            res_str += f"Test Dataset Size: {result['test_element_size']}\n"
            res_str += f"Train Download Size: {result['train_download_size']}\n"
            res_str += f"Train Dataset Size: {result['train_element_size']}\n"
            result_strs.append(res_str)
        return result_strs


class GWASCatalogSearch:
    def __init__(self):
        """
        初始化 GWAS 搜索类，按 trait 关键词获取 GWAS 研究，并进行筛选排序
        :param trait: 需要查询的疾病或性状
        :param min_year: 最小发表年份（默认 2000 年）
        :param min_sample_size: 最小样本量（默认 1000）
        """
        self.data = None
        self.mr_populations = [
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
                "multi-ancestry",
                "admixed",
                "unknown"
            ]
    # 分析单个路径
    def process_line_data(self, line_data):
        items = line_data.split("/")
        GCST_id = items[2]
        return GCST_id

    # 分析harmonised_list.txt文件
    def handle_list_data(self, path):
        GCST_id_dict = {}
        GCST_id_list = []
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                GCST_id = self.process_line_data(line.strip())  # GCST_id:'GCST90086659'
                GCST_id_dict[GCST_id] = line.strip()
                GCST_id_list.append(GCST_id)
        return GCST_id_list, GCST_id_dict

    # ★新增：从结构化字段中提取人群信息
    def extract_population_group(self, row):
        """
        从结构化字段 ancestries 中提取祖源人群信息（优先使用结构化字段）
        """
        try:
            ancestries = row.get("ancestries", [])
            for entry in ancestries:
                groups = entry.get("ancestralGroups", [])
                for group in groups:
                    pop = group.get("ancestralGroup", "")
                    if pop:
                        return str(pop)
        except Exception as e:
            print(f"种群提取失败：{e}")
        return "Unknown"

    def extract_numbers(self, text):
        """从字符串中提取所有整数（处理带逗号的情况）"""
        return [int(n.replace(",", "")) for n in re.findall(r'\d[\d,]*', text)]

    # 提取技术平台信息为字符串
    def _extract_platform_str(self, item):
        if isinstance(item, dict):
            return " ".join(str(v).lower() for v in item.values() if isinstance(v, str))
        return str(item).lower()
        return ""

    def score_study(self, row, weights=None):
        """
        对单条 GWAS Catalog 数据记录进行评分。

        参数:
            row: dict，对应一条 GWAS 研究记录
            weights: dict，可选，自定义权重配置（默认使用推荐权重）

        返回:
            score: float，总评分
            components: dict，各项评分明细，便于调试和可解释性
        """

        default_weights = {
            "sample_size": 0.25,
            "publication_year": 0.15,
            "full_pvalue": 0.10,
            "imputed": 0.05,
            "ukb": 0.10,
            "snp_count": 0.15,
            "platform_consistency": 0.10,
            "single_cohort": 0.10
        }
        if weights is None:
            weights = default_weights

        components = {}

        # 1.样本量评分（归一化，示例上限100万）
        try:
            initial = row.get("initialSampleSize", "")
            replication = row.get("replicationSampleSize", "")
            numbers = self.extract_numbers(initial) + self.extract_numbers(replication)
            sample_size = sum(numbers)
        except Exception as e:
            print(f"Error parsing sample size: {e}")
            sample_size = 0

        components["sample_size"] = min(sample_size / 1_000_000, 1.0)

        # 2.发表年份评分（线性递减，近五年满分）
        pub_date_str = row.get("publicationInfo.publicationDate", "")
        try:
            pub_year = int(pub_date_str[:4])
            current_year = datetime.datetime.now().year
            components["publication_year"] = max(0, min((pub_year - (current_year - 5)) / 5, 1.0))
        except:
            components["publication_year"] = 0

        # 3.是否提供全 P 值集
        components["full_pvalue"] = 1.0 if row.get("fullPvalueSet", False) else 0.0

        # 4.是否使用 imputed 样本
        components["imputed"] = 1.0 if row.get("imputed", False) else 0.0

        # 5.是否 UKB 队列
        cohort = row.get("cohort", [])
        # print(cohort, type(cohort))

        ukb_hit = False
        if isinstance(cohort, (list, tuple)):
            ukb_hit = any(isinstance(c, str) and ("ukb" in c.lower() or "uk biobank" in c.lower()) for c in cohort)
        elif isinstance(cohort, str):
            ukb_hit = "ukb" in cohort.lower() or "uk biobank" in cohort.lower()

        components["ukb"] = 1.0 if ukb_hit else 0.0

        # 6.SNP 数评分（归一化，示例上限 10M）
        snp_count = row.get("snpCount", 0)
        components["snp_count"] = min(snp_count / 10_000_000, 1.0)

        # 7.平台一致性评分
        platforms = row.get("platforms", [])
        techs = row.get("genotypingTechnologies", [])
        tech_all = platforms + techs

        tech_all_str = [self._extract_platform_str(p) for p in tech_all]

        if any("axiom" in p or "ukb" in p for p in tech_all_str):
            components["platform_consistency"] = 1.0
        elif all("illumina" in p for p in tech_all_str) and len(tech_all_str) > 0:
            components["platform_consistency"] = 1.0
        elif len(set(tech_all_str)) == 0:
            components["platform_consistency"] = 0.5
        else:
            components["platform_consistency"] = 0.3

        # 8.是否单队列
        # components["single_cohort"] = 1.0 if len(set(cohort)) <= 1 else 0.0
        # 8.是否单队列
        if isinstance(cohort, (list, tuple)):
            components["single_cohort"] = 1.0 if len(set(cohort)) <= 1 else 0.0
        elif isinstance(cohort, str):
            components["single_cohort"] = 1.0  # 单字符串也认为是单队列
        else:
            components["single_cohort"] = 0.0  # 不合法类型，默认给 0

        # 9.排除 replication-only 的研究（score 设置为 0）
        ancestries = row.get("ancestries", [])
        if ancestries and all(a.get("type", "").lower() == "replication" for a in ancestries):
            return 0.0, {"reason": "replication_only", **components}
        # 10.排除性状对不上的研究
        # trait = row.get("diseaseTrait.trait", "")

        # 计算加权总分  使用传入或默认的 weights 字典按项加权求和：
        score = sum(components[k] * weights.get(k, 0.0) for k in components)
        return score, components

    def _filter_and_sort_data(self, trait, data_type ,required_population="european", N=3):
        """
        过滤符合条件的数据，并按 年份 & 样本量 排序，同时筛选指定人群种群的数据
        :param trait: 需要查询的性状
        :param N: 返回的研究数量
        :param required_population: 可选，指定需要的人群种群（如 "European"）
        :return: 处理后的研究数据列表（N 个数据集）
        """

        # 获取 GWAS 研究数据
        studies = get_studies(efo_trait=trait)  # trait 可能是“乳腺癌”、“体重”或“血压”等，这些都是生物学上被认为有研究意义的表型。
        self.raw_data = studies.studies  # 原始数据
        # print(self.raw_data)
        if self.raw_data.empty:
            print("未找到相关 GWAS 研究")
            self.data = []
            return []

        items = list(self.raw_data["accessionId"])  # 获取与trait相关联的所有研究
        GCST_id_list, GCST_id_dict = self.handle_list_data("harmonised_list.txt")
        usable_GCST_id_list = []  # 可使用的研究id 有数据
        self.usable_GCST_id_dict = {}  # "id" : "下载链接"
        return_data = pd.DataFrame(columns=self.raw_data.columns)  # 能获取到
        for i in items:
            if i in GCST_id_dict:
                usable_GCST_id_list.append(i)
                self.usable_GCST_id_dict[i] = GCST_id_dict[i]
                item = self.raw_data.loc[self.raw_data['accessionId'] == i]
                return_data = pd.concat([return_data, item], ignore_index=True)

        filtered_data = []
        year_list = []
        sample_size_list = []
        rows_to_add = []
        rows_to_drop = []
        for idx, row in return_data.iterrows():
            # 解析样本量
            sample_size_str = str(row["initialSampleSize"])
            numbers = re.findall(r'\d{1,3}(?:,\d{3})*', sample_size_str)
            numbers_int = [int(num.replace(',', '')) for num in numbers]
            total_sample_size = sum(numbers_int) if numbers_int else 0
            sample_size_list.append(total_sample_size)
            # 解析发表年份
            date_format = "%Y-%m-%d"
            pub_year = str(row["publicationInfo.publicationDate"])
            pub_year = datetime.strptime(pub_year, date_format).year
            year_list.append(pub_year)
            # 种群匹配
            population_text = str(row["initialSampleSize"])
            # 匹配 ancestry 前面的族群名称，转小写
            # 构造正则 pattern，注意按长度降序排序，避免部分匹配冲突
            mr_populations_sorted = sorted(self.mr_populations, key=len, reverse=True)
            pattern = r"\b(" + "|".join(re.escape(p) for p in mr_populations_sorted) + r")\b"
            population_list = re.findall(pattern, population_text.lower())
            population_list = list(dict.fromkeys(population_list))  # 去重，保持顺序

            # openai_api_key = os.getenv('OPENAI_API_KEY')
            # population_list = self.population_identifier(population_text, "gpt-4o", openai_api_key=openai_api_key)
            #matches = re.findall(r'([A-Za-z ]+) ancestry', population_)
            # 转小写、去掉首尾空格、去重并保持顺序
            #unique_ethnicities = list(dict.fromkeys([m.strip().lower() for m in matches]))
            if data_type == "exposure":
                if len(population_list) == 1 and population_list[0] == required_population.lower():
                    if "case" in population_text and "control" in population_text:
                        pass
                    else:
                        rows_to_add.append(idx)
            else:
                if len(population_list) == 1 and population_list[0] == required_population.lower():
                    if "case" in population_text and "control" in population_text:
                        rows_to_add.append(idx)


        # 提取 rows_to_add 对应的行
        return_data = return_data.loc[rows_to_add].reset_index(drop=True)

        # 对年份和样本数量的重要关系进行综合考量 通过前面的系数 0.5+0.5=1
        # score_sample_list = [int(0.5 * 10 * year_list[i] + 0.5 * sample_size_list[i]) for i in range(len(year_list))]

        # 进行评分
        score_sample_list = []  # score_sample_list = [0.253674815,0.22352675,0.22352675,0.22354925,0.349255215,0.528302]
        for idx, row in return_data.iterrows():
            score, details = self.score_study(row.to_dict())
            score_sample_list.append(score)
        index_score_sample_list = [index for index, _ in sorted(enumerate(score_sample_list), key=lambda x: x[1])][
                                  ::-1]  # [-N:][::-1]
        # print(score_sample_list)
        # print(index_score_sample_list)
        if not index_score_sample_list:
            print("筛选后无有效数据")
            return []

        sort_data = return_data.iloc[index_score_sample_list]
        self.return_sample_sum = np.array(sample_size_list)[index_score_sample_list]
        return sort_data  # 整个数据行的描述

    # 种群识别模型（带混合人群判断）
    def population_identifier(self,sample_text, POP_LLM, openai_api_key=None):
        identify_sys = (
            "You are an automated population extraction agent in a Mendelian Randomization (MR) system.\n"
            "You are given a free-text sample description or population metadata.\n"
            "Your task is to identify which population/ethnic groups are mentioned.\n\n"
            "Output format:\n"
            "- Return a comma-separated list of populations (e.g., european, east asian, african american, hispanic).\n"
            "- All lowercase.\n"
            "- Use concise standardized labels (e.g., 'european', not 'british ancestry'; 'african american', not 'african american male').\n"
            "- If multiple populations exist, list them all separated by commas.\n"
            "- If you cannot determine, return 'unknown'.\n"
        )

        model_resp = query_model(
            openai_api_key=openai_api_key,
            model_str=f"{POP_LLM}",
            system_prompt=identify_sys,
            prompt=f"Here is the sample description:\n\n{sample_text}",
            temp=0.0
        )

        # 尝试解析输出，转成标准形式
        try:
            populations = [p.strip().lower() for p in model_resp.split(",") if p.strip()]
            populations = list(dict.fromkeys(populations))  # 去重并保持顺序
        except Exception:
            populations = ["unknown"]

        return populations

    def results_str(self, results,population):
        """
        将搜索结果格式化为可读字符串
        :param results: GWAS 研究结果列表
        :return: 可读格式的字符串列表
        """
        # results=self.data
        result_strs = []
        index = 0
        down_list = []
        if results is not None:
            for _, res in results.iterrows():
                # for res in results:
                # print(res['accessionId'])
                download_temp = self.usable_GCST_id_dict[res['accessionId']]
                download_path = "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics" + \
                                download_temp.split(".")[
                                    1] + ".h.tsv.gz"
                # print(download_path)
                res_str = (
                    f"Study ID: {res['accessionId']}\n"
                    f"Paper title: {res['publicationInfo.title']}\n"
                    f"Disease/Trait: {res['diseaseTrait.trait']}\n"
                    f"Population: {population}\n"
                    f"Publication Day: {res['publicationInfo.publicationDate']}\n"
                    # f"Sample Size: {self.return_sample_sum[index]}\n"
                    f"Downloads Path: {download_path}\n"
                    f"Sample description: {res['initialSampleSize']}\n"
                    f"PMID: {res['publicationInfo.pubmedId']}\n"
                    f"Journal: {res['publicationInfo.publication']}\n"
                    "-------------------------"
                )
                down_list.append(download_path)
                result_strs.append(res_str)
                index += 1
            return result_strs, down_list


class SemanticScholarSearch:
    def __init__(self):
        self.sch_engine = SemanticScholar(retry=False)

    def find_papers_by_str(self, query, N=10):
        paper_sums = list()
        results = self.sch_engine.search_paper(query, limit=N, min_citation_count=3, open_access_pdf=True)
        for _i in range(len(results)):
            paper_sum = f'Title: {results[_i].title}\n'
            paper_sum += f'Abstract: {results[_i].abstract}\n'
            paper_sum += f'Citations: {results[_i].citationCount}\n'
            paper_sum += f'Release Date: year {results[_i].publicationDate.year}, month {results[_i].publicationDate.month}, day {results[_i].publicationDate.day}\n'
            paper_sum += f'Venue: {results[_i].venue}\n'
            paper_sum += f'Paper ID: {results[_i].externalIds["DOI"]}\n'
            paper_sums.append(paper_sum)
        return paper_sums

    def retrieve_full_paper_text(self, query):
        pass


class TraitMatcher:
    """
    TraitMatcher 类根据用户查询短语，在 trait_list.txt 文件中查找最相关的 trait。
    匹配策略结合关键词子串匹配与 TF-IDF + 余弦相似度计算，提高匹配的准确性和鲁棒性。
    """

    def __init__(self, file_path="trait_list.txt", top_k=5):

        self.file_path = file_path  # 保存 trait 文件路径
        self.top_k = top_k  # 设置返回的候选个数
        self.trait_list = self._load_traits()  # 加载并缓存所有 trait 内容

    def _load_traits(self):
        """
        从文件中按行读取 trait 列表，每行一个 trait。
        返回:
            一个字符串列表，每项为一个 trait
        """
        with open(self.file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]  # 去除空行和首尾空格

    def _generate_word_combinations(self, phrase, min_words=2):
        """
        将多词查询短语生成长度大于等于 min_words 的所有组合（不考虑顺序）。
        用于增强匹配鲁棒性。
        参数:
            phrase: 原始短语（如 "blood pressure high"）
            min_words: 最小组合词数（默认 2）
        返回:
            所有可能组合的子短语列表
        """
        words = phrase.split()  # 将短语按空格分词
        combinations = []
        for r in range(min_words, len(words) + 1):  # 遍历组合长度从 min_words 到总词数
            for combo in itertools.combinations(words, r):  # 获取所有组合
                combinations.append(" ".join(combo))  # 组合成短语加入列表
        return combinations

    def _get_top_k_similar(self, query, candidates):
        """
        使用 TF-IDF + 余弦相似度对候选 trait 与 query 进行匹配，选出最相关的 top_k 条。
        参数:
            query: 用户查询字符串
            candidates: 候选 trait 列表
        返回:
            相似度最高的 top_k 个 trait 项
        """
        vectorizer = TfidfVectorizer().fit([query] + candidates)  # 拟合查询和候选 trait 构建词向量空间
        vectors = vectorizer.transform([query] + candidates)  # 转换为 TF-IDF 向量
        cosine_sim = cosine_similarity(vectors[0:1], vectors[1:]).flatten()  # 计算 query 与所有候选项的相似度
        top_indices = cosine_sim.argsort()[::-1][:self.top_k]  # 获取相似度排序前 top_k 的索引
        return [candidates[i] for i in top_indices]  # 返回对应的 trait 项

    def find_similar_traits(self, query):
        """
        主接口函数，根据查询字符串返回最相似的 trait 列表。
        针对单词与短语采用不同策略，提升匹配精度。
        参数:
            query: 用户输入查询短语（如 "blood glucose"）
        返回:
            最相似的 trait 列表（长度不超过 top_k）
        """
        if not self.trait_list:
            return []  # 若 trait 列表为空，则返回空列表

        filtered_traits_list = []  # 存储通过关键词匹配获得的候选 trait
        all_traits_list = []  # 存储无法关键词命中的情况下，通过语义匹配获得的候选 trait

        query_lower = query.lower()  # 转小写统一格式，增强匹配鲁棒性

        if len(query_lower.split()) == 1:
            # 查询为单个单词：直接对子串进行过滤匹配
            filtered_traits = [
                trait for trait in self.trait_list if query_lower in trait.lower()
            ]
            if filtered_traits:
                # 若关键词命中，则在匹配集中进一步排序
                filtered_traits_list = self._get_top_k_similar(query, filtered_traits)
            else:
                # 若无匹配项，则在全部 trait 中做相似度计算
                all_traits_list = self._get_top_k_similar(query, self.trait_list)

        else:
            # 多词短语：生成所有两个及以上词的组合进行关键词匹配
            query_combos = self._generate_word_combinations(query_lower)
            for keywords in query_combos:
                # 每个组合词作为关键词尝试匹配
                filtered_traits = [
                    trait for trait in self.trait_list if keywords in trait.lower()
                ]
                if filtered_traits:
                    # 若关键词匹配成功，则计算相似度并添加结果
                    filtered_traits_list += self._get_top_k_similar(query, filtered_traits)
                else:
                    # 若无匹配，则回退到在全体 trait 中做语义匹配
                    all_traits_list += self._get_top_k_similar(query, self.trait_list)

        # 优先返回通过关键词筛选的结果；若为空，则返回全局语义匹配结果
        return filtered_traits_list if filtered_traits_list else all_traits_list


# 定义一个类，用于下载并加载 GWAS 数据集
class GWASLoaderTool():
    def __init__(self):
        # 原来路径
        # self.data_dir = "./research_results/raw_gwas_data"

        # 验证任务完成率时的路径
        self.data_dir = "../research_results/raw_gwas_data"

    def get_session_with_retries(self, retries=5, backoff_factor=0.5, status_forcelist=(500, 502, 503, 504)):
        session = requests.Session()
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def get_remote_file_size(self, url):
        try:
            session = self.get_session_with_retries()
            response = session.head(url, timeout=10)
            if response.status_code == 200 and 'Content-Length' in response.headers:
                return int(response.headers['Content-Length'])
        except Exception as e:
            print(f"[GWAS Tool] Failed to fetch remote file size: {e}")
        return None

    def check_file_exists(self, url):
        file_name = url.split("/")[-1]
        os.makedirs(self.data_dir, exist_ok=True)
        file_path = os.path.join(self.data_dir, file_name)

        if os.path.exists(file_path):
            local_size = os.path.getsize(file_path)
            remote_size = self.get_remote_file_size(url)

            if remote_size is not None and local_size < remote_size:
                print(f"[GWAS Tool] File is incomplete (local: {local_size} < remote: {remote_size}). Deleting...")
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"[GWAS Tool] Failed to delete file: {e}")
                return False, file_path
            elif remote_size is not None and local_size == remote_size:
                print(f"[GWAS Tool] File exists and is complete.")
                return True, file_path
            else:
                print(f"[GWAS Tool] Remote file size unknown. Assuming local file is valid.")
                return True, file_path
        else:
            return False, file_path

    def download_file(self, url, file_path, chunk_size=1024):
        print(f"[GWAS Tool] Downloading from {url}...")

        session = self.get_session_with_retries()

        try:
            with session.get(url, stream=True, timeout=10) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('Content-Length', 0))
                with open(file_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))

            print(f"[GWAS Tool] Download completed: {file_path}")

        except Exception as e:
            print(f"[GWAS Tool] Download failed: {e}")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"[GWAS Tool] Deleted incomplete file: {file_path}")
                except Exception as delete_error:
                    print(f"[GWAS Tool] Failed to delete incomplete file: {delete_error}")
            raise  # 可选：向上传递异常供上层处理

    def run(self, url):
        exists, file_path = self.check_file_exists(url)
        if not exists:
            self.download_file(url, file_path)
        return file_path


# 定义一个 LD 剪枝的类，用于处理 GWAS 文件并使用 PLINK 工具进行 LD 剪枝

class LDPrunerAgent:
    """
    该类用于从GWAS数据中提取SNP列表，基于参考基因组利用PLINK软件进行连锁不平衡（LD）剪枝，
    并输出剪枝后的GWAS文件。整个过程包含文件校验、日志记录和状态管理。
    """

    def __init__(self, gwas_file, output_dir="./", r2=None, window_kb=None):
        self.plink_path = "plink"  # plink可执行文件路径
        self.gwas_file = gwas_file  # GWAS 输入文件

        # 原来路径
        # self.ref_panel_prefix = "./data/1000G_EUR/1000G_phase3_common_norel"  # 参考基因组文件前缀

        # 验证任务完成率时的路径
        self.ref_panel_prefix = "../data/1000G_EUR/1000G_phase3_common_norel"  # 参考基因组文件前缀

        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)
        self.output_prefix = os.path.join(self.output_dir, "ld_pruned")
        self.log_file = os.path.join(self.output_dir, "ld_pruner.log")
        self.status = {}  # 用于存储运行状态信息
        self.r2 = r2
        self.window_kb = window_kb
        self._check_files()

    def _check_files(self):
        # 检查必要文件是否存在
        missing = []
        if not os.path.exists(self.gwas_file):
            missing.append(self.gwas_file)
        for ext in [".bed", ".bim", ".fam"]:
            path = self.ref_panel_prefix + ext
            if not os.path.exists(path):
                missing.append(path)
        if missing:
            raise FileNotFoundError("以下文件未找到:\n" + "\n".join(missing))

    def _log(self, message):
        # 写日志并打印
        with open(self.log_file, "a") as f:
            f.write(message + "\n")
        # print(message)

    def _read_gwas(self):
        # 读取GWAS文件，支持gzip压缩
        if self.gwas_file.endswith(".gz"):
            df = pd.read_csv(self.gwas_file, sep="\t", compression="gzip", low_memory=False)
        else:
            df = pd.read_csv(self.gwas_file, sep="\t", low_memory=False)
        return df

    def _detect_snp_column(self, df):
        # 自动检测SNP列
        snp_column_candidates = ["SNP", "rsid", "variant_id", "hm_rsid"]
        for col in snp_column_candidates:
            if col in df.columns:
                return col
        raise ValueError("未找到可识别的 SNP ID 列，尝试了：" + ", ".join(snp_column_candidates))

    def _suggest_initial_params(self, snp_count):
        # 根据SNP数自动推测LD剪枝参数
        if snp_count > 10000:
            r2 = 0.02
            window_kb = 1000
        elif snp_count > 5000:
            r2 = 0.05
            window_kb = 800
        elif snp_count > 2000:
            r2 = 0.1
            window_kb = 600
        else:
            r2 = 0.2
            window_kb = 500
        self._log(f"    自动推测初始参数：r2={r2}, window_kb={window_kb}")
        return r2, window_kb

    def run_plink_prune(self, snplist_file, window_kb, step, r2):
        # 调用plink进行LD剪枝
        self._log(f"[2] 使用PLINK进行LD剪枝，参数：window_kb={window_kb}, step={step}, r2={r2}")
        cmd = [
            self.plink_path,
            "--bfile", self.ref_panel_prefix,
            "--extract", snplist_file,
            "--indep-pairwise", str(window_kb), str(step), str(r2),
            "--out", self.output_prefix
        ]
        subprocess.run(cmd, check=True)
        return self.output_prefix + ".prune.in"

    # def extract_pruned_gwas(self, pruned_snplist_file):
    #     # 根据plink输出的剪枝结果提取GWAS中对应的SNP记录
    #     self._log("[3] 提取剪枝后的GWAS SNPs")
    #     pruned_snps = set(line.strip() for line in open(pruned_snplist_file))
    #     df = getattr(self, "_gwas_df", self._read_gwas())
    #     pruned_df = df[df[self.snp_column].isin(pruned_snps)]
    #     # 仅保留hm_rsid和source_file列
    #     output_df = pruned_df[[self.snp_column, "source_file"]].drop_duplicates()
    #     output_file = os.path.join(self.output_dir, "gwas_pruned.tsv")
    #     output_df.to_csv(output_file, sep="\t", index=False)
    #     self._log(f"    输出 gwas_pruned.tsv，共 {len(output_df)} 个 SNP")
    #     return output_file, len(output_df)

    # 添加逻辑，如果snps数量小于10
    def run(self, min_snps_threshold=10, max_snps_threshold=100, max_attempts=10, step=10):
        # 整合流程：读取GWAS数据 -> 自动推参数 -> plink剪枝 -> 结果提取
        self._log("[1] 读取GWAS文件并提取所有SNP")
        df = self._read_gwas()
        self.snp_column = self._detect_snp_column(df)
        snps = df[self.snp_column].dropna().unique()
        snp_total = len(snps)
        self._log(f"总SNP数量: {snp_total}")

        # 如果 SNP 总数小于最小阈值，则不进行剪枝，直接返回原始 SNP 列表
        if snp_total < min_snps_threshold:
            self._log(f"SNP数量 ({snp_total}) 少于最小阈值 ({min_snps_threshold})，跳过剪枝")
            self.status.update({
                "completed": True,
                "output_file": snps,
                "snp_count": snp_total,
                "r2": None,
                "window_kb": None,
                "warning": "SNP数量过少，未进行剪枝"
            })
            return snps

        # 确定剪枝参数
        if self.r2 is not None and self.window_kb is not None:
            r2, window_kb = self.r2, self.window_kb
            self._log(f"使用用户指定参数：r2={r2}, window_kb={window_kb}")
        else:
            r2, window_kb = self._suggest_initial_params(snp_total)

        for attempt in range(max_attempts):
            try:
                pruned_snps_file = self.run_plink_prune(self.gwas_file, window_kb, step, r2)
                pruned_snps_file = pd.read_csv(pruned_snps_file, header=None)  # 这里是SNP列表，无表头
                snp_count = len(pruned_snps_file)
                print(f"剪枝后SNP数量: {snp_count}")

                # 检查 SNP 数是否在 [min_snps_threshold, max_snps_threshold] 范围内
                if min_snps_threshold <= snp_count <= max_snps_threshold:
                    self.status.update({
                        "completed": True,
                        "output_file": pruned_snps_file,
                        "snp_count": snp_count,
                        "r2": r2,
                        "window_kb": window_kb,
                        "warning": ""
                    })
                    return pruned_snps_file
                else:
                    if snp_count < min_snps_threshold:
                        self._log(f"    [尝试 {attempt + 1}] SNP数量不足（{snp_count}），放宽参数重试...")
                        r2 += 0.002  # 放宽 r2 阈值
                        window_kb = max(100, window_kb - 100)  # 减小窗口
                    elif snp_count > max_snps_threshold:
                        self._log(f"    [尝试 {attempt + 1}] SNP数量过多（{snp_count}），收紧参数重试...")
                        r2 = max(0.001, r2 - 0.002)  # 收紧 r2 阈值
                        window_kb += 200  # 扩大窗口范围
            except Exception as e:
                self._log(f"    发生错误：{e}")
                break

        self.status.update({
            "completed": False,
            "output_file": "",
            "warning": f"参数多次调整后仍未达到 {min_snps_threshold}-{max_snps_threshold} SNP 的要求"
        })
        return ""

    # 添加逻辑：10<snps<100
    # def run(self, min_snps_threshold=10, max_snps_threshold=100, max_attempts=10, step=10):
    #     # 整合流程：读取GWAS数据 -> 自动推参数 -> plink剪枝 -> 结果提取
    #     self._log("[1] 读取GWAS文件并提取所有SNP")
    #     df = self._read_gwas()
    #     self.snp_column = self._detect_snp_column(df)
    #     snps = df[self.snp_column].dropna().unique()
    #     snp_total = len(snps)
    #
    #     if self.r2 is not None and self.window_kb is not None:
    #         r2, window_kb = self.r2, self.window_kb
    #         self._log(f"使用用户指定参数：r2={r2}, window_kb={window_kb}")
    #     else:
    #         r2, window_kb = self._suggest_initial_params(snp_total)
    #
    #     for attempt in range(max_attempts):
    #         try:
    #             pruned_snps_file = self.run_plink_prune(self.gwas_file, window_kb, step, r2)
    #             pruned_snps_file = pd.read_csv(pruned_snps_file, header=None)  # 这里是SNP列表，无表头
    #             snp_count = len(pruned_snps_file)
    #             print(f"剪枝后snp数量为:{snp_count}")
    #
    #             # 检查 SNP 数是否在 [min_snps_threshold, max_snps_threshold] 范围内
    #             if min_snps_threshold <= snp_count <= max_snps_threshold:
    #                 self.status.update({
    #                     "completed": True,
    #                     "output_file": pruned_snps_file,
    #                     "snp_count": snp_count,
    #                     "r2": r2,
    #                     "window_kb": window_kb,
    #                     "warning": ""
    #                 })
    #                 return pruned_snps_file
    #             else:
    #                 if snp_count < min_snps_threshold:
    #                     self._log(f"    [尝试 {attempt + 1}] SNP数量不足（{snp_count}），放宽参数重试...")
    #                     r2 += 0.002  # 放宽 r2 阈值
    #                     window_kb = max(100, window_kb - 100)  # 减小窗口
    #                 elif snp_count > max_snps_threshold:
    #                     self._log(f"    [尝试 {attempt + 1}] SNP数量过多（{snp_count}），收紧参数重试...")
    #                     r2 = max(0.001, r2 - 0.002)  # 收紧 r2 阈值
    #                     window_kb += 200  # 扩大窗口范围
    #         except Exception as e:
    #             self._log(f"    发生错误：{e}")
    #             break
    #
    #     self.status.update({
    #         "completed": False,
    #         "output_file": "",
    #         "warning": f"参数多次调整后仍未达到 {min_snps_threshold}-{max_snps_threshold} SNP 的要求"
    #     })
    #     return ""

    # def run(self, min_snps_threshold=10, max_attempts=10, step=10):
    #     # 整合流程：读取GWAS数据 -> 自动推参数 -> plink剪枝 -> 结果提取
    #     self._log("[1] 读取GWAS文件并提取所有SNP")
    #     df = self._read_gwas()
    #     self.snp_column = self._detect_snp_column(df)
    #     snps = df[self.snp_column].dropna().unique()
    #     snp_total = len(snps)
    #     # 如果存在 source_file 列，统一替换反斜杠为正斜杠
    #     # if "source_file" in df.columns:
    #     #     df["source_file"] = df["source_file"].astype(str).str.replace("\\", "/", regex=False)
    #     # else:
    #     #     df["source_file"] = ""
    #
    #     # 构建包含 SNP ID 和 source_file 的 DataFrame
    #     # snps_df = df[[self.snp_column, "source_file"]].dropna().drop_duplicates()
    #
    #     # 写入 snplist.txt，使用制表符分隔，且不包含表头
    #     # snplist_file = os.path.join(self.output_dir, "snplist.txt")
    #     # snps_df[[self.snp_column]].to_csv(snplist_file, sep="\t", index=False, header=False)
    #
    #     # self._log(f"    提取 {len(snps_df)} 个 SNP 写入 snplist.txt （SNP列：{self.snp_column}）")
    #
    #     # self._gwas_df = df
    #     # snp_total = len(snps)
    #
    #     if self.r2 is not None and self.window_kb is not None:
    #         r2, window_kb = self.r2, self.window_kb
    #         self._log(f"使用用户指定参数：r2={r2}, window_kb={window_kb}")
    #     else:
    #         r2, window_kb = self._suggest_initial_params(snp_total)
    #
    #     for attempt in range(max_attempts):
    #         try:
    #             pruned_snps_file = self.run_plink_prune(self.gwas_file, window_kb, step, r2)
    #             pruned_snps_file = pd.read_csv(pruned_snps_file)
    #             snp_count = len(pruned_snps_file)
    #             print(f"剪枝后snp数量为:{snp_count}")
    #             # final_output, snp_count = self.extract_pruned_gwas(pruned_snps_file)
    #
    #             self.status.update({
    #                 "completed": snp_count >= min_snps_threshold,
    #                 "output_file": pruned_snps_file,
    #                 "snp_count": snp_count,
    #                 "r2": r2,
    #                 "window_kb": window_kb,
    #                 "warning": ""
    #             })
    #
    #             if snp_count >= min_snps_threshold:
    #                 return pruned_snps_file
    #             else:
    #                 self._log(f"    [尝试 {attempt + 1}] SNP数量不足（{snp_count}），调整参数后重试...")
    #                 r2 += 0.002  # 适当放宽r2阈值
    #                 window_kb = max(100, window_kb - 100)  # 适度减小 window，但不低于 5Mb
    #         except Exception as e:
    #             self._log(f"    发生错误：{e}")
    #             break
    #
    #     self.status.update({
    #         "completed": False,
    #         "output_file": "",
    #         "warning": "参数多次调整后仍未达到最小 SNP 数量阈值"
    #     })
    #     return ""

    def get_status(self):
        # 获取当前运行状态
        return self.status

    def command_run_auto(self, args):
        # 统一接口，接收参数并运行
        self.gwas_file = args.get("gwas_file", self.gwas_file)
        self.r2 = args.get("r2", self.r2)
        self.window_kb = args.get("window_kb", self.window_kb)
        min_snps = args.get("min_snps_threshold", 10)
        max_attempts = args.get("max_attempts", 5)
        step = args.get("step", 10)

        final_file = self.run(min_snps_threshold=min_snps, max_attempts=max_attempts, step=step)
        status = self.get_status()
        return {
            "pruned_gwas_file": final_file,
            "r2_used": status.get("r2"),
            "window_kb_used": status.get("window_kb"),
            "snp_count": status.get("snp_count"),
            "completed": status.get("completed"),
            "warning": status.get("warning", "")
        }


# 给 exposure 数据框添加 R² 和 F 统计量，并过滤掉弱工具变量
def add_r2_fstat_filter(df, N, f_threshold=10):
    """
    给 exposure 数据框添加 R² 和 F 统计量，并过滤掉弱工具变量 (F <= f_threshold)

    参数:
        df : pd.DataFrame
            必须包含以下列：
            - beta.exposure
            - se.exposure
            - eaf.exposure
        N : int
            样本量
        f_threshold : float
            F 统计量的阈值 (默认 10)

    返回:
        df_all : 带 R2 和 F_stat 的完整表格
        df_filtered : 过滤掉弱工具 (F <= f_threshold) 的子集
    """

    def calc_r2(beta, se, eaf, n):
        return (2 * eaf * (1 - eaf) * beta ** 2) / (
                2 * eaf * (1 - eaf) * beta ** 2 + 2 * eaf * (1 - eaf) * n * se ** 2
        )

    def calc_f(r2, n):
        return (r2 * (n - 2)) / (1 - r2)

    df_all = df.copy()

    df_all["R2"] = df_all.apply(
        lambda row: calc_r2(row["beta.exposure"], row["se.exposure"], row["eaf.exposure"], N),
        axis=1
    )
    df_all["F_stat"] = df_all.apply(
        lambda row: calc_f(row["R2"], N),
        axis=1
    )
    # print(df_all["F_stat"].dtype)

    # 过滤掉弱工具
    # 把 F ≤ 10 的 SNP 自动剔除
    # f_threshold = float(f_threshold)
    df_filtered = df_all[df_all["F_stat"] > f_threshold].reset_index(drop=True)

    return df_all, df_filtered


# 使用大模型判断 GWAS 数据集表型是否适合用户指定的性状，
def phenotype_matcher(user_trait, gwas_trait_description, LLM_model="gpt-4o", openai_api_key=None):
    """
    使用大模型判断 GWAS 数据集表型是否适合用户指定的性状，
    返回两个独立值：decision 和 explanation。
    """
    system_prompt = (
        "You are an expert in human genetics and Mendelian Randomization studies.\n"
        "You are given a user's target trait and a GWAS dataset trait description.\n"
        "Judge whether this GWAS dataset is suitable to study the user's target trait.\n"
        "Rules:\n"
        "1. If the GWAS dataset only covers a subset or specific subtype of the trait, respond 'no'.\n"
        "2. Otherwise, respond 'yes' if suitable.\n"
        "3. Provide a brief explanation (max 50 words).\n"
        "Output format: decision: <yes/no>, explanation: <brief explanation>"
    )

    prompt = (
        f"User target trait: {user_trait}\n"
        f"GWAS dataset trait description: {gwas_trait_description}\n"
        "Return your judgment in the specified format."
    )

    # 调用大模型
    response = query_model(
        openai_api_key=openai_api_key,
        model_str=LLM_model,
        system_prompt=system_prompt,
        prompt=prompt,
        temp=0.0
    )

    # 尝试解析为 decision 和 explanation
    try:
        parts = response.split("explanation:")
        decision = parts[0].replace("decision:", "").strip().replace(",", "").lower()
        explanation = parts[1].strip()
    except Exception:
        decision = "unknown"
        explanation = response.strip()

    return decision, explanation

class GWASMerger:
    def __init__(self,
                 snp_name_cols=('rsid', 'rsid'),
                 beta_hat_cols=('beta', 'beta'),
                 se_cols=('standard_error', 'standard_error'),
                 A1_cols=('effect_allele', 'effect_allele'),
                 A2_cols=('other_allele', 'other_allele')):
        """
        初始化 GWASMerger 实例，指定两个数据框中各类信息所对应的列名。

        参数：
        - snp_name_cols: 两个数据集中 SNP ID 的列名（X1, X2）。
        - beta_hat_cols: 两个数据集中效应值（beta hat）的列名。
        - se_cols: 两个数据集中标准误差（standard error）的列名。
        - A1_cols: 两个数据集中效应等位基因（effect allele）的列名。
        - A2_cols: 两个数据集中非效应等位基因的列名。
        """
        self.snp_name_cols = snp_name_cols
        self.beta_hat_cols = beta_hat_cols
        self.se_cols = se_cols
        self.A1_cols = A1_cols
        self.A2_cols = A2_cols

    @staticmethod
    def remove_ambiguous(df):
        """
        移除具有不确定互补碱基对（例如 A/T 或 G/C）记录的 SNP。
        这些 SNP 在不同平台上可能方向不一致，容易导致方向错误。

        参数：
        - df: 包含 A1 和 A2 的 DataFrame。

        返回值：
        - 过滤后的 DataFrame。
        """
        ambiguous = {('A', 'T'), ('T', 'A'), ('G', 'C'), ('C', 'G')}
        return df[~df[['A1', 'A2']].apply(tuple, axis=1).isin(ambiguous)]

    @staticmethod
    def align_beta(df, beta_col):
        """
        对 GWAS 数据中的效应值（beta）进行方向统一，确保 A1 为参考方向。
        如有必要，将 A1 和 A2 互换并对 beta 取负号。

        参数：
        - df: 包含 beta、A1、A2 列的 DataFrame。
        - beta_col: beta 值所在的列名。

        返回值：
        - 调整方向后的 DataFrame。
        """
        strand_flip = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}  # 碱基互补规则

        def flip_alleles(row):
            # 对于包含 T 的等位基因进行反向互补处理
            if row['A1'] == 'T' or row['A2'] == 'T':
                A1f = strand_flip.get(row['A1'], row['A1'])
                A2f = strand_flip.get(row['A2'], row['A2'])
            else:
                A1f, A2f = row['A1'], row['A2']

            # 若 A1f 不是 'A'，则翻转 beta 值方向
            if A1f == 'A':
                beta = row[beta_col]
            else:
                beta = -row[beta_col]
                A1f, A2f = A2f, A1f

            return pd.Series([beta, A1f, A2f], index=[beta_col, 'A1', 'A2'])

        df[[beta_col, 'A1', 'A2']] = df.apply(flip_alleles, axis=1)
        return df

    def merge(self, X1, X2):
        """
        主函数：标准化两个 GWAS 数据集，去除歧义 SNP，统一方向，并合并成一个用于下游分析的 DataFrame。

        参数：
        - X1, X2: 两个 GWAS summary statistics 数据框。

        返回值：
        - 合并清洗后的 DataFrame，包含可用于 CAUSE 分析的 SNP。
        """
        # 标准化两个数据集的列名
        X1 = X1.rename(columns={
            self.snp_name_cols[0]: 'snp',
            self.beta_hat_cols[0]: 'beta_hat',
            self.se_cols[0]: 'se',
            self.A1_cols[0]: 'A1',
            self.A2_cols[0]: 'A2'
        })

        X2 = X2.rename(columns={
            self.snp_name_cols[1]: 'snp',
            self.beta_hat_cols[1]: 'beta_hat',
            self.se_cols[1]: 'se',
            self.A1_cols[1]: 'A1',
            self.A2_cols[1]: 'A2'
        })

        print("Formatting X1 and X2...")

        # 去除互补模糊的 SNP
        X1 = self.remove_ambiguous(X1)
        X2 = self.remove_ambiguous(X2)

        # 方向校正
        X1 = self.align_beta(X1, 'beta_hat')
        X2 = self.align_beta(X2, 'beta_hat')

        # 合并两个数据集（以 SNP 为键）
        merged = pd.merge(X1, X2, on='snp', suffixes=('_1', '_2'))

        # 删除包含缺失值的记录
        merged = merged.dropna(subset=['beta_hat_1', 'beta_hat_2', 'se_1', 'se_2'])

        # 确保所有 beta 和 se 值都是数字并且非缺失
        merged = merged[merged[['beta_hat_1', 'beta_hat_2', 'se_1', 'se_2']].applymap(
            lambda x: pd.notnull(x) and pd.api.types.is_number(x)).all(axis=1)]

        # 只保留 se 大于 0 的记录
        merged = merged[(merged['se_1'] > 0) & (merged['se_2'] > 0)]

        # 确保 A2 一致，A1 可变
        merged = merged[merged['A2_1'] == merged['A2_2']]

        # 重命名为统一列名并筛选输出列
        merged = merged.rename(columns={
            'A1_1': 'A1',
            'A2_1': 'A2'
        })[['snp', 'beta_hat_1', 'se_1', 'beta_hat_2', 'se_2', 'A1', 'A2']]

        print(f"After merging and cleaning, {len(merged)} variants remain for MR analysis.")
        last_merge_data_path = "research_results/processed_data/1/harmonized_data.csv"
        merged.to_csv(last_merge_data_path, index=False)
        return merged, last_merge_data_path


class PubMedSearch:
    """
    PubMed文献检索工具类，基于Biopython Entrez接口实现PubMed文献检索与PMC全文下载
    """

    def __init__(self, email):
        """
        初始化Entrez客户端，需提供email
        """
        Entrez.email = email
        self.base_pmc_url = "https://www.ncbi.nlm.nih.gov/pmc/articles/"

    def _process_query(self, query: str) -> str:
        """
        处理查询字符串，保证长度适中，避免过长导致请求失败
        @param query: 原始查询字符串
        @return: 处理后的查询字符串
        """
        MAX_QUERY_LENGTH = 300
        if len(query) <= MAX_QUERY_LENGTH:
            return query

        words = query.split()
        processed_query = []
        current_length = 0
        for word in words:
            if current_length + len(word) + 1 <= MAX_QUERY_LENGTH:
                processed_query.append(word)
                current_length += len(word) + 1
            else:
                break
        return ' '.join(processed_query)

    def search_papers(self, query, max_results=20, sort_order="relevance"):
        """
        根据查询字符串在PubMed搜索文献
        @param query: 查询关键词
        @param max_results: 返回最大文献数量
        @param sort_order: 排序方式，常用'relevance'或'pubdate'
        @return: 文献列表，每个元素为字典包含title, abstract, pmid等信息
        """
        processed_query = self._process_query(query)

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                handle = Entrez.esearch(db="pubmed",
                                        term=processed_query,
                                        retmax=str(max_results),
                                        sort=sort_order,
                                        retmode="xml")
                record = Entrez.read(handle)
                handle.close()
                id_list = record.get("IdList", [])
                if not id_list:
                    return []

                # 获取详细信息
                handle = Entrez.efetch(db="pubmed",
                                       id=','.join(id_list),
                                       retmode="xml")
                records = Entrez.read(handle, validate=False)
                handle.close()

                papers = []
                for article in records.get("PubmedArticle", []):
                    try:
                        article_data = article['MedlineCitation']['Article']
                        title = article_data.get('ArticleTitle', 'NULL')
                        abstract = 'NULL'
                        if 'Abstract' in article_data and 'AbstractText' in article_data['Abstract']:
                            # AbstractText可能是列表或者字符串
                            abs_data = article_data['Abstract']['AbstractText']
                            if isinstance(abs_data, list):
                                abstract = ' '.join([str(x) for x in abs_data])
                            else:
                                abstract = str(abs_data)
                        pmid = article['MedlineCitation'].get('PMID', 'NULL')
                    except Exception:
                        title, abstract, pmid = 'NULL', 'NULL', 'NULL'

                    papers.append({
                        "title": title,
                        "abstract": abstract,
                        "pmid": str(pmid)
                    })
                time.sleep(1)  # 防止请求频率过高
                return papers

            except Exception as e:
                retry_count += 1
                time.sleep(2 ** retry_count)  # 指数回退重试

        return []

    def retrieve_full_paper_text(self, pmcid):
        if not pmcid.startswith("PMC"):
            pmcid = "PMC" + str(pmcid)

        xml_url = f"https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:{pmcid[3:]}&metadataPrefix=pmc"
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        try:
            response = requests.get(xml_url, headers=headers, timeout=15)
            if response.status_code != 200:
                return f"Failed to download XML. HTTP status code: {response.status_code}"

            soup = BeautifulSoup(response.content, 'lxml-xml')

            # 移除公式等不需要的内容
            for tag in soup.find_all(['inline-formula', 'tex-math', 'mml:math']):
                tag.decompose()

            # 查找 <body> 标签，若不存在则查找所有 <sec>
            body = soup.find("body")
            text_content = []

            def extract_from_sec(sec, depth=0):
                texts = []
                title = sec.find("title")
                if title:
                    texts.append("  " * depth + title.get_text(strip=True))
                for p in sec.find_all("p", recursive=False):
                    para_text = p.get_text(" ", strip=True)
                    if any(bad in para_text for bad in ["\\documentclass", "{-69pt}", "\\begin{", "\\end{"]):
                        continue
                    texts.append("  " * (depth + 1) + para_text)
                # 递归处理子 section
                for sub_sec in sec.find_all("sec", recursive=False):
                    texts.extend(extract_from_sec(sub_sec, depth + 1))
                return texts

            if body:
                # 有 <body> 时处理
                for p in body.find_all("p"):
                    text = p.get_text(" ", strip=True)
                    if text:
                        text_content.append(text)
            else:
                # 无 <body> 时处理 <sec>
                secs = soup.find_all("sec")
                for sec in secs:
                    if sec.find_parent("sec"):  # 只取顶级 <sec>
                        continue
                    text_content.extend(extract_from_sec(sec))

            return "\n\n".join(text_content) if text_content else "No readable text found in XML."

        except Exception as e:
            return f"Error occurred: {e}"

    def _clean_latex(self, text):
        import re
        # 清除LaTeX环境命令、数学公式等
        text = re.sub(r'\\\[.*?\\\]', '', text, flags=re.DOTALL)
        text = re.sub(r'\\begin\{.*?\}.*?\\end\{.*?\}', '', text, flags=re.DOTALL)
        text = re.sub(r'\$.*?\$', '', text)
        text = re.sub(r'\\[a-zA-Z]+\{.*?\}', '', text)
        return text.strip()

        # # 如果想下载PDF，解除注释：
        # pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
        # try:
        #     response = requests.get(pdf_url, timeout=15)
        #     if response.status_code == 200:
        #         filename = f"{pmcid}.pdf"
        #         with open(filename, "wb") as f:
        #             f.write(response.content)
        #         return filename
        #     else:
        #         return f"Failed to download PDF, HTTP status {response.status_code}"
        # except Exception as e:
        #     return f"Download error: {str(e)}"

    def get_paper_pmcid(self, pmid):
        """
        根据PMID查找对应的PMCID（如果存在）
        @param pmid: PubMed文章ID
        @return: PMC ID字符串，找不到则返回None
        """
        try:
            handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=str(pmid))
            record = Entrez.read(handle)
            handle.close()
            linksets = record[0]['LinkSetDb']
            for linkset in linksets:
                if linkset['DbTo'] == 'pmc':
                    pmcids = [link['Id'] for link in linkset['Link']]
                    if pmcids:
                        return "PMC" + pmcids[0]
            return None
        except Exception:
            return None


# Set the non-interactive backend early in the module
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def filter_significant_mutations(file_path, pvalue_threshold=5e-8, sep='\t', min_variants=10):
    """
    从 .gz 压缩文件中筛选显著变异，并根据阈值动态调整。

    参数：
    - file_path: 输入文件路径（支持 .gz），需包含 'p_value' 和 'variant_id' 两列
    - pvalue_threshold: 初始显著性阈值，默认 5e-8
    - sep: 文件分隔符，默认制表符 '\t'
    - min_variants: 最少显著位点数，默认 20

    返回：
    - significant_df: 筛选出的显著变异 DataFrame
    - used_threshold: 实际使用的 p 值阈值
    """
    try:
        df = pd.read_csv(file_path, sep=sep, compression='infer')
    except Exception as e:
        print(f"读取文件失败：{e}")
        return None, None

    required_columns = {'p_value', 'variant_id'}
    if not required_columns.issubset(df.columns):
        print(f"文件中缺少以下必要列：{required_columns - set(df.columns)}")
        return None, None

    # 定义一系列备用阈值
    thresholds = [pvalue_threshold, 1e-7, 1e-6, 1e-5]

    for thresh in thresholds:
        significant_df = df[df['p_value'] < thresh]
        len_snps = len(significant_df)
        if len_snps >= min_variants:
            print(f"使用阈值 {thresh} 共筛选出 {len_snps} 个显著变异")
            return significant_df, thresh

    # 如果所有阈值都不足，则返回最后一个阈值的结果
    print(f"即使使用最大阈值 {thresholds[-1]}，共筛选出 {len_snps} 个显著变异")
    return significant_df, thresholds[-1]


# def filter_significant_mutations(file_path, pvalue_threshold=5e-8, sep='\t'):
#     """
#     从 .gz 压缩文件中筛选显著变异。
#
#     参数：
#     - file_path: 输入文件路径（支持 .gz），需包含 'p_value' 和 'mutation_id' 两列
#     - pvalue_threshold: 显著性阈值，默认 0.05
#     - sep: 文件分隔符，默认制表符 '\t'
#
#     返回：
#     - 一个 DataFrame，仅包含显著变异记录
#     """
#     """
#     修改1：标准阈值小于5e-8显著位点（比如SNP位点数大于20个），可以直接进行后续的筛选，
#     如果阈值小于5e-8显著位点只有个位数，甚至为0，那么可以尝试1e-7、1e-6、1e-5作为阈值筛选条件
#     """
#     try:
#         df = pd.read_csv(file_path, sep=sep, compression='infer')
#     except Exception as e:
#         print(f"读取文件失败：{e}")
#         return None
#
#     required_columns = {'p_value', 'variant_id'}
#     if not required_columns.issubset(df.columns):
#         print(f"文件中缺少以下必要列：{required_columns - set(df.columns)}")
#         return None
#
#     # 筛选显著变异
#     significant_df = df[df['p_value'] < pvalue_threshold]
#     print(f"共筛选出 {len(significant_df)} 个显著变异（p < {pvalue_threshold}）")
#
#     return significant_df


# 自动识别主列名并标准化为 'rsid'
def standardize_rsid_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            if name == 'rsid':
                return df  # 已经是 rsid，无需改名
            df = df.rename(columns={name: 'rsid'})
            return df
    raise ValueError(f"未找到列名 {possible_names} 中的任意一个用于 rsid 匹配")


def worker_run_code(code_str, output_queue):
    output_capture = io.StringIO()
    sys.stdout = output_capture
    try:
        # Create a globals dictionary with __name__ set to "__main__"
        globals_dict = {"__name__": "__main__"}
        exec(code_str, globals_dict)
    except Exception as e:
        output_capture.write(f"[CODE EXECUTION ERROR]: {str(e)}\n")
        traceback.print_exc(file=output_capture)
    finally:
        sys.stdout = sys.__stdout__
    output_queue.put(output_capture.getvalue())


def extract_rsid(df, col_candidates=('rsid', 'variant_id')):
    """
    提取 rsid/variant_id 列并去重。

    返回：
    - Set[str] of rsids
    """
    for col in col_candidates:
        if col in df.columns:
            return set(df[col].dropna().astype(str).unique())
    print("[警告] 没有找到 rsid 或 variant_id 列。")
    return set()


import tempfile


def worker_run_r_code(code_str, output_queue):
    try:
        # 创建一个临时文件用于保存 R 代码，文件后缀为 .R，编码为 utf-8
        # delete=False 确保子进程运行时文件不会被提前删除
        with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(code_str)  # 将传入的 R 代码写入临时文件
            tmp_file_path = tmp_file.name  # 获取临时文件路径

        # 打印调试信息：执行状态和临时文件路径
        print("Executing R code...")
        print("Temporary R file:", tmp_file_path)

        # 设置子进程运行环境变量，确保使用 UTF-8 编码
        env = os.environ.copy()
        env["LANG"] = "en_US.UTF-8"  # 强制语言环境为 UTF-8
        # env["TF_ENABLE_ONEDNN_OPTS"] = "0"
        env["R_ENVIRON_USER"] = ""  # 清空用户 R 环境变量，避免外部干扰

        # 调用 Rscript 命令运行临时 R 脚本
        completed_process = subprocess.run(
            ['Rscript', tmp_file_path],  # 执行 Rscript 并传入脚本路径
            stdout=subprocess.PIPE,  # 捕获标准输出
            stderr=subprocess.PIPE,  # 捕获标准错误
            text=True,  # 将输出作为文本处理
            encoding="utf-8",  # 输出解码为 utf-8
            timeout=1000,  # 设置执行超时时间（秒）
            env=env,  # 使用自定义环境变量
        )
        # time.sleep(200)
        # print(output_queue.put(completed_process.stdout))

        # 删除临时文件，避免磁盘堆积
        os.remove(tmp_file_path)

        # 如果 Rscript 返回非零状态码，说明运行出错，将错误信息写入队列
        if completed_process.returncode != 0:
            output_queue.put(f"[CODE EXECUTION ERROR]\n{completed_process.stderr}")
        else:
            # 否则，将标准输出结果写入队列
            output_queue.put(completed_process.stdout)

    # 捕获超时异常，并返回超时错误提示
    except subprocess.TimeoutExpired:
        output_queue.put("[CODE EXECUTION ERROR]: Code execution exceeded the timeout limit.")
    # 捕获其他异常，并返回异常信息
    except Exception as e:
        output_queue.put(f"[CODE EXECUTION ERROR]: Unexpected error: {str(e)}")


def worker_run_r_code2(code_str, output_queue, timeout=1000):
    try:
        # 1️⃣ 创建临时 R 脚本文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(code_str)
            tmp_file_path = tmp_file.name

        print("Executing R code...")
        print("Temporary R file:", tmp_file_path)

        # 2️⃣ 设置环境变量
        env = os.environ.copy()
        env["LANG"] = "en_US.UTF-8"
        env["R_ENVIRON_USER"] = ""
        env["TF_ENABLE_ONEDNN_OPTS"] = "0"  # 禁用 TensorFlow oneDNN 优化，防止卡死

        # 3️⃣ 启动子进程
        process = subprocess.Popen(
            ['Rscript', tmp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            env=env
        )
        # ['Rscript', tmp_file_path],  # 执行 Rscript 并传入脚本路径
        # stdout = subprocess.PIPE,  # 捕获标准输出
        # stderr = subprocess.PIPE,  # 捕获标准错误
        # text = True,  # 将输出作为文本处理
        # encoding = "utf-8",  # 输出解码为 utf-8
        # timeout = 1000,  # 设置执行超时时间（秒）
        # env = env,  # 使用自定义环境变量
        # capture_output = False

        try:
            # 4️⃣ 等待输出，并设置超时
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # 超时处理：杀掉整个子进程树
            process.kill()
            stdout, stderr = process.communicate()
            output_queue.put("[CODE EXECUTION ERROR]: Code execution exceeded the timeout limit.")
            return

        # 5️⃣ 判断返回码
        if process.returncode != 0:
            output_queue.put(f"[CODE EXECUTION ERROR]\n{stderr}")
            process.kill()
        else:
            output_queue.put(stdout)
            process.kill()

    except Exception as e:
        output_queue.put(f"[CODE EXECUTION ERROR]: Unexpected error: {str(e)}")
    finally:
        # 删除临时文件
        try:
            os.remove(tmp_file_path)
        except Exception:
            pass


def execute_r_code1(code_str, timeout=1200):  # 原始代码
    # 定义允许的最大代码长度，防止输入代码过大导致问题
    MAX_LEN = 10000
    if len(code_str) > MAX_LEN:
        # 如果代码超过最大长度，直接返回错误信息
        return f"[CODE EXECUTION ERROR]: Code length exceeds max allowed length of {MAX_LEN} characters."

    # 检查是否包含禁止的命令，这里以 quit() 和 q() 为例，避免执行中断 R 进程的操作
    if "quit(" in code_str or "q(" in code_str:
        return "[CODE EXECUTION ERROR] The quit()/q() command is not allowed. Please remove it."

    # 创建进程间队列，用于获取子进程执行结果
    output_queue = multiprocessing.Queue()
    # 新建子进程执行 R 代码，目标函数是 worker_run_r_code
    proc = multiprocessing.Process(target=worker_run_r_code, args=(code_str, output_queue))
    proc.start()  # 启动子进程
    proc.join(timeout)  # 等待子进程完成，最多等待 timeout 秒

    # 如果子进程在超时时间后仍未结束，则终止子进程并返回超时错误
    if proc.is_alive():
        proc.terminate()  # 强制终止
        proc.join()  # 确保子进程资源回收
        return f"[CODE EXECUTION ERROR]: Code execution exceeded the timeout limit of {timeout} seconds."

    # 如果子进程有输出结果，则返回该结果
    if not output_queue.empty():
        return output_queue.get()
    else:
        # 如果没有任何输出，返回空字符串
        return ""


def execute_r_code(code_str, timeout=3000):
    # 限制代码长度
    MAX_LEN = 10000
    if len(code_str) > MAX_LEN:
        return f"[CODE EXECUTION ERROR]: Code length exceeds max allowed length of {MAX_LEN} characters."

    # 禁止使用退出 R 的命令
    if "quit(" in code_str or "q(" in code_str:
        return "[CODE EXECUTION ERROR]: The quit()/q() command is not allowed. Please remove it."

    # 写入临时 R 文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False, encoding='utf-8') as tmp_file:
        tmp_file.write(code_str)
        tmp_file_path = tmp_file.name

    print("Executing R code...")
    print("Temporary R file:", tmp_file_path)

    # 设置环境变量
    env = os.environ.copy()
    env["LANG"] = "en_US.UTF-8"
    env["R_ENVIRON_USER"] = ""  # 避免加载用户环境
    env["LC_ALL"] = "en_US.UTF-8"

    try:
        # 使用 Popen + communicate 避免缓冲区阻塞
        proc = subprocess.Popen(
            ['Rscript', tmp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            encoding='utf-8',
            errors="replace",
        )

        stdout, stderr = proc.communicate(timeout=timeout)

        if proc.returncode != 0:
            return f"[CODE EXECUTION ERROR]\n{stderr}"
        else:
            return stdout

    except subprocess.TimeoutExpired:
        proc.kill()
        return f"[CODE EXECUTION ERROR]: Code execution exceeded the timeout limit of {timeout} seconds."
    except Exception as e:
        return f"[CODE EXECUTION ERROR]: Unexpected error: {str(e)}"
    finally:
        # 删除临时文件
        if os.path.exists(tmp_file_path):
            for _ in range(3):
                try:
                    os.remove(tmp_file_path)
                    break
                except PermissionError:
                    time.sleep(0.2)
            # os.remove(tmp_file_path)


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
                    f"Outlined in the following text is the research plan that the algorithm engineer was tasked with building: {outlined_plan}\n\n"
                    f"The following text is the research latex that the model produced: \n{latex}\n\n"),
                temp=0.0)  # 设置温度为 0，保证模型输出稳定、确定
            # 从模型输出中提取 JSON（用于后续评分）
            review_json = extract_json_between_markers(scoring)

            # ====================== 3. 提取 JSON 中的各项分数并归一化 ======================
            overall = int(review_json["Overall"]) / 10  # 总体评分（1-10）
            soundness = int(review_json["Soundness"]) / 4  # 技术可靠性（1-4）
            confidence = int(review_json["Confidence"]) / 5  # 审稿人信心（1-5）
            contribution = int(review_json["Contribution"]) / 4  # 贡献度（1-4）
            presentation = int(review_json["Presentation"]) / 4  # 展示质量（1-4）
            clarity = int(review_json["Clarity"]) / 4  # 清晰度（1-4）
            originality = int(review_json["Originality"]) / 4  # 原创性（1-4）
            quality = int(review_json["Quality"]) / 4  # 技术质量（1-4）
            significance = int(review_json["Significance"]) / 4  # 重要性（1-4）

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
