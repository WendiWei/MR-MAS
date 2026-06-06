import json
from asyncio import timeout
from urllib.parse import urljoin
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
import re
import datetime
from pandasgwas.get_studies import get_studies
from urllib.parse import urljoin
import gzip
import tempfile

class ExperimentConfig:
    def __init__(self, file_path="experiment_config.json"):
        self.file_path = file_path
        self.config = {}
        if os.path.exists(file_path):
            self.load()

    def set(self, phase, key, value):
        if phase not in self.config:
            self.config[phase] = {}
        self.config[phase][key] = value
        self.save()

    def get(self, phase, key=None, default=None):
        if phase not in self.config:
            return default
        if key is None:
            return self.config[phase]
        return self.config[phase].get(key, default)

    def save(self):
        with open(self.file_path, "w") as f:
            json.dump(self.config, f, indent=4)

    def load(self):
        with open(self.file_path, "r") as f:
            self.config = json.load(f)



class GWASCatalogSearch:
    def __init__(self):
        self.data = None
        self.raw_data = None
        self.usable_GCST_id_dict = {}
        self.return_sample_sum = np.array([])
        self.search_debug = {}
        self._gwas_studies_download_table = None
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

        # 常用 trait 到 EFO ID 的映射。
        # 后续你可以继续往里面加，比如 sleep duration、BMI、CAD 等。
        self.trait_efo_map = {
            "smoking initiation": ["EFO_0005670"],
            "ever regular smoking": ["EFO_0005670"],
            "age at initiation of smoking": ["EFO_0021784"],
        }
    def process_line_data(self, line_data):
        line_data = str(line_data).strip()
        match = re.search(r"(GCST\d+)", line_data)
        if match:
            return match.group(1)
        return None

    def handle_list_data(self, path):
        GCST_id_dict = {}
        GCST_id_list = []
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                GCST_id = self.process_line_data(line)
                if not GCST_id:
                    continue
                GCST_id_dict[GCST_id] = line
                GCST_id_list.append(GCST_id)
        return GCST_id_list, GCST_id_dict

    def _build_gcst_range_dir(self, accession_id):
        accession_id = str(accession_id).strip()
        match = re.match(r"GCST(\d+)", accession_id)
        if not match:
            return None
        num_str = match.group(1)
        num = int(num_str)
        width = len(num_str)
        start = ((num - 1) // 1000) * 1000 + 1
        end = start + 999
        return f"GCST{str(start).zfill(width)}-GCST{str(end).zfill(width)}"
    def _is_true_value(self, value):
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        value_str = str(value).strip().lower()
        return value_str in ["true", "1", "yes", "y"]

    def _extract_hrefs_from_html(self, html):
        return re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)

    def _list_ftp_links(self, url, timeout=20):
        try:
            resp = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            if resp.status_code != 200:
                return []
            hrefs = self._extract_hrefs_from_html(resp.text)
            links = []
            for href in hrefs:
                if href in ["../", "./", "/"]:
                    continue
                full_url = urljoin(url, href)
                links.append(full_url)
            return links
        except Exception as e:
            print(f"[FTP List Warning] 无法读取 FTP 目录: {url}, error={e}")
            return []

    def _is_candidate_sumstats_file(self, url):
        lower_url = str(url).lower()
        bad_keywords = [
            "readme",
            "metadata",
            "manifest",
            "md5",
            "sha",
            "log",
            "license",
            "terms",
            ".json",
            ".yaml",
            ".yml",
            ".xml"
        ]
        if any(k in lower_url for k in bad_keywords):
            return False
        good_suffix = [
            ".tsv.gz",
            ".txt.gz",
            ".csv.gz",
            ".tsv",
            ".txt",
            ".csv",
            ".gz"
        ]
        return any(lower_url.endswith(s) for s in good_suffix)

    def _score_candidate_sumstats_url(self, url, accession_id):

        lower_url = str(url).lower()
        accession_lower = str(accession_id).lower()
        score = 0
        if accession_lower in lower_url:
            score += 5
        if "/harmonised/" in lower_url:
            score += 4
        if lower_url.endswith(".h.tsv.gz"):
            score += 3
        if lower_url.endswith(".tsv.gz"):
            score += 2
        if "munged" in lower_url or "formatted" in lower_url:
            score += 1
        return score

    def _url_exists(self, url, timeout=20):
        try:
            headers = {
                "Range": "bytes=0-1023",
                "User-Agent": "Mozilla/5.0"
            }
            resp = requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=timeout,
                allow_redirects=True
            )
            ok = resp.status_code in [200, 206]
            resp.close()
            return ok
        except Exception:
            return False

    def _find_real_summary_stats_url_from_ftp(self, accession_id):
        accession_id = str(accession_id).strip()
        range_dir = self._build_gcst_range_dir(accession_id)
        if not range_dir:
            return None
        base = "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics"
        root_url = f"{base}/{range_dir}/{accession_id}/"
        dirs_to_scan = [
            root_url,
            urljoin(root_url, "harmonised/")
        ]
        seen_dirs = set()
        candidate_files = []
        while dirs_to_scan and len(seen_dirs) < 10:
            current_dir = dirs_to_scan.pop(0)
            if current_dir in seen_dirs:
                continue
            seen_dirs.add(current_dir)
            links = self._list_ftp_links(current_dir)
            for link in links:
                lower_link = link.lower()
                if lower_link.endswith("/"):
                    if any(k in lower_link for k in ["harmonised", "formatted", "sumstat", "summary"]):
                        if link not in seen_dirs:
                            dirs_to_scan.append(link)
                    continue
                if self._is_candidate_sumstats_file(link):
                    candidate_files.append(link)
        if not candidate_files:
            return None
        candidate_files = sorted(
            candidate_files,
            key=lambda x: self._score_candidate_sumstats_url(x, accession_id),
            reverse=True
        )
        for url in candidate_files:
            if self._url_exists(url):
                return url

        return None

    def _get_download_url_for_accession(self, accession_id, GCST_id_dict=None, validate_url=True):
        accession_id = str(accession_id).strip()
        if GCST_id_dict and accession_id in GCST_id_dict:
            download_temp = GCST_id_dict[accession_id]
            candidate_urls = []
            try:
                candidate_urls.append(
                    "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics"
                    + download_temp.split(".")[1]
                    + ".h.tsv.gz"
                )
            except Exception:
                candidate_urls.append(
                    "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/"
                    + str(download_temp).lstrip("/")
                )
            for url in candidate_urls:
                if not validate_url or self._url_exists(url):
                    return url
        real_url = self._find_real_summary_stats_url_from_ftp(accession_id)
        if real_url:
            return real_url
        return None

    def _normalize_download_url(self, location):
        if location is None:
            return None
        location = str(location).strip()
        if location == "" or location.lower() in ["na", "nan", "none", "-", "not available"]:
            return None
        if location.startswith("ftp://ftp.ebi.ac.uk/"):
            location = location.replace("ftp://ftp.ebi.ac.uk/", "https://ftp.ebi.ac.uk/")
        if location.startswith("http://ftp.ebi.ac.uk/"):
            location = location.replace("http://ftp.ebi.ac.uk/", "https://ftp.ebi.ac.uk/")
        if location.startswith("https://"):
            return location
        if location.startswith("/pub/"):
            return "https://ftp.ebi.ac.uk" + location
        if location.startswith("pub/"):
            return "https://ftp.ebi.ac.uk/" + location
        if location.startswith("databases/"):
            return "https://ftp.ebi.ac.uk/pub/" + location
        if location.startswith("gwas/summary_statistics/"):
            return "https://ftp.ebi.ac.uk/pub/databases/" + location
        if location.startswith("summary_statistics/"):
            return "https://ftp.ebi.ac.uk/pub/databases/gwas/" + location
        return location

    def _load_gwas_studies_download_table(self):
        if self._gwas_studies_download_table is not None:
            return self._gwas_studies_download_table
        candidate_urls = [
            "https://www.ebi.ac.uk/gwas/api/search/downloads/studies/v1.0.3.1",
            "https://www.ebi.ac.uk/gwas/api/search/downloads/studies/v1.0.2.1",
            "https://www.ebi.ac.uk/gwas/api/search/downloads/studies",
            "https://www.ebi.ac.uk/gwas/api/search/downloads/studies_alternative",
        ]
        last_error = None
        for url in candidate_urls:
            try:
                print(f"[GWAS metadata] 正在读取 studies download 表: {url}")
                df = pd.read_csv(
                    url,
                    sep="\t",
                    dtype=str,
                    low_memory=False
                )
                df.columns = [
                    re.sub(r"\s+", " ", str(c).strip().upper())
                    for c in df.columns
                ]

                if "STUDY ACCESSION" not in df.columns:
                    print(f"[GWAS metadata] 当前表没有 STUDY ACCESSION 列，跳过: {url}")
                    continue

                if "SUMMARY STATS LOCATION" not in df.columns:
                    print(f"[GWAS metadata] 当前表没有 SUMMARY STATS LOCATION 列，跳过: {url}")
                    continue
                self._gwas_studies_download_table = df
                print(f"[GWAS metadata] studies download 表读取成功，共 {len(df)} 条记录")
                return df
            except Exception as e:
                last_error = e
                print(f"[GWAS metadata] 读取失败: {url}, error={e}")

        raise RuntimeError(f"无法读取 GWAS Catalog studies download 表: {last_error}")

    def _get_download_url_from_studies_metadata(self, accession_id):
        accession_id = str(accession_id).strip()
        try:
            df = self._load_gwas_studies_download_table()
        except Exception as e:
            print(f"[GWAS metadata] 无法加载 studies metadata: {e}")
            return None
        if "STUDY ACCESSION" not in df.columns:
            return None
        sub = df[df["STUDY ACCESSION"].astype(str).str.strip() == accession_id]
        if sub.empty:
            return None
        if "FULL SUMMARY STATISTICS" in sub.columns:
            full_mask = sub["FULL SUMMARY STATISTICS"].astype(str).str.lower().isin(
                ["true", "yes", "y", "1"]
            )
            if full_mask.any():
                sub = sub[full_mask]
        for _, row in sub.iterrows():
            location = row.get("SUMMARY STATS LOCATION", None)
            real_file_url = self._resolve_summary_stats_location(
                location=location,
                accession_id=accession_id
            )
            if real_file_url:
                return real_file_url
        return None

    def _looks_like_sumstats_file(self, url):
        lower = str(url).lower()
        bad_keywords = [
            "readme",
            "metadata",
            "manifest",
            "md5",
            "sha",
            "license",
            "log",
            ".json",
            ".yaml",
            ".yml",
            ".xml"
        ]

        if any(k in lower for k in bad_keywords):
            return False

        good_suffixes = [
            ".h.tsv.gz",
            ".tsv.gz",
            ".txt.gz",
            ".csv.gz",
            ".sumstats.gz",
            ".gz",
            ".tsv",
            ".txt",
            ".csv"
        ]

        return any(lower.endswith(s) for s in good_suffixes)

    def _extract_hrefs_from_html(self, html):
        return re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)

    def _list_ftp_dir(self, url, timeout=12):
        if not url:
            return []

        if not url.endswith("/"):
            url = url + "/"
        urls_to_try = [url]
        if url.startswith("https://ftp.ebi.ac.uk/"):
            urls_to_try.append(url.replace("https://ftp.ebi.ac.uk/", "http://ftp.ebi.ac.uk/"))

        for try_url in urls_to_try:
            try:
                resp = requests.get(
                    try_url,
                    timeout=timeout,
                    headers={
                        "User-Agent": "Mozilla/5.0",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Connection": "close"
                    }
                )
                if resp.status_code != 200:
                    continue

                hrefs = re.findall(
                    r'href=["\']([^"\']+)["\']',
                    resp.text,
                    flags=re.IGNORECASE
                )

                links = []

                for href in hrefs:
                    if href in ["../", "./", "/"]:
                        continue

                    full_url = urljoin(try_url, href)
                    if full_url.startswith("http://ftp.ebi.ac.uk/"):
                        full_url = full_url.replace("http://ftp.ebi.ac.uk/", "https://ftp.ebi.ac.uk/")

                    links.append(full_url)

                return links

            except Exception as e:
                print(f"[FTP List Warning] 读取目录失败: {try_url}, error={e}")

        return []

    def _get_study_dir_url(self, accession_id):
        accession_id = str(accession_id).strip()
        range_dir = self._build_gcst_range_dir(accession_id)
        if not range_dir:
            return None

        return (
            "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/"
            f"{range_dir}/{accession_id}/"
        )
    def _score_sumstats_file_url(self, url, accession_id):
        lower = str(url).lower()
        acc = str(accession_id).lower()
        score = 0
        if acc in lower:
            score += 10
        if "/harmonised/" in lower:
            score += 6
        if lower.endswith(".h.tsv.gz"):
            score += 5
        if lower.endswith(".tsv.gz"):
            score += 4
        if lower.endswith(".txt.gz"):
            score += 3
        if "harmon" in lower:
            score += 2
        if "buildgrch37" in lower or "grch37" in lower:
            score += 1
        return score

    def _resolve_summary_stats_location(self, location, accession_id, max_dirs=3):
        accession_id = str(accession_id).strip()
        accession_lower = accession_id.lower()
        url = self._normalize_download_url(location)

        if not url:
            return None
        if self._looks_like_sumstats_file(url):
            return url
        study_dir = self._get_study_dir_url(accession_id)
        if not study_dir:
            return None
        dirs_to_scan = []

        if accession_lower in str(url).lower():
            if not url.endswith("/"):
                url = url + "/"
            dirs_to_scan.append(url)
        else:
            print(
                f"[Resolve Guard] {accession_id} 的 SUMMARY STATS LOCATION 过宽，"
                f"改为只扫描 study 目录: {study_dir}"
            )
            dirs_to_scan.append(study_dir)

        extra_dirs = []
        for d in dirs_to_scan:
            extra_dirs.append(urljoin(d, "harmonised/"))
        dirs_to_scan = dirs_to_scan + extra_dirs
        clean_dirs = []
        for d in dirs_to_scan:
            if d not in clean_dirs:
                clean_dirs.append(d)

        clean_dirs = clean_dirs[:max_dirs]
        candidate_files = []

        for current_dir in clean_dirs:
            links = self._list_ftp_dir(current_dir)
            for link in links:
                lower_link = str(link).lower()
                if lower_link.endswith("/"):
                    continue
                if self._looks_like_sumstats_file(link):
                    candidate_files.append(link)

        if not candidate_files:
            print(
                f"[Resolve Warning] {accession_id} 的 study 目录中没有找到具体 summary statistics 文件: {study_dir}"
            )
            return None
        candidate_files = sorted(
            candidate_files,
            key=lambda x: self._score_sumstats_file_url(x, accession_id),
            reverse=True
        )
        best_url = candidate_files[0]
        print(f"[Resolve OK] {accession_id} 目录解析得到真实文件: {best_url}")

        return best_url
    def _build_usable_download_dict(
        self,
        raw_data,
        harmonised_list_path="harmonised_list.txt",
        require_full_summary_stats=True
    ):
        usable_dict = {}
        local_gcst_dict = {}

        if raw_data is None or raw_data.empty or "accessionId" not in raw_data.columns:
            return usable_dict
        if harmonised_list_path and os.path.exists(harmonised_list_path):
            try:
                _, local_gcst_dict = self.handle_list_data(harmonised_list_path)
            except Exception as e:
                print(f"读取 harmonised_list.txt 失败，将仅使用官方 studies metadata：{e}")

        for _, row in raw_data.iterrows():
            accession_id = str(row.get("accessionId", "")).strip()

            if not accession_id:
                continue
            if require_full_summary_stats and "fullPvalueSet" in row.index:
                if not self._is_true_value(row.get("fullPvalueSet")):
                    print(f"[Skip] {accession_id} fullPvalueSet=False，跳过")
                    continue

            url = self._get_download_url_from_studies_metadata(accession_id)
            if url:
                usable_dict[accession_id] = url
                print(f"[OK] {accession_id} 从 SUMMARY STATS LOCATION 获取下载地址: {url}")
                continue

            if accession_id in local_gcst_dict:
                try:
                    download_temp = local_gcst_dict[accession_id]
                    fallback_url = (
                        "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics"
                        + download_temp.split(".")[1]
                        + ".h.tsv.gz"
                    )
                    usable_dict[accession_id] = fallback_url
                    print(f"[OK] {accession_id} 从 harmonised_list.txt 获取下载地址: {fallback_url}")
                    continue
                except Exception as e:
                    print(f"[Skip] {accession_id} 本地 harmonised_list 路径解析失败: {e}")

            print(f"[Skip] {accession_id} 没有在 SUMMARY STATS LOCATION 中找到下载地址")

        return usable_dict
    def _url_exists(self, url, timeout=8):
        try:
            resp = requests.head(url, allow_redirects=True, timeout=timeout)
            if resp.status_code == 200:
                return True
            resp = requests.get(url, stream=True, timeout=timeout)
            return resp.status_code == 200

        except Exception:
            return False

    def _candidate_download_urls_from_gcst(self, accession_id):
        accession_id = str(accession_id).strip()
        range_dir = self._build_gcst_range_dir(accession_id)
        if not range_dir:
            return []
        base = "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics"
        candidate_urls = [
            f"{base}/{range_dir}/{accession_id}/harmonised/{accession_id}.h.tsv.gz",
            f"{base}/{range_dir}/{accession_id}/{accession_id}.tsv.gz",
            f"{base}/{range_dir}/{accession_id}/{accession_id}.h.tsv.gz",
            f"{base}/{range_dir}/{accession_id}/harmonised/{accession_id}.tsv.gz",
        ]
        return candidate_urls

    def _get_download_url_for_accession(self, accession_id, GCST_id_dict=None, validate_url=True):
        accession_id = str(accession_id).strip()
        if GCST_id_dict and accession_id in GCST_id_dict:
            download_temp = GCST_id_dict[accession_id]
            try:
                download_path = (
                    "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics"
                    + download_temp.split(".")[1]
                    + ".h.tsv.gz"
                )
                return download_path
            except Exception:
                return (
                    "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/"
                    + str(download_temp).lstrip("/")
                )

        candidate_urls = self._candidate_download_urls_from_gcst(accession_id)
        if not candidate_urls:
            return None
        if validate_url:
            for url in candidate_urls:
                if self._url_exists(url):
                    return url
            return None

        return candidate_urls[0]

    def extract_numbers(self, text):
        if text is None:
            return []
        return [
            int(n.replace(",", ""))
            for n in re.findall(r"\d[\d,]*", str(text))
        ]

    def _as_list(self, value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    def _extract_platform_str(self, item):
        if item is None:
            return ""
        if isinstance(item, dict):
            return " ".join(
                str(v).lower()
                for v in item.values()
                if v is not None
            )
        return str(item).lower()

    def _safe_get_studies(self, **kwargs):
        try:
            return get_studies(**kwargs, interactive=False)
        except TypeError:
            return get_studies(**kwargs)

    def _studies_to_df(self, studies):
        if studies is None:
            return pd.DataFrame()
        df = getattr(studies, "studies", None)
        if df is None:
            return pd.DataFrame()
        if isinstance(df, pd.DataFrame):
            return df.copy()
        try:
            return pd.DataFrame(df)
        except Exception:
            return pd.DataFrame()

    def _merge_study_dfs(self, dfs):
        valid_dfs = [
            df for df in dfs
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty
        ]
        if not valid_dfs:
            return pd.DataFrame()
        merged = pd.concat(valid_dfs, ignore_index=True)
        if "accessionId" in merged.columns:
            merged = merged.drop_duplicates(subset=["accessionId"], keep="first")
        else:
            merged = merged.drop_duplicates()

        return merged.reset_index(drop=True)

    def _get_studies_by_trait_fallback(self, trait):
        trait_raw = str(trait).strip()
        trait_key = trait_raw.lower()
        dfs = []
        query_logs = []

        efo_ids = self.trait_efo_map.get(trait_key, [])

        for efo_id in efo_ids:
            try:
                studies = self._safe_get_studies(efo_id=efo_id)
                df = self._studies_to_df(studies)
                dfs.append(df)

                query_logs.append({
                    "mode": "efo_id",
                    "query": efo_id,
                    "n": len(df)
                })

            except Exception as e:
                query_logs.append({
                    "mode": "efo_id",
                    "query": efo_id,
                    "error": str(e)
                })

        try:
            studies = self._safe_get_studies(efo_trait=trait_raw)
            df = self._studies_to_df(studies)
            dfs.append(df)
            query_logs.append({
                "mode": "efo_trait",
                "query": trait_raw,
                "n": len(df)
            })

        except Exception as e:
            query_logs.append({
                "mode": "efo_trait",
                "query": trait_raw,
                "error": str(e)
            })

        try:
            studies = self._safe_get_studies(reported_trait=trait_raw)
            df = self._studies_to_df(studies)
            dfs.append(df)

            query_logs.append({
                "mode": "reported_trait",
                "query": trait_raw,
                "n": len(df)
            })

        except Exception as e:
            query_logs.append({
                "mode": "reported_trait",
                "query": trait_raw,
                "error": str(e)
            })

        merged = self._merge_study_dfs(dfs)

        self.search_debug["query_logs"] = query_logs
        self.search_debug["raw_merged_n"] = len(merged)

        print("GWAS Catalog 查询日志：")
        for log in query_logs:
            print(log)

        return merged

    def extract_population_group(self, row):
        try:
            ancestries = row.get("ancestries", [])
            if isinstance(ancestries, list):
                for entry in ancestries:
                    groups = entry.get("ancestralGroups", [])
                    if isinstance(groups, list):
                        for group in groups:
                            pop = group.get("ancestralGroup", "")
                            if pop:
                                return str(pop)

        except Exception as e:
            print(f"种群提取失败：{e}")
        return "Unknown"

    def _extract_population_list(self, row):
        populations = []
        try:
            ancestries = row.get("ancestries", [])
            if isinstance(ancestries, list):
                for entry in ancestries:
                    groups = entry.get("ancestralGroups", [])
                    if isinstance(groups, list):
                        for group in groups:
                            pop = group.get("ancestralGroup", "")
                            if pop:
                                populations.append(str(pop).strip().lower())
        except Exception:
            pass

        if not populations:
            population_text = (
                str(row.get("initialSampleSize", "")) + " " +
                str(row.get("replicationSampleSize", ""))
            ).lower()
            mr_populations_sorted = sorted(
                self.mr_populations,
                key=len,
                reverse=True
            )
            pattern = (
                r"\b(" +
                "|".join(re.escape(p) for p in mr_populations_sorted) +
                r")\b"
            )
            populations = re.findall(pattern, population_text)

        norm_pops = []
        for p in populations:
            p = str(p).strip().lower()
            if p == "white british":
                p = "european"
            if p:
                norm_pops.append(p)
        norm_pops = list(dict.fromkeys(norm_pops))
        if not norm_pops:
            norm_pops = ["unknown"]
        return norm_pops

    def _population_match_level(self, population_list, required_population):
        required = str(required_population).strip().lower()
        population_list = [
            str(p).strip().lower()
            for p in population_list
            if str(p).strip()
        ]
        if not population_list:
            return 1

        text = " | ".join(population_list)
        european_aliases = [
            "european",
            "european ancestry",
            "european ancestries",
            "white british",
            "british",
            "caucasian",
            "eur",
            "uk biobank",
            "finnish",
            "icelandic",
            "swedish",
            "danish",
            "norwegian",
            "dutch",
            "german",
            "estonian"
        ]

        relaxed_aliases = [
            "unknown",
            "not reported",
            "not available",
            "multi-ancestry",
            "multi ancestry",
            "multiple ancestries",
            "mixed",
            "admixed",
            "trans-ancestry",
            "cross-ancestry"
        ]
        if required == "european":
            if any(alias in text for alias in european_aliases):
                return 2

        if required in population_list or any(required in p for p in population_list):
            return 2
        if any(alias in text for alias in relaxed_aliases):
            return 1
        return 0
    def _get_total_sample_size(self, row):
        initial = str(row.get("initialSampleSize", ""))
        replication = str(row.get("replicationSampleSize", ""))
        numbers = self.extract_numbers(initial) + self.extract_numbers(replication)
        return sum(numbers) if numbers else 0

    def _is_case_control_study(self, row):
        text = (
            str(row.get("initialSampleSize", "")) + " " +
            str(row.get("replicationSampleSize", "")) + " " +
            str(row.get("diseaseTrait.trait", ""))
        ).lower()
        has_case = "case" in text or "cases" in text
        has_control = "control" in text or "controls" in text
        return has_case and has_control

    def _trait_relevance_score(self, row, query_trait):
        query = str(query_trait).strip().lower()
        disease_trait = str(row.get("diseaseTrait.trait", "")).strip().lower()
        reported_trait = str(row.get("reportedTrait", "")).strip().lower()
        text = f"{disease_trait} {reported_trait}"
        if disease_trait == query or reported_trait == query:
            return 1.0
        if query in disease_trait or query in reported_trait:
            return 0.8
        query_tokens = [t for t in re.split(r"\s+", query) if t]
        if query_tokens and all(t in text for t in query_tokens):
            return 0.6
        if "pleiotropy" in text:
            return 0.1
        return 0.0
    def score_study(self, row, weights=None):
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
        try:
            sample_size = self._get_total_sample_size(row)
        except Exception as e:
            print(f"Error parsing sample size: {e}")
            sample_size = 0
        components["sample_size"] = min(sample_size / 1_000_000, 1.0)
        pub_date_str = str(row.get("publicationInfo.publicationDate", ""))

        try:
            pub_year = int(pub_date_str[:4])
            current_year = datetime.datetime.now().year
            components["publication_year"] = max(
                0,
                min((pub_year - (current_year - 5)) / 5, 1.0)
            )
        except Exception:
            components["publication_year"] = 0.0

        components["full_pvalue"] = 1.0 if row.get("fullPvalueSet", False) else 0.0
        components["imputed"] = 1.0 if row.get("imputed", False) else 0.0
        cohort = row.get("cohort", [])
        cohort_list = self._as_list(cohort)

        ukb_hit = any(
            isinstance(c, str) and
            ("ukb" in c.lower() or "uk biobank" in c.lower())
            for c in cohort_list
        )

        components["ukb"] = 1.0 if ukb_hit else 0.0

        try:
            snp_count = row.get("snpCount", 0)
            if pd.isna(snp_count):
                snp_count = 0
            snp_count = float(snp_count)
        except Exception:
            snp_count = 0

        components["snp_count"] = min(snp_count / 10_000_000, 1.0)

        platforms = self._as_list(row.get("platforms", []))
        techs = self._as_list(row.get("genotypingTechnologies", []))
        tech_all = platforms + techs

        tech_all_str = [
            self._extract_platform_str(p)
            for p in tech_all
            if self._extract_platform_str(p)
        ]

        if any("axiom" in p or "ukb" in p for p in tech_all_str):
            components["platform_consistency"] = 1.0
        elif len(tech_all_str) > 0 and all("illumina" in p for p in tech_all_str):
            components["platform_consistency"] = 1.0
        elif len(set(tech_all_str)) == 0:
            components["platform_consistency"] = 0.5
        else:
            components["platform_consistency"] = 0.3

        if isinstance(cohort, (list, tuple)):
            components["single_cohort"] = 1.0 if len(set(cohort)) <= 1 else 0.0
        elif isinstance(cohort, str):
            components["single_cohort"] = 1.0
        else:
            components["single_cohort"] = 0.0
        ancestries = row.get("ancestries", [])
        try:
            if (
                isinstance(ancestries, list) and
                len(ancestries) > 0 and
                all(
                    str(a.get("type", "")).lower() == "replication"
                    for a in ancestries
                    if isinstance(a, dict)
                )
            ):
                return 0.0, {"reason": "replication_only", **components}
        except Exception:
            pass
        score = sum(
            components[k] * weights.get(k, 0.0)
            for k in components
        )
        return score, components
    def _filter_and_sort_data(self,trait,data_type,required_population="european",N=10,harmonised_list_path="harmonised_list.txt",allow_binary_exposure=True, require_case_control_for_outcome=False, require_full_summary_stats=True):
        print(f"这是GWAS Catalog数据库的原始检索关键词：{trait}")
        trait = str(trait).strip()
        print(f"这是GWAS Catalog数据库的最终检索关键词：{trait.lower()}")

        self.raw_data = self._get_studies_by_trait_fallback(trait)
        if self.raw_data is None or self.raw_data.empty:
            print("未找到相关 GWAS 研究")
            self.data = []
            return []
        print(f"原始检索得到 studies 数量：{len(self.raw_data)}")

        if require_full_summary_stats and "fullPvalueSet" in self.raw_data.columns:
            before_n = len(self.raw_data)
            self.raw_data = self.raw_data[
                self.raw_data["fullPvalueSet"].apply(self._is_true_value)
            ].copy().reset_index(drop=True)
            print(f"Full summary statistics 可用 study 数量：{len(self.raw_data)} / {before_n}")
            if self.raw_data.empty:
                print("该 trait 在 GWAS Catalog 中有 study，但没有 full summary statistics 可下载数据")
                self.data = []
                return []
        if "accessionId" not in self.raw_data.columns:
            print("返回结果中没有 accessionId 字段，无法和 harmonised_list.txt 匹配")
            self.data = []
            return []

        self.usable_GCST_id_dict = self._build_usable_download_dict(
            raw_data=self.raw_data,
            harmonised_list_path=harmonised_list_path,
            require_full_summary_stats=require_full_summary_stats
        )

        if not self.usable_GCST_id_dict:
            print("GWAS Catalog 有相关 studies，但没有找到可访问的 summary statistics 下载链接")
            print(f"原始 studies 数量：{len(self.raw_data)}")
            print("建议：")
            print("1. 检查网络是否能访问 https://ftp.ebi.ac.uk")
            print("2. 检查这些 GCST 是否确实提供 summary statistics")
            print("3. 更新本地 harmonised_list.txt 或关闭 URL 验证")
            self.data = []
            return []

        matched_accessions = set(self.usable_GCST_id_dict.keys())
        return_data = self.raw_data[
            self.raw_data["accessionId"].isin(matched_accessions)
        ].copy().reset_index(drop=True)

        if return_data.empty:
            print("有下载链接，但 raw_data 中没有匹配到 accessionId")
            self.data = []
            return []
        print(f"匹配可下载 summary statistics 后数量：{len(return_data)}")

        if "accessionId" in return_data.columns:
            return_data = return_data.drop_duplicates(
                subset=["accessionId"],
                keep="first"
            )

        return_data = return_data.reset_index(drop=True)
        print(f"真实可下载 full summary statistics 数量：{len(return_data)}")

        strict_rows = []
        relaxed_rows = []
        fallback_rows = []
        debug_rows = []

        data_type = str(data_type).strip().lower()

        for idx, row in return_data.iterrows():
            population_list = self._extract_population_list(row)
            population_match = self._population_match_level(
                population_list,
                required_population
            )
            is_case_control = self._is_case_control_study(row)
            trait_rel = self._trait_relevance_score(row, trait)
            keep = True
            drop_reason = ""
            if trait_rel <= 0:
                keep = False
                drop_reason = "trait_not_relevant"

            if keep and data_type == "exposure":
                if is_case_control and not allow_binary_exposure:
                    keep = False
                    drop_reason = "binary_exposure_removed"

            if keep and data_type == "outcome":
                if require_case_control_for_outcome and not is_case_control:
                    keep = False
                    drop_reason = "outcome_not_case_control"

            debug_rows.append({
                "idx": idx,
                "accessionId": row.get("accessionId", ""),
                "trait": row.get("diseaseTrait.trait", ""),
                "reportedTrait": row.get("reportedTrait", ""),
                "population_list": population_list,
                "population_match": population_match,
                "is_case_control": is_case_control,
                "trait_relevance": trait_rel,
                "keep": keep,
                "drop_reason": drop_reason,
                "initialSampleSize": row.get("initialSampleSize", "")
            })

            if keep:
                if population_match == 2:
                    strict_rows.append(idx)
                elif population_match == 1:
                    relaxed_rows.append(idx)
                else:
                    fallback_rows.append(idx)


        if strict_rows:
            selected_rows = strict_rows
            print(f"严格匹配 {required_population} 人群后的数量：{len(selected_rows)}")

        elif relaxed_rows:
            selected_rows = relaxed_rows
            print(
                f"没有严格匹配 {required_population} 的数据，"
                f"使用 unknown/multi-ancestry 兜底数量：{len(selected_rows)}"
            )

        elif fallback_rows:
            selected_rows = fallback_rows
            print(
                f"没有 strict {required_population}，也没有 unknown/multi-ancestry；"
                f"使用其他 ancestry 的 trait 相关可下载数据作为兜底数量：{len(selected_rows)}"
            )

        else:
            selected_rows = []
            print("筛选后无有效数据")

        self.search_debug["filter_debug_rows"] = debug_rows

        if not selected_rows:
            print("筛选后无有效数据")
            print("可以打印 self.search_debug['filter_debug_rows'] 查看每条数据被删除的原因")
            self.data = []
            return []

        return_data = return_data.loc[selected_rows].copy().reset_index(drop=True)
        scores = []
        sample_sizes = []
        score_details = []

        for idx, row in return_data.iterrows():
            sample_size_value = self._get_total_sample_size(row)
            sample_sizes.append(sample_size_value)
            score, details = self.score_study(row.to_dict())
            trait_rel = self._trait_relevance_score(row, trait)
            details["trait_relevance"] = trait_rel
            score = score + 1.0 * trait_rel
            trait_text = (
                    str(row.get("diseaseTrait.trait", "")) + " " +
                    str(row.get("reportedTrait", ""))
            ).lower()
            if "pleiotropy" in trait_text:
                score -= 0.5
            population_list = self._extract_population_list(row)
            population_match = self._population_match_level(
                population_list,
                required_population
            )
            if population_match == 2:
                score += 0.05
            scores.append(score)
            score_details.append(details)
        return_data["__score"] = scores
        return_data["__sample_size"] = sample_sizes
        return_data["__score_details"] = score_details
        return_data = return_data.sort_values(
            by=["__score", "__sample_size"],
            ascending=[False, False]
        ).reset_index(drop=True)
        if return_data.empty:
            print("筛选后无有效数据")
            self.data = []
            return []
        sort_data = return_data.head(N).copy()
        self.return_sample_sum = sort_data["__sample_size"].to_numpy()
        sort_data = sort_data.drop(
            columns=["__score", "__sample_size", "__score_details"],
            errors="ignore"
        )
        self.data = sort_data
        print(f"最终返回数据数量：{len(sort_data)}")
        return sort_data

    def results_str(self, results, population):
        result_strs = []
        down_list = []
        index = 0
        if results is None:
            return result_strs, down_list
        if isinstance(results, list) and len(results) == 0:
            return result_strs, down_list
        if isinstance(results, pd.DataFrame) and results.empty:
            return result_strs, down_list
        for _, res in results.iterrows():
            accession_id = res.get("accessionId", "")
            if accession_id not in self.usable_GCST_id_dict:
                continue
            download_path = self.usable_GCST_id_dict[accession_id]
            sample_size_value = ""
            try:
                if index < len(self.return_sample_sum):
                    sample_size_value = self.return_sample_sum[index]
            except Exception:
                sample_size_value = ""

            res_str = (
                f"Study ID: {res.get('accessionId', '')}\n"
                f"Paper title: {res.get('publicationInfo.title', '')}\n"
                f"Disease/Trait: {res.get('diseaseTrait.trait', '')}\n"
                f"Population: {population}\n"
                f"Publication Day: {res.get('publicationInfo.publicationDate', '')}\n"
                f"Sample Size: {sample_size_value}\n"
                f"Downloads Path: {download_path}\n"
                f"Sample description: {res.get('initialSampleSize', '')}\n"
                f"PMID: {res.get('publicationInfo.pubmedId', '')}\n"
                f"Journal: {res.get('publicationInfo.publication', '')}\n"
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
    def __init__(self, file_path="trait_list.txt", top_k=5):

        self.file_path = file_path
        self.top_k = top_k
        self.trait_list = self._load_traits()
    def _load_traits(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _generate_word_combinations(self, phrase, min_words=2):
        words = phrase.split()
        combinations = []
        for r in range(min_words, len(words) + 1):
            for combo in itertools.combinations(words, r):
                combinations.append(" ".join(combo))
        return combinations

    def _get_top_k_similar(self, query, candidates):
        vectorizer = TfidfVectorizer().fit([query] + candidates)
        vectors = vectorizer.transform([query] + candidates)
        cosine_sim = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
        top_indices = cosine_sim.argsort()[::-1][:self.top_k]
        return [candidates[i] for i in top_indices]

    def find_similar_traits(self, query):
        if not self.trait_list:
            return []
        filtered_traits_list = []
        all_traits_list = []
        query_lower = query.lower()
        if len(query_lower.split()) == 1:
            filtered_traits = [
                trait for trait in self.trait_list if query_lower in trait.lower()
            ]
            if filtered_traits:
                filtered_traits_list = self._get_top_k_similar(query, filtered_traits)
            else:
                all_traits_list = self._get_top_k_similar(query, self.trait_list)

        else:
            query_combos = self._generate_word_combinations(query_lower)
            for keywords in query_combos:
                filtered_traits = [
                    trait for trait in self.trait_list if keywords in trait.lower()
                ]
                if filtered_traits:
                    filtered_traits_list += self._get_top_k_similar(query, filtered_traits)
                else:
                    all_traits_list += self._get_top_k_similar(query, self.trait_list)
        return filtered_traits_list if filtered_traits_list else all_traits_list


class GWASLoaderTool():
    def __init__(self):
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

class LDPrunerAgent:
    def __init__(self, gwas_file, output_dir="./", r2=None, window_kb=None):
        self.plink_path = "plink"
        self.gwas_file = gwas_file
        self.ref_panel_prefix = "./data/1000G_EUR/1000G_phase3_common_norel"
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_prefix = os.path.join(self.output_dir, "ld_pruned")
        self.log_file = os.path.join(self.output_dir, "ld_pruner.log")
        self.status = {}
        self.r2 = r2
        self.window_kb = window_kb
        self._check_files()

    def _check_files(self):
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
        with open(self.log_file, "a") as f:
            f.write(message + "\n")

    def _read_gwas(self):
        if self.gwas_file.endswith(".gz"):
            df = pd.read_csv(self.gwas_file, sep="\t", compression="gzip", low_memory=False)
        else:
            df = pd.read_csv(self.gwas_file, sep="\t", low_memory=False)
        return df

    def _detect_snp_column(self, df):
        snp_column_candidates = ["SNP", "rsid", "variant_id", "hm_rsid"]
        for col in snp_column_candidates:
            if col in df.columns:
                return col
        raise ValueError("未找到可识别的 SNP ID 列，尝试了：" + ", ".join(snp_column_candidates))

    def _suggest_initial_params(self, snp_count):
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


    def run(self, min_snps_threshold=10, max_snps_threshold=100, max_attempts=10, step=10):
        self._log("[1] 读取GWAS文件并提取所有SNP")
        df = self._read_gwas()
        self.snp_column = self._detect_snp_column(df)
        snps = df[self.snp_column].dropna().unique()
        snp_total = len(snps)
        self._log(f"总SNP数量: {snp_total}")
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
                        r2 += 0.002
                        window_kb = max(100, window_kb - 100)
                    elif snp_count > max_snps_threshold:
                        self._log(f"    [尝试 {attempt + 1}] SNP数量过多（{snp_count}），收紧参数重试...")
                        r2 = max(0.001, r2 - 0.002)
                        window_kb += 200
            except Exception as e:
                self._log(f"    发生错误：{e}")
                break

        self.status.update({
            "completed": False,
            "output_file": "",
            "warning": f"参数多次调整后仍未达到 {min_snps_threshold}-{max_snps_threshold} SNP 的要求"
        })
        return ""

    def get_status(self):
        return self.status

    def command_run_auto(self, args):
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


def add_r2_fstat_filter(df, N, f_threshold=10):
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
    df_filtered = df_all[df_all["F_stat"] > f_threshold].reset_index(drop=True)
    return df_all, df_filtered


def phenotype_matcher(user_trait, gwas_trait_description, LLM_model="gpt-4o", openai_api_key=None):
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

    response = query_model(
        openai_api_key=openai_api_key,
        model_str=LLM_model,
        system_prompt=system_prompt,
        prompt=prompt,
        temp=0.0
    )
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
        self.snp_name_cols = snp_name_cols
        self.beta_hat_cols = beta_hat_cols
        self.se_cols = se_cols
        self.A1_cols = A1_cols
        self.A2_cols = A2_cols

    @staticmethod
    def remove_ambiguous(df):
        ambiguous = {('A', 'T'), ('T', 'A'), ('G', 'C'), ('C', 'G')}
        return df[~df[['A1', 'A2']].apply(tuple, axis=1).isin(ambiguous)]

    @staticmethod
    def align_beta(df, beta_col):
        strand_flip = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
        def flip_alleles(row):
            if row['A1'] == 'T' or row['A2'] == 'T':
                A1f = strand_flip.get(row['A1'], row['A1'])
                A2f = strand_flip.get(row['A2'], row['A2'])
            else:
                A1f, A2f = row['A1'], row['A2']
            if A1f == 'A':
                beta = row[beta_col]
            else:
                beta = -row[beta_col]
                A1f, A2f = A2f, A1f
            return pd.Series([beta, A1f, A2f], index=[beta_col, 'A1', 'A2'])
        df[[beta_col, 'A1', 'A2']] = df.apply(flip_alleles, axis=1)
        return df

    def merge(self, X1, X2):
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

        X1 = self.remove_ambiguous(X1)
        X2 = self.remove_ambiguous(X2)
        X1 = self.align_beta(X1, 'beta_hat')
        X2 = self.align_beta(X2, 'beta_hat')
        merged = pd.merge(X1, X2, on='snp', suffixes=('_1', '_2'))
        merged = merged.dropna(subset=['beta_hat_1', 'beta_hat_2', 'se_1', 'se_2'])

        merged = merged[merged[['beta_hat_1', 'beta_hat_2', 'se_1', 'se_2']].applymap(
            lambda x: pd.notnull(x) and pd.api.types.is_number(x)).all(axis=1)]
        merged = merged[(merged['se_1'] > 0) & (merged['se_2'] > 0)]
        merged = merged[merged['A2_1'] == merged['A2_2']]
        merged = merged.rename(columns={
            'A1_1': 'A1',
            'A2_1': 'A2'
        })[['snp', 'beta_hat_1', 'se_1', 'beta_hat_2', 'se_2', 'A1', 'A2']]

        print(f"After merging and cleaning, {len(merged)} variants remain for MR analysis.")
        last_merge_data_path = "research_results/processed_data/1/harmonized_data.csv"
        merged.to_csv(last_merge_data_path, index=False)
        return merged, last_merge_data_path

matplotlib.use('Agg')

def _detect_sep(file_path):
    try:
        opener = gzip.open if str(file_path).endswith(".gz") else open
        with opener(file_path, "rt", encoding="utf-8", errors="ignore") as f:
            line = f.readline()
        if "\t" in line:
            return "\t"
        if "," in line:
            return ","
        return "\t"
    except Exception:
        return "\t"

def _read_gwas_table(file_path):
    sep = _detect_sep(file_path)
    return pd.read_csv(
        file_path,
        sep=sep,
        compression="infer",
        low_memory=False
    )

def _find_col(df, candidates):
    col_map = {}
    for c in df.columns:
        norm = str(c).strip().lower()
        norm = norm.replace("-", "_").replace(".", "_").replace(" ", "_")
        col_map[norm] = c
    for cand in candidates:
        norm_cand = str(cand).strip().lower()
        norm_cand = norm_cand.replace("-", "_").replace(".", "_").replace(" ", "_")
        if norm_cand in col_map:
            return col_map[norm_cand]
    return None

def _standardize_gwas_columns(df):
    col_variant = _find_col(df, [
        "variant_id",
        "hm_variant_id",
        "variant",
        "markername",
        "marker_name",
        "snp",
        "rsid",
        "hm_rsid",
        "rs_number"
    ])

    col_rsid = _find_col(df, [
        "rsid",
        "hm_rsid",
        "snp",
        "rs_number",
        "variant_id",
        "hm_variant_id",
        "markername",
        "marker_name"
    ])

    col_p = _find_col(df, [
        "p_value",
        "p-value",
        "pvalue",
        "p",
        "pval",
        "p_val",
        "p.value",
        "p_bolt_lmm_inf",
        "p_bolt_lmm"
    ])

    col_beta = _find_col(df, [
        "beta",
        "hm_beta",
        "effect",
        "effect_size",
        "estimate",
        "b"
    ])

    col_or = _find_col(df, [
        "odds_ratio",
        "or",
        "hm_odds_ratio"
    ])

    col_se = _find_col(df, [
        "standard_error",
        "se",
        "stderr",
        "std_error",
        "beta_se",
        "se_beta"
    ])

    col_ea = _find_col(df, [
        "effect_allele",
        "hm_effect_allele",
        "tested_allele",
        "allele1",
        "a1",
        "ea"
    ])

    col_oa = _find_col(df, [
        "other_allele",
        "hm_other_allele",
        "non_effect_allele",
        "allele2",
        "a2",
        "nea"
    ])

    col_eaf = _find_col(df, [
        "effect_allele_frequency",
        "eaf",
        "eaf_effect",
        "hm_effect_allele_frequency",
        "frequency",
        "freq"
    ])

    missing = []
    if col_variant is None and col_rsid is None:
        missing.append("variant_id/rsid/hm_rsid")
    if col_p is None:
        missing.append("p_value/pval/p")
    if missing:
        print(f"文件中缺少以下必要列：{set(missing)}")
        print(f"当前文件实际列名为：{list(df.columns)[:50]}")
        return pd.DataFrame()
    out = df.copy()
    if col_variant is not None:
        out["variant_id"] = out[col_variant].astype(str)
    elif col_rsid is not None:
        out["variant_id"] = out[col_rsid].astype(str)

    if col_rsid is not None:
        out["hm_rsid"] = out[col_rsid].astype(str)
    else:
        out["hm_rsid"] = out["variant_id"].astype(str)

    out["p_value"] = pd.to_numeric(out[col_p], errors="coerce")

    if col_beta is not None:
        out["beta"] = pd.to_numeric(out[col_beta], errors="coerce")
    elif col_or is not None:
        out["beta"] = np.log(pd.to_numeric(out[col_or], errors="coerce"))
    else:
        out["beta"] = np.nan
    if col_se is not None:
        out["standard_error"] = pd.to_numeric(out[col_se], errors="coerce")
    else:
        out["standard_error"] = np.nan

    if col_ea is not None:
        out["effect_allele"] = out[col_ea].astype(str)
    else:
        out["effect_allele"] = pd.NA

    if col_oa is not None:
        out["other_allele"] = out[col_oa].astype(str)
    else:
        out["other_allele"] = pd.NA

    if col_eaf is not None:
        out["effect_allele_frequency"] = pd.to_numeric(out[col_eaf], errors="coerce")
    else:
        out["effect_allele_frequency"] = np.nan

    return out

def filter_significant_mutations(file_path, thresholds=None, chunksize=200000):
    if thresholds is None:
        thresholds = [5e-8, 1e-6, 1e-5]
    for threshold in thresholds:
        filtered_chunks = []
        try:
            reader = pd.read_csv(
                file_path,
                sep="\t",
                compression="gzip" if str(file_path).endswith(".gz") else None,
                chunksize=chunksize,
                low_memory=False
            )
            for chunk in reader:
                chunk = _standardize_gwas_columns(chunk)
                if chunk is None or chunk.empty:
                    continue
                if "p_value" not in chunk.columns:
                    continue
                chunk["p_value"] = pd.to_numeric(chunk["p_value"], errors="coerce")
                chunk = chunk.dropna(subset=["p_value"])

                filtered = chunk[chunk["p_value"] < threshold].copy()

                if not filtered.empty:
                    filtered_chunks.append(filtered)
        except Exception as e:
            print(f"分块读取 GWAS 文件失败: {e}")
            return pd.DataFrame(), None
        if filtered_chunks:
            result = pd.concat(filtered_chunks, ignore_index=True)
            return result, threshold

    print("没有筛选到达到阈值的显著 SNP")
    return pd.DataFrame(), None
def standardize_rsid_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            if name == 'rsid':
                return df
            df = df.rename(columns={name: 'rsid'})
            return df
    raise ValueError(f"未找到列名 {possible_names} 中的任意一个用于 rsid 匹配")

def worker_run_code(code_str, output_queue):
    output_capture = io.StringIO()
    sys.stdout = output_capture
    try:
        globals_dict = {"__name__": "__main__"}
        exec(code_str, globals_dict)
    except Exception as e:
        output_capture.write(f"[CODE EXECUTION ERROR]: {str(e)}\n")
        traceback.print_exc(file=output_capture)
    finally:
        sys.stdout = sys.__stdout__
    output_queue.put(output_capture.getvalue())


def extract_rsid(df, col_candidates=('rsid', 'variant_id')):
    for col in col_candidates:
        if col in df.columns:
            return set(df[col].dropna().astype(str).unique())
    print("[警告] 没有找到 rsid 或 variant_id 列。")
    return set()

def worker_run_r_code(code_str, output_queue):
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(code_str)
            tmp_file_path = tmp_file.name

        print("Executing R code...")
        print("Temporary R file:", tmp_file_path)
        env = os.environ.copy()
        env["LANG"] = "en_US.UTF-8"
        env["R_ENVIRON_USER"] = ""

        completed_process = subprocess.run(
            ['Rscript', tmp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            timeout=1000,
            env=env,
        )

        os.remove(tmp_file_path)
        if completed_process.returncode != 0:
            output_queue.put(f"[CODE EXECUTION ERROR]\n{completed_process.stderr}")
        else:
            output_queue.put(completed_process.stdout)
    except subprocess.TimeoutExpired:
        output_queue.put("[CODE EXECUTION ERROR]: Code execution exceeded the timeout limit.")
    except Exception as e:
        output_queue.put(f"[CODE EXECUTION ERROR]: Unexpected error: {str(e)}")


def worker_run_r_code2(code_str, output_queue, timeout=1000):
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(code_str)
            tmp_file_path = tmp_file.name
        print("Executing R code...")
        print("Temporary R file:", tmp_file_path)

        env = os.environ.copy()
        env["LANG"] = "en_US.UTF-8"
        env["R_ENVIRON_USER"] = ""
        env["TF_ENABLE_ONEDNN_OPTS"] = "0"

        process = subprocess.Popen(
            ['Rscript', tmp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            env=env
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            output_queue.put("[CODE EXECUTION ERROR]: Code execution exceeded the timeout limit.")
            return

        if process.returncode != 0:
            output_queue.put(f"[CODE EXECUTION ERROR]\n{stderr}")
            process.kill()
        else:
            output_queue.put(stdout)
            process.kill()

    except Exception as e:
        output_queue.put(f"[CODE EXECUTION ERROR]: Unexpected error: {str(e)}")
    finally:
        try:
            os.remove(tmp_file_path)
        except Exception:
            pass

def execute_r_code1(code_str, timeout=1200):
    MAX_LEN = 10000
    if len(code_str) > MAX_LEN:
        return f"[CODE EXECUTION ERROR]: Code length exceeds max allowed length of {MAX_LEN} characters."

    if "quit(" in code_str or "q(" in code_str:
        return "[CODE EXECUTION ERROR] The quit()/q() command is not allowed. Please remove it."

    output_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=worker_run_r_code, args=(code_str, output_queue))
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return f"[CODE EXECUTION ERROR]: Code execution exceeded the timeout limit of {timeout} seconds."

    if not output_queue.empty():
        return output_queue.get()
    else:
        return ""


def execute_r_code(code_str, timeout=3000):
    MAX_LEN = 10000
    if len(code_str) > MAX_LEN:
        return f"[CODE EXECUTION ERROR]: Code length exceeds max allowed length of {MAX_LEN} characters."
    if "quit(" in code_str or "q(" in code_str:
        return "[CODE EXECUTION ERROR]: The quit()/q() command is not allowed. Please remove it."
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False, encoding='utf-8') as tmp_file:
        tmp_file.write(code_str)
        tmp_file_path = tmp_file.name
    print("Executing R code...")
    print("Temporary R file:", tmp_file_path)

    env = os.environ.copy()
    env["LANG"] = "en_US.UTF-8"
    env["R_ENVIRON_USER"] = ""
    env["LC_ALL"] = "en_US.UTF-8"

    try:
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
        if os.path.exists(tmp_file_path):
            for _ in range(3):
                try:
                    os.remove(tmp_file_path)
                    break
                except PermissionError:
                    time.sleep(0.2)

def extract_json_between_markers(llm_output):
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)
    if not matches:
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)
    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            try:
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue
    return None

def get_score(outlined_plan, latex, reward_model_llm, reviewer_type=None, attempts=3, openai_api_key=None):
    last_error = ""
    def _safe_int(value, field_name, default=1, min_value=1, max_value=10):
        try:
            if isinstance(value, int):
                parsed = value
            elif isinstance(value, float):
                parsed = int(value)
            elif isinstance(value, str):
                match = re.search(r"\d+", value)
                parsed = int(match.group(0)) if match else default
            else:
                parsed = default
            return max(min_value, min(parsed, max_value))
        except Exception:
            print(f"Warning: failed to parse field {field_name}: {value}")
            return default

    for _attempt in range(attempts):
        scoring = ""
        try:
            template_instructions = """
                Respond in the following format:
                
                THOUGHT:
                <THOUGHT>
                
                REVIEW JSON:
                ```json
                <JSON>
                ```
                
                In <THOUGHT>, briefly explain your reasoning for the evaluation. Be specific to the current manuscript and avoid generic comments.
                
                In <JSON>, provide the review using exactly the following fields:
                - "Summary": A summary of the manuscript content and its contributions.
                - "Strengths": A list of strengths.
                - "Weaknesses": A list of weaknesses.
                - "Originality": A rating from 1 to 4.
                - "Quality": A rating from 1 to 4.
                - "Clarity": A rating from 1 to 4.
                - "Significance": A rating from 1 to 4.
                - "Questions": Clarifying questions or suggestions for the authors.
                - "Limitations": Limitations of the work.
                - "Ethical Concerns": A boolean value.
                - "Soundness": A rating from 1 to 4.
                - "Presentation": A rating from 1 to 4.
                - "Contribution": A rating from 1 to 4.
                - "Overall": A rating from 1 to 10.
                - "Confidence": A rating from 1 to 5.
                - "Decision": Either Accept or Reject.
                
                The JSON must include all fields exactly as named:
                "Summary", "Strengths", "Weaknesses", "Originality", "Quality", "Clarity", "Significance",
                "Questions", "Limitations", "Ethical Concerns", "Soundness", "Presentation", "Contribution",
                "Overall", "Confidence", "Decision".
                
                Do not add, rename, or remove any field.
                For the "Decision" field, use only Accept or Reject.
                If a field has no content, use an empty string "" or a default numeric value.
                """

            strobe_mr_mas_review_form = (
                    """
                ## STROBE-MR-informed Review Form for MR-MAS-generated Manuscripts
    
                You are evaluating a manuscript generated by MR-MAS for a Mendelian Randomization (MR) study. 
                The evaluation should focus on manuscript reporting quality, methodological transparency, faithfulness to the executed MR analysis, and reproducibility. 
                Use the STROBE-MR checklist as the reporting guideline, but adapt it to the MR-MAS setting.
    
                Evaluate the manuscript from the following six perspectives:
    
                1. Alignment with the research plan and executed analysis:
                Assess whether the manuscript faithfully follows the provided MR research plan and accurately reflects the executed R-based MR analysis. 
                Penalize fabricated or unsupported datasets, numerical results, p-values, confidence intervals, figures, citations, or causal claims.
    
                2. Title, abstract, background, and objectives:
                Assess whether the manuscript clearly identifies MR as the study design, explains the exposure--outcome question, provides scientific and biological rationale, and states the causal objective or hypothesis.
    
                3. Study design, data sources, instruments, and assumptions:
                Assess whether the manuscript reports GWAS data sources, population or ancestry information when available, exposure and outcome definitions, genetic instrument selection, LD pruning, weak-instrument checks, allele harmonization, and the three core MR assumptions: relevance, independence, and exclusion restriction.
    
                4. Statistical methods and sensitivity analyses:
                Assess whether the manuscript reports the MR estimators and statistical methods used, such as IVW, MR-Egger, weighted median, weighted mode, MR-PRESSO, or other implemented methods. 
                Also assess whether heterogeneity, horizontal pleiotropy, outlier analysis, leave-one-out analysis, and other robustness checks are reported when applicable.
    
                5. Results, interpretation, and limitations:
                Assess whether the manuscript reports MR estimates, standard errors, confidence intervals, p-values, method-level consistency, and relevant plots or tables. 
                Evaluate whether the interpretation is cautious, biologically plausible, compared with prior evidence when available, and accompanied by limitations, uncertainty, potential bias, and generalizability discussion.
    
                6. Reproducibility and other reporting information:
                Assess whether the manuscript reports software, R packages, data access, code or workflow details, and reproducibility information. 
                Funding, ethics approval, pre-registration, and conflicts of interest should be checked when available; if this information is unavailable to MR-MAS, note it as requiring author completion rather than treating it as a major flaw.
    
                Scoring guidance:
                - Soundness should reflect whether the manuscript reports a coherent MR design, appropriate methods, and evidence-supported conclusions.
                - Presentation should reflect writing quality, organization, clarity of tables and figures, and completeness of STROBE-MR-style reporting.
                - Contribution should reflect the usefulness of the generated manuscript for MR analysis, automated scientific writing, or bioinformatics workflow support.
                - Overall should reflect the overall quality of the generated MR manuscript under STROBE-MR-informed reporting principles.
    
                Overall score:
                10: Exceptional manuscript with complete STROBE-MR-aligned reporting, faithful use of executed results, clear interpretation, and strong reproducibility.
                9: Very strong manuscript with high-quality MR reporting and only minor limitations.
                8: Strong manuscript with clear MR reporting and convincing executed evidence, but with minor weaknesses.
                7: Solid manuscript with generally adequate reporting, but with limitations that should be addressed.
                6: Marginally acceptable manuscript with useful content but notable concerns in reporting, clarity, or validation.
                5: Borderline manuscript with substantial weaknesses that limit confidence.
                4: Weak manuscript with important reporting, methodological, reproducibility, or interpretation problems.
                3: Poor manuscript with major flaws in MR reporting or evidence support.
                2: Very weak manuscript with severe technical or interpretive problems.
                1: Unacceptable manuscript with unsupported claims, fabricated results, or unusable reporting.
    
                Important:
                This review evaluates the quality of the generated MR manuscript and its alignment with STROBE-MR-informed reporting principles. 
                It is used as an internal quality-control signal for manuscript refinement and should not be interpreted as external validation of the causal conclusion.
                """
                    + template_instructions
            )

            if reviewer_type is None:
                reviewer_type = ""

            system_prompt = (
                  "You are a STROBE-MR-informed statistical genetics and bioinformatics reviewer "
                  "evaluating a manuscript about Mendelian Randomization study. "
                  "Be critical and cautious in your decision. "
                  "Focus on transparent MR reporting, methodological rigor, reproducibility, "
                  "and faithfulness to the executed MR analysis. "
                  f"{reviewer_type}\n"
                  ) + strobe_mr_mas_review_form

            scoring = query_model(
                model_str=f"{reward_model_llm}",
                system_prompt=system_prompt,
                openai_api_key=openai_api_key,
                prompt=(
                    "Outlined below is the MR research plan that the manuscript should follow:\n"
                    f"{outlined_plan}\n\n"
                    "The following text is the generated LaTeX manuscript to be reviewed:\n"
                    f"{latex}\n\n"
                    "Please evaluate the generated manuscript according to the STROBE-MR-informed review form."
                ),
                temp=0.0
            )

            review_json = extract_json_between_markers(scoring)

            overall = _safe_int(review_json.get("Overall", 1), "Overall", max_value=10) / 10
            soundness = _safe_int(review_json.get("Soundness", 1), "Soundness", max_value=4) / 4
            confidence = _safe_int(review_json.get("Confidence", 1), "Confidence", max_value=5) / 5
            contribution = _safe_int(review_json.get("Contribution", 1), "Contribution", max_value=4) / 4
            presentation = _safe_int(review_json.get("Presentation", 1), "Presentation", max_value=4) / 4
            clarity = _safe_int(review_json.get("Clarity", 1), "Clarity", max_value=4) / 4
            originality = _safe_int(review_json.get("Originality", 1), "Originality", max_value=4) / 4
            quality = _safe_int(review_json.get("Quality", 1), "Quality", max_value=4) / 4
            significance = _safe_int(review_json.get("Significance", 1), "Significance", max_value=4) / 4

            clarity_weight = 0.1
            quality_weight = 0.1
            overall_weight = 1.0
            soundness_weight = 0.1
            confidence_weight = 0.1
            originality_weight = 0.1
            significance_weight = 0.1
            contribution_weight = 0.4
            presentation_weight = 0.2

            max_score = (
                clarity_weight +
                quality_weight +
                overall_weight +
                soundness_weight +
                confidence_weight +
                originality_weight +
                significance_weight +
                contribution_weight +
                presentation_weight
            )

            performance = (
                (
                    soundness_weight * soundness +
                    presentation_weight * presentation +
                    confidence_weight * confidence +
                    contribution_weight * contribution +
                    overall_weight * overall +
                    originality_weight * originality +
                    significance_weight * significance +
                    clarity_weight * clarity +
                    quality_weight * quality
                ) / max_score
            ) * 10

            return performance, f"The performance of your submission is: {performance}\n\n{scoring}", True
        except Exception as e:
            last_error = str(e)
            print("评分模型原始输出:\n", scoring)
            print(f"评分出错了，错误信息是：{last_error}")
    return None, last_error, False
