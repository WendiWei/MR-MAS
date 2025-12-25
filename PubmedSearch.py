from Bio import Entrez
import time
import requests
from bs4 import BeautifulSoup
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
        根据查询字符串在PubMed搜索文献，并返回生成参考文献需要的完整信息
        @param query: 查询关键词
        @param max_results: 返回最大文献数量
        @param sort_order: 排序方式，常用'relevance'或'pubdate'
        @return: 文献列表，每个元素为字典包含title, abstract, pmid, authors, journal, year等信息
        """
        processed_query = self._process_query(query)

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # 搜索PubMed
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
                        # 标题
                        title = article_data.get('ArticleTitle', 'No title')
                        # 摘要
                        abstract = 'No abstract'
                        if 'Abstract' in article_data and 'AbstractText' in article_data['Abstract']:
                            abs_data = article_data['Abstract']['AbstractText']
                            if isinstance(abs_data, list):
                                abstract = ' '.join([str(x) for x in abs_data])
                            else:
                                abstract = str(abs_data)
                        # PMID
                        pmid = str(article['MedlineCitation'].get('PMID', 'NULL'))
                        # 作者列表
                        authors_list = []
                        for author in article_data.get('AuthorList', []):
                            last = author.get('LastName')
                            initials = author.get('Initials')
                            if last and initials:
                                authors_list.append(f"{last} {initials}")
                            elif last:
                                authors_list.append(last)
                        if not authors_list:
                            authors_list = ["Unknown"]
                        # 期刊
                        journal = article_data.get('Journal', {}).get('Title', 'No journal')
                        # 出版年份
                        year = 'No year'
                        pub_date = article_data.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
                        if 'Year' in pub_date:
                            year = pub_date['Year']
                        elif 'MedlineDate' in pub_date:
                            year = pub_date['MedlineDate'].split()[0]

                        papers.append({
                            "title": title,
                            "abstract": abstract,
                            "pmid": pmid,
                            "authors": authors_list,
                            "journal": journal,
                            "year": year
                        })

                    except Exception:
                        # 遇到异常也要返回基本信息
                        papers.append({
                            "title": 'No title',
                            "abstract": 'No abstract',
                            "pmid": 'NULL',
                            "authors": ["Unknown"],
                            "journal": "No journal",
                            "year": "No year"
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
