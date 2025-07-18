
# call requests to get the data by the url
def get_data(url, params=None, timeout=30, return_json=False):
    import requests
    response = requests.get(url, params=params, timeout=timeout)
    if response.status_code != 200:
        print(f"Failed to get data: {response.status_code=}, {response=}")
        return None
    if return_json:
        return response.json()
    else:
        return response.content

class KnowledgeBase:
    def __init__(self):
        self.base_url = "http://180.184.65.98:38880/atomgit"

    def metadata(self):
        """
        /metadata: 获取论文数据库的元数据信息。
        请求方式: GET
        参数: 无
        返回值: 元数据信息 (具体结构取决于 get_metadata 函数的实现)。
        """
        url = self.base_url + "/metadata"
        json_data = get_data(url, return_json=True)
        return json_data

    def search_papers(self, query, top_k=30):
        """
        /search_papers: 基于文本内容搜索相关论文片段。
        请求方式: GET
        参数:
        query (字符串, 必填): 要搜索的文本查询内容。
        top_k (整数, 可选, 默认值: 30): 返回最相关的论文片段数量。
        返回值: 论文片段信息列表。 每个元素是一个字典，包含 id (内部ID), distance (相关度距离), entity (包含 paper_id, paper_title, chunk_id, chunk_text, original_filename 等字段的字典)。
        """
        url = self.base_url + "/search_papers"
        params={"query": query, "top_k": top_k}
        json_data = get_data(url, params=params, return_json=True)
        return json_data

    def query_by_paper_id(self, paper_id, top_k=5):
        """
        /query_by_paper_id: 根据论文ID查询相关的论文片段。
        请求方式: GET
        参数:
        paper_id (字符串, 必填): 要查询的论文ID。
        top_k (整数, 可选, 默认值: 5): 返回最相关的论文片段数量。
        返回值: 论文片段信息列表。 结构同 /search_papers。
        """
        url = self.base_url + "/query_by_paper_id"
        params={"paper_id": paper_id, "top_k": top_k}
        json_data = get_data(url, params=params, return_json=True)
        return json_data
    
    def query_by_title(self, title, top_k=100):
        """
        /query_by_title: 根据论文标题查询相关的论文片段。
        请求方式: GET
        参数:
        title (字符串, 必填): 要查询的论文标题。
        top_k (整数, 可选, 默认值: 100): 返回最相关的论文片段数量。
        返回值: 论文片段信息列表。 结构同 /search_papers。
        """
        url = self.base_url + "/query_by_title"
        params={"title": title, "top_k": top_k}
        json_data = get_data(url, params=params, return_json=True)
        return json_data

    def query_by_title_contain(self, title, top_k=1000):
        """
        /query_by_title_contain: 搜索标题中包含特定关键词的论文片段。
        请求方式: GET
        参数:
        title (字符串, 必填): 要搜索的标题关键词。
        top_k (整数, 可选, 默认值: 1000): 返回最多 top_k 个结果。
        返回值: 论文片段信息列表。 结构同 /search_papers。
        """
        url = self.base_url + "/query_by_title_contain"
        params={"title": title, "top_k": top_k}
        json_data = get_data(url, params=params, return_json=True)
        return json_data

    def query_by_chunk_contain(self, chunk, top_k=1000):
        """
        /query_by_chunk_contain: 搜索论文片段内容中包含特定关键词的论文片段。
        请求方式: GET
        参数:
        chunk (字符串, 必填): 要搜索的片段关键词。
        top_k (整数, 可选, 默认值: 1000): 返回最多 top_k 个结果。
        返回值: 论文片段信息列表。 结构同 /search_papers。
        """
        url = self.base_url + "/query_by_chunk_contain"
        params={"chunk": chunk, "top_k": top_k}
        json_data = get_data(url, params=params, return_json=True)
        return json_data

    def query_by_title_like(self, title, top_k=1):
        """
        /query_by_title_like: 通过相似标题查询论文，并返回相关的论文片段。
        请求方式: GET
        参数:
        title (字符串, 必填): 要查询的标题关键词。
        top_k (整数, 可选, 默认值: 1): 查找最相似的标题数量 (用于初步筛选)。
        返回值: 嵌套的论文片段信息列表。 外层列表对应每个相似标题，内层列表是与该相似标题相关的论文片段信息列表 (结构同 /search_papers)。
        """
        url = self.base_url + "/query_by_title_like"
        params={"title": title, "top_k": top_k}
        json_data = get_data(url, params=params, return_json=True)
        return json_data

    def query_by_keyword(self, keyword):
        """
        /query_by_keyword: 通过关键词查询论文ID和标题。
        请求方式: GET
        参数:
        keyword (字符串, 必填): 要查询的关键词。
        返回值: 论文ID和标题列表。 每个元素是一个元组 (paper_id, title)。
        """
        url = self.base_url + "/query_by_keyword"
        params={"keyword": keyword}
        json_data = get_data(url, params=params, return_json=True)
        return json_data

    def query_whole_text_by_id(self, paper_id):
        """
        /query_whole_text_by_id: 通过论文ID查询论文全文。
        请求方式: GET
        参数:
        paper_id (字符串, 必填): 要查询的论文ID。
        返回值: 论文完整文本 (字符串) 或 None (如果未找到)。
        """
        url = self.base_url + "/query_whole_text_by_id"
        params={"paper_id": paper_id}
        json_data = get_data(url, params=params, return_json=True)
        return json_data

    def query_whole_text_by_title(self ,title):
        """
        /query_whole_text_by_title: 通过论文标题查询论文全文。
        请求方式: GET
        参数:
        title (字符串, 必填): 要查询的论文标题。
        返回值: 论文完整文本 (字符串) 或 None (如果未找到)。
        """
        url = self.base_url + "/query_whole_text_by_title"
        params={"title": title}
        json_data = get_data(url, params=params, return_json=True)
        return json_data

    def query_keywords_by_id(self, paper_id):
        """
        /query_keywords_by_id: 通过论文ID查询论文关键词。
        请求方式: GET
        参数:
        paper_id (字符串, 必填): 要查询的论文ID。
        返回值: 关键词列表 (字符串列表) 或 None (如果未找到)。
        """
        url = self.base_url + "/query_keywords_by_id"
        params={"paper_id": paper_id}
        json_data = get_data(url, params=params, return_json=True)
        return json_data

    def query_keywords_by_title(self, title):
        """
        /query_keywords_by_title: 通过论文标题查询论文关键词。
        请求方式: GET
        参数:
        title (字符串, 必填): 要查询的论文标题。
        返回值: 关键词列表 (字符串列表) 或 None (如果未找到)。
        """
        url = self.base_url + "/query_keywords_by_title"
        params={"title": title}
        json_data = get_data(url, params=params, return_json=True)
        return json_data

    def keywords_metadata(self):
        """
        /keywords_metadata: 获取所有关键词及其计数。
        请求方式: GET
        参数: 无
        返回值: 关键词统计信息。 JSON 字典，键为关键词字符串，值为该关键词的计数 (关联的论文数量)。 关键词按计数降序排列。
        """
        url = self.base_url + "/query_keywords_by_title"
        params={}
        json_data = get_data(url, params=params, return_json=True)
        return json_data


    def test(self):
        kb = KnowledgeBase()
        results = kb.search_papers("covid-19")
        print(f"kb.search_paper() return {len(results)} chunks.")
        results = kb.query_by_paper_id("6391560190e50fcafd9c4ead")
        print(f"kb.query_by_paper_id() return {len(results)} chunks.")
        results = kb.query_by_title("Copula Conformal Prediction for Multi-step Time Series Forecasting")
        print(f"kb.query_by_title() return {len(results)} chunks.")
        results = kb.query_by_title_contain("Time Series Forecasting")
        print(f"kb.query_by_title_contain() return {len(results)} chunks.")
        results = kb.query_by_chunk_contain("Time Series Forecasting")
        print(f"kb.query_by_chunk_contain() return {len(results)} chunks.")