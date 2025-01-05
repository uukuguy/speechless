
# call requests to get the data by the url
def get_data(url, params=None, timeout=30, return_json=False):
    import requests
    response = requests.get(url, params=params, timeout=timeout)
    if response.status_code != 200:
        print('Failed to get data:', response.status_code)
        return None
    if return_json:
        return response.json()
    else:
        return response.content

class KnowledgeBase:
    def __init__(self):
        self.base_url = "http://180.184.65.98:38880/atomgit"

    def metadata(self):
        url = self.base_url + "/metadata"
        json_data = get_data(url, return_json=True)
        return json_data

    def search_papers(self, query, top_k=30):
        url = self.base_url + "/search_papers"
        params={"query": query, "top_k": top_k}
        json_data = get_data(url, params=params, return_json=True)
        return json_data

    def query_by_paper_id(self, paper_id, top_k=5):
        url = self.base_url + "/query_by_paper_id"
        params={"paper_id": paper_id, "top_k": top_k}
        json_data = get_data(url, params=params, return_json=True)
        return json_data
    
    def query_by_title(self, title, top_k=100):
        url = self.base_url + "/query_by_title"
        params={"title": title, "top_k": top_k}
        json_data = get_data(url, params=params, return_json=True)
        return json_data

    def query_by_title_contain(self, title, top_k=1000):
        url = self.base_url + "/query_by_title_contain"
        params={"title": title, "top_k": top_k}
        json_data = get_data(url, params=params, return_json=True)
        return json_data

    def query_by_chunk_contain(self, chunk, top_k=1000):
        url = self.base_url + "/query_by_chunk_contain"
        params={"chunk": chunk, "top_k": top_k}
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