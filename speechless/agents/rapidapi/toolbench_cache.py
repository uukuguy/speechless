#!/usr/bin/env python

import os
import json, time
import requests
from tqdm import tqdm
from loguru import logger

"""
    payload = {
        "category": self.cate_names[k],
        "tool_name": self.tool_names[k],
        "api_name": pure_api_name,
        "tool_input": action_input,
        "strip": self.observ_compress_method,
        "toolbench_key": self.toolbench_key
    }

"""

from collections import defaultdict

def compare_dict(dict1, dict2):
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1.keys():
        if dict1[key] != dict2[key]:
            return False
    return True

class ToolBenchCache:
    def __init__(self, cache_file:str):
        self.cache_file = cache_file
        self.toolbench_cache = self._load_toolbench_cache()
        self.num_queries = 0
        self.num_hits = 0

    def _load_toolbench_cache(self):
        toolbench_cache = defaultdict(list)

        if os.path.exists(self.cache_file):
            lines = open(self.cache_file, 'r').readlines()
            cache_datas = [json.loads(line.strip()) for line in lines]

            num_queries = 0
            for cache_data in tqdm(cache_datas, ncols=100, desc="Loading toolbench cache"):
                k = cache_data['key']
                v = cache_data['value']
                query_response_list = toolbench_cache[k]
                has_same_input = False
                for query_response in query_response_list:
                    try:
                        dict1 = json.loads(v['tool_input'])
                        dict2 = json.loads(query_response['tool_input'])
                        if compare_dict(dict1, dict2):
                            has_same_input = True
                            break
                    except:
                        if query_response['tool_input'] == v['tool_input']:
                            has_same_input = True
                            break
                if not has_same_input:
                    num_queries += 1
                    toolbench_cache[k].append(v)

            if num_queries != len(cache_datas):
                with open(self.cache_file, 'w') as f:
                    for k, v in tqdm(toolbench_cache.items(), ncols=100, desc="Resaving"):
                        for item in v:
                            f.write(json.dumps({"key": k, "value": item}, ensure_ascii=False) + "\n")

            print(f"Loaded {len(toolbench_cache)} cached api calls ({num_queries} queries) from {self.cache_file}")

        return toolbench_cache

    def _get_payload_key(self, payload):
        key = f"{payload['category']}.{payload['tool_name']}.{payload['api_name']}.{payload['strip']}"
        return key
        
    def cache_query_response(self, payload, response, status_code):
        payload_key = self._get_payload_key(payload)
        value = {
            'tool_input': payload['tool_input'], 
            'status_code': status_code,
            'response': response, 
            }

        self.toolbench_cache[payload_key].append(value)

        with open(self.cache_file, 'a') as f:
            f.write(json.dumps({"key": payload_key, "value": value}) + "\n")

    def get_response_from_cache(self, payload):
        payload_key = self._get_payload_key(payload)
        self.num_queries += 1

        query_response_list = self.toolbench_cache.get(payload_key, [])
        tool_input = payload['tool_input']
        for query_response in query_response_list:
            try:
                dict1 = json.loads(tool_input)
                dict2 = json.loads(query_response['tool_input'])
                if compare_dict(dict1, dict2):
                    self.num_hits += 1
                    logger.debug(f"ToolBenchCache: {self.num_hits/self.num_queries:.3f}({self.num_hits}/{self.num_queries})")
                    return query_response['response'], query_response['status_code']
            except:
                if query_response['tool_input'] == tool_input:
                    self.num_hits += 1
                    logger.debug(f"ToolBenchCache: {self.num_hits/self.num_queries:.3f}({self.num_hits}/{self.num_queries})")
                    return query_response['response'], query_response['status_code']
        return None, None

def get_rapidapi_response(payload, api_customization=False):
    return ""

def toolbench_query_api(payload, service_url:str, toolbench_key:str = None, timeout:int = 15, rapidapi_key:str = None, api_customization:bool = False):
    if rapidapi_key:
        payload["rapidapi_key"] = rapidapi_key
        response = get_rapidapi_response(payload, api_customization=api_customization)
    else:
        response, status_code = get_response_from_cache(payload)
        if response is not None:
            return response, status_code

        time.sleep(2) # rate limit: 30 per minute
        headers = {"toolbench_key": toolbench_key}
        try:
            response = requests.post(service_url, json=payload, headers=headers, timeout=timeout)
        except Exception as e:
            response = json.dumps({"error": f"Timeout error...{e}", "response": ""}, ensure_ascii=False)
            cache_query_response(response, 5)
            return response, 5

    if response.status_code != 200:
        response = json.dumps({"error": f"request invalid, data error. status_code={response.status_code}", "response": ""}, ensure_ascii=False)
        cache_query_response(response, 12)
        return response, 12
    try:
        response = response.json()
    except:
        print(response)
        response = json.dumps({"error": f"request invalid, data error", "response": ""}, ensure_ascii=False)
        cache_query_response(response, 12)
        return response, 12

    # 1 Hallucinating function names
    # 4 means that the model decides to pruning by itself
    # 5 represents api call timeout
    # 6 for 404
    # 7 means not subscribed
    # 8 represents unauthorized
    # 9 represents too many requests
    # 10 stands for rate limit
    # 11 message contains "error" field
    # 12 error sending request
    if response["error"] == "API not working error...":
        status_code = 6
    elif response["error"] == "Unauthorized error...":
        status_code = 7
    elif response["error"] == "Unsubscribed error...":
        status_code = 8
    elif response["error"] == "Too many requests error...":
        status_code = 9
    elif response["error"] == "Rate limit per minute error...":
        print("Reach api calling limit per minute, sleeping...")
        time.sleep(10)
        status_code = 10
    elif response["error"] == "Message error...":
        status_code = 11
    else:
        status_code = 0

    response = json.dumps(response, ensure_ascii=False)
    cache_query_response(response, status_code)
    return response, status_code