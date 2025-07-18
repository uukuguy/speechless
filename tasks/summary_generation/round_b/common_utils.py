"""Utility functions for handling paper chunks, citations, and caching.

This module provides:
- ReviewType class with descriptions and metadata
- Data classes for paper chunks and citations
- Caching decorator with improved error handling
- Custom JSON encoders/decoders for numpy data
"""

import os
import json
import pickle
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union
import re
import numpy as np
from pydantic import validate_arguments

class ReviewType(Enum):
    """Enumeration of review types with metadata."""
    CONCEPT = "concept"  # Technical concept research
    STATUS = "status"    # Research status
    COMPARISON = "comparison"  # Method comparison
    TIMELINE = "timeline"   # Technical timeline

review_type_descriptions = {
    ReviewType.CONCEPT: "技术概念调研",
    ReviewType.STATUS: "研究现状",
    ReviewType.COMPARISON: "方法对比",
    ReviewType.TIMELINE: "技术脉络"
}
@dataclass
class PaperChunk:
    """Represents a chunk of text from a research paper."""
    paper_id: str
    title: str
    chunk_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Paper:
    """Represents a research paper with metadata."""
    paper_id: str
    title: str
    content: str = ""
    chunks: Dict[str, PaperChunk] = field(default_factory=dict)

    def add_chunk(self, chunk: PaperChunk) -> None:
        """Add a chunk to the paper."""
        self.chunks[chunk.chunk_id] = chunk

    def post_process(self) -> None:
        """
        Post-process paper content after adding chunks.
        Merge chunks into the main content ordered by chunk ID.
        """
        self.content = "\n".join(chunk.content for chunk in sorted(self.chunks.values(), key=lambda x: x.chunk_id))

@dataclass
class Citation:
    """Represents a citation reference within a paper."""
    paper_id: str
    chunk_id: str
    content: str
    title: str
    context: Optional[str] = None

@validate_arguments
def kb_chunk_to_paper_chunk(kb_chunk: Dict[str, Any]) -> PaperChunk:
    """Convert a knowledge base chunk to a PaperChunk object.
    
    Args:
        kb_chunk: Dictionary containing paper chunk data
        
    Returns:
        PaperChunk: Converted paper chunk object
        
    Raises:
        ValueError: If required fields are missing
    """
    chunk_entity = kb_chunk.get('entity', kb_chunk)
    try:
        return PaperChunk(
            paper_id=chunk_entity["paper_id"],
            title=chunk_entity["paper_title"],
            chunk_id=chunk_entity["chunk_id"],
            content=chunk_entity["chunk_text"]
        )
    except KeyError as e:
        raise ValueError(f"Missing required field in kb_chunk: {str(e)}")

class CacheManager:
    """Decorator for caching function results with improved error handling."""
    
    @staticmethod
    def validate_cache_file(cache_file: str) -> None:
        """Validate cache file path and format."""
        if not cache_file:
            raise ValueError("Cache file path must be provided")
            
        if not any(cache_file.endswith(ext) for ext in (".pkl", ".json", "jsonl")):
            raise ValueError("Cache file must be .pkl or .json")

    @classmethod
    def cache_or_rebuild(cls, cache_file: str):
        """Decorator to cache function results or rebuild if cache is not available."""
        cls.validate_cache_file(cache_file)
        is_pickle = cache_file.endswith(".pkl")
        is_json = cache_file.endswith(".json")
        is_jsonl = cache_file.endswith(".jsonl")

        def decorator(func):
            def wrapper(*args, **kwargs):
                if os.path.exists(cache_file):
                    try:
                        if is_pickle:
                            return pickle.load(open(cache_file, 'rb'))
                        elif is_json:
                            return json.load(open(cache_file, 'r', encoding='utf-8'), cls=LazyDecoder)
                        elif is_jsonl:
                            return [json.loads(line, cls=LazyDecoder) for line in open(cache_file, 'r', encoding='utf-8')]
                    except (pickle.PickleError, json.JSONDecodeError) as e:
                        os.remove(cache_file)  # Remove corrupted cache

                result = func(*args, **kwargs)
                if is_pickle:
                    pickle.dump(result, open(cache_file, 'wb'))
                elif is_json:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2, cls=NpEncoder)
                elif is_jsonl:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        for item in result:
                            json.dump(item, f, ensure_ascii=False, cls=NpEncoder)
                            f.write('\n')
                return result
            return wrapper
        return decorator

class LazyDecoder(json.JSONDecoder):
    """Custom JSON decoder to handle specific edge cases in JSON strings."""
    
    REGEX_REPLACEMENTS = [
        (re.compile(r'([^\\])\\([^\\])'), r'\1\\\\\2'),  # Fix single backslashes
        (re.compile(r',(\s*])'), r'\1'),  # Remove trailing commas
    ]

    def decode(self, s: str, **kwargs) -> Any:
        """Decode JSON string with preprocessing."""
        for regex, replacement in self.REGEX_REPLACEMENTS:
            s = regex.sub(replacement, s)
        return super().decode(s, **kwargs)

class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy data types."""
    
    def default(self, obj: Any) -> Union[int, float, list, Any]:
        """Convert numpy types to JSON-compatible types."""
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        return super().default(obj)

# Alias for backward compatibility
cache_or_rebuild = CacheManager.cache_or_rebuild
