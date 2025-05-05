"""
Factuality Reward Functions

This module provides reward functions for evaluating factual accuracy of responses:
- FactualityReward: Evaluates factual accuracy by cross-referencing with trusted sources
"""

import re
import logging
import numpy as np
from typing import List, Optional, Union, Dict, Any, Tuple

from .base import BaseReward

# Configure logging
logger = logging.getLogger(__name__)


class FactualityReward(BaseReward):
    """
    Reward function that evaluates factual accuracy of responses.
    
    This reward cross-references outputs with trusted sources to minimize hallucinations.
    """
    
    def __init__(self, 
                 reference_texts: Optional[List[str]] = None,
                 use_embeddings: bool = False,
                 embedding_model = None,
                 check_contradictions: bool = True,
                 weight: float = 1.0):
        """
        Initialize the factuality reward function.
        
        Args:
            reference_texts: List of reference texts containing factual information
            use_embeddings: Whether to use embeddings for semantic similarity (default: False)
            embedding_model: Model to use for embeddings if use_embeddings is True
            check_contradictions: Whether to check for contradictions (default: True)
            weight: Weight of this reward when combined with others (default: 1.0)
            
        Note:
            If use_embeddings is True, embedding_model must be provided.
        """
        super().__init__(name="factuality", weight=weight)
        self.reference_texts = reference_texts or []
        self.use_embeddings = use_embeddings
        self.embedding_model = embedding_model
        self.check_contradictions = check_contradictions
        
        if use_embeddings and embedding_model is None:
            raise ValueError("Embedding model must be provided when use_embeddings is True")
        
        # Precompute reference embeddings if using embeddings
        self.reference_embeddings = None
        if self.use_embeddings and self.embedding_model and self.reference_texts:
            self.reference_embeddings = self._compute_embeddings(self.reference_texts)
    
    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute embeddings for a list of texts.
        
        Args:
            texts: List of texts to compute embeddings for
            
        Returns:
            Array of embeddings
        """
        try:
            return self.embedding_model.encode(texts)
        except Exception as e:
            logger.error(f"Error computing embeddings: {e}")
            return np.zeros((len(texts), 768))  # Default embedding size
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        if self.use_embeddings and self.embedding_model:
            # Compute embeddings
            emb1 = self.embedding_model.encode([text1])[0]
            emb2 = self.embedding_model.encode([text2])[0]
            
            # Compute cosine similarity
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 > 0 and norm2 > 0:
                return np.dot(emb1, emb2) / (norm1 * norm2)
            return 0.0
        else:
            # Use simple text-based similarity
            # Tokenize and compute Jaccard similarity
            tokens1 = set(re.findall(r'\b\w+\b', text1.lower()))
            tokens2 = set(re.findall(r'\b\w+\b', text2.lower()))
            
            if not tokens1 or not tokens2:
                return 0.0
            
            intersection = len(tokens1.intersection(tokens2))
            union = len(tokens1.union(tokens2))
            
            return intersection / union if union > 0 else 0.0
    
    def _extract_facts(self, text: str) -> List[str]:
        """
        Extract factual statements from text.
        
        Args:
            text: Text to extract facts from
            
        Returns:
            List of extracted factual statements
        """
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter for likely factual statements
        facts = []
        for sentence in sentences:
            # Skip questions, commands, and very short sentences
            if re.search(r'\?$|!$', sentence) or len(sentence.split()) < 5:
                continue
            
            # Look for factual indicators
            if (re.search(r'\bis\b|\bwas\b|\bwere\b|\bare\b|\bhave\b|\bhas\b', sentence) or
                re.search(r'\bin\b|\bat\b|\bon\b|\bduring\b|\bfrom\b', sentence) or
                re.search(r'\d{4}|\d+%|\d+\s+\w+', sentence)):  # Years, percentages, quantities
                facts.append(sentence)
        
        return facts
    
    def _check_fact_support(self, fact: str, references: List[str]) -> float:
        """
        Check how well a fact is supported by reference texts.
        
        Args:
            fact: Factual statement to check
            references: List of reference texts
            
        Returns:
            Support score between 0 and 1
        """
        max_similarity = 0.0
        
        for ref in references:
            # Split reference into sentences
            ref_sentences = re.split(r'(?<=[.!?])\s+', ref)
            
            for ref_sentence in ref_sentences:
                similarity = self._compute_similarity(fact, ref_sentence)
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _detect_contradictions(self, facts: List[str], references: List[str]) -> float:
        """
        Detect contradictions between facts and references.
        
        Args:
            facts: List of factual statements
            references: List of reference texts
            
        Returns:
            Contradiction penalty between 0 and 1 (0 = many contradictions, 1 = no contradictions)
        """
        # This is a simplified implementation that looks for negation patterns
        # A more sophisticated approach would use natural language inference
        
        contradiction_count = 0
        total_comparisons = 0
        
        for fact in facts:
            fact_lower = fact.lower()
            
            # Extract key entities and predicates from the fact
            entities = set(re.findall(r'\b[A-Z][a-z]+\b', fact))  # Proper nouns
            predicates = set(re.findall(r'\b(is|was|were|are|have|has)\b\s+\w+', fact_lower))
            
            # Check for contradictions in references
            for ref in references:
                ref_lower = ref.lower()
                
                # Check if the reference mentions the same entities
                if any(entity.lower() in ref_lower for entity in entities):
                    # Look for negated versions of the predicates
                    for predicate in predicates:
                        negated = re.sub(r'\b(is|was|were|are|have|has)\b', r'is not|was not|were not|are not|have not|has not', predicate)
                        if re.search(negated, ref_lower):
                            contradiction_count += 1
                    
                    total_comparisons += 1
        
        if total_comparisons == 0:
            return 1.0  # No contradictions found
        
        # Calculate contradiction penalty
        contradiction_ratio = contradiction_count / total_comparisons
        return 1.0 - contradiction_ratio
    
    def compute_reward(self, 
                       response: Union[str, List[str]], 
                       prompt: Optional[Union[str, List[str]]] = None,
                       reference: Optional[Union[str, List[str]]] = None,
                       **kwargs) -> Union[float, List[float]]:
        """
        Compute the factuality reward for a response.
        
        Args:
            response: Model response(s) to evaluate
            prompt: Not used in this reward function
            reference: Optional reference text(s) containing factual information
            **kwargs: Additional arguments
            
        Returns:
            Normalized reward score(s) between 0 and 1
        """
        responses = self._ensure_list(response)
        
        # Combine provided references with pre-configured reference texts
        all_references = list(self.reference_texts)
        if reference is not None:
            all_references.extend(self._ensure_list(reference))
        
        if not all_references:
            logger.warning("No reference texts provided for factuality reward. Returning neutral score.")
            return [0.5] * len(responses)
        
        rewards = []
        for resp in responses:
            # Extract factual statements from the response
            facts = self._extract_facts(resp)
            
            if not facts:
                rewards.append(0.5)  # Neutral score if no facts found
                continue
            
            # Check fact support
            support_scores = [self._check_fact_support(fact, all_references) for fact in facts]
            avg_support = sum(support_scores) / len(support_scores) if support_scores else 0.5
            
            # Check for contradictions if enabled
            contradiction_penalty = 1.0
            if self.check_contradictions:
                contradiction_penalty = self._detect_contradictions(facts, all_references)
            
            # Calculate final score
            # Higher weight on contradiction penalty to discourage hallucinations
            final_score = 0.7 * avg_support + 0.3 * contradiction_penalty
            
            rewards.append(self._normalize_score(final_score))
        
        return rewards[0] if len(rewards) == 1 else rewards