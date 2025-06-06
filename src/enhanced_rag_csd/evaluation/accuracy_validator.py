"""
Accuracy validation framework for RAG systems.
This module provides tools for evaluating retrieval quality and answer accuracy.
"""

import os
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, ndcg_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

from enhanced_rag_csd.utils.logger import get_logger

logger = get_logger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval quality evaluation."""
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    ndcg_at_k: float
    map_score: float  # Mean Average Precision
    mrr_score: float  # Mean Reciprocal Rank
    hit_rate: float   # Percentage of queries with at least one relevant result
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AnswerMetrics:
    """Metrics for answer quality evaluation."""
    exact_match: float
    token_f1: float
    semantic_similarity: float
    rouge_1: float
    rouge_2: float
    rouge_l: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ValidationDataset:
    """Dataset for validation with ground truth."""
    queries: List[str]
    relevant_docs: List[List[str]]  # List of relevant doc IDs per query
    ground_truth_answers: Optional[List[str]] = None
    metadata: Optional[Dict] = None


class AccuracyValidator:
    """Validates RAG system accuracy with various metrics."""
    
    def __init__(self, stop_words: Optional[Set[str]] = None):
        self.stop_words = stop_words or set(stopwords.words('english'))
        self.results_cache = {}
    
    def evaluate_retrieval(self,
                         retrieved_docs: List[List[Dict[str, Any]]],
                         relevant_docs: List[List[str]],
                         k: int = 10) -> RetrievalMetrics:
        """Evaluate retrieval quality metrics."""
        if len(retrieved_docs) != len(relevant_docs):
            raise ValueError("Mismatched lengths for retrieved and relevant documents")
        
        precisions = []
        recalls = []
        f1_scores = []
        ndcg_scores = []
        aps = []  # Average precisions
        rrs = []  # Reciprocal ranks
        hits = []
        
        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            # Extract document IDs from retrieved results
            retrieved_ids = []
            for doc in retrieved[:k]:
                doc_id = doc.get('metadata', {}).get('doc_id', '')
                if not doc_id:
                    # Try to extract from source or other fields
                    doc_id = doc.get('metadata', {}).get('source', str(hash(doc['chunk'])))
                retrieved_ids.append(doc_id)
            
            # Calculate metrics for this query
            relevant_set = set(relevant)
            retrieved_set = set(retrieved_ids)
            
            # Precision and Recall
            true_positives = len(relevant_set & retrieved_set)
            precision = true_positives / len(retrieved_set) if retrieved_set else 0
            recall = true_positives / len(relevant_set) if relevant_set else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            
            # NDCG (Normalized Discounted Cumulative Gain)
            relevance_scores = [1 if doc_id in relevant_set else 0 for doc_id in retrieved_ids]
            ideal_scores = [1] * min(len(relevant_set), k) + [0] * (k - min(len(relevant_set), k))
            
            if sum(ideal_scores) > 0:
                ndcg = ndcg_score([ideal_scores], [relevance_scores])
            else:
                ndcg = 0
            ndcg_scores.append(ndcg)
            
            # Average Precision
            ap = self._calculate_average_precision(retrieved_ids, relevant_set)
            aps.append(ap)
            
            # Reciprocal Rank
            rr = 0
            for i, doc_id in enumerate(retrieved_ids):
                if doc_id in relevant_set:
                    rr = 1 / (i + 1)
                    break
            rrs.append(rr)
            
            # Hit Rate
            hits.append(1 if true_positives > 0 else 0)
        
        return RetrievalMetrics(
            precision_at_k=np.mean(precisions),
            recall_at_k=np.mean(recalls),
            f1_at_k=np.mean(f1_scores),
            ndcg_at_k=np.mean(ndcg_scores),
            map_score=np.mean(aps),
            mrr_score=np.mean(rrs),
            hit_rate=np.mean(hits)
        )
    
    def _calculate_average_precision(self, retrieved_ids: List[str], relevant_set: Set[str]) -> float:
        """Calculate average precision for a single query."""
        if not relevant_set:
            return 0
        
        num_relevant = 0
        sum_precision = 0
        
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                sum_precision += precision_at_i
        
        return sum_precision / len(relevant_set) if relevant_set else 0
    
    def evaluate_answers(self,
                        generated_answers: List[str],
                        ground_truth_answers: List[str]) -> AnswerMetrics:
        """Evaluate answer quality metrics."""
        if len(generated_answers) != len(ground_truth_answers):
            raise ValueError("Mismatched lengths for generated and ground truth answers")
        
        exact_matches = []
        token_f1_scores = []
        rouge_1_scores = []
        rouge_2_scores = []
        rouge_l_scores = []
        
        for generated, ground_truth in zip(generated_answers, ground_truth_answers):
            # Exact match
            exact_match = 1 if generated.strip().lower() == ground_truth.strip().lower() else 0
            exact_matches.append(exact_match)
            
            # Token-level F1
            token_f1 = self._calculate_token_f1(generated, ground_truth)
            token_f1_scores.append(token_f1)
            
            # ROUGE scores
            rouge_scores = self._calculate_rouge_scores(generated, ground_truth)
            rouge_1_scores.append(rouge_scores['rouge_1'])
            rouge_2_scores.append(rouge_scores['rouge_2'])
            rouge_l_scores.append(rouge_scores['rouge_l'])
        
        # Semantic similarity would require embeddings
        # For now, we'll use token overlap as a proxy
        semantic_similarities = token_f1_scores  # Simplified
        
        return AnswerMetrics(
            exact_match=np.mean(exact_matches),
            token_f1=np.mean(token_f1_scores),
            semantic_similarity=np.mean(semantic_similarities),
            rouge_1=np.mean(rouge_1_scores),
            rouge_2=np.mean(rouge_2_scores),
            rouge_l=np.mean(rouge_l_scores)
        )
    
    def _calculate_token_f1(self, predicted: str, ground_truth: str) -> float:
        """Calculate token-level F1 score."""
        pred_tokens = set(word_tokenize(predicted.lower()))
        truth_tokens = set(word_tokenize(ground_truth.lower()))
        
        # Remove stopwords
        pred_tokens = pred_tokens - self.stop_words
        truth_tokens = truth_tokens - self.stop_words
        
        if not pred_tokens and not truth_tokens:
            return 1.0
        if not pred_tokens or not truth_tokens:
            return 0.0
        
        common = pred_tokens & truth_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(truth_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_rouge_scores(self, predicted: str, ground_truth: str) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        pred_tokens = word_tokenize(predicted.lower())
        truth_tokens = word_tokenize(ground_truth.lower())
        
        # ROUGE-1 (unigram overlap)
        pred_unigrams = set(pred_tokens)
        truth_unigrams = set(truth_tokens)
        
        if truth_unigrams:
            rouge_1_recall = len(pred_unigrams & truth_unigrams) / len(truth_unigrams)
            rouge_1_precision = len(pred_unigrams & truth_unigrams) / len(pred_unigrams) if pred_unigrams else 0
            rouge_1_f1 = 2 * (rouge_1_precision * rouge_1_recall) / (rouge_1_precision + rouge_1_recall) \
                        if (rouge_1_precision + rouge_1_recall) > 0 else 0
        else:
            rouge_1_f1 = 0
        
        # ROUGE-2 (bigram overlap)
        pred_bigrams = set(zip(pred_tokens[:-1], pred_tokens[1:]))
        truth_bigrams = set(zip(truth_tokens[:-1], truth_tokens[1:]))
        
        if truth_bigrams:
            rouge_2_recall = len(pred_bigrams & truth_bigrams) / len(truth_bigrams)
            rouge_2_precision = len(pred_bigrams & truth_bigrams) / len(pred_bigrams) if pred_bigrams else 0
            rouge_2_f1 = 2 * (rouge_2_precision * rouge_2_recall) / (rouge_2_precision + rouge_2_recall) \
                        if (rouge_2_precision + rouge_2_recall) > 0 else 0
        else:
            rouge_2_f1 = 0
        
        # ROUGE-L (longest common subsequence)
        rouge_l_f1 = self._calculate_rouge_l(pred_tokens, truth_tokens)
        
        return {
            'rouge_1': rouge_1_f1,
            'rouge_2': rouge_2_f1,
            'rouge_l': rouge_l_f1
        }
    
    def _calculate_rouge_l(self, pred_tokens: List[str], truth_tokens: List[str]) -> float:
        """Calculate ROUGE-L using longest common subsequence."""
        if not pred_tokens or not truth_tokens:
            return 0.0
        
        # Dynamic programming for LCS
        m, n = len(pred_tokens), len(truth_tokens)
        lcs_table = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_tokens[i-1] == truth_tokens[j-1]:
                    lcs_table[i][j] = lcs_table[i-1][j-1] + 1
                else:
                    lcs_table[i][j] = max(lcs_table[i-1][j], lcs_table[i][j-1])
        
        lcs_length = lcs_table[m][n]
        
        if lcs_length == 0:
            return 0.0
        
        precision = lcs_length / len(pred_tokens)
        recall = lcs_length / len(truth_tokens)
        
        return 2 * (precision * recall) / (precision + recall)
    
    def create_validation_report(self,
                               retrieval_metrics: RetrievalMetrics,
                               answer_metrics: Optional[AnswerMetrics] = None,
                               system_name: str = "System") -> Dict[str, Any]:
        """Create a comprehensive validation report."""
        report = {
            "system_name": system_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "retrieval_metrics": retrieval_metrics.to_dict(),
            "overall_retrieval_score": self._calculate_overall_retrieval_score(retrieval_metrics)
        }
        
        if answer_metrics:
            report["answer_metrics"] = answer_metrics.to_dict()
            report["overall_answer_score"] = self._calculate_overall_answer_score(answer_metrics)
            report["combined_score"] = (report["overall_retrieval_score"] + 
                                      report["overall_answer_score"]) / 2
        
        return report
    
    def _calculate_overall_retrieval_score(self, metrics: RetrievalMetrics) -> float:
        """Calculate weighted overall retrieval score."""
        weights = {
            'precision_at_k': 0.2,
            'recall_at_k': 0.2,
            'ndcg_at_k': 0.25,
            'map_score': 0.25,
            'hit_rate': 0.1
        }
        
        score = (weights['precision_at_k'] * metrics.precision_at_k +
                weights['recall_at_k'] * metrics.recall_at_k +
                weights['ndcg_at_k'] * metrics.ndcg_at_k +
                weights['map_score'] * metrics.map_score +
                weights['hit_rate'] * metrics.hit_rate)
        
        return score
    
    def _calculate_overall_answer_score(self, metrics: AnswerMetrics) -> float:
        """Calculate weighted overall answer score."""
        weights = {
            'exact_match': 0.1,
            'token_f1': 0.2,
            'semantic_similarity': 0.3,
            'rouge_l': 0.4
        }
        
        score = (weights['exact_match'] * metrics.exact_match +
                weights['token_f1'] * metrics.token_f1 +
                weights['semantic_similarity'] * metrics.semantic_similarity +
                weights['rouge_l'] * metrics.rouge_l)
        
        return score


class DatasetLoader:
    """Loads standard datasets for validation."""
    
    @staticmethod
    def load_custom_dataset(file_path: str) -> ValidationDataset:
        """Load custom validation dataset from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return ValidationDataset(
            queries=data['queries'],
            relevant_docs=data['relevant_docs'],
            ground_truth_answers=data.get('ground_truth_answers'),
            metadata=data.get('metadata')
        )
    
    @staticmethod
    def create_sample_dataset() -> ValidationDataset:
        """Create a sample validation dataset for testing."""
        return ValidationDataset(
            queries=[
                "What is computational storage?",
                "How does RAG work?",
                "What are vector databases?",
                "Explain FAISS algorithm",
                "What is incremental indexing?"
            ],
            relevant_docs=[
                ["csd_intro.txt", "csd_overview.pdf"],
                ["rag_paper.pdf", "rag_tutorial.txt"],
                ["vector_db_guide.txt", "embeddings.pdf"],
                ["faiss_docs.txt", "similarity_search.pdf"],
                ["incremental_index.txt", "index_optimization.pdf"]
            ],
            ground_truth_answers=[
                "Computational storage brings computation closer to data storage devices.",
                "RAG combines retrieval with generation for better AI responses.",
                "Vector databases store and search high-dimensional embeddings efficiently.",
                "FAISS is a library for efficient similarity search and clustering.",
                "Incremental indexing updates search indices without full rebuilds."
            ],
            metadata={
                "dataset_name": "sample_validation",
                "version": "1.0"
            }
        )


class ValidationRunner:
    """Runs validation experiments across multiple systems."""
    
    def __init__(self, systems: Dict[str, Any], validator: AccuracyValidator):
        self.systems = systems
        self.validator = validator
        self.results = {}
    
    def run_validation(self,
                      dataset: ValidationDataset,
                      top_k: int = 10,
                      include_answer_eval: bool = False) -> Dict[str, Dict]:
        """Run validation across all systems."""
        logger.info(f"Running validation on {len(dataset.queries)} queries")
        
        for system_name, system in self.systems.items():
            logger.info(f"Evaluating {system_name}...")
            
            try:
                # Retrieve documents for all queries
                retrieved_docs = []
                generated_answers = []
                
                for query in dataset.queries:
                    result = system.query(query, top_k=top_k)
                    retrieved_docs.append(result.get('retrieved_docs', []))
                    
                    if include_answer_eval:
                        # Extract answer from augmented query (simplified)
                        augmented = result.get('augmented_query', '')
                        # In real scenario, this would be passed to an LLM
                        generated_answers.append(augmented.split('\n')[0])
                
                # Evaluate retrieval
                retrieval_metrics = self.validator.evaluate_retrieval(
                    retrieved_docs,
                    dataset.relevant_docs,
                    k=top_k
                )
                
                # Evaluate answers if ground truth available
                answer_metrics = None
                if include_answer_eval and dataset.ground_truth_answers:
                    answer_metrics = self.validator.evaluate_answers(
                        generated_answers,
                        dataset.ground_truth_answers
                    )
                
                # Create report
                report = self.validator.create_validation_report(
                    retrieval_metrics,
                    answer_metrics,
                    system_name
                )
                
                self.results[system_name] = report
                
            except Exception as e:
                logger.error(f"Error evaluating {system_name}: {e}")
                self.results[system_name] = {"error": str(e)}
        
        return self.results
    
    def save_results(self, output_path: str) -> None:
        """Save validation results to file."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Validation results saved to {output_path}")
    
    def print_summary(self) -> None:
        """Print summary of validation results."""
        print("\n" + "="*80)
        print("VALIDATION RESULTS SUMMARY")
        print("="*80)
        
        # Sort by overall score
        sorted_systems = sorted(
            [(name, res) for name, res in self.results.items() if 'error' not in res],
            key=lambda x: x[1].get('overall_retrieval_score', 0),
            reverse=True
        )
        
        print("\nRetrieval Performance:")
        print("-"*50)
        print(f"{'System':<20} {'Precision@k':<12} {'Recall@k':<12} {'NDCG@k':<12} {'MAP':<12}")
        print("-"*50)
        
        for name, result in sorted_systems:
            metrics = result['retrieval_metrics']
            print(f"{name:<20} {metrics['precision_at_k']:<12.3f} "
                  f"{metrics['recall_at_k']:<12.3f} {metrics['ndcg_at_k']:<12.3f} "
                  f"{metrics['map_score']:<12.3f}")
        
        if any('answer_metrics' in res for _, res in sorted_systems):
            print("\nAnswer Quality:")
            print("-"*50)
            print(f"{'System':<20} {'Token F1':<12} {'ROUGE-L':<12} {'Semantic Sim':<12}")
            print("-"*50)
            
            for name, result in sorted_systems:
                if 'answer_metrics' in result:
                    metrics = result['answer_metrics']
                    print(f"{name:<20} {metrics['token_f1']:<12.3f} "
                          f"{metrics['rouge_l']:<12.3f} {metrics['semantic_similarity']:<12.3f}")