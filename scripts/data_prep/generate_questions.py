#!/usr/bin/env python
"""
Generate comprehensive question sets from downloaded documents for RAG evaluation.
Creates questions of different types and difficulty levels for thorough testing.
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class QuestionGenerator:
    """Generate questions from documents using rule-based and template approaches."""
    
    def __init__(self):
        self.question_templates = {
            "factual": [
                "What is {entity}?",
                "Define {entity}.",
                "Explain {entity}.",
                "What does {entity} mean?",
                "How would you describe {entity}?"
            ],
            "comparison": [
                "What is the difference between {entity1} and {entity2}?", 
                "How do {entity1} and {entity2} compare?",
                "Compare {entity1} with {entity2}.",
                "What are the similarities and differences between {entity1} and {entity2}?"
            ],
            "application": [
                "How is {entity} used in {domain}?",
                "What are the applications of {entity}?",
                "Where is {entity} applied?",
                "How can {entity} be utilized?",
                "What are practical uses of {entity}?"
            ],
            "causal": [
                "Why is {entity} important?",
                "What are the benefits of {entity}?",
                "What problems does {entity} solve?",
                "What causes {entity}?",
                "What are the advantages of {entity}?"
            ],
            "procedural": [
                "How does {entity} work?",
                "What is the process of {entity}?",
                "How is {entity} implemented?",
                "What are the steps in {entity}?",
                "How do you {action} {entity}?"
            ]
        }
        
        self.domains = [
            "artificial intelligence", "machine learning", "computer science",
            "research", "technology", "healthcare", "finance", "industry"
        ]
        
        self.actions = [
            "implement", "use", "apply", "develop", "design", "optimize", "evaluate"
        ]

    def extract_entities(self, text: str) -> List[str]:
        """Extract key entities/concepts from text."""
        # Simple entity extraction using capitalized words and technical terms
        entities = []
        
        # Find capitalized phrases (potential proper nouns/concepts)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend(capitalized)
        
        # Find technical terms (words ending in -ing, -tion, -ity, etc.)
        technical_patterns = [
            r'\b\w+(?:ing|tion|sion|ity|ism|ogy|ics)\b',
            r'\b\w+(?:algorithm|system|method|model|approach|technique)\b'
        ]
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.extend(matches)
        
        # Clean and filter entities
        entities = [e.strip() for e in entities if len(e) > 3 and len(e) < 50]
        entities = list(set(entities))  # Remove duplicates
        
        return entities[:20]  # Limit to top 20 entities

    def extract_sentences_with_entities(self, text: str, entities: List[str]) -> Dict[str, List[str]]:
        """Extract sentences containing specific entities."""
        sentences = re.split(r'[.!?]+', text)
        entity_sentences = defaultdict(list)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            for entity in entities:
                if entity.lower() in sentence.lower():
                    entity_sentences[entity].append(sentence)
        
        return entity_sentences

    def generate_factual_questions(self, entities: List[str], text: str) -> List[Dict[str, Any]]:
        """Generate factual questions about entities."""
        questions = []
        
        for entity in entities[:10]:  # Limit to top 10 entities
            for template in self.question_templates["factual"]:
                question = template.format(entity=entity)
                
                # Find relevant context sentences
                context_sentences = []
                sentences = re.split(r'[.!?]+', text)
                for sentence in sentences:
                    if entity.lower() in sentence.lower() and len(sentence.strip()) > 20:
                        context_sentences.append(sentence.strip())
                
                if context_sentences:
                    questions.append({
                        "question": question,
                        "type": "factual",
                        "entity": entity,
                        "context": context_sentences[:3],  # Top 3 relevant sentences
                        "difficulty": "easy"
                    })
        
        return questions

    def generate_comparison_questions(self, entities: List[str]) -> List[Dict[str, Any]]:
        """Generate comparison questions between entities."""
        questions = []
        
        # Pair entities for comparison
        for i in range(len(entities)):
            for j in range(i+1, min(i+3, len(entities))):  # Compare with next 2 entities
                entity1, entity2 = entities[i], entities[j]
                
                for template in self.question_templates["comparison"]:
                    question = template.format(entity1=entity1, entity2=entity2)
                    questions.append({
                        "question": question,
                        "type": "comparison", 
                        "entities": [entity1, entity2],
                        "difficulty": "medium"
                    })
        
        return questions[:10]  # Limit comparisons

    def generate_application_questions(self, entities: List[str]) -> List[Dict[str, Any]]:
        """Generate application-focused questions."""
        questions = []
        
        for entity in entities[:8]:
            for domain in random.sample(self.domains, 3):  # Random domains
                for template in self.question_templates["application"]:
                    if "{domain}" in template:
                        question = template.format(entity=entity, domain=domain)
                    else:
                        question = template.format(entity=entity)
                    
                    questions.append({
                        "question": question,
                        "type": "application",
                        "entity": entity,
                        "domain": domain,
                        "difficulty": "medium"
                    })
        
        return questions[:15]  # Limit applications

    def generate_causal_questions(self, entities: List[str]) -> List[Dict[str, Any]]:
        """Generate causal/reasoning questions.""" 
        questions = []
        
        for entity in entities[:8]:
            for template in self.question_templates["causal"]:
                question = template.format(entity=entity)
                questions.append({
                    "question": question,
                    "type": "causal",
                    "entity": entity,
                    "difficulty": "hard"
                })
        
        return questions

    def generate_procedural_questions(self, entities: List[str]) -> List[Dict[str, Any]]:
        """Generate procedural/how-to questions."""
        questions = []
        
        for entity in entities[:6]:
            for action in random.sample(self.actions, 2):
                for template in self.question_templates["procedural"]:
                    if "{action}" in template:
                        question = template.format(entity=entity, action=action)
                    else:
                        question = template.format(entity=entity)
                    
                    questions.append({
                        "question": question,
                        "type": "procedural",
                        "entity": entity,
                        "action": action,
                        "difficulty": "hard"
                    })
        
        return questions[:12]  # Limit procedural

    def generate_questions_from_document(self, doc_path: str, doc_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate comprehensive questions from a single document."""
        
        try:
            with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading {doc_path}: {e}")
            return []
        
        # Extract entities from the document
        entities = self.extract_entities(text)
        if not entities:
            return []
        
        print(f"  Found {len(entities)} entities: {entities[:5]}...")
        
        # Generate different types of questions
        all_questions = []
        
        # Factual questions (easy)
        factual_q = self.generate_factual_questions(entities, text)
        all_questions.extend(factual_q)
        
        # Comparison questions (medium)
        comparison_q = self.generate_comparison_questions(entities)
        all_questions.extend(comparison_q)
        
        # Application questions (medium)
        application_q = self.generate_application_questions(entities)
        all_questions.extend(application_q)
        
        # Causal questions (hard)
        causal_q = self.generate_causal_questions(entities)
        all_questions.extend(causal_q)
        
        # Procedural questions (hard)
        procedural_q = self.generate_procedural_questions(entities)
        all_questions.extend(procedural_q)
        
        # Add document metadata to all questions
        for question in all_questions:
            question.update({
                "source_document": doc_path,
                "document_title": doc_metadata.get("title", ""),
                "document_category": doc_metadata.get("category", ""),
                "document_type": "paper" if "papers" in doc_path else "literature" if "literature" in doc_path else "wikipedia"
            })
        
        return all_questions

def load_document_metadata() -> Dict[str, Any]:
    """Load document metadata."""
    metadata_path = "data/documents/metadata.json"
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    return {"documents": []}

def generate_all_questions() -> Dict[str, Any]:
    """Generate questions from all downloaded documents."""
    
    print("🤔 Generating Questions from Documents")
    print("=" * 50)
    
    # Load document metadata
    metadata = load_document_metadata()
    documents = metadata.get("documents", [])
    
    # Initialize question generator
    generator = QuestionGenerator()
    
    all_questions = []
    question_stats = {
        "total_questions": 0,
        "by_type": defaultdict(int),
        "by_difficulty": defaultdict(int),
        "by_document_type": defaultdict(int),
        "by_category": defaultdict(int)
    }
    
    # Process each document
    for doc in documents:
        if "local_path" not in doc:
            continue
            
        doc_path = doc["local_path"]
        if not os.path.exists(doc_path):
            continue
        
        print(f"\n📄 Processing: {doc.get('title', doc_path)}")
        
        # Generate questions for this document
        doc_questions = generator.generate_questions_from_document(doc_path, doc)
        
        print(f"  Generated {len(doc_questions)} questions")
        
        # Update statistics
        for q in doc_questions:
            question_stats["by_type"][q["type"]] += 1
            question_stats["by_difficulty"][q["difficulty"]] += 1
            question_stats["by_document_type"][q["document_type"]] += 1
            question_stats["by_category"][q["document_category"]] += 1
        
        all_questions.extend(doc_questions)
    
    # Add Wikipedia documents
    wiki_docs = [
        {"title": "Artificial Intelligence", "category": "AI", "path": "data/documents/wikipedia/Artificial_Intelligence.txt"},
        {"title": "Machine Learning", "category": "ML", "path": "data/documents/wikipedia/Machine_Learning.txt"},
        {"title": "Information Retrieval", "category": "IR", "path": "data/documents/wikipedia/Information_Retrieval.txt"}
    ]
    
    for doc in wiki_docs:
        if os.path.exists(doc["path"]):
            print(f"\n📖 Processing Wikipedia: {doc['title']}")
            doc_questions = generator.generate_questions_from_document(doc["path"], doc)
            print(f"  Generated {len(doc_questions)} questions")
            
            for q in doc_questions:
                question_stats["by_type"][q["type"]] += 1
                question_stats["by_difficulty"][q["difficulty"]] += 1
                question_stats["by_document_type"]["wikipedia"] += 1
                question_stats["by_category"][q["document_category"]] += 1
            
            all_questions.extend(doc_questions)
    
    # Shuffle questions for variety
    random.shuffle(all_questions)
    
    # Update final statistics
    question_stats["total_questions"] = len(all_questions)
    
    # Create question sets by difficulty
    question_sets = {
        "all_questions": all_questions,
        "easy_questions": [q for q in all_questions if q["difficulty"] == "easy"],
        "medium_questions": [q for q in all_questions if q["difficulty"] == "medium"],
        "hard_questions": [q for q in all_questions if q["difficulty"] == "hard"],
        "factual_questions": [q for q in all_questions if q["type"] == "factual"],
        "comparison_questions": [q for q in all_questions if q["type"] == "comparison"],
        "application_questions": [q for q in all_questions if q["type"] == "application"],
        "statistics": dict(question_stats)
    }
    
    return question_sets

def main():
    """Main function to generate all questions."""
    
    # Generate questions
    question_sets = generate_all_questions()
    
    # Save question sets
    output_path = "data/questions/generated_questions.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(question_sets, f, indent=2, ensure_ascii=False)
    
    # Print summary
    stats = question_sets["statistics"]
    
    print(f"\n✅ Question Generation Complete!")
    print(f"   Total questions: {stats['total_questions']}")
    print(f"   Easy: {stats['by_difficulty']['easy']}")
    print(f"   Medium: {stats['by_difficulty']['medium']}")
    print(f"   Hard: {stats['by_difficulty']['hard']}")
    print(f"\n   By type:")
    for qtype, count in stats['by_type'].items():
        print(f"     {qtype}: {count}")
    print(f"\n   Saved to: {output_path}")
    
    # Save sample questions for manual review
    sample_questions = random.sample(question_sets["all_questions"], min(20, len(question_sets["all_questions"])))
    
    sample_path = "data/questions/sample_questions.json"
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(sample_questions, f, indent=2, ensure_ascii=False)
    
    print(f"   Sample questions saved to: {sample_path}")

if __name__ == "__main__":
    main()