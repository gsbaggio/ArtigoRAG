from typing import List, Dict, Any, Tuple
import os
import json
import random
from datetime import datetime

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv

load_dotenv()

class CodeContestsRAG:
    def __init__(self, train_path: str, test_path: str):
        """
        Initialize the CodeContests RAG system.
        
        Args:
            train_path: Path to codecontests_train.json
            test_path: Path to codecontests_test.json
        """
        self.train_path = train_path
        self.test_path = test_path
        
        # Load datasets
        self.train_data = self._load_jsonl(train_path)
        self.test_data = self._load_jsonl(test_path)
        
        print(f"Loaded {len(self.train_data)} training problems")
        print(f"Loaded {len(self.test_data)} test problems")
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Build FAISS index for training data
        self._build_indices()
        
    def _load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def _build_indices(self):
        """Build FAISS index using descriptions from training data."""
        descriptions = [problem['description'] for problem in self.train_data]
        
        print("Encoding training descriptions...")
        self.train_embeddings = self.embedding_model.encode(descriptions, show_progress_bar=True)
        
        # Build FAISS index
        dimension = self.train_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(self.train_embeddings).astype('float32'))
        
        print(f"Built FAISS index with {len(descriptions)} training problems")
    
    def retrieve_similar_questions(self, test_description: str, top_k: int = 3, 
                                  min_threshold: float = 0.7,
                                  use_adaptive_threshold: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve similar questions from training set based on description similarity.
        """
        # Encode test question description
        query_embedding = self.embedding_model.encode([test_description])
        
        # Search for similar problems
        candidate_k = min(top_k * 3, len(self.train_data))
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), candidate_k
        )
        
        # Convert L2 distances to cosine similarities
        # For normalized embeddings: cosine_sim = 1 - (L2_dist^2 / 2)
        similarities = [1 - (dist/2) for dist in distances[0]]
        similarities = [max(0, min(s, 1)) for s in similarities]
        
        # Adaptive threshold
        threshold = min_threshold  
        if use_adaptive_threshold and similarities:
            max_similarity = similarities[0]
            if max_similarity > 0.85:
                threshold = max(min_threshold, 0.8 * max_similarity)
            elif max_similarity > 0.75:
                threshold = max(min_threshold, 0.7 * max_similarity)
            else:
                threshold = max(min_threshold, 0.85 * max_similarity)
        
        # Filter by threshold and prepare results
        similar_questions = []
        for i, idx in enumerate(indices[0]):
            similarity_score = similarities[i]
            
            if similarity_score >= threshold:
                train_problem = self.train_data[idx]
                # Get only the first solution
                first_solution = ""
                if 'solutions' in train_problem and train_problem['solutions']:
                    if isinstance(train_problem['solutions'], dict) and 'solution' in train_problem['solutions']:
                        solutions_list = train_problem['solutions']['solution']
                        if solutions_list and len(solutions_list) > 0:
                            first_solution = solutions_list[0]
                    elif isinstance(train_problem['solutions'], list) and len(train_problem['solutions']) > 0:
                        first_solution = train_problem['solutions'][0]
                
                similar_questions.append({
                    'name': train_problem['name'],
                    'cf_tags': train_problem['cf_tags'],
                    'description': train_problem['description'],
                    'difficulty': train_problem['difficulty'],
                    'solutions': first_solution,
                    'similarity_score': float(similarity_score)  # Convert to Python float
                })
        
        return similar_questions[:top_k]
    
    def retrieve_random_questions(self) -> List[Dict[str, Any]]:
        """Retrieve 1-3 random questions from training set."""
        num_examples = random.choice([1, 2, 3])
        max_examples = min(num_examples, len(self.train_data))
        
        random_indices = random.sample(range(len(self.train_data)), max_examples)
        
        random_questions = []
        for idx in random_indices:
            train_problem = self.train_data[idx]
            # Get only the first solution
            first_solution = ""
            if 'solutions' in train_problem and train_problem['solutions']:
                if isinstance(train_problem['solutions'], dict) and 'solution' in train_problem['solutions']:
                    solutions_list = train_problem['solutions']['solution']
                    if solutions_list and len(solutions_list) > 0:
                        first_solution = solutions_list[0]
                elif isinstance(train_problem['solutions'], list) and len(train_problem['solutions']) > 0:
                    first_solution = train_problem['solutions'][0]
            
            random_questions.append({
                'name': train_problem['name'],
                'cf_tags': train_problem['cf_tags'],
                'description': train_problem['description'],
                'difficulty': train_problem['difficulty'],
                'solutions': first_solution
            })
        
        return random_questions
    
    def _build_context_with_rag(self, test_problem: Dict[str, Any], 
                               similar_questions: List[Dict[str, Any]]) -> str:
        """Build context with RAG for a test problem."""
        context = f"CURRENT QUESTION:\n{test_problem['description']}\n\n"
        
        if similar_questions:
            context += "SIMILAR QUESTIONS:\n"
            for i, q in enumerate(similar_questions):
                context += f"Similar question {i+1} (Similarity score: {q['similarity_score']:.2f}):\n"
                context += f"Name: {q['name']}\n"
                context += f"Tags: {q['cf_tags']}\n"
                context += f"Description: {q['description']}\n"
                context += f"Difficulty: {q['difficulty']}\n"
                context += f"Solutions: {q['solutions']}\n\n"
        else:
            context += "No sufficiently similar questions found in the database.\n\n"
        
        context += """
INSTRUCTIONS:
Solve the current question.
Provide:
1. A complete and efficient code solution, optimized for both time and space complexity.
2. A detailed explanation of the solution, including:
   - The intuition behind the approach;
   - Time and space complexity;
   - Important considerations about the algorithm.
3. If the language has classes, implement in 'Solution' class. Any language is accepted.
"""
        
        if similar_questions:
            context += "4. Use the similar questions as references to improve the solution, but only if they are relevant.\n"
        else:
            context += "4. Solve the problem from first principles as no similar questions were found.\n"
        
        context += "5. Don't use any external libraries. Don't need to import any libraries.\n"
        
        return context
    
    def _build_context_without_rag(self, test_problem: Dict[str, Any]) -> str:
        """Build context without RAG for a test problem."""
        context = f"CURRENT QUESTION:\n{test_problem['description']}\n\n"
        
        context += """
INSTRUCTIONS:
Solve the current question.
Provide:
1. A complete and efficient code solution, optimized for both time and space complexity.
2. A detailed explanation of the solution, including:
   - The intuition behind the approach;
   - Time and space complexity;
   - Important considerations about the algorithm.
3. If the language has classes, implement in 'Solution' class. Any language is accepted.
4. Don't use any external libraries. Don't need to import any libraries.
"""
        
        return context
    
    def _build_context_with_random(self, test_problem: Dict[str, Any], 
                                  random_questions: List[Dict[str, Any]]) -> str:
        """Build context with random examples for a test problem."""
        context = f"CURRENT QUESTION:\n{test_problem['description']}\n\n"
        
        if random_questions:
            context += "EXAMPLE QUESTIONS:\n"
            for i, q in enumerate(random_questions):
                context += f"Example question {i+1}:\n"
                context += f"Name: {q['name']}\n"
                context += f"Tags: {q['cf_tags']}\n"
                context += f"Description: {q['description']}\n"
                context += f"Difficulty: {q['difficulty']}\n"
                context += f"Solutions: {q['solutions']}\n\n"
        else:
            context += "No example questions available.\n\n"
        
        context += """
INSTRUCTIONS:
Solve the current question.
Provide:
1. A complete and efficient code solution, optimized for both time and space complexity.
2. A detailed explanation of the solution, including:
   - The intuition behind the approach;
   - Time and space complexity;
   - Important considerations about the algorithm.
3. If the language has classes, implement in 'Solution' class. Any language is accepted.
"""
        
        if random_questions:
            context += "4. You can use the example questions as general references for coding patterns and structure, but solve the current problem independently.\n"
        else:
            context += "4. Solve the problem from first principles.\n"
        
        context += "5. Don't use any external libraries. Don't need to import any libraries.\n"
        
        return context
    
    def _safe_filename(self, name: str) -> str:
        """Convert problem name to safe filename."""
        # Remove or replace characters that are problematic for filenames
        safe_chars = []
        for char in name:
            if char.isalnum() or char in ' -_()[]{}':
                safe_chars.append(char)
            else:
                safe_chars.append('_')
        return ''.join(safe_chars).strip()
    
    def process_test_problems(self, output_base_dir: str = "data/CodeContest") -> Dict[str, Any]:
        """
        Process all test problems and create directories with RAG outputs.
        """
        if not os.path.exists(output_base_dir):
            os.makedirs(output_base_dir)
        
        processed_count = 0
        skipped_count = 0
        results = []
        
        for i, test_problem in enumerate(self.test_data):
            print(f"Processing test problem {i+1}/{len(self.test_data)}: {test_problem['name']}")
            
            # Retrieve similar questions
            similar_questions = self.retrieve_similar_questions(
                test_problem['description'],
                top_k=3,
                min_threshold=0.7,
                use_adaptive_threshold=True
            )
            
            # Skip if no similar questions found
            if not similar_questions:
                print(f"  No similar questions found, skipping...")
                skipped_count += 1
                continue
            
            # Create safe directory name
            safe_name = self._safe_filename(test_problem['name'])
            problem_dir = os.path.join(output_base_dir, safe_name)
            
            if not os.path.exists(problem_dir):
                os.makedirs(problem_dir)
            
            # Create info.txt
            info_content = f"{test_problem['cf_rating']}\n{test_problem['cf_tags']}"
            with open(os.path.join(problem_dir, "info.txt"), 'w', encoding='utf-8') as f:
                f.write(info_content)
            
            # Create problem.txt
            with open(os.path.join(problem_dir, "problem.txt"), 'w', encoding='utf-8') as f:
                f.write(test_problem['description'])
            
            # Create prompt.txt (with RAG)
            context_with_rag = self._build_context_with_rag(test_problem, similar_questions)
            with open(os.path.join(problem_dir, "prompt.txt"), 'w', encoding='utf-8') as f:
                f.write(context_with_rag)
            
            # Create prompt_sem_rag.txt (without RAG)
            context_without_rag = self._build_context_without_rag(test_problem)
            with open(os.path.join(problem_dir, "prompt_sem_rag.txt"), 'w', encoding='utf-8') as f:
                f.write(context_without_rag)
            
            # Create prompt_com_aleatorios.txt (with random examples)
            random_questions = self.retrieve_random_questions()
            context_with_random = self._build_context_with_random(test_problem, random_questions)
            with open(os.path.join(problem_dir, "prompt_com_aleatorios.txt"), 'w', encoding='utf-8') as f:
                f.write(context_with_random)
            
            processed_count += 1
            
            # Store results
            results.append({
                'test_problem_name': test_problem['name'],
                'safe_directory_name': safe_name,
                'similar_questions_count': len(similar_questions),
                'random_questions_count': len(random_questions),
                'similarity_scores': [float(q['similarity_score']) for q in similar_questions],  # Convert to Python float
                'cf_rating': test_problem['cf_rating'],
                'cf_tags': test_problem['cf_tags']
            })
            
            print(f"  Created directory: {problem_dir}")
            print(f"  Found {len(similar_questions)} similar questions")
        
        summary = {
            'total_test_problems': len(self.test_data),
            'processed_problems': processed_count,
            'skipped_problems': skipped_count,
            'output_directory': output_base_dir,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        
        # Save summary
        summary_path = os.path.join(output_base_dir, "processing_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nProcessing complete!")
        print(f"Total test problems: {len(self.test_data)}")
        print(f"Processed: {processed_count}")
        print(f"Skipped (no similar questions): {skipped_count}")
        print(f"Summary saved to: {summary_path}")
        
        return summary


if __name__ == "__main__":
    # Paths to the datasets
    train_path = "data/codecontests_train.json"
    test_path = "data/codecontests_test.json"
    
    # Initialize RAG system
    print("Initializing CodeContests RAG system...")
    rag_system = CodeContestsRAG(train_path, test_path)
    
    # Process all test problems
    print("\nProcessing test problems...")
    summary = rag_system.process_test_problems()
    
    print(f"\nDone! Processed {summary['processed_problems']} problems.")
