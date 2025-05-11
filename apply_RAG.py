from typing import List, Dict, Any, Tuple
import os
import json
from datetime import datetime

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv

load_dotenv()

class ProgrammingQuestionRAG:
    def __init__(self, questions_db_path: str):
        questions_db_path = questions_db_path.replace('.csv', '.json')
        
        with open(questions_db_path, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
            self.questions_df = pd.DataFrame(questions_data)
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self._build_indices()
        
        print(f"{len(self.questions_df)} total questions.")
    
    def _build_indices(self):
        question_texts = self.questions_df['question_text'].tolist()
        self.question_embeddings = self.embedding_model.encode(question_texts)
        
        dimension = self.question_embeddings.shape[1]
        self.question_index = faiss.IndexFlatL2(dimension)
        self.question_index.add(np.array(self.question_embeddings).astype('float32'))
    
    def retrieve_similar_questions(self, query: str, top_k: int = 3, 
                                  min_threshold: float = 0.65,
                                  use_adaptive_threshold: bool = True) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode([query])
        
        candidate_k = min(top_k * 3, len(self.questions_df))
        
        distances, indices = self.question_index.search(
            np.array(query_embedding).astype('float32'), candidate_k
        )
        
        similarities = [1 - (dist/2) for dist in distances[0]]
        similarities = [max(0, min(s, 1)) for s in similarities]
        
        threshold = min_threshold  
        if use_adaptive_threshold and similarities:
            max_similarity = similarities[0]
            if max_similarity > 0.8:
                threshold = max(min_threshold, 0.75 * max_similarity)
            elif max_similarity > 0.7:
                threshold = max(min_threshold, 0.8 * max_similarity)
            else:
                threshold = max(min_threshold, 0.85 * max_similarity)
        
        similar_questions = []
        for i, idx in enumerate(indices[0]):
            similarity_score = similarities[i]
            
            if similarity_score >= threshold:
                similar_questions.append({
                    'question_id': self.questions_df.iloc[idx]['question_id'],
                    'title': self.questions_df.iloc[idx]['title'],
                    'difficulty': self.questions_df.iloc[idx]['difficulty'],
                    'category': self.questions_df.iloc[idx]['category'],
                    'question_text': self.questions_df.iloc[idx]['question_text'],
                    'solution': self.questions_df.iloc[idx]['solution'],
                    'explanation': self.questions_df.iloc[idx]['explanation'],
                    'similarity_score': similarity_score
                })
        
        return similar_questions[:top_k]

    def save_context_to_file(self, context: str, output_path: str = "prompt.txt"):
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(context)
            print(f"Context saved {output_path}")
        except Exception as e:
            print(f"Error: {e}")
    
    def _build_context_without_rag(self, question: str) -> str:
        context = f"CURRENT QUESTION:\n{question}\n\n"
        
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

    def save_context_without_rag(self, question: str, output_path: str = "prompt_sem_rag.txt"):
        context = self._build_context_without_rag(question)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(context)
            print(f"Context saved {output_path}")
        except Exception as e:
            print(f"Error: {e}")

    def generate_solution(self, question: str) -> Dict[str, Any]:
        similar_questions = self.retrieve_similar_questions(
            question, 
            top_k=3, 
            min_threshold=0.65,  
            use_adaptive_threshold=True 
        )
        
        context_with_rag = self._build_context(question, similar_questions)
        
        self.save_context_to_file(context_with_rag)
        
        self.save_context_without_rag(question)
        
        return {
            "question": question,
            "timestamp": datetime.now().isoformat(),
            "context_with_rag_path": "resposta.txt",
            "context_without_rag_path": "resposta_sem_rag.txt",
            "similar_questions_found": len(similar_questions),
            "similarity_scores": [q['similarity_score'] for q in similar_questions] if similar_questions else []
        }
    
    def _build_context(self, question: str, similar_questions: List[Dict]) -> str:
        context = f"CURRENT QUESTION:\n{question}\n\n"
        
        if similar_questions:
            context += "SIMILAR QUESTIONS:\n"
            for i, q in enumerate(similar_questions):
                context += f"Similar question {i+1} (Similarity score: {q['similarity_score']:.2f}):\n"
                context += f"Title: {q['title']}\n"
                context += f"Categorys: {q['category']}\n"
                context += f"Question text: {q['question_text']}\n"
                context += f"Solution: {q['solution']}\n"
                context += f"Explanation: {q['explanation']}\n\n"
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
        
        context += "        5. Don't use any external libraries. Don't need to import any libraries.\n"
        
        return context

    def process_problem(self, problem_path: str, output_dir: str = None) -> Dict[str, Any]:
        try:
            with open(problem_path, "r", encoding="utf-8") as f:
                question = f.read().strip()
        except FileNotFoundError:
            print(f"Erro: Non existing file '{problem_path}'.")
            return None
        
        if output_dir is None:
            output_dir = os.path.dirname(problem_path)
            
        prompt_with_rag_path = os.path.join(output_dir, "prompt.txt")
        prompt_without_rag_path = os.path.join(output_dir, "prompt_sem_rag.txt")
        
        similar_questions = self.retrieve_similar_questions(
            question, 
            top_k=3, 
            min_threshold=0.65,
            use_adaptive_threshold=True
        )
        
        context_with_rag = self._build_context(question, similar_questions)
        self.save_context_to_file(context_with_rag, prompt_with_rag_path)
        
        context_without_rag = self._build_context_without_rag(question)
        self.save_context_to_file(context_without_rag, prompt_without_rag_path)
        
        return {
            "question": question,
            "timestamp": datetime.now().isoformat(),
            "context_with_rag_path": prompt_with_rag_path,
            "context_without_rag_path": prompt_without_rag_path,
            "similar_questions_found": len(similar_questions),
            "similarity_scores": [q['similarity_score'] for q in similar_questions] if similar_questions else []
        }
    
    def process_all_problems_in_answers(self, answers_dir: str = "data/answers") -> List[Dict[str, Any]]:
        results = []
        
        if not os.path.exists(answers_dir) or not os.path.isdir(answers_dir):
            print(f"'{answers_dir}' non existing folder.")
            return results
        
        subdirs = [d for d in os.listdir(answers_dir) if os.path.isdir(os.path.join(answers_dir, d))]
        
        for subdir in subdirs:
            subdir_path = os.path.join(answers_dir, subdir)
            problem_path = os.path.join(subdir_path, "problem.txt")
            
            if os.path.exists(problem_path):
                print(f"Processing in: {subdir_path}")
                result = self.process_problem(problem_path, subdir_path)
                if result:
                    results.append(result)
            else:
                print(f"problem.txt file not found in: {subdir_path}")
        
        return results

if __name__ == "__main__":
    questions_db = "data/programming_questions.json"
    rag_system = ProgrammingQuestionRAG(questions_db)
    
    try:
        top_level_result = rag_system.process_problem("problem.txt")
        if top_level_result:
            print("Test question processed:")
            print(f"RAG: {top_level_result['context_with_rag_path']}")
            print(f"No RAG: {top_level_result['context_without_rag_path']}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nProcessing questions in data/answers/...")
    answers_results = rag_system.process_all_problems_in_answers()
    print(f"\n{len(answers_results)} questions processed in data/answers/")