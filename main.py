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
        
        print(f"Sistema RAG inicializado com {len(self.questions_df)} questões.")
    
    def _build_indices(self):
        question_texts = self.questions_df['question_text'].tolist()
        self.question_embeddings = self.embedding_model.encode(question_texts)
        
        dimension = self.question_embeddings.shape[1]
        self.question_index = faiss.IndexFlatL2(dimension)
        self.question_index.add(np.array(self.question_embeddings).astype('float32'))
    
    def retrieve_similar_questions(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode([query])
        
        distances, indices = self.question_index.search(
            np.array(query_embedding).astype('float32'), top_k
        )
        
        similar_questions = []
        for i, idx in enumerate(indices[0]):
            similar_questions.append({
                'question_id': self.questions_df.iloc[idx]['question_id'],
                'title': self.questions_df.iloc[idx]['title'],
                'difficulty': self.questions_df.iloc[idx]['difficulty'],
                'category': self.questions_df.iloc[idx]['category'],
                'question_text': self.questions_df.iloc[idx]['question_text'],
                'solution': self.questions_df.iloc[idx]['solution'],
                'explanation': self.questions_df.iloc[idx]['explanation'],
                'similarity_score': 1 - distances[0][i]/100  
            })
        
        return similar_questions

    def save_context_to_file(self, context: str, output_path: str = "prompt.txt"):
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(context)
            print(f"Contexto RAG salvo em {output_path}")
        except Exception as e:
            print(f"Erro ao salvar o contexto RAG: {e}")
    
    def _build_context_without_rag(self, question: str) -> str:
        context = f"CURRENT QUESTION:\n{question}\n\n"
        
        # Adicionar instrução para o LLM
        context += """
        INSTRUCTIONS:
        Solve the current question.
        Provide:
        1. A complete and efficient code solution, optimized for both time and space complexity
        2. A detailed explanation of the solution, including:
           - The intuition behind the approach
           - Time and space complexity
           - Important considerations about the algorithm
        3. If the language has classes, implement in 'Solution' class. Any language is accepted
        4. Don't use any external libraries. Don't need to import any libraries.
        """
        
        return context

    def save_context_without_rag(self, question: str, output_path: str = "prompt_sem_rag.txt"):
        context = self._build_context_without_rag(question)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(context)
            print(f"Contexto sem RAG salvo em {output_path}")
        except Exception as e:
            print(f"Erro ao salvar o contexto sem RAG: {e}")

    def generate_solution(self, question: str) -> Dict[str, Any]:
        similar_questions = self.retrieve_similar_questions(question, top_k=3)
        
        context_with_rag = self._build_context(question, similar_questions)
        
        self.save_context_to_file(context_with_rag)
        
        self.save_context_without_rag(question)
        
        return {
            "question": question,
            "timestamp": datetime.now().isoformat(),
            "context_with_rag_path": "resposta.txt",
            "context_without_rag_path": "resposta_sem_rag.txt"
        }
    
    def _build_context(self, question: str, similar_questions: List[Dict]) -> str:
        context = f"CURRENT QUESTION:\n{question}\n\n"
        
        context += "SIMILAR QUESTIONS:\n"
        for i, q in enumerate(similar_questions):
            context += f"Similar question {i+1}:\n"
            context += f"Title: {q['title']}\n"
            context += f"Categorys: {q['category']}\n"
            context += f"Question text: {q['question_text']}\n"
            context += f"Solution: {q['solution']}\n"
            context += f"Explanation: {q['explanation']}\n\n"
        
        context += """
        INSTRUCTIONS:
        Solve the current question.
        Provide:
        1. A complete and efficient code solution, optimized for both time and space complexity
        2. A detailed explanation of the solution, including:
           - The intuition behind the approach
           - Time and space complexity
           - Important considerations about the algorithm
        3. If the language has classes, implement in 'Solution' class. Any language is accepted
        4. Don't use any external libraries. Don't need to import any libraries.
        """
        
        return context

if __name__ == "__main__":
    questions_db = "data/programming_questions.json"

    rag_system = ProgrammingQuestionRAG(questions_db)
    
    try:
        with open("query.txt", "r", encoding="utf-8") as f:
            sample_question = f.read().strip()
    except FileNotFoundError:
        print("Erro: O arquivo 'query.txt' não foi encontrado.")
        exit(1)
    
    result = rag_system.generate_solution(sample_question)
    
    print("Questão:")
    print(sample_question)
    print("\nContextos salvos em:")
    print(f"Com RAG: {result['context_with_rag_path']}")
    print(f"Sem RAG: {result['context_without_rag_path']}")