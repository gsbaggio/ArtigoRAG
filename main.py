from typing import List, Dict, Any, Tuple
import os
import json
from datetime import datetime

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv

# Carregando variáveis de ambiente (mantido por compatibilidade)
load_dotenv()

class ProgrammingQuestionRAG:
    def __init__(self, questions_db_path: str):
        """
        Inicializa o sistema RAG para questões de programação.
        
        Args:
            questions_db_path: Caminho para o banco de dados de questões
        """
        # Assegurar que trabalhamos com extensões .json
        questions_db_path = questions_db_path.replace('.csv', '.json')
        
        # Carregar os dados
        with open(questions_db_path, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
            self.questions_df = pd.DataFrame(questions_data)
        
        # Inicializar o modelo de embedding
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Construir índices de similaridade
        self._build_indices()
        
        print(f"Sistema RAG inicializado com {len(self.questions_df)} questões.")
    
    def _build_indices(self):
        """Constrói índices FAISS para pesquisa rápida de similaridade."""
        # Processando embeddings de questões
        question_texts = self.questions_df['question_text'].tolist()
        self.question_embeddings = self.embedding_model.encode(question_texts)
        
        # Criar índice FAISS para questões
        dimension = self.question_embeddings.shape[1]
        self.question_index = faiss.IndexFlatL2(dimension)
        self.question_index.add(np.array(self.question_embeddings).astype('float32'))
    
    def retrieve_similar_questions(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Recupera as questões mais similares à consulta.
        
        Args:
            query: Texto da consulta/questão
            top_k: Número de questões similares a retornar
            
        Returns:
            Lista de questões similares com metadados
        """
        # Obter embedding da consulta
        query_embedding = self.embedding_model.encode([query])
        
        # Pesquisar questões similares
        distances, indices = self.question_index.search(
            np.array(query_embedding).astype('float32'), top_k
        )
        
        # Formatar resultados
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
                'similarity_score': 1 - distances[0][i]/100  # Normalizar para score de similaridade
            })
        
        return similar_questions

    def save_context_to_file(self, context: str, output_path: str = "resposta.txt"):
        """
        Salva o contexto completo do RAG em um arquivo de texto.
        
        Args:
            context: O contexto RAG completo
            output_path: Caminho para o arquivo de saída
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(context)
            print(f"Contexto RAG salvo em {output_path}")
        except Exception as e:
            print(f"Erro ao salvar o contexto RAG: {e}")
    
    def _build_context_without_rag(self, question: str) -> str:
        """
        Constrói o contexto para o LLM apenas com a questão atual, sem as questões similares do RAG.
        
        Args:
            question: Texto da questão original
            
        Returns:
            Contexto formatado para o LLM sem os documentos recuperados
        """
        context = f"QUESTÃO ATUAL:\n{question}\n\n"
        
        # Adicionar instrução para o LLM
        context += """
        INSTRUÇÕES:
        Resolva a questão atual.
        Forneça:
        1. Uma solução de código completa e eficiente
        2. Uma explicação detalhada da solução, incluindo:
           - A intuição por trás da abordagem
           - A complexidade de tempo e espaço
           - Considerações importantes sobre o algoritmo
        3. Possíveis otimizações ou abordagens alternativas
        """
        
        return context

    def save_context_without_rag(self, question: str, output_path: str = "resposta_sem_rag.txt"):
        """
        Salva um contexto sem os documentos recuperados pelo RAG em um arquivo de texto.
        
        Args:
            question: A questão original
            output_path: Caminho para o arquivo de saída
        """
        # Construir contexto sem RAG
        context = self._build_context_without_rag(question)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(context)
            print(f"Contexto sem RAG salvo em {output_path}")
        except Exception as e:
            print(f"Erro ao salvar o contexto sem RAG: {e}")

    def generate_solution(self, question: str) -> Dict[str, Any]:
        """
        Gera os contextos RAG e salva em arquivos.
        
        Args:
            question: Texto da questão
            
        Returns:
            Dicionário contendo os caminhos para os arquivos salvos
        """
        # Recuperar questões similares
        similar_questions = self.retrieve_similar_questions(question, top_k=3)
        
        # Construir contexto para o LLM com RAG
        context_with_rag = self._build_context(question, similar_questions)
        
        # Salvar o contexto completo com RAG em um arquivo
        self.save_context_to_file(context_with_rag)
        
        # Salvar o contexto sem RAG em um arquivo separado
        self.save_context_without_rag(question)
        
        return {
            "question": question,
            "timestamp": datetime.now().isoformat(),
            "context_with_rag_path": "resposta.txt",
            "context_without_rag_path": "resposta_sem_rag.txt"
        }
    
    def _build_context(self, question: str, similar_questions: List[Dict]) -> str:
        """
        Constrói o contexto para o LLM com base nas questões similares.
        
        Args:
            question: Texto da questão original
            similar_questions: Lista de questões similares
            
        Returns:
            Contexto formatado para o LLM
        """
        context = f"QUESTÃO ATUAL:\n{question}\n\n"
        
        # Adicionar questões similares
        context += "QUESTÕES SIMILARES:\n"
        for i, q in enumerate(similar_questions):
            context += f"Questão Similar {i+1}:\n"
            context += f"Título: {q['title']}\n"
            context += f"Categorias: {q['category']}\n"
            context += f"Descrição: {q['question_text']}\n"
            context += f"Solução: {q['solution']}\n"
            context += f"Explicação: {q['explanation']}\n\n"
        
        # Adicionar instrução para o LLM
        context += """
        INSTRUÇÕES:
        Com base nas questões similares acima, resolva a questão atual.
        Forneça:
        1. Uma solução de código completa e eficiente
        2. Uma explicação detalhada da solução, incluindo:
           - A intuição por trás da abordagem
           - A complexidade de tempo e espaço
           - Considerações importantes sobre o algoritmo
        3. Possíveis otimizações ou abordagens alternativas
        """
        
        return context

# Exemplo de uso
if __name__ == "__main__":
    # Caminhos para os bancos de dados
    questions_db = "data/programming_questions.json"
    
    # Inicializar o sistema RAG
    rag_system = ProgrammingQuestionRAG(questions_db)
    
    # Exemplo de questão
    sample_question = """
    Dado um array nums contendo n números inteiros, encontre três números em nums de forma que a soma seja a mais próxima possível de um número alvo target.
    Retorne a soma dos três números.
    
    Exemplo:
    Input: nums = [-1,2,1,-4], target = 1
    Output: 2
    Explicação: A soma que é mais próxima do alvo é 2. (-1 + 2 + 1 = 2).
    """
    
    # Gerar e salvar contextos
    result = rag_system.generate_solution(sample_question)
    
    # Exibir resultado
    print("Questão:")
    print(sample_question)
    print("\nContextos salvos em:")
    print(f"Com RAG: {result['context_with_rag_path']}")
    print(f"Sem RAG: {result['context_without_rag_path']}")