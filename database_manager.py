import os
import pandas as pd
import uuid
import json
from typing import Dict, Any, List, Optional
import argparse

class RAGDatabaseManager:
    def __init__(self, questions_db_path: str, theory_db_path: str):
        """
        Inicializa o gerenciador de banco de dados para o sistema RAG.
        
        Args:
            questions_db_path: Caminho para o banco de dados de questões
            theory_db_path: Caminho para o banco de dados teórico
        """
        # Assegurar que as extensões dos arquivos são .json
        self.questions_db_path = questions_db_path.replace('.csv', '.json')
        self.theory_db_path = theory_db_path.replace('.csv', '.json')
        
        # Carregar ou criar os DataFrames
        self._load_or_create_dataframes()
        
    def _load_or_create_dataframes(self):
        """Carrega os DataFrames existentes ou cria novos se não existirem."""
        # Estrutura para questões
        questions_columns = ['question_id', 'title', 'difficulty', 'category', 'question_text', 'solution', 'explanation']
        
        # Estrutura para teoria
        theory_columns = ['theory_id', 'title', 'category', 'content']
        
        # Carregar ou criar DataFrame de questões
        if os.path.exists(self.questions_db_path):
            with open(self.questions_db_path, 'r', encoding='utf-8') as f:
                questions_data = json.load(f)
                self.questions_df = pd.DataFrame(questions_data)
        else:
            self.questions_df = pd.DataFrame(columns=questions_columns)
            os.makedirs(os.path.dirname(self.questions_db_path), exist_ok=True)
        
        # Carregar ou criar DataFrame de teoria
        if os.path.exists(self.theory_db_path):
            with open(self.theory_db_path, 'r', encoding='utf-8') as f:
                theory_data = json.load(f)
                self.theory_df = pd.DataFrame(theory_data)
        else:
            self.theory_df = pd.DataFrame(columns=theory_columns)
            os.makedirs(os.path.dirname(self.theory_db_path), exist_ok=True)
    
    def add_question(self, title: str, difficulty: str, category: str, question_text: str, 
                     solution: str, explanation: str) -> str:
        """
        Adiciona uma nova questão ao banco de dados.
        
        Args:
            title: Título da questão
            difficulty: Dificuldade (e.g., "Easy", "Medium", "Hard")
            category: Categorias/tags da questão (e.g., "Arrays, Recursão, Ordenação")
            question_text: Texto completo da questão
            solution: Solução em código
            explanation: Explicação detalhada da solução
            
        Returns:
            ID da questão adicionada
        """
        # Validar entradas
        if not all([title, difficulty, category, question_text, solution, explanation]):
            raise ValueError("Todos os campos são obrigatórios para adicionar uma questão")
        
        # Gerar ID único
        question_id = str(uuid.uuid4())
        
        # Criar novo registro
        new_question = {
            'question_id': question_id,
            'title': title,
            'difficulty': difficulty,
            'category': category,
            'question_text': question_text,
            'solution': solution,
            'explanation': explanation
        }
        
        # Adicionar ao DataFrame
        self.questions_df = pd.concat([self.questions_df, pd.DataFrame([new_question])], ignore_index=True)
        
        # Salvar DataFrame atualizado como JSON
        with open(self.questions_db_path, 'w', encoding='utf-8') as f:
            json.dump(self.questions_df.to_dict('records'), f, ensure_ascii=False, indent=4)
        
        print(f"Questão '{title}' adicionada com sucesso com ID: {question_id}")
        return question_id
    
    def add_theory(self, title: str, category: str, content: str) -> str:
        """
        Adiciona um novo documento teórico ao banco de dados.
        
        Args:
            title: Título do documento teórico
            category: Categoria (e.g., "Algoritmos", "Estruturas de Dados")
            content: Conteúdo completo do documento
            
        Returns:
            ID do documento teórico adicionado
        """
        # Validar entradas
        if not all([title, category, content]):
            raise ValueError("Todos os campos são obrigatórios para adicionar um documento teórico")
        
        # Gerar ID único
        theory_id = str(uuid.uuid4())
        
        # Criar novo registro
        new_theory = {
            'theory_id': theory_id,
            'title': title,
            'category': category,
            'content': content
        }
        
        # Adicionar ao DataFrame
        self.theory_df = pd.concat([self.theory_df, pd.DataFrame([new_theory])], ignore_index=True)
        
        # Salvar DataFrame atualizado como JSON
        with open(self.theory_db_path, 'w', encoding='utf-8') as f:
            json.dump(self.theory_df.to_dict('records'), f, ensure_ascii=False, indent=4)
        
        print(f"Documento teórico '{title}' adicionado com sucesso com ID: {theory_id}")
        return theory_id
    
    def list_questions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Lista as questões no banco de dados."""
        questions = self.questions_df.head(limit) if limit else self.questions_df
        return questions[['question_id', 'title', 'difficulty']].to_dict('records')
    
    def list_theory(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Lista os documentos teóricos no banco de dados."""
        theory = self.theory_df.head(limit) if limit else self.theory_df
        return theory[['theory_id', 'title', 'category']].to_dict('records')
    
    def get_question(self, question_id: str) -> Dict[str, Any]:
        """Obtém os detalhes de uma questão específica."""
        question = self.questions_df[self.questions_df['question_id'] == question_id]
        if len(question) == 0:
            raise ValueError(f"Questão com ID {question_id} não encontrada")
        return question.iloc[0].to_dict()
    
    def get_theory(self, theory_id: str) -> Dict[str, Any]:
        """Obtém os detalhes de um documento teórico específico."""
        theory = self.theory_df[self.theory_df['theory_id'] == theory_id]
        if len(theory) == 0:
            raise ValueError(f"Documento teórico com ID {theory_id} não encontrado")
        return theory.iloc[0].to_dict()
    

def interactive_mode(db_manager: RAGDatabaseManager):
    """Interface interativa para gerenciar o banco de dados."""
    while True:
        print("\n=== Sistema de Gerenciamento de Banco de Dados RAG ===")
        print("1. Adicionar nova questão")
        print("2. Adicionar novo documento teórico")
        print("3. Listar questões")
        print("4. Listar documentos teóricos")
        print("5. Ver detalhes de uma questão")
        print("6. Ver detalhes de um documento teórico")
        print("7. Importar questão de arquivo")
        print("8. Importar documento teórico de arquivo")
        print("0. Sair")
        
        choice = input("\nEscolha uma opção: ")
        
        if choice == "1":
            print("\n--- Adicionar Nova Questão ---")
            title = input("Título: ")
            difficulty = input("Dificuldade (Easy/Medium/Hard): ")
            category = input("Categorias/tags (separadas por vírgulas): ")
            print("Texto da questão (termine com uma linha vazia):")
            question_text = ""
            while True:
                line = input()
                if line.strip() == "":
                    break
                question_text += line + "\n"
            
            print("Solução em código (termine com uma linha vazia):")
            solution = ""
            while True:
                line = input()
                if line.strip() == "":
                    break
                solution += line + "\n"
            
            print("Explicação (termine com uma linha vazia):")
            explanation = ""
            while True:
                line = input()
                if line.strip() == "":
                    break
                explanation += line + "\n"
            
            try:
                db_manager.add_question(title, difficulty, category, question_text.strip(), 
                                      solution.strip(), explanation.strip())
            except ValueError as e:
                print(f"Erro: {e}")
        
        elif choice == "2":
            print("\n--- Adicionar Novo Documento Teórico ---")
            title = input("Título: ")
            category = input("Categoria: ")
            print("Conteúdo (termine com uma linha vazia):")
            content = ""
            while True:
                line = input()
                if line.strip() == "":
                    break
                content += line + "\n"
            
            try:
                db_manager.add_theory(title, category, content.strip())
            except ValueError as e:
                print(f"Erro: {e}")
        
        elif choice == "3":
            print("\n--- Lista de Questões ---")
            questions = db_manager.list_questions()
            for i, q in enumerate(questions, 1):
                print(f"{i}. [{q['difficulty']}] {q['title']} (ID: {q['question_id']})")
        
        elif choice == "4":
            print("\n--- Lista de Documentos Teóricos ---")
            theories = db_manager.list_theory()
            for i, t in enumerate(theories, 1):
                print(f"{i}. [{t['category']}] {t['title']} (ID: {t['theory_id']})")
        
        elif choice == "5":
            question_id = input("\nInforme o ID da questão: ")
            try:
                question = db_manager.get_question(question_id)
                print(f"\n--- Detalhes da Questão: {question['title']} ---")
                print(f"Dificuldade: {question['difficulty']}")
                print(f"Categorias: {question['category']}")
                print(f"Texto da questão:\n{question['question_text']}")
                print(f"Solução:\n{question['solution']}")
                print(f"Explicação:\n{question['explanation']}")
            except ValueError as e:
                print(f"Erro: {e}")
        
        elif choice == "6":
            theory_id = input("\nInforme o ID do documento teórico: ")
            try:
                theory = db_manager.get_theory(theory_id)
                print(f"\n--- Detalhes do Documento Teórico: {theory['title']} ---")
                print(f"Categoria: {theory['category']}")
                print(f"Conteúdo:\n{theory['content']}")
            except ValueError as e:
                print(f"Erro: {e}")
        
        elif choice == "7":
            print("\n--- Importar Questão de Arquivo ---")
            file_path = input("Caminho para o arquivo de texto: ")
            try:
                import_question_from_file(db_manager, file_path)
            except Exception as e:
                print(f"Erro ao importar questão: {e}")
        
        elif choice == "8":
            print("\n--- Importar Documento Teórico de Arquivo ---")
            file_path = input("Caminho para o arquivo de texto: ")
            try:
                import_theory_from_file(db_manager, file_path)
            except Exception as e:
                print(f"Erro ao importar documento teórico: {e}")
        
        elif choice == "0":
            print("Saindo do sistema...")
            break
        
        else:
            print("Opção inválida, tente novamente.")

def import_question_from_file(db_manager: RAGDatabaseManager, file_path: str):
    """
    Importa uma questão de um arquivo de texto.
    
    O arquivo deve estar no formato:
    TITLE: [título da questão]
    DIFFICULTY: [dificuldade]
    CATEGORY: [categorias/tags]
    QUESTION:
    [texto da questão]
    SOLUTION:
    [solução em código]
    EXPLANATION:
    [explicação da solução]
    
    Args:
        db_manager: Instância do gerenciador de banco de dados
        file_path: Caminho para o arquivo de texto
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Extrair informações do arquivo
    try:
        # Dividir o conteúdo em seções
        sections = {}
        current_section = None
        current_content = []
        
        # Primeiro, encontrar o título e a dificuldade
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('TITLE:'):
                sections['title'] = line[6:].strip()
            elif line.startswith('DIFFICULTY:'):
                sections['difficulty'] = line[11:].strip()
            elif line.startswith('CATEGORY:'):
                sections['category'] = line[9:].strip()
            elif line.startswith('QUESTION:'):
                current_section = 'question_text'
                lines = lines[i+1:]  # Continuar a partir da próxima linha
                break
        
        # Agora processar o restante do texto
        for line in lines:
            if line.startswith('SOLUTION:'):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'solution'
                current_content = []
            elif line.startswith('EXPLANATION:'):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'explanation'
                current_content = []
            else:
                current_content.append(line)
        
        # Adicionar a última seção
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        # Verificar se todas as seções necessárias estão presentes
        required_fields = ['title', 'difficulty', 'category', 'question_text', 'solution', 'explanation']
        for field in required_fields:
            if field not in sections or not sections[field]:
                raise ValueError(f"Campo obrigatório ausente ou vazio: {field}")
        
        # Adicionar questão ao banco de dados
        db_manager.add_question(
            sections['title'],
            sections['difficulty'],
            sections['category'],
            sections['question_text'],
            sections['solution'],
            sections['explanation']
        )
        
        print(f"Questão '{sections['title']}' importada com sucesso!")
        
    except Exception as e:
        raise ValueError(f"Erro ao processar o arquivo: {e}")

def import_theory_from_file(db_manager: RAGDatabaseManager, file_path: str):
    """
    Importa um documento teórico de um arquivo de texto.
    
    O arquivo deve estar no formato:
    TITLE: [título do documento]
    CATEGORY: [categoria]
    CONTENT:
    [conteúdo do documento]
    
    Args:
        db_manager: Instância do gerenciador de banco de dados
        file_path: Caminho para o arquivo de texto
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Extrair informações do arquivo
    try:
        # Dividir o conteúdo em seções
        sections = {}
        current_section = None
        current_content = []
        
        # Primeiro, encontrar o título e a categoria
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('TITLE:'):
                sections['title'] = line[6:].strip()
            elif line.startswith('CATEGORY:'):
                sections['category'] = line[9:].strip()
            elif line.startswith('CONTENT:'):
                current_section = 'content'
                lines = lines[i+1:]  # Continuar a partir da próxima linha
                break
        
        # Processar o conteúdo
        for line in lines:
            current_content.append(line)
        
        # Adicionar o conteúdo à seção
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        # Verificar se todas as seções necessárias estão presentes
        required_fields = ['title', 'category', 'content']
        for field in required_fields:
            if field not in sections or not sections[field]:
                raise ValueError(f"Campo obrigatório ausente ou vazio: {field}")
        
        # Adicionar documento teórico ao banco de dados
        db_manager.add_theory(
            sections['title'],
            sections['category'],
            sections['content']
        )
        
        print(f"Documento teórico '{sections['title']}' importado com sucesso!")
        
    except Exception as e:
        raise ValueError(f"Erro ao processar o arquivo: {e}")

def batch_mode(db_manager: RAGDatabaseManager, args):
    """Modo em lote para adicionar entradas via linha de comando."""
    if args.add_question:
        with open(args.question_text, 'r', encoding='utf-8') as f:
            question_text = f.read()
        
        with open(args.solution, 'r', encoding='utf-8') as f:
            solution = f.read()
        
        with open(args.explanation, 'r', encoding='utf-8') as f:
            explanation = f.read()
        
        db_manager.add_question(args.title, args.difficulty, args.category, question_text, solution, explanation)
    
    elif args.add_theory:
        with open(args.content, 'r', encoding='utf-8') as f:
            content = f.read()
        
        db_manager.add_theory(args.title, args.category, content)

def main():
    parser = argparse.ArgumentParser(description='Gerenciador de Banco de Dados para sistema RAG')
    
    # Caminhos para os bancos de dados
    parser.add_argument('--questions-db', type=str, default='data/programming_questions.json',
                        help='Caminho para o arquivo JSON de questões')
    parser.add_argument('--theory-db', type=str, default='data/programming_theory.json',
                        help='Caminho para o arquivo JSON de teoria')
    
    # Subcomandos
    subparsers = parser.add_subparsers(dest='command')
    
    # Para adicionar questões em lote
    question_parser = subparsers.add_parser('add-question', help='Adicionar uma nova questão')
    question_parser.add_argument('--title', required=True, help='Título da questão')
    question_parser.add_argument('--difficulty', required=True, choices=['Easy', 'Medium', 'Hard'], 
                               help='Dificuldade da questão')
    question_parser.add_argument('--category', required=True, 
                               help='Categorias/tags da questão (separadas por vírgulas)')
    question_parser.add_argument('--question-text', required=True, 
                               help='Arquivo contendo o texto da questão')
    question_parser.add_argument('--solution', required=True, 
                               help='Arquivo contendo a solução em código')
    question_parser.add_argument('--explanation', required=True, 
                               help='Arquivo contendo a explicação')
    question_parser.set_defaults(add_question=True, add_theory=False)
    
    # Para adicionar teoria em lote
    theory_parser = subparsers.add_parser('add-theory', help='Adicionar um novo documento teórico')
    theory_parser.add_argument('--title', required=True, help='Título do documento teórico')
    theory_parser.add_argument('--category', required=True, 
                             help='Categoria do documento teórico')
    theory_parser.add_argument('--content', required=True, 
                             help='Arquivo contendo o conteúdo do documento teórico')
    theory_parser.set_defaults(add_question=False, add_theory=True)
    
    args = parser.parse_args()
    
    # Inicializar o gerenciador de banco de dados
    db_manager = RAGDatabaseManager(args.questions_db, args.theory_db)
    
    # Decidir modo de operação
    if args.command:
        batch_mode(db_manager, args)
    else:
        interactive_mode(db_manager)

if __name__ == "__main__":
    main()
