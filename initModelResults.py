import os
import sys

def create_model_structure():
    base_dir = os.path.join("data", "CodeContest")
    
    if not os.path.exists(base_dir):
        print(f"ERRO: Diretório {base_dir} não encontrado!")
        return False
    
    models = [
        'Claude 3.7 Sonnet',
        'DeepSeek R1', 
        'Gemini 2.0',
        'GPT 4o',
        'GPT-3.5',
        'Qwen2.5-Coder'
    ]
    
    files = [
        'answer_com_aleatorios.txt',
        'answer_sem_rag.txt', 
        'answer.txt',
        'results.txt'
    ]
    
    processed_dirs = 0
    skipped_items = 0
    
    print(f"Iniciando criação da estrutura de modelos em {base_dir}...")
    print(f"Modelos: {', '.join(models)}")
    print(f"Arquivos por modelo: {', '.join(files)}")
    print("-" * 70)
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        
        if not os.path.isdir(item_path) or item == "processing_summary.json":
            skipped_items += 1
            continue
            
        print(f"Processando: {item}")
        
        for model in models:
            model_dir = os.path.join(item_path, model)
            
            try:
                os.makedirs(model_dir, exist_ok=True)
                
                for file_name in files:
                    file_path = os.path.join(model_dir, file_name)
                    
                    if not os.path.exists(file_path):
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write("")  
                
                print(f"   {model}")
                
            except Exception as e:
                print(f"   ERRO ao criar {model}: {e}")
                
        processed_dirs += 1
        
    print("-" * 70)
    print(f"Processamento concluído!")
    print(f"Diretórios processados: {processed_dirs}")
    print(f"Itens ignorados: {skipped_items}")
    print(f"Total de pastas de modelos criadas: {processed_dirs * len(models)}")
    print(f"Total de arquivos criados: {processed_dirs * len(models) * len(files)}")
    
    return True

def main():
    print("=" * 70)
    print("INICIALIZADOR DE ESTRUTURA DE RESULTADOS DE MODELOS")
    print("=" * 70)
    
    try:
        success = create_model_structure()
        if success:
            print("\n Estrutura criada com sucesso!")
        else:
            print("\n Falha na criação da estrutura!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nProcesso interrompido pelo usuário.")
        sys.exit(1)
    except Exception as e:
        print(f"\n Erro inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
