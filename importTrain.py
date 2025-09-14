from datasets import load_dataset
import json

def extract_train_data():
    ds = load_dataset("deepmind/code_contests")
    
    train_data = ds['train']
    
    print(f"Total de problemas no train split: {len(train_data)}")
    
    extracted_data = []
    
    print("Extraindo campos cf_tags e cf_rating...")
    
    for i, problem in enumerate(train_data):
        problem_data = {
            'cf_tags': problem.get('cf_tags', []),
            'cf_rating': problem.get('cf_rating', 0)
        }
        
        extracted_data.append(problem_data)
        
        if (i + 1) % 1000 == 0:
            print(f"Processados {i + 1} problemas...")
    
    print("Salvando dados no arquivo codecontests_train.json...")
    with open('codecontests_train.json', 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=2)
    
    print(f"Dados salvos com sucesso! Total de problemas extraídos: {len(extracted_data)}")
    
    problems_with_cf_rating = sum(1 for p in extracted_data if p['cf_rating'] != 0)
    print(f"Problemas com cf_rating != 0 (do Codeforces): {problems_with_cf_rating}")
    print(f"Problemas com cf_rating == 0 (não do Codeforces): {len(extracted_data) - problems_with_cf_rating}")

if __name__ == "__main__":
    extract_train_data()