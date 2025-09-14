import json
import matplotlib.pyplot as plt
import ast
import os
from collections import Counter

def process_problem(problem, tag_counter, difficulty_counter):
    cf_rating = problem.get('cf_rating', 0)
    
    cf_tags = problem.get('cf_tags', [])
    
    if isinstance(cf_tags, str):
        try:
            cf_tags = ast.literal_eval(cf_tags)
        except:
            cf_tags = [cf_tags] if cf_tags else []
    elif not isinstance(cf_tags, list):
        cf_tags = []
    
    if cf_tags:
        for tag in cf_tags:
            if tag and tag.strip():  
                tag_counter[tag.strip()] += 1
    
    if cf_rating != 0:
        if 0 < cf_rating <= 1000:
            difficulty_counter['Easy'] += 1
        elif 1000 < cf_rating <= 2000:
            difficulty_counter['Medium'] += 1
        elif cf_rating > 2000:
            difficulty_counter['Hard'] += 1
        
        return True
    return False

def analyze_test_data():
    tag_counter = Counter()
    difficulty_counter = {'Easy': 0, 'Medium': 0, 'Hard': 0}
    total_problems = 0
    
    codecontest_path = 'data/CodeContest'
    
    if not os.path.exists(codecontest_path):
        print(f"Pasta {codecontest_path} não encontrada!")
        return tag_counter, difficulty_counter, total_problems
    
    print("Analisando dados de teste...")
    
    for folder_name in os.listdir(codecontest_path):
        folder_path = os.path.join(codecontest_path, folder_name)
        
        if os.path.isdir(folder_path):
            info_file = os.path.join(folder_path, 'info.txt')
            
            if os.path.exists(info_file):
                try:
                    with open(info_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        
                    if len(lines) >= 2:
                        rating = int(lines[0].strip())
                        tags_str = lines[1].strip()
                        
                        try:
                            tags = ast.literal_eval(tags_str)
                        except:
                            tags = []
                        
                        if tags:  
                            for tag in tags:
                                if tag and tag.strip():  
                                    tag_counter[tag.strip()] += 1
                        
                        if 0 < rating <= 1000:
                            difficulty_counter['Easy'] += 1
                        elif 1000 < rating <= 2000:
                            difficulty_counter['Medium'] += 1
                        elif rating > 2000:
                            difficulty_counter['Hard'] += 1
                        
                        total_problems += 1
                        
                except Exception as e:
                    print(f"Erro ao processar {info_file}: {e}")
    
    print(f"Dados de teste processados: {total_problems} problemas")
    return tag_counter, difficulty_counter, total_problems

def analyze_codecontests_train():
    
    tag_counter_train = Counter()
    difficulty_counter_train = {'Easy': 0, 'Medium': 0, 'Hard': 0}
    total_problems_train = 0
    processed_count = 0
    
    try:
        with open('codecontests_train.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        
        for i, problem in enumerate(data):
            process_problem(problem, tag_counter_train, difficulty_counter_train)
            if problem.get('cf_rating', 0) != 0:
                total_problems_train += 1
            processed_count += 1
            
            if processed_count % 1000 == 0:
                print(f"Processados {processed_count} problemas...")
                
        
    except FileNotFoundError:
        print("Arquivo codecontests_train.json não encontrado!")
        return
    except Exception as e:
        print(f"Erro ao processar arquivo: {e}")
        return
    
    tag_counter_test, difficulty_counter_test, total_problems_test = analyze_test_data()
    
    print(f"\nTotal de problemas processados: {processed_count}")
    
    print(f"\n=== ANÁLISE DE DIFICULDADE - DADOS DE TREINO ===")
    print(f"Total de problemas do Codeforces (cf_rating != 0): {total_problems_train}")
    print(f"\nDistribuição por dificuldade (TREINO):")
    
    for difficulty, count in difficulty_counter_train.items():
        percentage = (count / total_problems_train * 100) if total_problems_train > 0 else 0
        print(f"{difficulty}: {count} questões ({percentage:.2f}%)")
    
    print(f"\n=== ANÁLISE DE DIFICULDADE - DADOS DE TESTE ===")
    print(f"Total de problemas de teste: {total_problems_test}")
    print(f"\nDistribuição por dificuldade (TESTE):")
    
    for difficulty, count in difficulty_counter_test.items():
        percentage = (count / total_problems_test * 100) if total_problems_test > 0 else 0
        print(f"{difficulty}: {count} questões ({percentage:.2f}%)")
    
    if tag_counter_train:
        sorted_tags_train = tag_counter_train.most_common(20)
        
        tags_train = [tag for tag, count in sorted_tags_train]
        counts_train = [count for tag, count in sorted_tags_train]
        
        plt.figure(figsize=(16, 8))
        
        bars = plt.bar(range(len(tags_train)), counts_train, 
                      color='lightblue', 
                      edgecolor='black', 
                      linewidth=0.5,
                      width=0.8)
        
        plt.title('CodeContests Tags Distribution (Knowledge Dataset)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Tags', fontsize=13, fontweight='bold')
        plt.ylabel('Number of Problems', fontsize=13, fontweight='bold')
        
        plt.xticks(range(len(tags_train)), tags_train, rotation=45, ha='right', fontsize=13)
        
        plt.ylim(0, max(counts_train) * 1.1)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(counts_train)*0.01,
                    f'{int(height)}', ha='center', va='bottom', 
                    fontsize=13, fontweight='bold')
        
        plt.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        plt.show()
    
    if tag_counter_test:
        sorted_tags_test = tag_counter_test.most_common(20)
        
        tags_test = [tag for tag, count in sorted_tags_test]
        counts_test = [count for tag, count in sorted_tags_test]
        
        plt.figure(figsize=(16, 8))
        
        bars = plt.bar(range(len(tags_test)), counts_test, 
                      color='orange', 
                      edgecolor='black', 
                      linewidth=0.5,
                      width=0.8)
        
        plt.title('CodeContests Tags Distribution (Test Dataset)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Tags', fontsize=13, fontweight='bold')
        plt.ylabel('Number of Problems', fontsize=13, fontweight='bold')
        
        plt.xticks(range(len(tags_test)), tags_test, rotation=45, ha='right', fontsize=13)
        
        plt.ylim(0, max(counts_test) * 1.1)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(counts_test)*0.01,
                    f'{int(height)}', ha='center', va='bottom', 
                    fontsize=13, fontweight='bold')
        
        plt.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        plt.show()
        
    print(f"\n=== TOP 20 TAGS MAIS FREQUENTES (TREINO) ===")
    for i, (tag, count) in enumerate(tag_counter_train.most_common(20), 1):
        print(f"{i:2d}. {tag}: {count} questões")
    
    print(f"\n=== TOP 20 TAGS MAIS FREQUENTES (TESTE) ===")
    for i, (tag, count) in enumerate(tag_counter_test.most_common(20), 1):
        print(f"{i:2d}. {tag}: {count} questões")

    else:
        print("Nenhuma tag encontrada nos dados.")

if __name__ == "__main__":
    analyze_codecontests_train()