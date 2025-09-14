import json
import matplotlib.pyplot as plt
import ast
from collections import Counter

def process_problem(problem, tag_counter, difficulty_counter):
    cf_rating = problem.get('cf_rating', 0)
    
    if cf_rating != 0:
        cf_tags = problem.get('cf_tags', [])
        
        if isinstance(cf_tags, str):
            try:
                cf_tags = ast.literal_eval(cf_tags)
            except:
                cf_tags = [cf_tags] if cf_tags else []
        
        for tag in cf_tags:
            tag_counter[tag] += 1
        
        if 0 < cf_rating <= 1000:
            difficulty_counter['Easy'] += 1
        elif 1000 < cf_rating <= 2000:
            difficulty_counter['Medium'] += 1
        elif cf_rating > 2000:
            difficulty_counter['Hard'] += 1
        
        return True
    return False

def analyze_codecontests_train():
    
    tag_counter = Counter()
    difficulty_counter = {'Easy': 0, 'Medium': 0, 'Hard': 0}
    total_problems = 0
    processed_count = 0
    
    try:
        with open('codecontests_train.json', 'r', encoding='utf-8') as file:
            try:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if line:
                        try:
                            problem = json.loads(line)
                            if process_problem(problem, tag_counter, difficulty_counter):
                                total_problems += 1
                            processed_count += 1
                            
                            if processed_count % 1000 == 0:
                                print(f"Processados {processed_count} problemas...")
                                
                        except json.JSONDecodeError:
                            break
                            
            except:
                file.seek(0)
                
                buffer = ""
                chunk_size = 1024 * 1024  
                
                while True:
                    chunk = file.read(chunk_size)
                    if not chunk:
                        break
                    
                    buffer += chunk
                    
                    while True:
                        start = buffer.find('{')
                        if start == -1:
                            break
                            
                        brace_count = 0
                        end = start
                        in_string = False
                        escape_next = False
                        
                        for i in range(start, len(buffer)):
                            char = buffer[i]
                            
                            if escape_next:
                                escape_next = False
                                continue
                                
                            if char == '\\':
                                escape_next = True
                                continue
                                
                            if char == '"' and not escape_next:
                                in_string = not in_string
                                
                            if not in_string:
                                if char == '{':
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        end = i + 1
                                        break
                        
                        if brace_count == 0 and end > start:
                            try:
                                json_str = buffer[start:end]
                                problem = json.loads(json_str)
                                if process_problem(problem, tag_counter, difficulty_counter):
                                    total_problems += 1
                                processed_count += 1
                                
                                if processed_count % 1000 == 0:
                                    print(f"Processados {processed_count} problemas...")
                                    
                            except json.JSONDecodeError:
                                pass
                            
                            buffer = buffer[end:]
                        else:
                            break
                    
                    if len(buffer) > 10240:
                        buffer = buffer[-10240:]
                        
    except FileNotFoundError:
        print("Arquivo codecontests_train.json não encontrado!")
        return
    except Exception as e:
        print(f"Erro ao processar arquivo: {e}")
        return
    
    print(f"\nTotal de problemas processados: {processed_count}")
    
    print(f"\n=== ANÁLISE DE DIFICULDADE ===")
    print(f"Total de problemas do Codeforces (cf_rating != 0): {total_problems}")
    print(f"\nDistribuição por dificuldade:")
    
    for difficulty, count in difficulty_counter.items():
        percentage = (count / total_problems * 100) if total_problems > 0 else 0
        print(f"{difficulty}: {count} questões ({percentage:.2f}%)")
    
    if tag_counter:
        sorted_tags = tag_counter.most_common()
        
        tags = [tag for tag, count in sorted_tags]
        counts = [count for tag, count in sorted_tags]
        
        plt.figure(figsize=(15, 8))
        bars = plt.bar(range(len(tags)), counts)
        
        plt.title('Distribuição de Tags nos Problemas do CodeContests (cf_rating != 0)', fontsize=16, fontweight='bold')
        plt.xlabel('Tags', fontsize=12)
        plt.ylabel('Número de Questões', fontsize=12)
        
        plt.xticks(range(len(tags)), tags, rotation=45, ha='right')
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        plt.show()
        
        print(f"\n=== TOP 10 TAGS MAIS FREQUENTES ===")
        for i, (tag, count) in enumerate(sorted_tags[:10], 1):
            print(f"{i:2d}. {tag}: {count} questões")
    
    else:
        print("Nenhuma tag encontrada nos dados.")

if __name__ == "__main__":
    analyze_codecontests_train()