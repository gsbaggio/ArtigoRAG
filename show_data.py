import os
from pathlib import Path
import statistics
import matplotlib.pyplot as plt
import numpy as np

def safe_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0

root_dir = Path("data/answers")

llms = ["Claude 3.7 Sonnet", "Gemini 2.0", "GPT 4o", "GPT-3.5", "DeepSeek R1", "Qwen2.5-Coder"]

images_dir = Path("images")
os.makedirs(images_dir, exist_ok=True)

results = {
    llm: {
        "without_rag": {
            "correct": [],
            "speed": [],
            "memory": []
        },
        "random_rag": {
            "correct": [],
            "speed": [],
            "memory": []
        },
        "with_rag": {
            "correct": [],
            "speed": [],
            "memory": []
        }
    } for llm in llms
}

results_by_category = {
    "Easy": {
        "without_rag": {"correct": [], "speed": [], "memory": []},
        "random_rag": {"correct": [], "speed": [], "memory": []},
        "with_rag": {"correct": [], "speed": [], "memory": []}
    },
    "Medium": {
        "without_rag": {"correct": [], "speed": [], "memory": []},
        "random_rag": {"correct": [], "speed": [], "memory": []},
        "with_rag": {"correct": [], "speed": [], "memory": []}
    },
    "Hard": {
        "without_rag": {"correct": [], "speed": [], "memory": []},
        "random_rag": {"correct": [], "speed": [], "memory": []},
        "with_rag": {"correct": [], "speed": [], "memory": []}
    },
    "Before 2023": {
        "without_rag": {"correct": [], "speed": [], "memory": []},
        "random_rag": {"correct": [], "speed": [], "memory": []},
        "with_rag": {"correct": [], "speed": [], "memory": []}
    },
    "2023-2025": {
        "without_rag": {"correct": [], "speed": [], "memory": []},
        "random_rag": {"correct": [], "speed": [], "memory": []},
        "with_rag": {"correct": [], "speed": [], "memory": []}
    }
}

problems = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

for problem in problems:
    problem_dir = root_dir / problem
    
    info_file = problem_dir / "info.txt"
    difficulty = None
    year = None
    
    if info_file.exists():
        try:
            with open(info_file, "r") as f:
                lines = [line.strip() for line in f.readlines()]
                if len(lines) >= 2:
                    difficulty = lines[0]  # E, M, or H
                    year = int(lines[1])
        except Exception as e:
            print(f"Error reading {info_file}: {e}")
    
    difficulty_map = {"E": "Easy", "M": "Medium", "H": "Hard"}
    difficulty_name = difficulty_map.get(difficulty, None)
    
    year_category = None
    if year:
        year_category = "Before 2023" if year < 2023 else "2023-2025"
    
    for llm in llms:
        llm_dir = problem_dir / llm
        results_file = llm_dir / "results.txt"
    
        if not results_file.exists():
            continue

        try:
            with open(results_file, "r") as f:
                lines = [line.strip() for line in f.readlines()]
                
                if len(lines) >= 9:
                    without_correct = 1 if lines[0] == "Y" else 0
                    without_speed = safe_float(lines[1])
                    without_memory = safe_float(lines[2])
                    
                    with_correct = 1 if lines[3] == "Y" else 0
                    with_speed = safe_float(lines[4])
                    with_memory = safe_float(lines[5])
                    
                    random_correct = 1 if lines[6] == "Y" else 0
                    random_speed = safe_float(lines[7])
                    random_memory = safe_float(lines[8])
                    
                    results[llm]["without_rag"]["correct"].append(without_correct)
                    if without_speed > 0:
                        results[llm]["without_rag"]["speed"].append(without_speed)
                    if without_memory > 0:
                        results[llm]["without_rag"]["memory"].append(without_memory)
                    
                    results[llm]["with_rag"]["correct"].append(with_correct)
                    if with_speed > 0:
                        results[llm]["with_rag"]["speed"].append(with_speed)
                    if with_memory > 0:
                        results[llm]["with_rag"]["memory"].append(with_memory)
                    
                    results[llm]["random_rag"]["correct"].append(random_correct)
                    if random_speed > 0:
                        results[llm]["random_rag"]["speed"].append(random_speed)
                    if random_memory > 0:
                        results[llm]["random_rag"]["memory"].append(random_memory)
                    
                    if difficulty_name:
                        results_by_category[difficulty_name]["without_rag"]["correct"].append(without_correct)
                        results_by_category[difficulty_name]["with_rag"]["correct"].append(with_correct)
                        results_by_category[difficulty_name]["random_rag"]["correct"].append(random_correct)
                        if without_speed > 0:
                            results_by_category[difficulty_name]["without_rag"]["speed"].append(without_speed)
                        if with_speed > 0:
                            results_by_category[difficulty_name]["with_rag"]["speed"].append(with_speed)
                        if random_speed > 0:
                            results_by_category[difficulty_name]["random_rag"]["speed"].append(random_speed)
                        if without_memory > 0:
                            results_by_category[difficulty_name]["without_rag"]["memory"].append(without_memory)
                        if with_memory > 0:
                            results_by_category[difficulty_name]["with_rag"]["memory"].append(with_memory)
                        if random_memory > 0:
                            results_by_category[difficulty_name]["random_rag"]["memory"].append(random_memory)
                    
                    if year_category:
                        results_by_category[year_category]["without_rag"]["correct"].append(without_correct)
                        results_by_category[year_category]["with_rag"]["correct"].append(with_correct)
                        results_by_category[year_category]["random_rag"]["correct"].append(random_correct)
                        if without_speed > 0:
                            results_by_category[year_category]["without_rag"]["speed"].append(without_speed)
                        if with_speed > 0:
                            results_by_category[year_category]["with_rag"]["speed"].append(with_speed)
                        if random_speed > 0:
                            results_by_category[year_category]["random_rag"]["speed"].append(random_speed)
                        if without_memory > 0:
                            results_by_category[year_category]["without_rag"]["memory"].append(without_memory)
                        if with_memory > 0:
                            results_by_category[year_category]["with_rag"]["memory"].append(with_memory)
                        if random_memory > 0:
                            results_by_category[year_category]["random_rag"]["memory"].append(random_memory)
                    
        except Exception as e:
            print(f"Error processing {results_file}: {e}")

without_rag_correct_data = []
random_rag_correct_data = []
with_rag_correct_data = []
without_rag_speed_data = []
random_rag_speed_data = []
with_rag_speed_data = []
without_rag_memory_data = []
random_rag_memory_data = []
with_rag_memory_data = []

print("Performance Metrics for LLMs on LeetCode Problems")
print("=" * 50)

for llm in llms:
    print(f"\n{llm}:")
    
    without_rag_correct = statistics.mean(results[llm]["without_rag"]["correct"]) * 100 if results[llm]["without_rag"]["correct"] else 0
    random_rag_correct = statistics.mean(results[llm]["random_rag"]["correct"]) * 100 if results[llm]["random_rag"]["correct"] else 0
    with_rag_correct = statistics.mean(results[llm]["with_rag"]["correct"]) * 100 if results[llm]["with_rag"]["correct"] else 0
    
    without_rag_speed = statistics.mean(results[llm]["without_rag"]["speed"]) if results[llm]["without_rag"]["speed"] else 0
    random_rag_speed = statistics.mean(results[llm]["random_rag"]["speed"]) if results[llm]["random_rag"]["speed"] else 0
    with_rag_speed = statistics.mean(results[llm]["with_rag"]["speed"]) if results[llm]["with_rag"]["speed"] else 0
    
    without_rag_memory = statistics.mean(results[llm]["without_rag"]["memory"]) if results[llm]["without_rag"]["memory"] else 0
    random_rag_memory = statistics.mean(results[llm]["random_rag"]["memory"]) if results[llm]["random_rag"]["memory"] else 0
    with_rag_memory = statistics.mean(results[llm]["with_rag"]["memory"]) if results[llm]["with_rag"]["memory"] else 0
    
    print(f"  Without RAG:")
    print(f"    Correct: {without_rag_correct:.1f}% (Questions processed: {len(results[llm]['without_rag']['correct'])})")
    print(f"    Speed Percentile: {without_rag_speed:.1f}% (Questions processed: {len(results[llm]['without_rag']['speed'])})")
    print(f"    Memory Percentile: {without_rag_memory:.1f}% (Questions processed: {len(results[llm]['without_rag']['memory'])})")
    
    print(f"  Random RAG:")
    print(f"    Correct: {random_rag_correct:.1f}% (Questions processed: {len(results[llm]['random_rag']['correct'])})")
    print(f"    Speed Percentile: {random_rag_speed:.1f}% (Questions processed: {len(results[llm]['random_rag']['speed'])})")
    print(f"    Memory Percentile: {random_rag_memory:.1f}% (Questions processed: {len(results[llm]['random_rag']['memory'])})")
    
    print(f"  With RAG:")
    print(f"    Correct: {with_rag_correct:.1f}% (Questions processed: {len(results[llm]['with_rag']['correct'])})")
    print(f"    Speed Percentile: {with_rag_speed:.1f}% (Questions processed: {len(results[llm]['with_rag']['speed'])})")
    print(f"    Memory Percentile: {with_rag_memory:.1f}% (Questions processed: {len(results[llm]['with_rag']['memory'])})")
    
    without_rag_correct_data.append(without_rag_correct)
    random_rag_correct_data.append(random_rag_correct)
    with_rag_correct_data.append(with_rag_correct)
    without_rag_speed_data.append(without_rag_speed)
    random_rag_speed_data.append(random_rag_speed)
    with_rag_speed_data.append(with_rag_speed)
    without_rag_memory_data.append(without_rag_memory)
    random_rag_memory_data.append(random_rag_memory)
    with_rag_memory_data.append(with_rag_memory)

print("\nSummary Table:")
print("-" * 180)
print(f"{'Model':<20} | {'Without RAG':^58} | {'Random RAG':^58} | {'With RAG':^58}")
print(f"{'':<20} | {'Correct':^18} | {'Speed':^18} | {'Memory':^18} | {'Correct':^18} | {'Speed':^18} | {'Memory':^18} | {'Correct':^18} | {'Speed':^18} | {'Memory':^18}")
print("-" * 180)

for llm in llms:
    without_rag_correct = statistics.mean(results[llm]["without_rag"]["correct"]) * 100 if results[llm]["without_rag"]["correct"] else 0
    random_rag_correct = statistics.mean(results[llm]["random_rag"]["correct"]) * 100 if results[llm]["random_rag"]["correct"] else 0
    with_rag_correct = statistics.mean(results[llm]["with_rag"]["correct"]) * 100 if results[llm]["with_rag"]["correct"] else 0
    
    without_rag_speed = statistics.mean(results[llm]["without_rag"]["speed"]) if results[llm]["without_rag"]["speed"] else 0
    random_rag_speed = statistics.mean(results[llm]["random_rag"]["speed"]) if results[llm]["random_rag"]["speed"] else 0
    with_rag_speed = statistics.mean(results[llm]["with_rag"]["speed"]) if results[llm]["with_rag"]["speed"] else 0
    
    without_rag_memory = statistics.mean(results[llm]["without_rag"]["memory"]) if results[llm]["without_rag"]["memory"] else 0
    random_rag_memory = statistics.mean(results[llm]["random_rag"]["memory"]) if results[llm]["random_rag"]["memory"] else 0
    with_rag_memory = statistics.mean(results[llm]["with_rag"]["memory"]) if results[llm]["with_rag"]["memory"] else 0
    
    print(f"{llm:<20} | {without_rag_correct:^10.1f}% ({len(results[llm]['without_rag']['correct'])}) | {without_rag_speed:^10.1f}% ({len(results[llm]['without_rag']['speed'])}) | {without_rag_memory:^10.1f}% ({len(results[llm]['without_rag']['memory'])}) | {random_rag_correct:^10.1f}% ({len(results[llm]['random_rag']['correct'])}) | {random_rag_speed:^10.1f}% ({len(results[llm]['random_rag']['speed'])}) | {random_rag_memory:^10.1f}% ({len(results[llm]['random_rag']['memory'])}) | {with_rag_correct:^10.1f}% ({len(results[llm]['with_rag']['correct'])}) | {with_rag_speed:^10.1f}% ({len(results[llm]['with_rag']['speed'])}) | {with_rag_memory:^10.1f}% ({len(results[llm]['with_rag']['memory'])})")

def create_grouped_bar_chart(data1, data2, data3, title, ylabel):
    plt.figure(figsize=(14, 6))
    
    x = np.arange(len(llms))
    width = 0.25
    
    plt.bar(x - width, data1, width, label='Without RAG', color='skyblue')
    plt.bar(x, data2, width, label='Random RAG', color='lightcoral')
    plt.bar(x + width, data3, width, label='With RAG', color='orange')
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('LLM Model')
    plt.xticks(x, llms, rotation=45, ha='right')
    plt.ylim(0, 100)  
    plt.legend()
    plt.tight_layout()
    
    for i, v in enumerate(data1):
        plt.text(i - width, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(data2):
        plt.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(data3):
        plt.text(i + width, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.savefig(images_dir / f'{title.replace(" ", "_").lower()}.png', dpi=300)
    plt.show()

print("\nDisplaying charts one by one. Close each chart to see the next one.")

create_grouped_bar_chart(without_rag_correct_data, random_rag_correct_data, with_rag_correct_data, 
                        'Correctness by LLM', 'Correct (%)')

create_grouped_bar_chart(without_rag_speed_data, random_rag_speed_data, with_rag_speed_data, 
                        'Speed Percentile by LLM', 'Speed Percentile (%)')

create_grouped_bar_chart(without_rag_memory_data, random_rag_memory_data, with_rag_memory_data, 
                        'Memory Percentile by LLM', 'Memory Percentile (%)')

plt.figure(figsize=(18, 10))
x = np.arange(len(llms))
width = 0.1

plt.bar(x - width*3, without_rag_correct_data, width, label='Correctness (Without RAG)', color='skyblue')
plt.bar(x - width*2, without_rag_speed_data, width, label='Speed (Without RAG)', color='lightgreen')
plt.bar(x - width, without_rag_memory_data, width, label='Memory (Without RAG)', color='lightpink')
plt.bar(x, random_rag_correct_data, width, label='Correctness (Random RAG)', color='lightcoral')
plt.bar(x + width, random_rag_speed_data, width, label='Speed (Random RAG)', color='gold')
plt.bar(x + width*2, random_rag_memory_data, width, label='Memory (Random RAG)', color='mediumpurple')
plt.bar(x + width*3, with_rag_correct_data, width, label='Correctness (With RAG)', color='blue')
plt.bar(x + width*4, with_rag_speed_data, width, label='Speed (With RAG)', color='green')
plt.bar(x + width*5, with_rag_memory_data, width, label='Memory (With RAG)', color='red')

plt.title('LLM Performance Metrics Comparison')
plt.ylabel('Percentage (%)')
plt.xlabel('LLM Model')
plt.xticks(x + width, llms)
plt.ylim(0, 100)  
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

plt.tight_layout()
plt.savefig(images_dir / 'llm_combined_metrics.png', dpi=300)
plt.show()

correctness_improvements = []
speed_improvements = []
memory_improvements = []
random_correctness_improvements = []
random_speed_improvements = []
random_memory_improvements = []

print("\n" + "="*60)
print("AVERAGE IMPROVEMENTS WITH RAG")
print("="*60)

for llm in llms:
    without_rag_correct = statistics.mean(results[llm]["without_rag"]["correct"]) * 100 if results[llm]["without_rag"]["correct"] else 0
    random_rag_correct = statistics.mean(results[llm]["random_rag"]["correct"]) * 100 if results[llm]["random_rag"]["correct"] else 0
    with_rag_correct = statistics.mean(results[llm]["with_rag"]["correct"]) * 100 if results[llm]["with_rag"]["correct"] else 0
    
    without_rag_speed = statistics.mean(results[llm]["without_rag"]["speed"]) if results[llm]["without_rag"]["speed"] else 0
    random_rag_speed = statistics.mean(results[llm]["random_rag"]["speed"]) if results[llm]["random_rag"]["speed"] else 0
    with_rag_speed = statistics.mean(results[llm]["with_rag"]["speed"]) if results[llm]["with_rag"]["speed"] else 0
    
    without_rag_memory = statistics.mean(results[llm]["without_rag"]["memory"]) if results[llm]["without_rag"]["memory"] else 0
    random_rag_memory = statistics.mean(results[llm]["random_rag"]["memory"]) if results[llm]["random_rag"]["memory"] else 0
    with_rag_memory = statistics.mean(results[llm]["with_rag"]["memory"]) if results[llm]["with_rag"]["memory"] else 0
    
    correctness_improvement = with_rag_correct - without_rag_correct
    speed_improvement = with_rag_speed - without_rag_speed
    memory_improvement = with_rag_memory - without_rag_memory
    
    random_correctness_improvement = random_rag_correct - without_rag_correct
    random_speed_improvement = random_rag_speed - without_rag_speed
    random_memory_improvement = random_rag_memory - without_rag_memory
    
    if results[llm]["without_rag"]["correct"] and results[llm]["with_rag"]["correct"]:
        correctness_improvements.append(correctness_improvement)
    if results[llm]["without_rag"]["correct"] and results[llm]["random_rag"]["correct"]:
        random_correctness_improvements.append(random_correctness_improvement)
    if with_rag_speed > 0 and without_rag_speed > 0:
        speed_improvements.append(speed_improvement)
    if random_rag_speed > 0 and without_rag_speed > 0:
        random_speed_improvements.append(random_speed_improvement)
    if with_rag_memory > 0 and without_rag_memory > 0:
        memory_improvements.append(memory_improvement)
    if random_rag_memory > 0 and without_rag_memory > 0:
        random_memory_improvements.append(random_memory_improvement)
    
    print(f"{llm}:")
    print(f"  Random RAG improvements over No RAG:")
    print(f"    Correctness improvement: {random_correctness_improvement:+.1f} percentage points")
    print(f"    Speed improvement: {random_speed_improvement:+.1f} percentage points")
    print(f"    Memory improvement: {random_memory_improvement:+.1f} percentage points")
    print(f"  Similar RAG improvements over No RAG:")
    print(f"    Correctness improvement: {correctness_improvement:+.1f} percentage points")
    print(f"    Speed improvement: {speed_improvement:+.1f} percentage points")
    print(f"    Memory improvement: {memory_improvement:+.1f} percentage points")

avg_correctness_improvement = statistics.mean(correctness_improvements) if correctness_improvements else 0
avg_speed_improvement = statistics.mean(speed_improvements) if speed_improvements else 0
avg_memory_improvement = statistics.mean(memory_improvements) if memory_improvements else 0

avg_random_correctness_improvement = statistics.mean(random_correctness_improvements) if random_correctness_improvements else 0
avg_random_speed_improvement = statistics.mean(random_speed_improvements) if random_speed_improvements else 0
avg_random_memory_improvement = statistics.mean(random_memory_improvements) if random_memory_improvements else 0

print("\n" + "-"*60)
print("OVERALL AVERAGE IMPROVEMENTS:")
print(f"Random RAG over No RAG:")
print(f"  Correctness: {avg_random_correctness_improvement:+.1f} percentage points")
print(f"  Speed: {avg_random_speed_improvement:+.1f} percentage points")
print(f"  Memory: {avg_random_memory_improvement:+.1f} percentage points")
print(f"Similar RAG over No RAG:")
print(f"  Correctness: {avg_correctness_improvement:+.1f} percentage points")
print(f"  Speed: {avg_speed_improvement:+.1f} percentage points")
print(f"  Memory: {avg_memory_improvement:+.1f} percentage points")
print("-"*60)

print("\n" + "="*80)
print("PERFORMANCE BY DIFFICULTY AND YEAR")
print("="*80)

print(f"{'Category':<15} | {'No RAG':^30} | {'Random RAG':^30} | {'With RAG':^30}")
print(f"{'':<15} | {'Correct':^9} | {'Speed':^9} | {'Memory':^9} | {'Correct':^9} | {'Speed':^9} | {'Memory':^9} | {'Correct':^9} | {'Speed':^9} | {'Memory':^9}")
print("-" * 110)

categories = ["Easy", "Medium", "Hard", "Before 2023", "2023-2025"]

for category in categories:
    if category in results_by_category:
        data = results_by_category[category]
        
        without_correct = statistics.mean(data["without_rag"]["correct"]) * 100 if data["without_rag"]["correct"] else 0
        without_speed = statistics.mean(data["without_rag"]["speed"]) if data["without_rag"]["speed"] else 0
        without_memory = statistics.mean(data["without_rag"]["memory"]) if data["without_rag"]["memory"] else 0
        
        random_correct = statistics.mean(data["random_rag"]["correct"]) * 100 if data["random_rag"]["correct"] else 0
        random_speed = statistics.mean(data["random_rag"]["speed"]) if data["random_rag"]["speed"] else 0
        random_memory = statistics.mean(data["random_rag"]["memory"]) if data["random_rag"]["memory"] else 0
        
        with_correct = statistics.mean(data["with_rag"]["correct"]) * 100 if data["with_rag"]["correct"] else 0
        with_speed = statistics.mean(data["with_rag"]["speed"]) if data["with_rag"]["speed"] else 0
        with_memory = statistics.mean(data["with_rag"]["memory"]) if data["with_rag"]["memory"] else 0
        
        print(f"{category:<15} | {without_correct:^9.1f} | {without_speed:^9.1f} | {without_memory:^9.1f} | {random_correct:^9.1f} | {random_speed:^9.1f} | {random_memory:^9.1f} | {with_correct:^9.1f} | {with_speed:^9.1f} | {with_memory:^9.1f}")

print("-" * 110)