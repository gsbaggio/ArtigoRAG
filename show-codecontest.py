import os
from pathlib import Path
import statistics
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import json

def elo_map_rating(results, problem_elos, codeforces_ratings=None, mu0=None, sigma0=None, eps=0.1):
    if codeforces_ratings is not None and mu0 is None and sigma0 is None:
        mu0 = np.mean(codeforces_ratings)
        sigma0 = np.std(codeforces_ratings)
        print(f"Using empirical prior: mu0={mu0:.1f}, sigma0={sigma0:.1f}")
    elif mu0 is None or sigma0 is None:
        mu0 = 1500
        sigma0 = 400
        print(f"Using traditional prior: mu0={mu0:.1f}, sigma0={sigma0:.1f}")
    
    y = np.array(results)
    d = np.array(problem_elos)

    def pi(r):
        return 1.0 / (1.0 + 10.0 ** ((d - r) / 400.0))

    def L(r):
        P = pi(r)
        ll = np.sum(y * np.log(P + 1e-12) + (1 - y) * np.log(1 - P + 1e-12))
        prior = -((r - mu0) ** 2) / (2 * sigma0 ** 2)
        return ll + prior

    def negL(r):
        return -L(r[0])

    res = minimize(negL, x0=np.array([mu0]), method="L-BFGS-B")
    r_hat = res.x[0]

    L_plus = L(r_hat + eps)
    L_mid = L(r_hat)
    L_minus = L(r_hat - eps)

    I = -(L_plus - 2 * L_mid + L_minus) / (eps ** 2)

    std = 1.0 / np.sqrt(I) if I > 0 else np.inf

    return r_hat, std

def get_problem_elo(problem_dir):
    info_file = problem_dir / "info.txt"
    if info_file.exists():
        try:
            with open(info_file, "r", encoding='utf-8') as f:
                first_line = f.readline().strip()
                return int(first_line)
        except (ValueError, FileNotFoundError, UnicodeDecodeError):
            return None
    return None

def get_problem_categories(problem_dir):
    info_file = problem_dir / "info.txt"
    if info_file.exists():
        try:
            with open(info_file, "r", encoding='utf-8') as f:
                f.readline()  
                second_line = f.readline().strip()
                import ast
                categories = ast.literal_eval(second_line)
                return categories if isinstance(categories, list) else []
        except (ValueError, FileNotFoundError, UnicodeDecodeError, SyntaxError):
            return []
    return []

def categorize_difficulty(elo):
    if elo <= 1000:
        return "Easy"
    elif elo <= 2000:
        return "Medium"
    else:
        return "Hard"

def calculate_difficulty_breakdown(detailed_results):
    breakdown = {"Easy": {"solved": 0, "total": 0}, 
                 "Medium": {"solved": 0, "total": 0}, 
                 "Hard": {"solved": 0, "total": 0}}
    
    for result, elo in detailed_results:
        difficulty = categorize_difficulty(elo)
        breakdown[difficulty]["total"] += 1
        if result == 1:
            breakdown[difficulty]["solved"] += 1
    
    for difficulty in breakdown:
        total = breakdown[difficulty]["total"]
        solved = breakdown[difficulty]["solved"]
        breakdown[difficulty]["percentage"] = (solved / total * 100) if total > 0 else 0
    
    return breakdown

def load_codeforces_ratings():
    try:
        with open('usersRating.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        ratings = [user.get('rating', 0) for user in data['result'] if 'rating' in user]
        ratings.sort()  
        
        print(f"Loaded {len(ratings)} CodeForces user ratings")
        print(f"Rating range: {min(ratings)} to {max(ratings)}")
        
        return ratings
    except FileNotFoundError:
        print("Warning: usersRating.json not found. Percentile calculation will be skipped.")
        return None
    except Exception as e:
        print(f"Error loading CodeForces ratings: {e}")
        return None

def calculate_codeforces_percentile(model_rating, codeforces_ratings):
    if codeforces_ratings is None:
        return None
    
    lower_count = sum(1 for rating in codeforces_ratings if rating < model_rating)
    
    percentile = (lower_count / len(codeforces_ratings)) * 100
    
    return percentile

def safe_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0

def calculate_fair_averages(llm_results):
    problem_results = llm_results["problem_results"]
    
    common_solved_problems = []
    for problem, scenarios in problem_results.items():
        if (len(scenarios) == 3 and 
            scenarios.get("without_rag", {}).get("correct", 0) == 1 and
            scenarios.get("random_rag", {}).get("correct", 0) == 1 and
            scenarios.get("with_rag", {}).get("correct", 0) == 1):
            common_solved_problems.append(problem)
    
    fair_averages = {
        "without_rag": {"time": 0, "memory": 0, "count": 0},
        "random_rag": {"time": 0, "memory": 0, "count": 0},
        "with_rag": {"time": 0, "memory": 0, "count": 0}
    }
    
    for scenario in ["without_rag", "random_rag", "with_rag"]:
        times = []
        memories = []
        
        for problem in common_solved_problems:
            problem_data = problem_results[problem][scenario]
            if problem_data["time"] > 0:
                times.append(problem_data["time"])
            if problem_data["memory"] > 0:
                memories.append(problem_data["memory"])
        
        fair_averages[scenario]["time"] = statistics.mean(times) if times else 0
        fair_averages[scenario]["memory"] = statistics.mean(memories) if memories else 0
        fair_averages[scenario]["count"] = len(common_solved_problems)
    
    return fair_averages, common_solved_problems

root_dir = Path("data/CodeContest")

llms = ["Claude 3.7 Sonnet", "Gemini 2.0", "GPT 4o", "GPT-3.5", "DeepSeek R1", "Qwen2.5-Coder"]

images_dir = Path("images-contest")
os.makedirs(images_dir, exist_ok=True)

codeforces_ratings = load_codeforces_ratings()

if codeforces_ratings:
    empirical_mean = np.mean(codeforces_ratings)
    empirical_std = np.std(codeforces_ratings)
    print(f"Empirical CodeForces prior: mu0={empirical_mean:.1f}, sigma0={empirical_std:.1f}")
    print(f"Traditional prior would be: mu0=1500, sigma0=400")
    print(f"Difference: mu0 is {empirical_mean-1500:+.1f}, sigma0 is {empirical_std-400:+.1f}")

results = {
    llm: {
        "without_rag": {
            "correct": [],
            "time": [],  # Raw time values in ms
            "memory": [],  # Raw memory values in KB
            "detailed_results": [],  # For Elo calculation: list of (result, problem_elo) tuples
            "problem_elos": []  # Just the elos for solved problems
        },
        "random_rag": {
            "correct": [],
            "time": [],  # Raw time values in ms
            "memory": [],  # Raw memory values in KB
            "detailed_results": [],
            "problem_elos": []
        },
        "with_rag": {
            "correct": [],
            "time": [],  # Raw time values in ms
            "memory": [],  # Raw memory values in KB
            "detailed_results": [],
            "problem_elos": []
        },
        "problem_results": {}  # problem_name -> {scenario -> {correct, time, memory}}
    } for llm in llms
}

problems = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

print("Analyzing problem categories...")
category_counts = {}
category_to_problems = {}

for problem in problems:
    problem_dir = root_dir / problem
    
    problem_elo = get_problem_elo(problem_dir)
    problem_categories = get_problem_categories(problem_dir)
    
    if problem_elo is None or not problem_categories:
        continue
    
    for category in problem_categories:
        if category not in category_counts:
            category_counts[category] = 0
            category_to_problems[category] = []
        category_counts[category] += 1
        category_to_problems[category].append((problem, problem_elo))

top_15_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:15]
print(f"\nTop 15 categories by frequency:")
for i, (category, count) in enumerate(top_15_categories, 1):
    print(f"{i:2d}. {category}: {count} problems")

top_15_category_names = [cat for cat, count in top_15_categories]

deepseek_category_results = {
    category: {
        "without_rag": {"detailed_results": []},
        "random_rag": {"detailed_results": []}, 
        "with_rag": {"detailed_results": []}
    } for category in top_15_category_names
}

for problem in problems:
    problem_dir = root_dir / problem
    
    problem_elo = get_problem_elo(problem_dir)
    problem_categories = get_problem_categories(problem_dir)
    
    if problem_elo is None:
        continue  
    
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
                    without_time = safe_float(lines[1])  # Raw time in ms
                    without_memory = safe_float(lines[2])  # Raw memory in KB
                    
                    with_correct = 1 if lines[3] == "Y" else 0
                    with_time = safe_float(lines[4])  # Raw time in ms
                    with_memory = safe_float(lines[5])  # Raw memory in KB
                    
                    random_correct = 1 if lines[6] == "Y" else 0
                    random_time = safe_float(lines[7])  # Raw time in ms
                    random_memory = safe_float(lines[8])  # Raw memory in KB
                    
                    # Store original data
                    results[llm]["without_rag"]["correct"].append(without_correct)
                    if without_time > 0:
                        results[llm]["without_rag"]["time"].append(without_time)
                    if without_memory > 0:
                        results[llm]["without_rag"]["memory"].append(without_memory)
                    
                    results[llm]["with_rag"]["correct"].append(with_correct)
                    if with_time > 0:
                        results[llm]["with_rag"]["time"].append(with_time)
                    if with_memory > 0:
                        results[llm]["with_rag"]["memory"].append(with_memory)
                    
                    results[llm]["random_rag"]["correct"].append(random_correct)
                    if random_time > 0:
                        results[llm]["random_rag"]["time"].append(random_time)
                    if random_memory > 0:
                        results[llm]["random_rag"]["memory"].append(random_memory)
                    
                    if problem not in results[llm]["problem_results"]:
                        results[llm]["problem_results"][problem] = {}
                    
                    results[llm]["problem_results"][problem]["without_rag"] = {
                        "correct": without_correct,
                        "time": without_time,
                        "memory": without_memory
                    }
                    results[llm]["problem_results"][problem]["with_rag"] = {
                        "correct": with_correct,
                        "time": with_time,
                        "memory": with_memory
                    }
                    results[llm]["problem_results"][problem]["random_rag"] = {
                        "correct": random_correct,
                        "time": random_time,
                        "memory": random_memory
                    }
                    
                    results[llm]["without_rag"]["detailed_results"].append((without_correct, problem_elo))
                    results[llm]["with_rag"]["detailed_results"].append((with_correct, problem_elo))
                    results[llm]["random_rag"]["detailed_results"].append((random_correct, problem_elo))
                    
                    results[llm]["without_rag"]["problem_elos"].append(problem_elo)
                    results[llm]["with_rag"]["problem_elos"].append(problem_elo)
                    results[llm]["random_rag"]["problem_elos"].append(problem_elo)
                    
                    if llm == "DeepSeek R1" and problem_categories:
                        for category in problem_categories:
                            if category in top_15_category_names:
                                deepseek_category_results[category]["without_rag"]["detailed_results"].append((without_correct, problem_elo))
                                deepseek_category_results[category]["random_rag"]["detailed_results"].append((random_correct, problem_elo))
                                deepseek_category_results[category]["with_rag"]["detailed_results"].append((with_correct, problem_elo))
                    
        except Exception as e:
            print(f"Error processing {results_file}: {e}")

without_rag_correct_data = []
random_rag_correct_data = []
with_rag_correct_data = []
without_rag_time_data = []  # Average time in ms
random_rag_time_data = []
with_rag_time_data = []
without_rag_memory_data = []  # Average memory in KB
random_rag_memory_data = []
with_rag_memory_data = []

print("Performance Metrics for LLMs on LeetCode Problems")
print("=" * 50)

for llm in llms:
    print(f"\n{llm}:")
    
    without_rag_correct = statistics.mean(results[llm]["without_rag"]["correct"]) * 100 if results[llm]["without_rag"]["correct"] else 0
    random_rag_correct = statistics.mean(results[llm]["random_rag"]["correct"]) * 100 if results[llm]["random_rag"]["correct"] else 0
    with_rag_correct = statistics.mean(results[llm]["with_rag"]["correct"]) * 100 if results[llm]["with_rag"]["correct"] else 0
    
    fair_averages, common_solved_problems = calculate_fair_averages(results[llm])
    
    without_rag_time = fair_averages["without_rag"]["time"]
    random_rag_time = fair_averages["random_rag"]["time"]
    with_rag_time = fair_averages["with_rag"]["time"]
    
    without_rag_memory = fair_averages["without_rag"]["memory"]
    random_rag_memory = fair_averages["random_rag"]["memory"]
    with_rag_memory = fair_averages["with_rag"]["memory"]
    
    common_count = len(common_solved_problems)
    
    print(f"  Without RAG:")
    print(f"    Correct: {without_rag_correct:.1f}% (Questions processed: {len(results[llm]['without_rag']['correct'])})")
    print(f"    Average Time: {without_rag_time:.1f} ms (Fair comparison on {common_count} common problems)")
    print(f"    Average Memory: {without_rag_memory:.1f} KB (Fair comparison on {common_count} common problems)")
    
    print(f"  Random RAG:")
    print(f"    Correct: {random_rag_correct:.1f}% (Questions processed: {len(results[llm]['random_rag']['correct'])})")
    print(f"    Average Time: {random_rag_time:.1f} ms (Fair comparison on {common_count} common problems)")
    print(f"    Average Memory: {random_rag_memory:.1f} KB (Fair comparison on {common_count} common problems)")
    
    print(f"  With RAG:")
    print(f"    Correct: {with_rag_correct:.1f}% (Questions processed: {len(results[llm]['with_rag']['correct'])})")
    print(f"    Average Time: {with_rag_time:.1f} ms (Fair comparison on {common_count} common problems)")
    print(f"    Average Memory: {with_rag_memory:.1f} KB (Fair comparison on {common_count} common problems)")
    
    without_rag_correct_data.append(without_rag_correct)
    random_rag_correct_data.append(random_rag_correct)
    with_rag_correct_data.append(with_rag_correct)
    without_rag_time_data.append(without_rag_time)
    random_rag_time_data.append(random_rag_time)
    with_rag_time_data.append(with_rag_time)
    without_rag_memory_data.append(without_rag_memory)
    random_rag_memory_data.append(random_rag_memory)
    with_rag_memory_data.append(with_rag_memory)

print("\nSummary Table:")
print("-" * 200)
print(f"{'Model':<20} | {'Without RAG':^58} | {'Random RAG':^58} | {'With RAG':^58}")
print(f"{'':<20} | {'Correct':^18} | {'Time (ms)*':^18} | {'Memory (KB)*':^18} | {'Correct':^18} | {'Time (ms)*':^18} | {'Memory (KB)*':^18} | {'Correct':^18} | {'Time (ms)*':^18} | {'Memory (KB)*':^18}")
print("-" * 200)

for llm in llms:
    without_rag_correct = statistics.mean(results[llm]["without_rag"]["correct"]) * 100 if results[llm]["without_rag"]["correct"] else 0
    random_rag_correct = statistics.mean(results[llm]["random_rag"]["correct"]) * 100 if results[llm]["random_rag"]["correct"] else 0
    with_rag_correct = statistics.mean(results[llm]["with_rag"]["correct"]) * 100 if results[llm]["with_rag"]["correct"] else 0
    
    fair_averages, common_solved_problems = calculate_fair_averages(results[llm])
    
    without_rag_time = fair_averages["without_rag"]["time"]
    random_rag_time = fair_averages["random_rag"]["time"]
    with_rag_time = fair_averages["with_rag"]["time"]
    
    without_rag_memory = fair_averages["without_rag"]["memory"]
    random_rag_memory = fair_averages["random_rag"]["memory"]
    with_rag_memory = fair_averages["with_rag"]["memory"]
    
    common_count = len(common_solved_problems)
    
    print(f"{llm:<20} | {without_rag_correct:^8.1f}% ({len(results[llm]['without_rag']['correct'])}) | {without_rag_time:^8.0f} ({common_count}) | {without_rag_memory:^8.0f} ({common_count}) | {random_rag_correct:^8.1f}% ({len(results[llm]['random_rag']['correct'])}) | {random_rag_time:^8.0f} ({common_count}) | {random_rag_memory:^8.0f} ({common_count}) | {with_rag_correct:^8.1f}% ({len(results[llm]['with_rag']['correct'])}) | {with_rag_time:^8.0f} ({common_count}) | {with_rag_memory:^8.0f} ({common_count})")

print("\n* Time and Memory averages calculated only on problems solved correctly by ALL three approaches for fair comparison.")

print("\n" + "="*80)
print("MAP ELO RATINGS FOR EACH MODEL AND SCENARIO")
print("="*80)

for llm in llms:
    print(f"\n{llm}:")
    
    for scenario in ["without_rag", "random_rag", "with_rag"]:
        scenario_name = {
            "without_rag": "Without RAG",
            "random_rag": "Random RAG", 
            "with_rag": "With RAG"
        }[scenario]
        
        detailed_results = results[llm][scenario]["detailed_results"]
        
        if len(detailed_results) > 0:
            model_results = [result for result, elo in detailed_results]
            problem_elos = [elo for result, elo in detailed_results]
            
            try:
                elo_rating, elo_std = elo_map_rating(model_results, problem_elos, codeforces_ratings)
                
                solved_count = sum(model_results)
                total_count = len(model_results)
                success_rate = (solved_count / total_count * 100) if total_count > 0 else 0
                
                cf_percentile = calculate_codeforces_percentile(elo_rating, codeforces_ratings)
                
                print(f"  {scenario_name}:")
                print(f"    MAP Elo Rating: {elo_rating:.1f} ± {elo_std:.1f}")
                if cf_percentile is not None:
                    print(f"    CodeForces Percentile: {cf_percentile:.1f}% (better than {cf_percentile:.1f}% of CF users)")
                print(f"    Problems Solved: {solved_count}/{total_count} ({success_rate:.1f}%)")
                print(f"    Average Problem Elo: {np.mean(problem_elos):.1f}")
                
            except Exception as e:
                print(f"  {scenario_name}: Error calculating Elo - {e}")
        else:
            print(f"  {scenario_name}: No data available")

print("\n" + "="*140)
print("DETAILED PERFORMANCE AND MAP ELO BREAKDOWN BY DIFFICULTY")
print("="*140)

all_scenario_data = {}

for scenario in ["without_rag", "random_rag", "with_rag"]:
    scenario_name = {
        "without_rag": "Without RAG",
        "random_rag": "Random RAG", 
        "with_rag": "With RAG"
    }[scenario]
    
    all_scenario_data[scenario] = {"name": scenario_name, "models": {}}
    
    for llm in llms:
        detailed_results = results[llm][scenario]["detailed_results"]
        
        if len(detailed_results) > 0:
            model_results = [result for result, elo in detailed_results]
            problem_elos = [elo for result, elo in detailed_results]
            
            try:
                elo_rating, elo_std = elo_map_rating(model_results, problem_elos, codeforces_ratings)
                elo_str = f"{elo_rating:.0f}±{elo_std:.0f}"
                
                cf_percentile = calculate_codeforces_percentile(elo_rating, codeforces_ratings)
                cf_percentile_str = f"({cf_percentile:.1f}%)" if cf_percentile is not None else ""
                
                elo_display = f"{elo_str} {cf_percentile_str}".strip()
            except:
                elo_display = "Error"
            
            breakdown = calculate_difficulty_breakdown(detailed_results)
            
            all_scenario_data[scenario]["models"][llm] = {
                "elo": elo_display,
                "breakdown": breakdown
            }
        else:
            all_scenario_data[scenario]["models"][llm] = {
                "elo": "No data",
                "breakdown": {"Easy": {"percentage": 0}, "Medium": {"percentage": 0}, "Hard": {"percentage": 0}}
            }

for scenario in ["without_rag", "random_rag", "with_rag"]:
    scenario_data = all_scenario_data[scenario]
    
    print(f"\n{scenario_data['name']}:")
    print("-" * 140)
    print(f"{'Model':<20} | {'Easy (0-1000)':<15} | {'Medium (1001-2000)':<20} | {'Hard (2001+)':<15} | {'MAP Elo Rating (CF %ile)':<25}")
    print("-" * 140)
    
    for llm in llms:
        model_data = scenario_data["models"][llm]
        easy_pct = model_data["breakdown"]["Easy"]["percentage"]
        medium_pct = model_data["breakdown"]["Medium"]["percentage"]
        hard_pct = model_data["breakdown"]["Hard"]["percentage"]
        elo = model_data["elo"]
        
        print(f"{llm:<20} | {easy_pct:>12.1f}% | {medium_pct:>17.1f}% | {hard_pct:>12.1f}% | {elo:<25}")

print(f"\n" + "="*140)
print("SUMMARY COMPARISON TABLE")
print("="*140)
print(f"{'Scenario':<15} | {'Model':<20} | {'Easy':<8} | {'Medium':<8} | {'Hard':<8} | {'Overall':<8} | {'MAP Elo (CF %ile)':<20}")
print("-" * 140)

for scenario in ["without_rag", "random_rag", "with_rag"]:
    scenario_data = all_scenario_data[scenario]
    
    for i, llm in enumerate(llms):
        model_data = scenario_data["models"][llm]
        easy_pct = model_data["breakdown"]["Easy"]["percentage"]
        medium_pct = model_data["breakdown"]["Medium"]["percentage"]
        hard_pct = model_data["breakdown"]["Hard"]["percentage"]
        elo = model_data["elo"]
        
        detailed_results = results[llm][scenario]["detailed_results"]
        if len(detailed_results) > 0:
            overall_pct = sum(result for result, _ in detailed_results) / len(detailed_results) * 100
        else:
            overall_pct = 0
        
        scenario_display = scenario_data['name'] if i == 0 else ""
        print(f"{scenario_display:<15} | {llm:<20} | {easy_pct:>6.1f}% | {medium_pct:>6.1f}% | {hard_pct:>6.1f}% | {overall_pct:>6.1f}% | {elo:<20}")
    
    if scenario != "with_rag":  
        print("-" * 140)

def create_grouped_bar_chart(data1, data2, data3, title, ylabel):
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(llms))
    width = 0.25
    
    plt.bar(x - width, data1, width, label='Zero-shot', color='skyblue')
    plt.bar(x, data2, width, label='Random-retrieval', color='lightcoral')
    plt.bar(x + width, data3, width, label='RAG approach', color='orange')

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.xlabel('Model', fontsize=14, fontweight='bold')
    plt.xticks(x, llms, rotation=45, ha='right', fontsize=13)
    plt.yticks(fontsize=13)
    
    if 'Correctness' in title:
        plt.ylim(0, 100)
        value_format = lambda v: f'{v:.1f}%'
        text_offset = 2
    elif 'Time' in title:
        all_data = [v for v in data1 + data2 + data3 if v > 0]
        max_val = max(all_data) if all_data else 100
        plt.ylim(0, max_val * 1.1) 
        value_format = lambda v: f'{v:.0f}' if v > 0 else '0'
        text_offset = max_val * 0.02  
    elif 'Memory' in title:
        all_data = [v for v in data1 + data2 + data3 if v > 0]
        max_val = max(all_data) if all_data else 1000
        plt.ylim(0, max_val * 1.1)  
        value_format = lambda v: f'{v:.0f}' if v > 0 else '0'
        text_offset = max_val * 0.02  
    else:
        plt.ylim(0, 100)
        value_format = lambda v: f'{v:.1f}%'
        text_offset = 2
    
    plt.legend()
    plt.tight_layout()
    
    for i, v in enumerate(data1):
        plt.text(i - width, v + text_offset, value_format(v), ha='center', va='bottom', fontsize=11)
    for i, v in enumerate(data2):
        plt.text(i, v + text_offset, value_format(v), ha='center', va='bottom', fontsize=11)
    for i, v in enumerate(data3):
        plt.text(i + width, v + text_offset, value_format(v), ha='center', va='bottom', fontsize=11)

    plt.savefig(images_dir / f'{title.replace(" ", "_").lower()}.png', dpi=300)
    plt.show()

print("\nDisplaying charts one by one. Close each chart to see the next one.")

create_grouped_bar_chart(without_rag_correct_data, random_rag_correct_data, with_rag_correct_data, 
                        'Correctness by LLM', 'Correct (%)')

create_grouped_bar_chart(without_rag_time_data, random_rag_time_data, with_rag_time_data, 
                        'Time by LLM', 'Time (ms)')

create_grouped_bar_chart(without_rag_memory_data, random_rag_memory_data, with_rag_memory_data, 
                        'Memory by LLM', 'Memory (KB)')

correctness_improvements = []
time_improvements = []
memory_improvements = []
random_correctness_improvements = []
random_time_improvements = []
random_memory_improvements = []

print("\n" + "="*60)
print("AVERAGE IMPROVEMENTS WITH RAG")
print("="*60)

for llm in llms:
    without_rag_correct = statistics.mean(results[llm]["without_rag"]["correct"]) * 100 if results[llm]["without_rag"]["correct"] else 0
    random_rag_correct = statistics.mean(results[llm]["random_rag"]["correct"]) * 100 if results[llm]["random_rag"]["correct"] else 0
    with_rag_correct = statistics.mean(results[llm]["with_rag"]["correct"]) * 100 if results[llm]["with_rag"]["correct"] else 0
    
    fair_averages, common_solved_problems = calculate_fair_averages(results[llm])
    
    without_rag_time = fair_averages["without_rag"]["time"]
    random_rag_time = fair_averages["random_rag"]["time"]
    with_rag_time = fair_averages["with_rag"]["time"]
    
    without_rag_memory = fair_averages["without_rag"]["memory"]
    random_rag_memory = fair_averages["random_rag"]["memory"]
    with_rag_memory = fair_averages["with_rag"]["memory"]
    
    correctness_improvement = with_rag_correct - without_rag_correct
    time_improvement = with_rag_time - without_rag_time  
    memory_improvement = with_rag_memory - without_rag_memory  
    
    random_correctness_improvement = random_rag_correct - without_rag_correct
    random_time_improvement = random_rag_time - without_rag_time
    random_memory_improvement = random_rag_memory - without_rag_memory
    
    if results[llm]["without_rag"]["correct"] and results[llm]["with_rag"]["correct"]:
        correctness_improvements.append(correctness_improvement)
    if results[llm]["without_rag"]["correct"] and results[llm]["random_rag"]["correct"]:
        random_correctness_improvements.append(random_correctness_improvement)
    if with_rag_time > 0 and without_rag_time > 0:
        time_improvements.append(time_improvement)
    if random_rag_time > 0 and without_rag_time > 0:
        random_time_improvements.append(random_time_improvement)
    if with_rag_memory > 0 and without_rag_memory > 0:
        memory_improvements.append(memory_improvement)
    if random_rag_memory > 0 and without_rag_memory > 0:
        random_memory_improvements.append(random_memory_improvement)
    
    common_count = len(common_solved_problems)
    
    print(f"{llm}:")
    print(f"  Random RAG improvements over No RAG (based on {common_count} common problems):")
    print(f"    Correctness improvement: {random_correctness_improvement:+.1f} percentage points")
    print(f"    Time change: {random_time_improvement:+.1f} ms (negative = faster)")
    print(f"    Memory change: {random_memory_improvement:+.1f} KB (negative = less memory)")
    print(f"  Similar RAG improvements over No RAG (based on {common_count} common problems):")
    print(f"    Correctness improvement: {correctness_improvement:+.1f} percentage points")
    print(f"    Time change: {time_improvement:+.1f} ms (negative = faster)")
    print(f"    Memory change: {memory_improvement:+.1f} KB (negative = less memory)")

print("\n" + "="*80)
print("ANÁLISE COMPARATIVA DAS PORCENTAGENS MÉDIAS DE ACERTOS")
print("="*80)

avg_without_rag = statistics.mean(without_rag_correct_data) if without_rag_correct_data else 0
avg_random_rag = statistics.mean(random_rag_correct_data) if random_rag_correct_data else 0
avg_with_rag = statistics.mean(with_rag_correct_data) if with_rag_correct_data else 0

print(f"\nPorcentagens médias de acerto entre todos os modelos:")
print(f"  Sem RAG: {avg_without_rag:.2f}%")
print(f"  RAG Aleatório: {avg_random_rag:.2f}%")
print(f"  RAG Similar: {avg_with_rag:.2f}%")

print(f"\nGanhos comparativos:")
print(f"  RAG Aleatório vs Sem RAG:")
random_vs_without = avg_random_rag - avg_without_rag
print(f"    Ganho absoluto: {random_vs_without:+.2f} pontos percentuais")
print(f"    Ganho relativo: {(random_vs_without/avg_without_rag)*100:+.1f}%" if avg_without_rag > 0 else "    Ganho relativo: N/A")

print(f"  RAG Similar vs Sem RAG:")
with_vs_without = avg_with_rag - avg_without_rag
print(f"    Ganho absoluto: {with_vs_without:+.2f} pontos percentuais")
print(f"    Ganho relativo: {(with_vs_without/avg_without_rag)*100:+.1f}%" if avg_without_rag > 0 else "    Ganho relativo: N/A")

print(f"  RAG Similar vs RAG Aleatório:")
with_vs_random = avg_with_rag - avg_random_rag
print(f"    Ganho absoluto: {with_vs_random:+.2f} pontos percentuais")
print(f"    Ganho relativo: {(with_vs_random/avg_random_rag)*100:+.1f}%" if avg_random_rag > 0 else "    Ganho relativo: N/A")

print(f"\nConclusões:")
if with_vs_without > 0:
    print(f" RAG Similar melhora a precisão em {with_vs_without:.2f} pontos percentuais comparado ao Sem RAG")
else:
    print(f" RAG Similar reduz a precisão em {abs(with_vs_without):.2f} pontos percentuais comparado ao Sem RAG")

if random_vs_without > 0:
    print(f" RAG Aleatório melhora a precisão em {random_vs_without:.2f} pontos percentuais comparado ao Sem RAG")
else:
    print(f" RAG Aleatório reduz a precisão em {abs(random_vs_without):.2f} pontos percentuais comparado ao Sem RAG")

if with_vs_random > 0:
    print(f" RAG Similar é {with_vs_random:.2f} pontos percentuais melhor que RAG Aleatório")
else:
    print(f" RAG Similar é {abs(with_vs_random):.2f} pontos percentuais pior que RAG Aleatório")

print(f"\nEfetividade das abordagens RAG:")
best_approach = max([("Sem RAG", avg_without_rag), ("RAG Aleatório", avg_random_rag), ("RAG Similar", avg_with_rag)], key=lambda x: x[1])
print(f"  Melhor abordagem: {best_approach[0]} com {best_approach[1]:.2f}% de acerto")

improvement_threshold = 1.0  
if with_vs_without > improvement_threshold:
    print(f"  ✓ RAG Similar apresenta melhoria significativa (>{improvement_threshold}pp)")
elif with_vs_without > 0:
    print(f"  ⚠ RAG Similar apresenta melhoria marginal (<{improvement_threshold}pp)")
else:
    print(f"  ✗ RAG Similar não apresenta melhoria")

if random_vs_without > improvement_threshold:
    print(f"  ✓ RAG Aleatório apresenta melhoria significativa (>{improvement_threshold}pp)")
elif random_vs_without > 0:
    print(f"  ⚠ RAG Aleatório apresenta melhoria marginal (<{improvement_threshold}pp)")
else:
    print(f"  ✗ RAG Aleatório não apresenta melhoria")

print("\n" + "="*100)
print("DEEPSEEK R1 ELO RATINGS BY CATEGORY ACROSS ALL SCENARIOS")
print("="*100)

print(f"\n{'Category':<25} | {'Without RAG':<20} | {'Random RAG':<20} | {'With RAG':<20}")
print("-" * 90)

for category in top_15_category_names:
    category_data = deepseek_category_results[category]
    elo_results = {}
    
    for scenario in ["without_rag", "random_rag", "with_rag"]:
        detailed_results = category_data[scenario]["detailed_results"]
        
        if len(detailed_results) > 0:
            model_results = [result for result, elo in detailed_results]
            problem_elos = [elo for result, elo in detailed_results]
            
            try:
                elo_rating, elo_std = elo_map_rating(model_results, problem_elos, codeforces_ratings)
                cf_percentile = calculate_codeforces_percentile(elo_rating, codeforces_ratings)
                
                solved_count = sum(model_results)
                total_count = len(model_results)
                
                if cf_percentile is not None:
                    elo_display = f"{elo_rating:.0f}±{elo_std:.0f} ({solved_count}/{total_count}, {cf_percentile:.1f}%)"
                else:
                    elo_display = f"{elo_rating:.0f}±{elo_std:.0f} ({solved_count}/{total_count})"
                    
            except Exception as e:
                elo_display = f"Error ({len(detailed_results)} problems)"
        else:
            elo_display = "No data"
        
        elo_results[scenario] = elo_display
    
    print(f"{category:<25} | {elo_results['without_rag']:<20} | {elo_results['random_rag']:<20} | {elo_results['with_rag']:<20}")

print("\nLegend: Elo±StdDev (Solved/Total, CodeForces_Percentile%)")
print("Note: Each problem can belong to multiple categories, so the same problem result")
print("      may be counted in multiple category calculations.")
