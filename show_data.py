import os
from pathlib import Path
import statistics
import matplotlib.pyplot as plt
import numpy as np

def safe_float(s):
    """Safely convert string to float, returning 0.0 if conversion fails."""
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0

# Define the root directory
root_dir = Path("data/answers")

# Define LLM names
llms = ["Claude 3.7 Sonnet", "Gemini 2.0", "GPT 4o", "GPT-3.5"]

# Create images directory if it doesn't exist
images_dir = Path("images")
os.makedirs(images_dir, exist_ok=True)

# Create data structures to store the results
results = {
    llm: {
        "without_rag": {
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

# Walk through the directory structure
problems = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

for problem in problems:
    problem_dir = root_dir / problem
    
    for llm in llms:
        llm_dir = problem_dir / llm
        results_file = llm_dir / "results.txt"
        
        if not results_file.exists():
            continue
        
        try:
            with open(results_file, "r") as f:
                lines = [line.strip() for line in f.readlines()]
                
                if len(lines) >= 6:
                    # Without RAG
                    results[llm]["without_rag"]["correct"].append(1 if lines[0] == "Y" else 0)
                    speed = safe_float(lines[1])
                    memory = safe_float(lines[2])
                    if speed > 0:  # Only include non-zero values
                        results[llm]["without_rag"]["speed"].append(speed)
                    if memory > 0:  # Only include non-zero values
                        results[llm]["without_rag"]["memory"].append(memory)
                    
                    # With RAG
                    results[llm]["with_rag"]["correct"].append(1 if lines[3] == "Y" else 0)
                    speed = safe_float(lines[4])
                    memory = safe_float(lines[5])
                    if speed > 0:  # Only include non-zero values
                        results[llm]["with_rag"]["speed"].append(speed)
                    if memory > 0:  # Only include non-zero values
                        results[llm]["with_rag"]["memory"].append(memory)
        except Exception as e:
            print(f"Error processing {results_file}: {e}")

# Calculate averages and prepare data for visualization
without_rag_correct_data = []
with_rag_correct_data = []
without_rag_speed_data = []
with_rag_speed_data = []
without_rag_memory_data = []
with_rag_memory_data = []

# Print the results in terminal first
print("Performance Metrics for LLMs on LeetCode Problems")
print("=" * 50)

for llm in llms:
    print(f"\n{llm}:")
    
    # Calculate averages
    without_rag_correct = statistics.mean(results[llm]["without_rag"]["correct"]) * 100 if results[llm]["without_rag"]["correct"] else 0
    with_rag_correct = statistics.mean(results[llm]["with_rag"]["correct"]) * 100 if results[llm]["with_rag"]["correct"] else 0
    
    without_rag_speed = statistics.mean(results[llm]["without_rag"]["speed"]) if results[llm]["without_rag"]["speed"] else 0
    with_rag_speed = statistics.mean(results[llm]["with_rag"]["speed"]) if results[llm]["with_rag"]["speed"] else 0
    
    without_rag_memory = statistics.mean(results[llm]["without_rag"]["memory"]) if results[llm]["without_rag"]["memory"] else 0
    with_rag_memory = statistics.mean(results[llm]["with_rag"]["memory"]) if results[llm]["with_rag"]["memory"] else 0
    
    # Print results with question counts
    print(f"  Without RAG:")
    print(f"    Correct: {without_rag_correct:.1f}% (Questions processed: {len(results[llm]['without_rag']['correct'])})")
    print(f"    Speed Percentile: {without_rag_speed:.1f}% (Questions processed: {len(results[llm]['without_rag']['speed'])})")
    print(f"    Memory Percentile: {without_rag_memory:.1f}% (Questions processed: {len(results[llm]['without_rag']['memory'])})")
    
    print(f"  With RAG:")
    print(f"    Correct: {with_rag_correct:.1f}% (Questions processed: {len(results[llm]['with_rag']['correct'])})")
    print(f"    Speed Percentile: {with_rag_speed:.1f}% (Questions processed: {len(results[llm]['with_rag']['speed'])})")
    print(f"    Memory Percentile: {with_rag_memory:.1f}% (Questions processed: {len(results[llm]['with_rag']['memory'])})")
    
    # Store for visualization
    without_rag_correct_data.append(without_rag_correct)
    with_rag_correct_data.append(with_rag_correct)
    without_rag_speed_data.append(without_rag_speed)
    with_rag_speed_data.append(with_rag_speed)
    without_rag_memory_data.append(without_rag_memory)
    with_rag_memory_data.append(with_rag_memory)

# Also print a summary table for easy comparison
print("\nSummary Table:")
print("-" * 120)
print(f"{'Model':<20} | {'Without RAG':^58} | {'With RAG':^58}")
print(f"{'':<20} | {'Correct':^18} | {'Speed':^18} | {'Memory':^18} | {'Correct':^18} | {'Speed':^18} | {'Memory':^18}")
print("-" * 120)

for llm in llms:
    without_rag_correct = statistics.mean(results[llm]["without_rag"]["correct"]) * 100 if results[llm]["without_rag"]["correct"] else 0
    with_rag_correct = statistics.mean(results[llm]["with_rag"]["correct"]) * 100 if results[llm]["with_rag"]["correct"] else 0
    
    without_rag_speed = statistics.mean(results[llm]["without_rag"]["speed"]) if results[llm]["without_rag"]["speed"] else 0
    with_rag_speed = statistics.mean(results[llm]["with_rag"]["speed"]) if results[llm]["with_rag"]["speed"] else 0
    
    without_rag_memory = statistics.mean(results[llm]["without_rag"]["memory"]) if results[llm]["without_rag"]["memory"] else 0
    with_rag_memory = statistics.mean(results[llm]["with_rag"]["memory"]) if results[llm]["with_rag"]["memory"] else 0
    
    print(f"{llm:<20} | {without_rag_correct:^10.1f}% ({len(results[llm]['without_rag']['correct'])}) | {without_rag_speed:^10.1f}% ({len(results[llm]['without_rag']['speed'])}) | {without_rag_memory:^10.1f}% ({len(results[llm]['without_rag']['memory'])}) | {with_rag_correct:^10.1f}% ({len(results[llm]['with_rag']['correct'])}) | {with_rag_speed:^10.1f}% ({len(results[llm]['with_rag']['speed'])}) | {with_rag_memory:^10.1f}% ({len(results[llm]['with_rag']['memory'])})")

# Create visualization function with Y limit at 100
def create_grouped_bar_chart(data1, data2, title, ylabel):
    """Create a grouped bar chart comparing two sets of data across LLMs"""
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(llms))
    width = 0.35
    
    plt.bar(x - width/2, data1, width, label='Without RAG', color='skyblue')
    plt.bar(x + width/2, data2, width, label='With RAG', color='orange')
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('LLM Model')
    plt.xticks(x, llms, rotation=45, ha='right')
    plt.ylim(0, 100)  # Set Y limit to 100 for all charts
    plt.legend()
    plt.tight_layout()
    
    # Add value labels on top of bars
    for i, v in enumerate(data1):
        plt.text(i - width/2, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(data2):
        plt.text(i + width/2, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.savefig(images_dir / f'{title.replace(" ", "_").lower()}.png', dpi=300)
    plt.show()  # Show each chart individually

# Create and display charts one by one
print("\nDisplaying charts one by one. Close each chart to see the next one.")

# Chart 1: Correctness
create_grouped_bar_chart(without_rag_correct_data, with_rag_correct_data, 
                        'Correctness by LLM', 'Correct (%)')

# Chart 2: Speed
create_grouped_bar_chart(without_rag_speed_data, with_rag_speed_data, 
                        'Speed Percentile by LLM', 'Speed Percentile (%)')

# Chart 3: Memory
create_grouped_bar_chart(without_rag_memory_data, with_rag_memory_data, 
                        'Memory Percentile by LLM', 'Memory Percentile (%)')

# Chart 4: Combined view
plt.figure(figsize=(15, 10))
x = np.arange(len(llms))
width = 0.15

# Plot each metric with different positions
plt.bar(x - width*1.5, without_rag_correct_data, width, label='Correctness (Without RAG)', color='skyblue')
plt.bar(x - width/2, without_rag_speed_data, width, label='Speed (Without RAG)', color='lightgreen')
plt.bar(x + width/2, without_rag_memory_data, width, label='Memory (Without RAG)', color='lightpink')
plt.bar(x + width*1.5, with_rag_correct_data, width, label='Correctness (With RAG)', color='blue')
plt.bar(x + width*2.5, with_rag_speed_data, width, label='Speed (With RAG)', color='green')
plt.bar(x + width*3.5, with_rag_memory_data, width, label='Memory (With RAG)', color='red')

plt.title('LLM Performance Metrics Comparison')
plt.ylabel('Percentage (%)')
plt.xlabel('LLM Model')
plt.xticks(x + width, llms)
plt.ylim(0, 100)  # Set Y limit to 100
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

plt.tight_layout()
plt.savefig(images_dir / 'llm_combined_metrics.png', dpi=300)
plt.show()