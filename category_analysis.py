import json
import matplotlib.pyplot as plt
import re
from collections import defaultdict

def load_questions(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_categories(questions):
    category_counts = defaultdict(int)

    category_mapping = {
        # Bit Manipulation variations
        'Bit Manipulation': 'Bit Manipulation',
        'Bitmask': 'Bit Manipulation',
        
        # Sort variations  
        'Sorting': 'Sorting',
        'Sort': 'Sorting',
        'Bucket Sort': 'Sorting',
        'Merge Sort': 'Sorting',
        
        # Math variations
        'Math': 'Math',
        'Geometry': 'Math',
        'Number Theory': 'Math',
        'Counting': 'Math',
        
        # Stack variations
        'Stack': 'Stack',
        'Monotonic Stack': 'Stack',
        
        # String variations
        'String': 'String',
        'String Trie': 'String',
        'Trie': 'String',
        
        # Tree variations
        'Tree': 'Tree',
        'Binary Tree': 'Tree',
        'Binary Indexed Tree': 'Tree',
        'Segment Tree': 'Tree',
        'Indexed Tree': 'Tree',
        
        # Queue variations
        'Queue': 'Queue',
        'Design Queue': 'Queue',
        'Priority Queue': 'Heap',
        'Heap (Priority Queue)': 'Heap',
        'Heap': 'Heap',
        
        # Dynamic Programming variations
        'Dynamic Programming': 'Dynamic Programming',
        'Memoization': 'Dynamic Programming',
        
        # Design variations - keep as separate category
        'Design': 'Design',
        'Hash Function': 'Design',
        'Doubly-Linked List': 'Design',
        'Linked List': 'Design',
        'Rolling Hash': 'Design',
        
        # Search variations
        'Binary Search': 'Binary Search',
        'Depth-First Search': 'Depth-First Search', 
        'Breadth-First Search': 'Breadth-First Search',
        
        # Other specific categories
        'Array': 'Array',
        'Hash Table': 'Hash Table',
        'Two Pointers': 'Two Pointers',
        'Sliding Window': 'Sliding Window',
        'Divide and Conquer': 'Divide and Conquer',
        'Greedy': 'Greedy',
        'Quickselect': 'Quickselect',
        'Ordered Set': 'Ordered Set',
        'Recursion': 'Recursion',
        'Backtracking': 'Backtracking',
        'Simulation': 'Simulation',
        'Randomized': 'Randomized',
        'Union Find': 'Union Find'
    }
    

    for question in questions:
        categories = []
        for item in category_mapping.items():
            if item[0] in question['category']:
                categories.append(item[1])

        print(categories)

        categories = set(categories)  
       
        for category in categories:
            category_counts[category] += 1
    
    return category_counts

def create_visualization(category_counts):
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    
    filtered_categories = [(cat, count) for cat, count in sorted_categories if count > 3]
    
    categories = [item[0] for item in filtered_categories]
    counts = [item[1] for item in filtered_categories]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(categories, counts, color='skyblue', edgecolor='navy', alpha=0.7)
    
    plt.title('Distribution of Programming Questions by Category')
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Number of Questions', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    plt.grid(axis='y', alpha=0.3)
    
    plt.show()
    
    return filtered_categories

if __name__ == "__main__":
    file_path = "data/programming_questions.json"
    questions = load_questions(file_path)
    
    category_counts = extract_categories(questions)
    
    filtered_categories = create_visualization(category_counts)
