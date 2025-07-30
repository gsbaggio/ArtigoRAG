import json
import matplotlib.pyplot as plt
import re
from collections import defaultdict

def load_questions(file_path):
    """Load questions from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_categories(questions):
    """Extract and count categories from all questions"""
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
    """Create a bar chart of category counts"""
    # Sort categories by count (descending)
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Filter categories with more than 3 questions
    filtered_categories = [(cat, count) for cat, count in sorted_categories if count > 3]
    
    categories = [item[0] for item in filtered_categories]
    counts = [item[1] for item in filtered_categories]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(categories, counts, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Customize the plot
    plt.title('Distribution of Programming Questions by Category')
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Number of Questions', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3)
    
    # Show the plot
    plt.show()
    
    return filtered_categories

if __name__ == "__main__":
    # Load the questions
    file_path = "data/programming_questions.json"
    questions = load_questions(file_path)
    
    # Extract and count categories
    category_counts = extract_categories(questions)
    
    # Create visualization
    filtered_categories = create_visualization(category_counts)
