import json
import os

def count_difficulty_levels():
    """
    Counts and prints the number of programming questions for each difficulty level
    from the 'data/programming_questions.json' file.
    """
    # Construct the path to the JSON file relative to this script
    script_dir = os.path.dirname(__file__)  # Gets the directory where the script is located
    file_path = os.path.join(script_dir, 'data', 'programming_questions.json')

    difficulty_counts = {
        "Easy": 0,
        "Medium": 0,
        "Hard": 0
    }

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file {file_path}.")
        return

    if not isinstance(questions_data, list):
        print(f"Error: Expected a list of questions in {file_path}, but got {type(questions_data)}.")
        return

    for question in questions_data:
        if isinstance(question, dict):
            difficulty = question.get("difficulty")
            if difficulty in difficulty_counts:
                difficulty_counts[difficulty] += 1
        else:
            print(f"Warning: Found an item in the JSON data that is not a dictionary: {question}")

    total_questions_with_difficulty = sum(difficulty_counts.values())

    print("\nAnalysis for 'data/programming_questions.json':")
    print("Number of questions by difficulty:")
    if total_questions_with_difficulty == 0:
        print("No questions with specified difficulty levels found.")
        for level, count in difficulty_counts.items():
            print(f"- {level}: {count} (0.00%)")
    else:
        for level, count in difficulty_counts.items():
            percentage = (count / total_questions_with_difficulty) * 100
            print(f"- {level}: {count} ({percentage:.2f}%)")

def analyze_test_dataset():
    """
    Analyzes the test dataset in 'data/answers/' by reading 'info.txt' from subdirectories.
    Counts and prints questions by difficulty and release year.
    """
    script_dir = os.path.dirname(__file__)
    answers_base_dir = os.path.join(script_dir, 'data', 'answers')

    difficulty_map = {'E': "Easy", 'M': "Medium", 'H': "Hard"}
    test_difficulty_counts = {"Easy": 0, "Medium": 0, "Hard": 0}
    test_year_counts = {"Before 2020": 0, "2020-2022": 0, "2023-2025": 0}
    
    if not os.path.isdir(answers_base_dir):
        print(f"Error: Test dataset directory {answers_base_dir} not found.")
        return

    question_dirs = [d for d in os.listdir(answers_base_dir) if os.path.isdir(os.path.join(answers_base_dir, d))]

    for question_dir_name in question_dirs:
        question_path = os.path.join(answers_base_dir, question_dir_name)
        info_file_path = os.path.join(question_path, 'info.txt')

        try:
            with open(info_file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines()]
                
                if len(lines) >= 2:
                    # Process difficulty
                    difficulty_code = lines[0]
                    difficulty = difficulty_map.get(difficulty_code)
                    if difficulty:
                        test_difficulty_counts[difficulty] += 1
                    else:
                        print(f"Warning: Unknown difficulty code '{difficulty_code}' in {info_file_path}")

                    # Process year
                    year_str = lines[1]
                    try:
                        year = int(year_str)
                        if year < 2020:
                            test_year_counts["Before 2020"] += 1
                        elif 2020 <= year <= 2022:
                            test_year_counts["2020-2022"] += 1
                        elif 2023 <= year <= 2025:
                            test_year_counts["2023-2025"] += 1
                        # else:
                        #     print(f"Warning: Year {year} in {info_file_path} is outside defined ranges.")
                    except ValueError:
                        print(f"Warning: Invalid year format '{year_str}' in {info_file_path}")
                else:
                    print(f"Warning: Not enough lines in {info_file_path} (expected at least 2).")
        except FileNotFoundError:
            print(f"Warning: info.txt not found in {question_path}")
        except Exception as e:
            print(f"Error processing {info_file_path}: {e}")

    print("\nAnalysis for Test Dataset ('data/answers/'):")
    
    # Print difficulty analysis
    total_test_questions_by_difficulty = sum(test_difficulty_counts.values())
    print("Number of test questions by difficulty:")
    if total_test_questions_by_difficulty == 0:
        print("No test questions with specified difficulty levels found.")
        for level, count in test_difficulty_counts.items():
            print(f"- {level}: {count} (0.00%)")
    else:
        for level, count in test_difficulty_counts.items():
            percentage = (count / total_test_questions_by_difficulty) * 100
            print(f"- {level}: {count} ({percentage:.2f}%)")

    # Print year analysis
    total_test_questions_by_year = sum(test_year_counts.values())
    print("\nNumber of test questions by release year:")
    if total_test_questions_by_year == 0:
        print("No test questions with specified year information found.")
        for year_range, count in test_year_counts.items():
            print(f"- {year_range}: {count} (0.00%)")
    else:
        for year_range, count in test_year_counts.items():
            percentage = (count / total_test_questions_by_year) * 100
            print(f"- {year_range}: {count} ({percentage:.2f}%)")


if __name__ == "__main__":
    count_difficulty_levels()
    analyze_test_dataset()
