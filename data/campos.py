import json

with open("codecontests_train.json", "r", encoding="utf-8") as f:
    first_line = json.loads(f.readline())  # lê só a primeira linha
    print(list(first_line.keys()))