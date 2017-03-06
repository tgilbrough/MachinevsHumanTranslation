import sys
from math import floor

def parse_training_data():
    # Snag lines from text file
    with open('A5.train.labeled', 'r', encoding='utf-8') as f:
        raw_lines = f.read().split('\n')

    lines = []
    for i in range(len(raw_lines) // 6):
        line = {}
        line["chinese"] = raw_lines[i * 6 + 0]
        line["human"] = raw_lines[i * 6 + 1]
        line["?"] = raw_lines[i * 6 + 2]
        line["bleu"] = float(raw_lines[i * 6 + 3])
        line["label"] = raw_lines[i * 6 + 4]
        lines.append(line)

    with open('A5.train_trees.labeled', 'r', encoding='utf-8') as f:
        raw_lines = f.read().split('\n')

    for i in range(len(raw_lines) // 3):
        lines[i]["h_tree"] = raw_lines[i * 3]
        lines[i]["q_tree"] = raw_lines[i * 3 + 1]

    return lines

def parse_testing_data():
    # Snag lines from text file
    with open('A5.test.unlabeled', 'r', encoding='utf-8') as f:
        raw_lines = f.read().split('\n')

    lines = []
    for i in range(len(raw_lines) // 6):
        line = {}
        line["chinese"] = raw_lines[i * 6 + 0]
        line["human"] = raw_lines[i * 6 + 1]
        line["?"] = raw_lines[i * 6 + 2]
        line["bleu"] = float(raw_lines[i * 6 + 3])
        line["label"] = raw_lines[i * 6 + 4]
        lines.append(line)

    with open('A5.test_trees.unlabeled', 'r', encoding='utf-8') as f:
        raw_lines = f.read().split('\n')

    for i in range(len(raw_lines) // 3):
        lines[i]["h_tree"] = raw_lines[i * 3]
        lines[i]["q_tree"] = raw_lines[i * 3 + 1]

    return lines