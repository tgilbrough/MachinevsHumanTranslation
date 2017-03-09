import sys
from nltk.parse import stanford

filename = 'A5.train.labeled'

# Parser
parser = stanford.StanfordParser()

# Snag lines from text file
with open(filename, 'r', encoding='utf-8') as f:
    raw_lines = f.read().split('\n')

lines = []
for i in range(len(raw_lines) // 6):
    line = {}
    line["human"] = raw_lines[i * 6 + 1]
    line["?"] = raw_lines[i * 6 + 2]
    lines.append(line)

count = 0
for line in lines:
    tree = next(parser.raw_parse(line["human"]))
    print(tree.pformat().replace("\n", ""))
    
    tree = next(parser.raw_parse(line["?"]))
    print(tree.pformat().replace("\n", ""))

    if (count % 10 == 0) :
        print(count, len(lines), file=sys.stderr)

    count += 1
    print()

    