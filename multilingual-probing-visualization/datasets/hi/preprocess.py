import re

in_file = 'dev.conllu'
out_file = 'dev_roman.conllu'

new_lines = []
with open(in_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if len(line) > 0 and line[0].isdigit():
            columns = line.split('\t')
            match = re.search(r'Translit=([^|]+)', columns[9])
            if match:
                translit_value = match.group(1)
                columns[1] = translit_value
            else:
                raise Exception("No translit!")
            new_lines.append('\t'.join(columns).replace('\n', ''))
        else:
            new_lines.append(line.replace('\n', ''))
# print(new_lines)
with open(out_file, 'w') as f:
    f.write('\n'.join(new_lines))