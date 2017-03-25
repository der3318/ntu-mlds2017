import csv
import numpy as np

def get_test_data(fname='data/testing_data.csv'):
    questions = []

    for row in csv.DictReader(open(fname, 'r')):
        questions.append({'question': row['question'],
                          'options': [row[c+')'] for c in 'abcde']})

    return questions


def write_submit_file(ans, output):
    ans_list = list('abcde')

    with open(output, 'w') as fw:
        fw.write('id,answer\n')

        for i, answer in enumerate(ans):
            fw.write('{:d},{:s}\n'.format(i+1, ans_list[answer]))


def load_glove(fname):
    model = {}
    for i, line in enumerate(open(fname)):
        splits = line.strip().split()
        word = splits[0]
        vector = splits[1:]

        model[word] = np.array([float(num) for num in vector])

        if i % 50000 == 0:
            print('processed {:d} words'.format(i))
    
    return model
