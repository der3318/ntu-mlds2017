import sys
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from utility import *

model = load_glove('embeddings.txt')

questions = get_test_data()

ans = []
for i, q in enumerate(questions):
    words = q['question'].lower().strip().strip('.').split()
    #print(words)
    vectors = [model[w] for w in words if w in model]

    totalsims = []
    for option in q['options']:
        if option in model:
            vector_option = model[option]
            totalsims.append(cosine_similarity(vector_option.reshape(1, -1), vectors).sum())
        else:
            totalsims.append(-10.0)

    #print(q, '\n', totalsims)
    ans.append(np.argmax(totalsims))

write_submit_file(ans, 'output.csv')
