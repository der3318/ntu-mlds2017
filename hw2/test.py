import os
import sys
import json
import numpy as np

from preprocessing_for_test import data
from model import S2VTmodel

#######################################
# Parameters specifically for this task
#######################################

frames = 80
dim_image = 4096

#######################################
# Paths
#######################################

model_path = './models/'

def test(testing_id_file, feature_path):
    Data = data(
            testing_id_file=testing_id_file,
            testing_dir=feature_path,
            word_count_threshold=3
            )
    
    n_words = Data.get_vocab_size()

    model = S2VTmodel(
                n_hidden=512,
                n_step1=frames,
                n_step2=20,
                use_ss=False,
                use_att=True,
                use_bn=False,
                beam_size=5,
                alpha_c=0.0,
                n_words=n_words,
                dim_image=dim_image,
                seed=3318
                )


    test_X, test_id = Data.gen_test_data()
    
    pred = model.predict(test_X,
                        model_dir=model_path,
                        name='model-h512-att-lrschedule-mask-count3-lencost5e-1', 
                        model_epoch=None)

    predictions = []
    for p, id in zip(pred, test_id):
        sent = Data.get_sentence_by_indices(p[0])
        print(id, sent)
        predictions.append({"caption": sent, "id": id})

    with open('output.json', 'w') as fw:
        fw.write(json.dumps(predictions))


if __name__ == '__main__':
    testing_id_file = sys.argv[1]
    feature_path = sys.argv[2]

    test(testing_id_file, feature_path)
