import sys
sys.path.append('./skip-thoughts')

import skipthoughts

class Embedding:

    support_models = ['skipthoughts']

    def __init__(self, model='skipthoughts', **kwargs):
        """
        Arguments:
        model       embedding model name, support 'skipthoughts'.

        """
        
        if not model in self.support_models:
            raise ValueError('the model name %s is not supported.' % model)

        self.model_name = model

        if model == 'skipthoughts':
            self.model = Skipthoughts()


    def encode(self, sentence):
        return self.model.encode(sentence)



class Skipthoughts:

    dim = 4800

    def __init__(self): 
        self.model = skipthoughts.load_model()
        self.encoder = skipthoughts.Encoder(self.model)


    def encode(self, sentence):
        return self.encoder.encode(sentence, verbose=False)
