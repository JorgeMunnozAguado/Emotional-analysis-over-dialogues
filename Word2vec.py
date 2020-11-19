import gensim
import numpy as np

from Transformation import Transformation


class Word2vec(Transformation):

    def __init__(self, path='model/', verbose=1):

        if verbose:  print('Loading Word2vec model ...', end='\r')
        
        
        self.model = gensim.models.KeyedVectors.load_word2vec_format(path + 'GoogleNews-vectors-negative300.bin', binary=True)
        
        
        if verbose:
            print('Model Word2vec Loaded (OK)', end='\r')
            print()

    
    def transform(self, text, verbose=0):
    
        newT = []
    
        for word in text:
    
            try:
                newT.append( self.model[word] )
    
            except KeyError:
                if verbose: print(word)
                pass

    
        return np.asarray(newT)
    
    
    def shape(self):
        
        return 300