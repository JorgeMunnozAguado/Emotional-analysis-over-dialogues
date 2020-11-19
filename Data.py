import numpy as np
import os


class DataLoader:


    def __init__(self, trans, path='data/'):

        self.path = path

        self.dialogues_file = 'dialogues_text.txt'
        self.emotions_file = 'dialogues_emotion.txt'
        self.single_file = '-scores.txt'

        self.topic    = {1: 'Ordinary Life', 2: 'School Life', 3: 'Culture & Education', 4: 'Attitude & Emotion', 5: 'Relationship', 6: 'Tourism' , 7: 'Health', 8: 'Work', 9: 'Politics', 10: 'Finance'}
        self.act      = {1: 'inform', 2: 'question', 3: 'directive', 4: 'commissive'}
        self.emotions = {0: 'no_emotion', 1: 'anger', 2: 'disgust', 3: 'fear', 4: 'happiness', 5: 'sadness', 6: 'surprise'}  # , 7: 'anticipation', 8: 'trust'


        self.trans = trans
        




    def preprocessText(self, text, verbose=0):


        # Split the data
        split = text.lower().split(' ')
        split = np.array(split)

        # Delete mark-points - TODO (need to be reviewed)
        split, dot = DataLoader.find4del(split, '.')
        split, dot = DataLoader.find4del(split, ',')
        split, que = DataLoader.find4del(split, '?')
        split, exc = DataLoader.find4del(split, '!')

        # change don't -> do not
        split = DataLoader.doNot(split)

        split = self.trans.transform(text, verbose=verbose)
        
        
        return split, dot, que, exc
        
        



    def dialoguesLoader(self, path='dialogues/', verbose=0):

        y_data = []
        X_data = []

        with open(self.path + path + 'dialogues_text.txt', "r", encoding='utf-8') as f:

            for line in f:

                dialog = line.split('__eou__')[:-1]

                for speech in dialog:

                    X_data.append( self.preprocessText(speech)[0] )



        with open(self.path + path + 'dialogues_emotion.txt', "r", encoding='utf-8') as f:

            for line in f:

                emotions = line.split()[:-1]

                for emot in emotions:

                    y_data.append(emot)


        return np.asarray(X_data), np.asarray(y_data)




    def sigleLoader(self, path='single_emotions/', verbose=0):
        
        X_data = []
        y_data = []
        
        for i, emotion in self.emotions.items():
            
            file_name = self.path + path + emotion + self.single_file
            
            if os.path.isfile(file_name):
            
                if verbose: print('Loading', file_name)
                
                with open(file_name, "r", encoding='utf-8') as f:
                    
                    for line in f:
                        
                        split = line.split()
                        
                        trans = self.trans.transform([split[0]], verbose=verbose)
                        
                        if len(trans) == 0: continue
                        
                        X_data.append( trans[0] )
                        y_data.append( self.OneHotEncode(i, split[1]) )


        return np.asarray(X_data), np.asarray(y_data)
    
    
    
    
    def OneHotEncode(self, classe, data):
        
        matrix = np.zeros(len(self.emotions))
        matrix[classe] = data
        
        return matrix




########################################################################


    @staticmethod
    def find4del(text, word):

        location = np.where(text == word)[0]
        text = np.delete(text, location)

        return text, location


    @staticmethod
    def doNot(text):
        '''
        Change a don't by do not.
        '''

        neg = np.where(text == '’')[0]

        for i in neg[::-1]:

            text[i-1] = text[i-1][:-1]
            text[i+1] = 'not'

        split = np.delete(text, neg)

        return split