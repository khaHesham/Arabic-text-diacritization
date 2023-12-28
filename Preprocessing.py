import numpy as np
import pandas as pd
import re
from nltk.tokenize import word_tokenize

class cleaning:
    """
        This class provides methods for cleaning and preprocessing Arabic text data.
    """

    def __init__(self):
        '''
            Args :
                data (str): The data to be processed.  -> plain text
                
        '''
        self.data = None
        self.diacritics = [0x064B,0x065F,0x0670,0x0674,
                           0x06D4,0x06D6,0x06ED,0x06F0,
                           0x06FC,0x06FF,0x0750,0x077F,
                           0x08A0,0x08B4,0x08B6,0x08BD,
                           0x08D4,0x08E1,0x08E3,0x08FF,
                           0xFB50,0xFBC1,0xFBD3,0xFD3D,
                           0xFD50,0xFD8F,0xFD92,0xFDC7,
                           0xFDF0,0xFDFC,0xFE70,0xFEFC]
        
        
    def _tokinizer(self, sentence):
        """
            Tokenizes a given sentence into words.
            Args:
                sentence (str): The sentence to be tokenized.

            Returns:
                list: A list of tokens (words) in the sentence.
        """
        tokens = []
        for sentence in self.data.split('.'):       # split over full stop
            tokens.append(word_tokenize(sentence))  # tokenize each sentence
        return tokens
    
    def _restore_diacritics(self, sentence, index_diacritics):
        """
            Restores diacritics in a given sentence based on the provided index of diacritics.
            Args:
                sentence (str): The sentence to restore diacritics in.
                index_diacritics (dict): A dictionary containing the index of diacritics in the sentence.  (index:diacritic)

            Returns:
                str: The sentence with restored diacritics.
        """
        for i,char in index_diacritics.items():
            sentence = sentence[:i] + char + sentence[i:]
        return sentence   
        
    def _extract_diacritics(self,sentence):
        """
            Extracts diacritics from a given sentence.
            Args:
                sentence (str): The sentence to extract diacritics from.

            Returns:
                tuple: A tuple containing a dictionary of index-diacritics pairs and a list of characters in the sentence.
        """
        index_diacritics = {}
        _list = []
        for i,char in enumerate(sentence):
            if char in [char(diacritic) for diacritic in self.diacritics]:
                index_diacritics[i] = char
            _list.append(char)  
        return index_diacritics, _list
    
    def _clean(self):
        # remove anything that is not Arabic letters or diacritics
        self.data = re.sub(r'[^ء-ي]',' ',self.data)
        
    
    
    def pipeline(self,data):
        """
            The main pipeline for cleaning and preprocessing the data.
            Returns:
                str: The cleaned and preprocessed data. as tokenized words
        """
        self.data = data
        text = self._clean()
        tokens = self._tokinizer(text)

        # for sentence in tokens:
            # index_diacritics, _list = self.extract_diacritics(sentence)
            # cleaned_sentence = self.restore_diacritics(''.join(_list), index_diacritics)
            # cleaned_data.append(cleaned_sentence)
            
        return text
    
    
    
    
    