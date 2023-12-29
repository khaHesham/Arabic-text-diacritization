import pickle
import re 
import numpy as np
from nltk.tokenize import word_tokenize

class DiacriticsDataset: 
    def __init__(self):
        with open('linguistic_resources/diacritic2id.pickle', 'rb') as file:
            # Load the object from the pickle file
            self.diacritic2id = pickle.load(file)
            self.diacritic_classes = list(self.diacritic2id.keys())
            
        with open('linguistic_resources/diacritics.pickle', 'rb') as file:
            self.diacritics = pickle.load(file)
            
        with open('linguistic_resources/arabic_letters.pickle', 'rb') as file:
            self.arabic_letters = pickle.load(file)
            
    def load(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            self.corpus = file.read()
        return self.corpus
                    
    def is_diacritic(self, char):
        return char in self.diacritics
    
    def is_arabic(self, char):
        return char in self.arabic_letters
    
    def clean(self, save:bool=False, file:str=None) -> str:
        separators = {' ', '.', ',', ';', ':', '\n'}
        allowed_chars = self.diacritics | self.arabic_letters | separators
        pattern = f'[^{"".join(allowed_chars)}]'
        self.cleaned_corpus = re.sub(pattern, '', self.corpus)
        
        if save:
            with open(file, 'w', encoding='utf-8') as file:
                file.write(self.cleaned_corpus)

        return self.cleaned_corpus
           
    # def separate_diacritics(self, word: str) -> (list[str], list[int]):
    #     prev_is_diacritic = False
    #     prev_is_char = False

    #     curr_diacritic = ''
    #     diacritics = []
    #     characters = []
        
    #     for char in word:
    #         if self.is_diacritic(char):
    #             if(prev_is_diacritic):
    #                 curr_diacritic += char
    #                 diacritics[-1] = self.diacritic2id[curr_diacritic]
    #             else:
    #                 diacritics.append(self.diacritic2id[char])
    #                 curr_diacritic = char
    #             prev_is_diacritic = True
    #             prev_is_char = False
    #         else:
    #             characters.append(char)
    #             if(prev_is_char):
    #                 diacritics.append(self.diacritic2id[''])
    #             prev_is_char = True
    #             prev_is_diacritic = False
            
    #     return characters, diacritics
    
    def separate_diacritics(self, word: str) -> (list[str], list[int]):
        n = len(word)
        
        diacritics = []
        characters = []
        
        for i, char in enumerate(word):
            if not self.is_arabic(char): continue
            
            characters.append(char)
            
            diacritic = ''
            if i+1 < n and self.is_diacritic(word[i+1]):
                diacritic += word[i+1]
                if i+2 < n and self.is_diacritic(word[i+2]):
                    diacritic += word[i+2]
                
            diacritics.append(self.diacritic2id[diacritic])
            
        return characters, diacritics
    
    def tokenize(self):
        self.tokenized_sentences = []
        sentences = self.cleaned_corpus.split('.')
        for sentence in sentences:
            self.tokenized_sentences.append(word_tokenize(sentence))  # tokenize each sentence
        return self.tokenized_sentences

def main():
    dataset = DiacriticsDataset()
    corpus = dataset.load('dataset/val.txt')
    cleaned_corpus = dataset.clean(save=True, file='cleaned.txt')
    tokenized_corpus = dataset.tokenize()
    
    for i, diacritic in enumerate(dataset.diacritic_classes):
        print(f"{i}:", 'ت' + diacritic)
    
    sentence = tokenized_corpus[2]
    print(sentence)
    for word in sentence:
        characters, diacritics = dataset.separate_diacritics(word)
        print("Characters:", characters)
        print("Diacritics:", diacritics)

if __name__ == "__main__":
    main()