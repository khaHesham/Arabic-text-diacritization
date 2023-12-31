import torch
from typing import List
import pickle

class Diacritizer():
    def __init__(self):
        with open('linguistic_resources/diacritic2id.pickle', 'rb') as file:
            diacritic2id = pickle.load(file)
            self.id2diacritic = {id:diacritic for diacritic, id in diacritic2id.items()}
            
        with open('linguistic_resources/arabic_letters.pickle', 'rb') as file:
            self.arabic_letters = pickle.load(file)
            self.characters2id = {char:i for i, char in enumerate(self.arabic_letters)}

    def decode_diacritics(self, encoded_diacritics: torch.Tensor) -> List[str]:
        diacritics = [self.id2diacritic[diacritic.item()] for diacritic in encoded_diacritics]
        return diacritics
    
    def encode_chars(self, sentence:List[str]) -> List[int]: 
        encoded_chars = [self.characters2id[char] for char in sentence]
        return encoded_chars

    # def diacritize(self, sentence: str, model:nn.Module) -> str:
    #     characters = [char for char in sentence if char in self.arabic_letters]
    #     encoded_characters = self.encode_chars(characters)
        
    #     inputs = torch.tensor(encoded_characters)
        
    #     model.eval()
    #     with torch.no_grad():
    #         outputs = model(inputs)
    #     encoded_diacritics = outputs['diacritics']

    #     diacritics = decode_diacritics(encoded_diacritics)

    #     diacritized_sentence = ""
    #     diacritic_idx = 0
    #     for char in sentence:
    #         diacritized_sentence += char
    #         if char in self.arabic_letters:
    #             diacritized_sentence += diacritics[diacritic_idx]
    #             diacritic_idx += 1

    #     return diacritized_sentence
    
    def diacritize(self, corpus: str, encoded_diacritics: torch.Tensor) -> str:
        diacritics = self.decode_diacritics(encoded_diacritics)

        diacritized_corpus = ""
        diacritic_idx = 0
        for char in corpus:
            diacritized_corpus += char
            if char in self.arabic_letters:
                diacritized_corpus += diacritics[diacritic_idx]
                diacritic_idx += 1

        return diacritized_corpus