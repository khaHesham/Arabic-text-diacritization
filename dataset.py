import pickle
import re 
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from nltk.tokenize import word_tokenize
from typing import List, Tuple

class DiacriticsDataset(Dataset):
    def __init__(self):
        with open('linguistic_resources/diacritic2id.pickle', 'rb') as file:
            self.diacritic2id = pickle.load(file)
        self.diacritic_classes = list(self.diacritic2id.keys())
            
        with open('linguistic_resources/diacritics.pickle', 'rb') as file:
            self.diacritics = pickle.load(file)
            
        with open('linguistic_resources/arabic_letters.pickle', 'rb') as file:
            self.arabic_letters = pickle.load(file)
            
        self.characters2id = {char:i for i, char in enumerate(self.arabic_letters)}
        self.id2characters = {i:char for char, i in self.characters2id.items()}
            
        self.pad = -1
                    
    def load(self, path:str):
        with open(path, 'r', encoding='utf-8') as file:
            corpus = file.read()
            
        cleaned_corpus = self.clean(corpus)
        sentences = self.segment_sentences(cleaned_corpus)
        characters, diacritics = self.separate_chars_from_diacritics(sentences)
        
        encoded_characters = self.encode_chars(characters)
            
        tensor_characters = [torch.tensor(sentence) for sentence in encoded_characters]
        tensor_diacritics = [torch.tensor(sentence) for sentence in diacritics]

        # Pad sequences to the maximum length
        self.character_sentences = pad_sequence(tensor_characters, batch_first=True, padding_value=self.pad)
        self.diacritic_sentences = pad_sequence(tensor_diacritics, batch_first=True, padding_value=self.pad)
                    
    def is_diacritic(self, char:str) -> bool:
        return char in self.diacritics
    
    def is_arabic(self, char:str) -> bool:
        return char in self.arabic_letters
    
    def clean(self, corpus:str, save:bool=False, file:str=None) -> str:
        # separators = {' ', '.', ',', ';', ':', '\n'}
        separators = {'.'}
        allowed_chars = self.diacritics | self.arabic_letters | separators
        pattern = f'[^{"".join(allowed_chars)}]'
        cleaned_corpus = re.sub(pattern, '', corpus)
        
        if save:
            with open(file, 'w', encoding='utf-8') as file:
                file.write(cleaned_corpus)

        return cleaned_corpus
    
    def __len__(self) -> int:
        """
        This function should return the length of the dataset (the number of sentences)
        """
        return self.character_sentences.shape[0]

    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This function returns a subset of the whole dataset
        """
        return self.character_sentences[idx], self.diacritic_sentences[idx]
           
    def encode_chars(self, sentences:List[List[str]]) -> List[List[int]]: 
        encoded_sentences = []
        for sentence in sentences:
            encoded_chars = [self.characters2id[char] for char in sentence]
            encoded_sentences.append(encoded_chars)
        return encoded_sentences
    
    def decode_chars(self, sentence: torch.Tensor) -> List[str]:
        # Remove padding
        unpadded_sentence = sentence[sentence != self.pad]
        
        decoded_chars = [self.id2characters[encoded_char.item()] for encoded_char in unpadded_sentence]
        return decoded_chars
    
    def segment_sentences(self,corpus:str) -> List[str]:
        sentences = corpus.split('.')
        #TODO if sentence length exceeds certain threshold, then split on [, ; :]
        return sentences
    
    def separate_chars_from_diacritics(self, sentences:List[str]) -> Tuple[List[List[str]], List[List[int]]]:
        character_sentences = []
        diacritics_sentences = []
        
        for sentence in sentences:
            characters, diacritics = self.separate_diacritics(sentence)
            character_sentences.append(characters)
            diacritics_sentences.append(diacritics)
            
        return character_sentences, diacritics_sentences
        
    def separate_diacritics(self, sentence: str) -> Tuple[List[str], List[int]]:
        n = len(sentence)
        
        diacritics = []
        characters = []
        
        for i, char in enumerate(sentence):
            if not self.is_arabic(char): continue
            
            characters.append(char)
            
            diacritic = ''
            if i+1 < n and self.is_diacritic(sentence[i+1]):
                diacritic += sentence[i+1]
                if i+2 < n and self.is_diacritic(sentence[i+2]):
                    diacritic += sentence[i+2]
                
            diacritics.append(self.diacritic2id[diacritic])
            
        return characters, diacritics
    
    # def tokenize(self):
    #     self.tokenized_sentences = []
    #     sentences = self.cleaned_corpus.split('.')
    #     for sentence in sentences:
    #         self.tokenized_sentences.append(word_tokenize(sentence))  # tokenize each sentence
    #     return self.tokenized_sentences

def main():
    dataset = DiacriticsDataset()
    dataset.load('dataset/val.txt')

    characters, diacritics = dataset[0]
    decoded_chars = dataset.decode_chars(characters)
    print(decoded_chars)
    print(diacritics[:len(decoded_chars)])

if __name__ == "__main__":
    main()