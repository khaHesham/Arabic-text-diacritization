import pickle
import re 
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from nltk.tokenize import word_tokenize
from typing import List, Tuple
from more_itertools import chunked


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
        
            
        self.pad_char = len(self.arabic_letters)
        self.pad_diacritic = 14

        self.MAX_SENTENCE_SIZE = 400
                    
    def load(self, path:str, train:bool=True):
        with open(path, 'r', encoding='utf-8') as file:
            corpus = file.read()
            
        cleaned_corpus = self.clean(corpus)
        sentences = self.segment_sentences(cleaned_corpus, train)
        
        self.train = train
        
        if self.train:
            characters, diacritics = self.separate_chars_from_diacritics(sentences)
            tensor_diacritics = [torch.tensor(sentence) for sentence in diacritics]
            self.diacritic_sentences = pad_sequence(tensor_diacritics, batch_first=True, padding_value=self.pad_diacritic)
        else:
            characters = [list(sentence) for sentence in sentences]
        
        encoded_characters = self.encode_chars(characters)     
        tensor_characters = [torch.tensor(sentence) for sentence in encoded_characters]
        
        self.character_sentences = pad_sequence(tensor_characters, batch_first=True, padding_value=self.pad_char)

                    
    def is_diacritic(self, char:str) -> bool:
        return char in self.diacritics
    
    def is_arabic(self, char:str) -> bool:
        return char in self.arabic_letters
    
    def clean(self, corpus:str, test:bool=False) -> str:        
        allowed_chars = self.diacritics | self.arabic_letters | {'.'}
        pattern = f'[^{"".join(allowed_chars)}]'
        cleaned_corpus = re.sub(pattern, '', corpus)
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
        if self.train:
            return self.character_sentences[idx], self.diacritic_sentences[idx]
        else:
            return self.character_sentences[idx]
           
    def encode_chars(self, sentences:List[List[str]]) -> List[List[int]]: 
        encoded_sentences = []
        for sentence in sentences:
            encoded_chars = [self.characters2id[char] for char in sentence]
            encoded_sentences.append(encoded_chars)
        return encoded_sentences
    
    def decode_chars(self, sentence: torch.Tensor) -> List[str]:
        # Remove padding
        unpadded_sentence = sentence[sentence != self.pad_char]
        
        decoded_chars = [self.id2characters[encoded_char.item()] for encoded_char in unpadded_sentence]
        return decoded_chars
    
    def decode_diacritics(self, diacritics: torch.Tensor) -> List[str]:
        decoded_diacritics = [self.diacritic_classes[diacritic] for diacritic in diacritics]
        return decoded_diacritics
            
    def segment_sentences(self, corpus:str, train:bool) -> List[str]:
        splitted_sentences = corpus.split('.')

        sentences = []
        for sentence in splitted_sentences:
            if len(sentence) > self.MAX_SENTENCE_SIZE:
                sentences += list(chunked(sentence, self.MAX_SENTENCE_SIZE))
            else:
                sentences.append(sentence)
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