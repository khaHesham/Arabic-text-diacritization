import pickle
import re 

class Preprocessor: 
    def __init__(self):
        with open('diacritic2id.pickle', 'rb') as file:
            # Load the object from the pickle file
            self.diacritic2id = pickle.load(file)
            self.diacritic_classes = list(self.diacritic2id.keys())
            
        with open('diacritics.pickle', 'rb') as file:
            self.diacritics = pickle.load(file)
            
        with open('arabic_letters.pickle', 'rb') as file:
            self.arabic_letters = pickle.load(file)
                    
    def is_diacritic(self, char):
        return (char in self.diacritics)
    
    def clean(self, corpus: str) -> str:
        separators = {' ', '.', ',', ';', ':'}
        allowed_chars = self.diacritics | self.arabic_letters | separators
        pattern = f'[^{"".join(allowed_chars)}]'
        cleaned_corpus = re.sub(pattern, '', corpus)

        return cleaned_corpus
           
    def separate_diacritics(self, sentence: str) -> (list[str], list[int]):
        prev_is_diacritic = False
        prev_is_char = False

        curr_diacritic = ''
        diacritics = []
        characters = []
        
        for char in sentence:
            if self.is_diacritic(char):
                if(prev_is_diacritic):
                    curr_diacritic += char
                    print(ord(curr_diacritic[0]), ord(curr_diacritic[1]))
                    diacritics[-1] = self.diacritic2id[curr_diacritic]
                else:
                    diacritics.append(self.diacritic2id[char])
                    curr_diacritic = char
                prev_is_diacritic = True
            else:
                characters.append(char)
                if(prev_is_char):
                    diacritics.append(self.diacritic2id[''])
                prev_is_char = True
            
        return characters, diacritics



def main():
    # Test the clean method
    corpus = 'وَحَيَوَانٌ غَيْرُ مَوْجُودٍ'
    preprocessor = Preprocessor()

    # print(ord(preprocessor.diacritic_classes[12][0]), ord(preprocessor.diacritic_classes[12][1]))
    
    cleaned_corpus = preprocessor.clean(corpus)

    characters, diacritics = preprocessor.separate_diacritics(cleaned_corpus)
    print(characters)
    print(diacritics)

    # Test the separate_diacritics method
    # sentence = "وَحَيَوَانٌ غَيْرُ مَوْجُودٍ ."
    # characters, diacritics = preprocessor.separate_diacritics(sentence)
    # print("Characters:", characters)
    # print("Diacritics:", diacritics)

if __name__ == "__main__":
    main()