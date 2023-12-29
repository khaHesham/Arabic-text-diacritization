import pickle
import re 

class Preprocessor: 
    def __init__(self):
        with open('diacritic2id.pickle', 'rb') as file:
            # Load the object from the pickle file
            self.diacritic2id = pickle.load(file)
            self.diacritic_classes = self.diacritic2id.keys()
            
        with open('diacritics.pickle', 'rb') as file:
            self.diacritics = pickle.load(file)
            
        with open('arabic_letters.pickle', 'rb') as file:
            self.arabic_letters = pickle.load(file)
                    
    def is_diacritic(self, char):
        return (char in self.diacritics)
    
    def clean(self, corpus: str) -> str:
        separators = {' ', '.', ',', ';', ':'}
        allowed_chars = self.diacritics | self.arabic_letters | separators
        pattern = f'[^{" ".join(allowed_chars)}]'
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


# try the clean method 
corpus = '''وْلُهُ : ( أَوْ قَطَعَ الْأَوَّلُ يَدَهُ إلَخْ ) قَالَ الزَّرْكَشِيُّ( 14 / 123 )
ابْنُ عَرَفَةَ : قَوْلُهُ : بِلَفْظٍ يَقْتَضِيه كَإِنْكَارِ غَيْرِ حَدِيثٍ بِالْإِسْلَامِ وُجُوبَ مَا عُلِمَ وُجُوبُهُ مِنْ الدِّينِ ضَرُورَةً ( كَإِلْقَاءِ مُصْحَفٍ بِقَذَرٍ وَشَدِّ زُنَّارٍ ) ابْنُ عَرَفَةَ : قَوْلُ ابْنِ شَاسٍ : أَوْ بِفِعْلٍ يَتَضَمَّنُهُ هُوَ كَلُبْسِ الزُّنَّارِ وَإِلْقَاءِ الْمُصْحَفِ فِي صَرِيحِ النَّجَاسَةِ وَالسُّجُودِ لِلصَّنَمِ وَنَحْوِ ذَلِكَ ( وَسِحْرٍ ) مُحَمَّدٌ : قَوْلُ مَالِكٍ وَأَصْحَابِهِ أَنَّ السَّاحِرَ كَافِرٌ بِاَللَّهِ تَعَالَى قَالَ مَالِكٌ : هُوَ كَالزِّنْدِيقِ إذَا عَمِلَ السِّحْرَ بِنَفْسِهِ قُتِلَ وَلَمْ يُسْتَتَبْ .
( قَوْلُهُ لِعَدَمِ مَا تَتَعَلَّقُ إلَخْ ) أَيْ الْوَصِيَّةُ ( قَوْلُهُ مَا مَرَّ ) أَيْ قُبَيْلَ قَوْلِ الْمَتْنِ لَغَتْ وَلَوْ اقْتَصَرَ عَلَى أَوْصَيْت لَهُ بِشَاةٍ أَوْ أَعْطُوهُ شَاةً وَلَا غَنَمَ لَهُ عِنْدَ الْمَوْتِ هَلْ تَبْطُلُ الْوَصِيَّةُ أَوْ يُشْتَرَى لَهُ شَاةٌ وَيُؤْخَذُ مِنْ قَوْلِهِ الْآتِي كَمَا لَوْ لَمْ يَقُلْ مِنْ مَالِي وَلَا مِنْ غَنَمِي أَنَّهَا لَا تَبْطُلُ ، وَعِبَارَةُ الْكَنْزِ وَلَوْ لَمْ يَقُلْ مِنْ مَالِي وَلَا مِنْ غَنَمِي لَمْ يَتَعَيَّنْ غَنَمُهُ إنْ كَانَتْ انْتَهَتْ ا ه سم ( قَوْلُهُ فَيُعْطَى وَاحِدَةً مِنْهَا إلَخْ ) كَمَا لَوْ كَانَتْ مَوْجُودَةً عِنْدَ الْوَصِيَّةِ وَالْمَوْتِ ، وَلَا يَجُوزُ أَنْ يُعْطَى وَاحِدَةً مِنْ غَيْرِ غَنَمِهِ فِي الصُّورَتَيْنِ وَإِنْ تَرَاضَيَا ؛ لِأَنَّهُ صُلْحٌ عَلَى مَجْهُولٍ مُغْنِي وَنِهَايَةٌ قَالَ ع ش قَوْلُهُ وَاحِدَةً مِنْهَا أَيْ كَامِلَةً ، وَلَا يَجُوزُ أَنْ يُعْطَى نِصْفَيْنِ مِنْ شَاتَيْنِ ؛ لِأَنَّهُ لَا يُسَمَّى شَاةً وَقَوْلُهُ وَلَا يَجُوزُ أَنْ يُعْطَى وَاحِدَةً مِنْ غَيْرِ غَنَمِهِ وَيَنْبَغِي أَنْ يُقَالَ مِثْلُ ذَلِكَ فِي الْأَرِقَّاءِ اه .
وَحَيَوَانٌ غَيْرُ مَوْجُودٍ .'''

print(corpus)

# preprocessor = Preprocessor()
# print (preprocessor.clean(corpus))