import torch
from CBHG import CBHGModel
from diacritizer import Diacritizer
from dataset import DiacriticsDataset
import pandas as pd


# test_dataset_path = 'test_no_diacritics.txt'
test_dataset_path = 'dataset/sample_test_no_diacritics.txt'
model_path = 'models/CBHG_EP20_BS256.pth'
# input_csv_path = 'test_set_without_labels.csv'
output_csv_path = 'output/labels.csv'
test_dataset_diacritized_path = 'output/diacritized.txt'

def main():
    # Load dataset
    test_dataset = DiacriticsDataset()
    test_dataset.load(test_dataset_path, train=False)
    inputs = test_dataset.character_sentences
    
    # Load the model
    model = CBHGModel(
        inp_vocab_size = 37,
        targ_vocab_size = 15,
    )
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    
    # Predict
    with torch.no_grad():
        outputs = model(inputs)
    diacritics = torch.argmax(outputs['diacritics'], dim=-1)
    
    # Remove padding
    mask_no_pad = inputs != test_dataset.pad_char
    output_diacritics = diacritics[mask_no_pad]
    
    # Save diacritics to CSV
    df = pd.DataFrame(output_diacritics.numpy(), columns=["label"])
    df = df.rename_axis('ID').reset_index()
    df.to_csv(output_csv_path, index=False)
    
    # Diacritize the sentence
    with open(test_dataset_path, 'r', encoding='utf-8') as file:
        corpus = file.read()
        
    diacritizer = Diacritizer()
    diacritized_corpus = diacritizer.diacritize(corpus, output_diacritics)
    
    with open(test_dataset_diacritized_path, 'w', encoding='utf-8') as file:
        corpus = file.write(diacritized_corpus)

if __name__=='__main__':
    main()