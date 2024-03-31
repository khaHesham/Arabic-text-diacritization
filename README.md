# Arabic Text Diacritization Project

## Overview
This project aims to develop a system for Arabic text diacritization using natural language processing (NLP) techniques. Diacritization involves adding diacritical marks (e.g., vowels, short vowels, etc.) to Arabic text, which aids in pronunciation and comprehension, particularly for learners or in automated text processing tasks.

## Features
- Diacritize Arabic text input.
- Support for various diacritical marks commonly used in Arabic.
- Evaluate diacritization accuracy through metrics such as accuracy, precision, and recall.
- Trainable model for improving diacritization performance.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/khaHesham/arabic-diacritization.git
   ```
2. Install dependencies:
   ```bash
   cd arabic-diacritization
   pip install -r requirements.txt
   ```

## Usage
1. Prepare your Arabic text data.
2. Run the diacritization script:
   ```bash
   python diacritize.py --input input.txt --output output.txt
   ```
   Replace `input.txt` with the path to your input file and `output.txt` with the desired output file path.
3. Evaluate diacritization accuracy:
   ```bash
   python evaluate.py --predicted predicted.txt --gold gold.txt
   ```
   Replace `predicted.txt` with the path to the predicted diacritized text file and `gold.txt` with the path to the gold standard diacritized text file.

## Training
If you wish to train your own diacritization model:
1. Prepare a training dataset with diacritized Arabic text.
2. Train the model:
   ```bash
   python train.py --train_data train.txt --dev_data dev.txt --model_dir model/
   ```
   Replace `train.txt` with the path to your training data, `dev.txt` with the path to your development data, and `model/` with the desired directory for saving the trained model.

## Contributors
- Abdelaziz Salah
- Abdelrahman Noaman
- Khaled Hesham
- Kirollos Samy

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any inquiries or feedback, please contact [AEyeTeam](mailto:abdelaziz132001@gmail.com).
```

This Markdown file can be saved as `README.md` in the root directory of your project. Adjust any paths or details as needed.
