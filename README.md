# Generative AI with LSTM - Text Generation

## Project Overview

This project implements a text generation system using a Long Short-Term Memory (LSTM) neural network. The model is trained on the public-domain works of William Shakespeare and generates text based on a given seed phrase.

The project follows an end-to-end Generative AI text generation workflow including text preprocessing, tokenization, sequence preparation, model training, and text generation.

## Dataset

The dataset contains the complete works of William Shakespeare from Project Gutenberg.

Dataset source:

https://www.gutenberg.org/ebooks/100

The dataset is stored locally as:

```text
dataset/shakespeare.txt

Technologies Used
- Python 3.13
- TensorFlow / Keras
- NumPy
- Scikit-learn
- Matplotlib
Workflow
1. Load the Shakespeare text dataset.
2. Convert the text to lowercase.
3. Remove punctuation.
4. Tokenize the text into words.
5. Create input-output sequences for next-word prediction.
6. Pad sequences to a common length.
7. Split the data into training and validation sets.
8. Build an LSTM neural network.
9. Train the model using the training and validation data.
10. Generate text from different seed phrases.
Dataset Statistics
- Total characters: 5,378,701
- Unique words: 33,515
- Total input sequences: 812,482
- Maximum sequence length: 20
- Training samples: 731,233
- Validation samples: 81,249
Model Architecture
The project uses a Sequential neural network with the following architecture:
Input Sequence
      |
      v
Embedding Layer
128-dimensional embeddings
      |
      v
LSTM Layer
128 LSTM units
      |
      v
Dense Layer
33,515 output classes
      |
      v
Softmax
Next Word Prediction

Model Configuration
- Embedding dimension: 128
- LSTM units: 128
- Output classes: 33,515
- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy
- Batch size: 128
- Training epochs completed: 5
The model contains approximately 8.74 million trainable parameters.
Training
The dataset was divided into training and validation sets.
Early stopping and model checkpointing were configured to monitor validation loss.
The best model checkpoint was saved at:
model/best_lstm_model.keras

The model was trained for 5 completed epochs. Training was stopped after the fifth epoch because the model checkpoint had already been saved and the text generation pipeline was working successfully.
Text Generation
After training, the saved LSTM model is used to generate text from a seed phrase.
The generation process works as follows:
1. Take a seed phrase as input.
2. Convert the seed phrase into token IDs.
3. Pad the sequence to the required length.
4. Use the trained LSTM model to predict the next word.
5. Add the predicted word to the generated text.
6. Repeat the process to generate additional words.
Sample Seed Phrases
The model was tested using multiple seed phrases:
- to be or not
- shall i compare thee
- the king
- love is
Sample Generated Output
Seed: "to be or not"
to be or not to be so much as i am a man of my heart and i am a man of my heart and i am a man and i am a man

Seed: "shall i compare thee"
shall i compare thee to the king and the king of the king of the king of the king and the king of the king of the king and the king of the king

Seed: "the king"
the king is not the king of the king of the king of the king and the duke of york and the duke of york and the duke of york and the

Seed: "love is"
love is the matter of the king and the king of the king of the king and the duke of york and the duke of york and the duke of york and

Generated Outputs
The generated text samples are saved in:
outputs/generated_text.txt

Project Structure
GenerativeAI_LSTM_Text_Generator/
|
├── dataset/
│   └── shakespeare.txt
|
├── model/
│   └── best_lstm_model.keras
|
├── outputs/
│   └── generated_text.txt
|
├── lstm_text_generator.py
├── README.md
└── requirements.txt

How to Run
1. Create a Virtual Environment
python -m venv venv

2. Activate the Virtual Environment
For Windows PowerShell:
.\venv\Scripts\Activate.ps1

3. Install Dependencies
pip install -r requirements.txt

4. Run the Project
python lstm_text_generator.py

The script performs the preprocessing, sequence preparation, model construction, training, and text generation workflow.
Model Checkpoint
The trained model checkpoint is stored at:
model/best_lstm_model.keras

Future Improvements
- Train the model for more epochs.
- Experiment with different sequence lengths.
- Tune the LSTM architecture.
- Experiment with different vocabulary sizes.
- Use temperature-based sampling for more diverse text generation.
- Reduce repetitive generated text.
- Compare different LSTM configurations.
Conclusion
This project demonstrates an end-to-end Generative AI text-generation pipeline using an LSTM neural network trained on the public-domain works of William Shakespeare.
The system successfully performs text preprocessing, tokenization, sequence preparation, LSTM model training, model checkpointing, and next-word text generation from multiple seed phrases.