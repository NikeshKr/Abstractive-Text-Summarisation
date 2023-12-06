# Abstractive-Text-Summarisation
# Abstractive Text Summarization with Word Sense Disambiguation and Seq2Seq Model

## Introduction
This repository contains the code for an Abstractive Text Summarization model that incorporates Word Sense Disambiguation (WSD) using WordNet and utilizes a Seq2Seq architecture.

## Overview
The code includes the following components:

1. **Data Preprocessing:**
   - Utilizes the CNN/Daily Mail dataset for training.
   - Implements WSD using WordNet to disambiguate word senses in the text.

2. **Model Architecture:**
   - Implements a Seq2Seq model for abstractive text summarization.
   - Utilizes an Encoder-Decoder architecture with LSTM layers.

3. **Tokenization and Padding:**
   - Tokenizes and preprocesses the text data.
   - Converts text to sequences and pads sequences to a specified length.

4. **Training:**
   - Compiles the model using the RMSprop optimizer and sparse categorical crossentropy loss.
   - Trains the model on the provided dataset for a specified number of epochs.

5. **Model Saving and Loading:**
   - Includes functions to save and load both the preprocessed texts and the trained Seq2Seq model.

6. **Summarization Function:**
   - Defines a function to generate abstractive summaries for input articles.

## Instructions
To use this code, follow these steps:

1. Install necessary libraries: `tensorflow`, `nltk`, `datasets`. Ensure you have the required dependencies by running the provided installation commands.

2. Run the notebook: Execute the notebook `Abstractive_Text_Summarisation_with_wsd&seq2seq_model.ipynb` to train the model and perform text summarization.

3. Save and Load Model: The code includes functions to save and load both preprocessed texts and the trained model. Customize the paths according to your preferences.

4. Summarize Text: Use the `summarize_text` function to generate abstractive summaries for input articles.

## Example Usage
```python
input_article = '''[Your input article here]'''
summary = summarize_text(input_article)
print("Generated Summary:", summary)
```


## References
- [CNN/Daily Mail Dataset](https://huggingface.co/datasets/cnn_dailymail)
- [WordNet](https://www.nltk.org/howto/wordnet.html)

## Author
Nikesh Kumar
