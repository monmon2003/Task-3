import json
from keras.models import load_model
from keras.preprocessing.text import tokenizer_from_json
from keras.utils import pad_sequences
import numpy as np



#load  French model
model_1 = load_model('english_to_french_model')

#load  Hindi model
model_2 = load_model('english_to_hindi_model')

#load Tokenizer
with open('english_tokenizer.json') as f:
    data = json.load(f)
    english_tokenizer = tokenizer_from_json(data)
    
with open('French_tokenizer.json') as f:
    data = json.load(f)
    french_tokenizer = tokenizer_from_json(data)

with open('hindi_tokenizer.json') as f:
    data = json.load(f)
    hindi_tokenizer = tokenizer_from_json(data)
    

#load max length
with open('sequence_length.json') as f:
    max_length_fr = json.load(f)
with open('sequence_length_hindi.json') as f:
    max_length_hn = json.load(f)
    
def pad(x, length=None):
    return pad_sequences(x, maxlen=length, padding='post')

def translate_to_french(english_sentence):
    english_sentence = english_sentence.lower()
    
    english_sentence = english_sentence.replace(".", '')
    english_sentence = english_sentence.replace("?", '')
    english_sentence = english_sentence.replace("!", '')
    english_sentence = english_sentence.replace(",", '')
    
    english_sentence = english_tokenizer.texts_to_sequences([english_sentence])
    english_sentence = pad(english_sentence, max_length_fr)
    
    english_sentence = english_sentence.reshape((-1,max_length_fr))
    
    french_sentence = model_1.predict(english_sentence)[0]
    
    french_sentence = [np.argmax(word) for word in french_sentence]

    french_sentence = french_tokenizer.sequences_to_texts([french_sentence])[0]
    
    
    return french_sentence

def translate_to_hindi(english_sentence):
    english_sentence = english_sentence.lower()
    
    english_sentence = english_sentence.replace(".", '')
    english_sentence = english_sentence.replace("?", '')
    english_sentence = english_sentence.replace("!", '')
    english_sentence = english_sentence.replace(",", '')
    
    english_sentence = english_tokenizer.texts_to_sequences([english_sentence])
    english_sentence = pad(english_sentence, max_length_hn)
    
    english_sentence = english_sentence.reshape((-1,max_length_hn))
    
    hindi_sentence = model_2.predict(english_sentence)[0]
    
    hindi_sentence = [np.argmax(word) for word in hindi_sentence]

    hindi_sentence = hindi_tokenizer.sequences_to_texts([hindi_sentence])[0]
    
    return hindi_sentence

def translate_to_French_and_Hindi(word):
    if len(word) != 10:
        print("Cannot translate as number of letter is not equal to 10. Try again!")
    else:
        print(f"French Translation: {translate_to_french(word)}")
        print(f"Hindi Translation: {translate_to_hindi(word)}")

def main():
    word = input("Enter word to translate")
    translate_to_French_and_Hindi(word)

if __name__ == "__main__":
    main()
