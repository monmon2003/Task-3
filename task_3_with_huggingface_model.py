from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer1 = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
model1 = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

tokenizer2 = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
model2 = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")

def translate(sentence, tokenizer, model):
    english_sentence = tokenizer([sentence], return_tensors = "pt")
    generated_ids = model.generate(**english_sentence)
    french_sentence = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return french_sentence

def translate_to_French_and_Hindi(word):
    if len(word) != 10:
        print("Cannot translate as number of letter is not equal to 10. Try again!")
    else:
        print(f"French Translation: {translate(word, tokenizer1, model1)}")
        print(f"Hindi Translation: {translate(word, tokenizer2, model2)}")

def main():
    word = input("Enter an English word: ")
    translate_to_French_and_Hindi(word)

if __name__ == "__main__":
    main()