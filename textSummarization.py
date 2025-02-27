import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation


text = """The Marvel Cinematic Universe (MCU) is an American media franchise and shared universe centered on a series of superhero films produced by Marvel Studios. The films are based on characters that appear in American comic books published by Marvel Comics. The franchise also includes several television series, short films, digital series, and literature. The shared universe, much like the original Marvel Universe in comic books, was established by crossing over common plot elements, settings, cast, and characters."""

stopwords = list(STOP_WORDS)
#print(stopwords)
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)
#print(doc)
tokens = [token.text for token in doc]
#print(tokens)
word_freq = {}
for word in doc:
    if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
        if word.text not in word_freq.keys():
            word_freq[word.text] = 1
        else:
            word_freq[word.text] += 1
#print(word_freq)

max_freq = max(word_freq.values())
#print(max_freq)

for word in word_freq.keys():
    word_freq[word] = word_freq[word]/max_freq

#print(word_freq)

sent_tokens = [sent for sent in doc.sents]
print(sent_tokens)
