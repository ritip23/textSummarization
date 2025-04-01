# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 20:31:21 2025

@author: RITI
"""

from flask import Flask, render_template, request
from transformers import pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.luhn import LuhnSummarizer
from heapq import nlargest
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation

app = Flask(__name__)

# BERT Summarizer pipeline
bert_summarizer = pipeline("summarization")

# SpaCy model
nlp = spacy.load('en_core_web_sm')

def luhn_summarizer(text, sentences_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LuhnSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return " ".join(str(sentence) for sentence in summary)

def spacy_summarizer(rawdocs):
    stopwords = list(STOP_WORDS)
    doc = nlp(rawdocs)
    word_freq = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_freq.keys():
                word_freq[word.text] = 1
            else:
                word_freq[word.text] += 1

    if not word_freq:
        return ""  # if text is empty or stopwords only

    max_freq = max(word_freq.values())

    for word in word_freq.keys():
        word_freq[word] = word_freq[word] / max_freq

    sent_tokens = [sent for sent in doc.sents]
    sent_scores = {}
    for sent in sent_tokens:
        for word in sent:
            if word.text in word_freq.keys():
                if sent not in sent_scores.keys():
                    sent_scores[sent] = word_freq[word.text]
                else:
                    sent_scores[sent] += word_freq[word.text]

    select_len = max(1, int(len(sent_tokens) * 0.3))  # always at least 1

    summary = nlargest(select_len, sent_scores, key=sent_scores.get)
    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)
    return summary

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    text = request.form["text"]
    algorithm = request.form["algorithm"]
    summary = ""

    if algorithm == "luhn":
        summary = luhn_summarizer(text)
    elif algorithm == "bert":
        summary = bert_summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    elif algorithm == "spacy":
        summary = spacy_summarizer(text)

    return render_template("index.html", original_text=text, summary=summary, selected_algorithm=algorithm)

if __name__ == "__main__":
    app.run(debug=True)
