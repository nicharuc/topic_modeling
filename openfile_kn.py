# packages
from sklearn.feature_extraction import stop_words
#from tools import Extractor
import spacy
import nltk
import re
import string
import codecs
import numpy as np
import pandas as pd
#from user_definitions import *
import json
from sklearn.cluster import MiniBatchKMeans
import collections
from collections import OrderedDict
from sklearn.cluster import KMeans
import itertools



nlp = spacy.load('en')
STOPWORDS_DICT = {lang: set(nltk.corpus.stopwords.words(lang)) for lang in nltk.corpus.stopwords.fileids()}


def lemma_all(df, col_name):
    """
    Combines all functions used to clean and lemmatize the comments.
    :param df: data frame with comments
    :param col_name: column name in data frame containing comments
    :return: data frame with comments column lemmatized
    """

    # encode for only ascii characters
    df[col_name] = df[col_name].map(ascii_lower)


    # lemmatize words
    df[col_name] = df[col_name].map(lemma)

    # remove punctuation
    df[col_name] = df[col_name].map(punc_n)


    # filter only english comments
    df['language'] = df[col_name].map(get_language)
    df = df.loc[df['language'] == 'english']
    df = df.drop('language', axis=1)
    df[col_name] = df[col_name].map(noun_only)
    df = df[df[col_name] != ""]
    df[col_name] = df[col_name].map(lambda x: x.lower())

    return df


def ascii_lower(comment):
    """
    Parses comments and keep only ascii characters
    :param comment: a comment
    :return: comment with only ascii characters
    """
    comment = comment.encode('ascii', errors = 'ignore')
    return comment

def get_language(text):
    """
    Determines what language the comment is written in and filters only English comments.
    :param text: comment
    :return: language of comment
    """
    words = set(nltk.wordpunct_tokenize(text.lower()))
    return max(((lang, len(words & stopwords)) for lang, stopwords in STOPWORDS_DICT.items()), key = lambda x: x[1])[0]


def punc_n(comment):
    """
    Removes punctuations from comments.
    :param comment: a comment
    :return: comment without punctuations
    """
    regex = re.compile('[' + re.escape('!"#%&\'()*+,-./:;<=>?@[\\]^_`{|}~')+'0-9\\r\\t\\n]')
    nopunct = regex.sub(" ", comment)
    nopunct_words = nopunct.split(' ')
    filter_words = [word.strip() for word in nopunct_words if word != '']
    words = ' '.join(filter_words)
    return words



def lemma(comment):
    """
    Lemmatize comments using spacy lemmatizer.
    :param comment: a comment
    :return: lemmatized comment
    """
    lemmatized = nlp(unicode(comment, 'utf-8'))
    lemmatized2 = ' '.join([t.lemma_ for t in lemmatized if t.lemma_ != '\'s'])
    return lemmatized2


#function to filter only NN/NNS or JJ/NN
def rightOrdering(list_of_tuples):
    type_of_words = zip(*list_of_tuples)[1]
    acceptable_types = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'CC')
    if all(type in acceptable_types for type in type_of_words):
        all_nouns = ('NNS', 'NN', 'NNP', 'NNPS', 'CC')
        satisfy_all_nouns = all(type in all_nouns for type in type_of_words)
        satisfy_jj_condition = type_of_words[0] in ('JJ','JJR','JJS') and all(type not in ('JJ','JJR','JJS') for type in type_of_words[1:])
        return satisfy_all_nouns or satisfy_jj_condition
    return False

def noun_only(x):
    x = x.split(" ")

    if len(x[0])> 0:
        pos_comment = nltk.pos_tag(x)
        filtered = [word[0] for word in pos_comment if word[1] in ['NN', 'NNP', 'VB', 'VBD', 'VBG', 'VBN', 'VBZ']]
        #filtered = [word[0] for word in pos_comment if word[1] in ['NN', 'NNP']]
        words = ' '.join(filtered)
    return words



