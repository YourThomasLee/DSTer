#!/usr/bin python
#-*- encoding: utf-8 -*-
import nltk

BOS_WORD ='BOS'
EOS_WORD ='EOS'
BLANK_WORD = 'PAD'
UNK_WORD = 'UNK'

class Vocab:
    __slots__=['word_counter','n_words','word2idx','idx2word']
    def __init__(self):
        self.word_counter=dict()
        self.n_words=0
        self.word2idx=dict()
        self.idx2word=dict()
        self.add_words_list([BLANK_WORD,BOS_WORD,EOS_WORD,UNK_WORD])
    
    def add_word(self, word):
        self.word_counter[word] = self.word_counter.get(word,0) + 1
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.n_words += 1
    
    def add_words_list(self,words_list):
        for word in words_list:
            self.add_word(word)
    
    def add_sentence(self, sentence):
        pattern = r'''(?x) # set flag to allow verbose regexps
            (?:[A-Z]\.)+[A-Z]   # abbreviations, e.g. U.S.A
            | \w+(?:-\w+)* # words with optional internal hyphens
            | \$?\d+(?:\.\d+)?\%? # currency and percentages, e.g. $12.40, 82%
            | \.\.\.      # ellipsis
            |(?:[.,;"'?():-_`!])  # these are separate tokens; includes ], [
        '''
        tokenlist = nltk.regexp_tokenize(sentence, pattern)
        for word in tokenlist:
            if len(word.strip())>0:
                self.add_word(word.strip())

    def merge(self,vocab_list):
        for vocab in vocab_list:
            for word in vocab.word2idx:
                self.add_word(word)
    
    def get_index(self, word):
        return self.word2idx.get(word,3)

    def get_word(self, idx):
        return self.idx2word.get(idx, UNK_WORD)
    
    def size(self):
        return self.n_words
