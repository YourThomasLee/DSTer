#-*- encoding: utf-8 -*-
import re

def restore_common_abbr(caption):
    # 还原常见缩写单词
    pat_is = re.compile("(it|he|she|that|this|there|here)(\'s)", re.I)
    pat_blank = re.compile("[\t ]+")
    pat_s = re.compile("(?<=[a-zA-Z])\'s")  # 找出字母后面的字母
    pat_s2 = re.compile("(?<=s)\'s?")
    pat_not = re.compile("(?<=[a-zA-Z])n\'t")  # not的缩写
    pat_would = re.compile("(?<=[a-zA-Z])\'d")  # would的缩写
    pat_will = re.compile("(?<=[a-zA-Z])\'ll")  # will的缩写
    pat_am = re.compile("(?<=[I|i])\'m")  # am的缩写
    pat_are = re.compile("(?<=[a-zA-Z])\'re")  # are的缩写
    pat_ve = re.compile("(?<=[a-zA-Z])\'ve")  # have的缩写

    new_text = caption
    new_text = pat_is.sub(r"\1 is", new_text)
    new_text = pat_blank.sub(" ", new_text)#remove redundant blank space
    new_text = pat_s.sub("", new_text)
    new_text = pat_s2.sub("", new_text)
    new_text = pat_not.sub(" not", new_text)
    new_text = pat_would.sub(" would", new_text)
    new_text = pat_will.sub(" will", new_text)
    new_text = pat_am.sub(" am", new_text)
    new_text = pat_are.sub(" are", new_text)
    new_text = pat_ve.sub(" have", new_text)
    new_text = new_text.replace('\'', ' ')
    return new_text