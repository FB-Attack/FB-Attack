import utils
import os
import email
import glob
import mailbox
import os
import random
import pandas as pd
import nltk
import numpy as np
import tqdm
import math
from collections import Counter
import json
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
def split_df(dframe, frac=0.5):
    first_split = dframe.sample(frac=frac)     
    second_split = dframe.drop(first_split.index)     
    return first_split, second_split
def get_body_from_enron_email(mail):
    """Extract the content from raw Enron email"""
    msg = email.message_from_string(mail)
    parts = []
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            parts.append(part.get_payload())
    return "".join(parts)
def extract_sent_mail_contents(maildir_directory="../maildir/") -> pd.DataFrame:
    """Extract the emails from the _sent_mail folder of each Enron mailbox."""
    path = os.path.expanduser(maildir_directory)
    mails = glob.glob(f"{path}/*/_sent_mail/*")
    mail_contents = []
    mail_lengths = []
    for mailfile_path in tqdm.tqdm(iterable=mails, desc="Reading the emails"):         
        with open(mailfile_path, "r") as mailfile:
            raw_mail = mailfile.read()
            mail_contents.append(get_body_from_enron_email(raw_mail))
        mail_lengths.append(utils.getFileSize(mailfile_path))
    return pd.DataFrame(data={"filename": mails, "mail_body": mail_contents, "mail_length": mail_lengths})
class KeywordExtractor:
    """Class to extract the keyword from a corpus/email set"""
    def __init__(self, corpus_df, file_name, min_freq=1):
        freq_dict, glob_freq_dict = KeywordExtractor.extract_email_voc(corpus_df)
        glob_freq_dict_json = json.dumps(glob_freq_dict)
        f = open(file_name, 'w')
        f.write(glob_freq_dict_json)
    @staticmethod     
    def get_voc_from_one_email(email_text, freq=False):
        stopwords_list = stopwords.words("english")         
        stopwords_list.extend(["subject", "cc", "from", "to", "forward"])         
        stemmer = PorterStemmer()         
        stemmed_word_list = [
            stemmer.stem(word.lower())             
            for sentence in sent_tokenize(email_text)             
            for word in word_tokenize(sentence)             
            if word.lower() not in stopwords_list and word.isalnum()         
            ]
        if freq:             
            return nltk.FreqDist(stemmed_word_list)
        else:             
            return stemmed_word_list
    @staticmethod     
    def extract_email_voc(dframe, one_occ_per_doc=True):
        freq_dict = {}
        glob_freq_list = {}
        for row_tuple in tqdm.tqdm(iterable=dframe.itertuples(), total=len(dframe)):
            temp_freq_dist = KeywordExtractor.get_voc_from_one_email(row_tuple.mail_body, freq=True)
            freq_dict[row_tuple.filename] = []             
            for word, freq in temp_freq_dist.items():                 
                freq_to_add = 1 if one_occ_per_doc else freq                 
                freq_dict[row_tuple.filename].append(word)                
                try:
                    glob_freq_list[word]["size"] += freq_to_add                     
                    glob_freq_list[word]["length"].append(row_tuple.mail_length)
                except KeyError:
                    glob_freq_list[word] = {}
                    glob_freq_list[word]["size"] = freq_to_add
                    glob_freq_list[word]["length"] = []
                    glob_freq_list[word]["length"].append(row_tuple.mail_length)
        return freq_dict, glob_freq_list 
    def consist(a, b):
        a = dict(Counter(a))
        b = dict(Counter(b))
        for key1, val1 in a.items():
            if key1 in b.keys() and val1 <= b[key1]:
                continue
            else:
                return False
        return True
def inter_list(a, b):
    a = dict(Counter(a))
    b = dict(Counter(b))
    count = 0
    for key1, val1 in a.items():
        if key1 in b.keys():
            count += min(val1, b[key1])
    return count
def test_data_top(edb_filename, partial_filename, frac1, frac2, word_size=500):
    f1 = open(edb_filename, 'r')
    edb_data = json.load(f1)
    f2 = open(partial_filename, 'r')
    partial_data = json.load(f2)
    edb_sorted = sorted(edb_data.items(), key=lambda x: x[1]['size'], reverse=True)
    partial_sorted = sorted(partial_data.items(), key=lambda x: x[1]['size'], reverse=True)
    edb_keyword = {}
    for tup in edb_sorted[0:word_size]:
        edb_keyword[tup[0]] = {"size": tup[1]["size"], "length": tup[1]["length"]}
    partial_keyword = {}
    for tup in partial_sorted[0:word_size]:
        partial_keyword[tup[0]] = {"size": tup[1]["size"], "length": tup[1]["length"]}
    count_uni = 0
    for ed_key, ed_value in tqdm.tqdm(edb_keyword.items()):
        count_in = 0
        for par_key, par_value in partial_keyword.items():
            res = inter_list(ed_value["length"], par_value["length"])
            ed_length = len(ed_value["length"])
            par_length = len(par_value["length"])
            res_percent = res * 1.0 / math.sqrt(ed_length * par_length)
            if res_percent >= (frac1 + frac2) - 1:
                count_in += 1
        if count_in == 1:
            count_uni += 1
    print(frac1, frac2, "containing unique result", count_uni)
frac1 = [0.9, 0.8, 0.7, 0.6, 0.5]
frac2 = [0.9, 0.8, 0.7]
for i in frac1:
    for j in frac2:
        filename1 = './edb/' + str(i) + '.json'
        filename2 = './partial/' + str(j) + '.json'
        test_data_top(filename1, filename2, i, j, 3000)