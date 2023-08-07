import datetime
import os
from collections import Counter
from pytrends.request import TrendReq
import nltk
from nltk.corpus import stopwords, words
from nltk.stem.porter import PorterStemmer
import json
import gzip
import pickle
import time
import numpy as np
import tarfile
import email
import mailbox
import csv
import pandas as pd
import re
import gtab
from random import randrange, sample
 
def load_stem_trends(dataset_name, nkw):
    pro_dataset_path = "../datasets/trends/" + str(dataset_name) + "_trends_" + str(nkw) + ".pkl"
    if not os.path.exists(pro_dataset_path):
        raise ValueError("The file {} does not exist".format(pro_dataset_path))
    with open(pro_dataset_path, "rb") as f:
        stem, stem_trends = pickle.load(f)
    return stem, stem_trends


def remove_n_random_items(lst1, lst2, n):
    to_delete = set(sample(range(len(lst1)), int(len(lst1) * n)))
    delete_lst1 = [item for index, item in enumerate(lst1) if not index in to_delete]
    delete_lst2 = [item for index, item in enumerate(lst2) if not index in to_delete]
    return delete_lst1, delete_lst2

def get_size_frompkl(datasetname, year, month, nkw=500):
    save_path = '../datasets/' + str(datasetname)
    with open(save_path + '/' + str(year) + "_" + str(month) + ".pkl", 'rb') as f:
        dataset, res_dataset_length = pickle.load(f)
    stems, stems_trends = load_stem_trends(datasetname, nkw)

    keyword_size = {}
    temp_keyword_size = {}

    unique_word = np.unique([kw for document in dataset for kw in document])
    for word in unique_word:
        temp_keyword_size[word] = []

    for indexd, document in enumerate(dataset):
        for word in document:
            temp_keyword_size[word].append(res_dataset_length[indexd])

    for word in stems:
        if word in temp_keyword_size:
            keyword_size[word] = temp_keyword_size[word]
    # print(len(keyword_size.keys()))
    # print(stems)
    return keyword_size

def delete_get_size_frompkl(datasetname, year, month, nkw, delete_n):
    save_path = '../datasets/' + str(datasetname)
    with open(save_path + '/' + str(year) + "_" + str(month) + ".pkl", 'rb') as f:
        dataset, res_dataset_length = pickle.load(f)
    stems, stems_trends = load_stem_trends(datasetname, nkw)

    keyword_size = {}
    temp_keyword_size = {}

    dataset, res_dataset_length = remove_n_random_items(dataset, res_dataset_length, delete_n)

    unique_word = np.unique([kw for document in dataset for kw in document])
    for word in unique_word:
        temp_keyword_size[word] = []

    for indexd, document in enumerate(dataset):
        for word in document:
            temp_keyword_size[word].append(res_dataset_length[indexd])

    for word in stems:
        if word in temp_keyword_size:
            keyword_size[word] = temp_keyword_size[word]
    # print(len(keyword_size.keys()))
    return keyword_size


if __name__ == "__main__":
    dataset_list = ["enron-full", "lucene"]
    year_list = [1999, 2000, 2001, 2002]
    month_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # get_size_length_frompkl("enron-full", 2000, 1)
    # save_path = '../../datasets_mon_alter/' + "enron-full" + "_dict/" + str(2000) + str(1) + ".pkl"
    # with open(save_path, 'rb') as f:
    #     keyword_size_length_trends = pickle.load(f)
    # print(keyword_size_length_trends)

    nkw = 500
    res = get_size_frompkl("enron-full", 2001, 1, 500)


