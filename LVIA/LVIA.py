from multiprocessing import Pool
import math
import os
import pickle
import numpy
import numpy as np
import time
import multiprocessing
import pickle
import email
import glob
import mailbox
import os
from collections import Counter
import random
import pandas as pd
import nltk
import tqdm
import json
import datetime
import pytz
import gtab
from collections import Counter
from random import randrange, sample
np.set_printoptions(suppress=True)


def load_sent_mail_contents(dataset_name, year=1999, month=1):
    pro_dataset_path = "../datasets/" + \
        str(dataset_name) + str(year) + "_" + str(month) + ".pkl"
    if not os.path.exists(pro_dataset_path):
        raise ValueError("The file {} does not exist".format(pro_dataset_path))
    with open(pro_dataset_path, "rb") as f:
        dataset, res_dataset_length = pickle.load(f)
    return dataset, res_dataset_length


def load_stem_trends(dataset_name, nkw):
    pro_dataset_path = "../datasets/trends/" + \
        str(dataset_name) + "_trends_" + str(nkw) + ".pkl"
    if not os.path.exists(pro_dataset_path):
        raise ValueError("The file {} does not exist".format(pro_dataset_path))
    with open(pro_dataset_path, "rb") as f:
        stem, stem_trends = pickle.load(f)
    return stem, stem_trends


def static_enron_data_info(dataset_name, nkw, n_month, alpha=0.5):
    keywordFrequency = Counter()
    if dataset_name == 'enron-full':
        year = 1999
        month = 12
    else:
        year = 2004
        month = 6
    dataset_by_month = []
    dataset_by_month.append([])
    i_count = 0
    for i in range(18):
        if month == 13:
            month = 1
            year += 1
        save_path = '../datasets/' + str(dataset_name)
        with open(save_path + '/' + str(year) + "_" + str(month) + ".pkl", 'rb') as f:
            dataset, res_dataset_length = pickle.load(f)
        for doc in dataset:
            for word in set(doc):
                keywordFrequency[word] += 1
        if i % n_month != 0 or i == 0:
            dataset_by_month[i_count].extend(dataset)
        else:
            i_count += 1
            dataset_by_month.append([])
            dataset_by_month[i_count].extend(dataset)
        month += 1
    chosen_keywords = set([kw for kw, _ in keywordFrequency.most_common(nkw)])
    dataset_split_known = []
    dataset_split_attack = []
    for i in range(i_count + 1):
        target_index = set(
            sample(range(len(dataset_by_month[i])), int(len(dataset_by_month[i]) * 0.5)))
        sampleSetSize = len(dataset_by_month[i]) - len(target_index)
        known_index = set(
            sample(set(range(len(dataset_by_month[i]))) - target_index, int(sampleSetSize * alpha / 0.5)))
        dataset_split_known.append(
            [dataset_by_month[i][index] for index in known_index])
        dataset_split_attack.append(
            [dataset_by_month[i][index] for index in target_index])
    word_to_id = {}
    for index, keyword in enumerate(chosen_keywords):
        word_to_id[keyword] = index
    keyword_split_known = np.zeros([i_count + 1, nkw])
    keyword_split_attack = np.zeros([i_count + 1, nkw])
    for i in range(i_count + 1):
        for email in dataset_split_known[i]:
            for keyword in chosen_keywords.intersection(email):
                keyword_split_known[i][word_to_id[keyword]] += 1
        for email in dataset_split_attack[i]:
            for keyword in chosen_keywords.intersection(email):
                keyword_split_attack[i][word_to_id[keyword]] += alpha / 0.5
    candicate_keyword = [[] for i in range(nkw)]
    for i in range(i_count + 1):
        for j in range(nkw):
            cand_np = np.abs(
                keyword_split_attack[i][j] - keyword_split_known[i])
            if keyword_split_attack[i][j] == 0:
                cand_index = np.where(cand_np <= 1)
            else:
                cand_index = np.where(
                    cand_np/keyword_split_attack[i][j] <= 0.4)
            candicate_keyword[j].extend(cand_index[0].tolist())
    maxCandidateSetSize_ = 0
    counter = []
    maxCandidateSetSize = 0
    threshold = 2
    keyword2query = [[] for i in range(nkw)]
    for i in range(0, nkw):
        curCounter = Counter(candicate_keyword[i])
        kvs = curCounter.most_common()
        maxCandidateSetSize_ = max(maxCandidateSetSize_, len(kvs))
        candicate_keyword[i] = []
        vset = set([v for k, v in kvs])
        vset = sorted(list(vset), reverse=True)
        if len(vset) > threshold:
            vset = vset[:threshold]
        for k, v in kvs:
            if v in vset:
                for z in range(v):
                    candicate_keyword[i].append(k)
                keyword2query[k].append(i)
            else:
                break
        counter.append(Counter(candicate_keyword[i]))
        maxCandidateSetSize = max(
            maxCandidateSetSize, len(candicate_keyword[i]))
    known_queries = {}
    mapped_keywords = set()
    flag = True
    while flag:
        flag = False
        for i in range(0, nkw):
            if i in known_queries:
                continue
            counter_keys = list(counter[i].keys())
            for counter_key in counter_keys:
                if counter_key in mapped_keywords:
                    del counter[i][counter_key]
            if len(counter[i].most_common()) != 1:
                continue
            else:
                cand = counter[i].most_common()[0][0]
                known_queries[i] = cand
                mapped_keywords.add(cand)
                for kw in candicate_keyword[i]:
                    if i in keyword2query[kw]:
                        keyword2query[kw].remove(i)
                keyword2query[cand].clear()
                candicate_keyword[i].clear()
                flag = True
        if not flag:
            target_keyword = -1
            for i in range(0, nkw):
                if i in mapped_keywords:
                    continue
                if target_keyword == -1 or len(keyword2query[i]) > 0 and len(keyword2query[i]) <= len(keyword2query[target_keyword]):
                    target_keyword = i
            if target_keyword == -1:
                break
            target_query = -1
            target_dist = 0
            for q in keyword2query[target_keyword]:
                if q in known_queries:
                    continue
                cur_dist = 0
                for interval in range(i_count+1):
                    cur_dist += np.abs(
                        keyword_split_attack[interval][target_keyword] - keyword_split_known[interval][q])
                if target_query == -1 or cur_dist < target_dist:
                    target_query = q
                    target_dist = cur_dist
            if target_query == -1:
                break
            known_queries[target_query] = target_keyword
            mapped_keywords.add(target_keyword)
            for kw in candicate_keyword[target_query]:
                if target_query in keyword2query[kw]:
                    keyword2query[kw].remove(target_query)
            keyword2query[target_keyword].clear()
            candicate_keyword[target_query].clear()
            flag = True
    same_count = 0
    for i in known_queries.keys():
        if known_queries[i] == i:
            same_count += 1
    return same_count


if __name__ == "__main__":
    repeat_cnt = 1
    nkw_list = [500, 1000, 2000, 3000]
    alphas = np.linspace(0.4, 0, 9)[:-1]
    dataset_names = ['enron-full', 'lucene']
    query_month = [18, 6, 3, 2, 1]
    start = time.perf_counter()
    with Pool(32) as pool:
        results = pool.starmap(static_enron_data_info, [
            (dataset_name, nkw, n_month, alpha) for dataset_name in dataset_names for nkw in nkw_list for n_month in query_month for alpha in alphas for _ in range(repeat_cnt)
        ])
        pool.close()
        pool.join()
    end = time.perf_counter()
    print("time: ", end - start, " s")
    for dataset_name in dataset_names:
        print("dataset_name: ", dataset_name)
        for nkw in nkw_list:
            for n_month in query_month:
                for alpha in alphas:
                    list_res = []
                    for _ in range(repeat_cnt):
                        list_res.append(results.pop(0) / nkw)
                    print("threshold: ", alpha, "nkw: ", nkw, "n_month: ", n_month,
                          "# intervals: ", 18 // n_month)
                    print("average accuracy: ", np.mean(
                        list_res), ", std: ", np.std(list_res))
        print()
