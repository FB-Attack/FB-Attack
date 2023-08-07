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
from config import PRE_DATASET_FOLDER, RAW_DATASET_FOLDER, PRO_DATASET_FOLDER
import numpy as np
import tarfile
import email
import mailbox
import csv
import pandas as pd
import re
import gtab
from utils import *
def find_most_popular_keyword(keywords, nweeks):
    pytrend = TrendReq()
    most_popular = keywords[0]
    for i in range(1, len(keywords), 4):
        time.sleep(5)         
        if i + 4 <= len(keywords):
            keywords_to_search = keywords[i:i + 4]
        else:
            keywords_to_search = keywords[i:]
        keywords_to_search += [most_popular]
        nread = len(keywords_to_search)
        print("({:d}/{:d}) Searching for:".format(i, len(keywords)), keywords_to_search, flush=True)
        pytrend.build_payload(kw_list=keywords_to_search)
        pytrend.build_payload(kw_list=keywords_to_search, geo='US', timeframe='2020-01-01 2020-12-31')
        data = pytrend.interest_over_time().to_numpy()
        mini_matrix = np.zeros((nread, nweeks))
        for i_kw in range(nread):
            for j, i_week in enumerate(range(-nweeks - 1, -1)):
                mini_matrix[i_kw, j] = data[i_week][i_kw]
        idx_max = np.argmax(np.max(mini_matrix, axis=1))
        if idx_max < nread - 1:
            print("{:s} replaced {:s} as most popular!".format(keywords_to_search[idx_max], most_popular), flush=True)
            most_popular = keywords_to_search[idx_max]
    return most_popular
def get_keyword_trends(keywords, baseline_keyword, nweeks):
    pytrend = TrendReq()
    nkw = len(keywords)
    trend_matrix = np.zeros((nkw, nweeks))
    for i in range(0, nkw, 4):
        time.sleep(5)         
        if i + 4 <= nkw:
            keywords_to_search = keywords[i:i + 4]
        else:
            keywords_to_search = keywords[i:]
        nread = len(keywords_to_search)
        print(keywords_to_search)
        if baseline_keyword in keywords_to_search:
            i_baseline = keywords_to_search.index(baseline_keyword)
            pytrend.build_payload(kw_list=keywords_to_search)
            data = pytrend.interest_over_time().to_numpy()
            mini_matrix = np.zeros((nread, nweeks))
            for i_kw in range(nread):
                for j, i_week in enumerate(range(-nweeks - 1, -1)):
                    mini_matrix[i_kw, j] = data[i_week][i_kw]
            norm_factor = np.sum(mini_matrix[i_baseline])
            if norm_factor == 0:
                norm_factor = 1
        else:
            pytrend.build_payload(kw_list=keywords_to_search + [baseline_keyword])
            data = pytrend.interest_over_time().to_numpy()
            mini_matrix = np.zeros((nread, nweeks))
            for i_kw in range(nread):
                for j, i_week in enumerate(range(-nweeks - 1, -1)):
                    mini_matrix[i_kw, j] = data[i_week][i_kw]
            baseline_trend = np.zeros(nweeks)
            for j, i_week in enumerate(range(-nweeks - 1, -1)):
                baseline_trend[j] = data[i_week][nread]
            norm_factor = np.sum(baseline_trend)
            if norm_factor == 0:
                norm_factor = 1
        print(mini_matrix)
        trend_matrix[i:i + nread] = mini_matrix / norm_factor
        for k in range(0, nread):
            print("{:s} ({:d}/{:d}) :".format(keywords[i + k], i + k, nkw), end='')
            print(trend_matrix[i + k][-5:], end='')
            print("    sum={:.2f}".format(sum(trend_matrix[i + k])), flush=True)
    return trend_matrix
def dataset_of_words_to_ids(dataset):
    unique_keywords = np.unique([kw for document in dataset for kw in document])
    kw_to_id = {kw: i for i, kw in enumerate(unique_keywords)}
    dataset = [[kw_to_id[kw] for kw in document] for document in dataset]
    return unique_keywords
def extract_words_from_original_dataset(dataset_original, dataset_date, dataset_length):
    """Receives an original dataset: a list of strings, where each string is a document.
    Extracts the words from the document using a regular expression, converting to lower case, and keeping only unique alpha-only words"""
    dataset_keywords = []
    dataset_newdate = []
    dataset_newlength = []
    for index, document in enumerate(dataset_original):
        unique_words_this_doc = list(set(re.findall(r'\w+', document)))
        unique_words_this_doc = list(set([word.lower() for word in unique_words_this_doc if word.isalpha()]))
        if len(unique_words_this_doc) > 0:
            dataset_keywords.append(unique_words_this_doc)
            dataset_newdate.append(dataset_date[index])
            dataset_newlength.append(dataset_length[index])
    return dataset_keywords, dataset_newdate, dataset_newlength
def process_email_enron(message):
    global email_date
    payload = []
    for part in message.walk():
        if part.get_content_type() == 'text/plain':
            payload.append(part.get_payload())
        email_date = email.utils.parsedate_to_datetime(part.get("Date"))
    payload = "".join(payload)
    return payload, email_date
def process_email(message):
    payload = []
    for part in message.walk():
        if part.get_content_type() == 'text/plain':
            payload.append(part.get_payload())
    payload = "".join(payload)
    return payload
def save_enron_data_bymonth(dataset_name, dataset, dataset_date, dataset_length):
    save_path = '../datasets_mon_alter/' + dataset_name + '/'
    year_list = [1999, 2000, 2001, 2002]
    month_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for i in year_list:
        for j in month_list:
            locals()[str(i) + "_" + str(j) + "dataset"] = []
            locals()[str(i) + "_" + str(j) + "date"] = []
            locals()[str(i) + "_" + str(j) + "length"] = []
    for index, data in enumerate(dataset_date):
        year = data.year
        month = data.month
        locals()[str(year) + "_" + str(month) + "dataset"].append(dataset[index])
        locals()[str(year) + "_" + str(month) + "date"].append(dataset_date[index])
        locals()[str(year) + "_" + str(month) + "length"].append(dataset_length[index])
    res_dataset = []
    res_dataset_date = []
    res_dataset_length = []
    for i in year_list:
        for j in month_list:
            res_dataset.extend(locals()[str(i) + "_" + str(j) + "dataset"])
            res_dataset_date.extend(locals()[str(i) + "_" + str(j) + "dataset"])
            res_dataset_length.extend(locals()[str(i) + "_" + str(j) + "length"])
            results = (res_dataset, res_dataset_date, res_dataset_length)
            with open(save_path + str(i) + "_" + str(j) + ".pkl", "wb") as f:
                pickle.dump(results, f)
def save_lucene_data_bymonth(dataset_name, dataset, dataset_date, dataset_length):
    save_path = '../datasets_mon_alter/' + dataset_name + '/'
    year_list = [2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012]
    month_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for i in year_list:
        for j in month_list:
            locals()[str(i) + "_" + str(j) + "dataset"] = []
            locals()[str(i) + "_" + str(j) + "date"] = []
            locals()[str(i) + "_" + str(j) + "length"] = []
    for index, data in enumerate(dataset_date):
        year = data.year
        month = data.month
        locals()[str(year) + "_" + str(month) + "dataset"].append(dataset[index])
        locals()[str(year) + "_" + str(month) + "date"].append(dataset_date[index])
        locals()[str(year) + "_" + str(month) + "length"].append(dataset_length[index])
    res_dataset = []
    res_dataset_date = []
    res_dataset_length = []
    for i in year_list:
        for j in month_list:
            res_dataset.extend(locals()[str(i) + "_" + str(j) + "dataset"])
            res_dataset_date.extend(locals()[str(i) + "_" + str(j) + "dataset"])
            res_dataset_length.extend(locals()[str(i) + "_" + str(j) + "length"])
            results = (res_dataset, res_dataset_date, res_dataset_length)
            with open(save_path + str(i) + "_" + str(j) + ".pkl", "wb") as f:
                pickle.dump(results, f)
def preprocess_raw_dataset(dataset_name, force_recompute=True):
    path_to_pre_dataset = os.path.join(PRE_DATASET_FOLDER, dataset_name + '.pkl')
    if os.path.exists(path_to_pre_dataset) and not force_recompute:
        print("Pre-processed dataset {:s} already existed, not processing again.".format(dataset_name))
        return
    time0 = time.time()
    print("Going to process {:s}".format(dataset_name))
    if dataset_name == 'enron-full':
        path_to_raw_dataset = os.path.join(RAW_DATASET_FOLDER, 'enron_mail_20150507.tar.gz')
        dataset_original = []
        dataset_date = []
        dataset_length = []
        count = 0
        with tarfile.open(path_to_raw_dataset, mode='r') as tar:
            for member in tar.getmembers():
                if '_sent_mail' in member.path:
                    if member.isfile():
                        file_object = tar.extractfile(member)
                        email_data_binary = file_object.read()
                        email_data_string = email_data_binary.decode("utf-8")
                        message = email.message_from_string(email_data_string)
                        email_context, email_date = process_email_enron(message)
                        dataset_original.append(email_context) 
                        dataset_date.append(email_date)
                        dataset_length.append(len(email_data_binary))
                        count += 1
                        if count % 1000 == 0:
                            print("{:d} documents processed".format(count))
        print("Done reading, we have {:d} documents".format(len(dataset_original)))
        dataset_keywords, dataset_newdate, dataset_newlength = extract_words_from_original_dataset(
            dataset_original, dataset_date, dataset_length)
        print("After removing non-alpha keywords, we have {:d} documents in {:s}".format(len(dataset_keywords),
                                                                                         dataset_name))
        unique_keywords = dataset_of_words_to_ids(dataset_keywords)
        results = (dataset_keywords, unique_keywords, dataset_newdate, dataset_newlength)
        print(len(dataset_keywords), len(unique_keywords), len(dataset_newdate))
    elif dataset_name == 'lucene':
        path_to_raw_dataset = os.path.join(RAW_DATASET_FOLDER, 'apache_ml/')
        dataset_original = []
        dataset_date = []
        dataset_length = []
        count = 0
        for filename in os.listdir(path_to_raw_dataset):
            if filename.endswith('mbox'):
                email_date = datetime.datetime(year=int(filename[0:4]), month=int(filename[4:6]), day=1)
                mbox = mailbox.mbox(os.path.join(path_to_raw_dataset, filename))
                for message in mbox:
                    dataset_original.append(process_email(message).split("To unsubscribe")[0])
                    dataset_length.append(len(process_email(message).split("To unsubscribe")[0].encode()))
                    dataset_date.append(email_date)
                    count += 1
                    if count % 1000 == 0:
                        print("{:d} documents processed".format(count))
        print("Done reading, we have {:d} documents".format(len(dataset_original)))
        dataset_keywords, dataset_newdate, dataset_newlength = extract_words_from_original_dataset(dataset_original,
                                                                                                   dataset_date,
                                                                                                   dataset_length)
        print("After removing non-alpha keywords, we have {:d} documents in {:s}".format(len(dataset_keywords),
                                                                                         dataset_name))
        unique_keywords = dataset_of_words_to_ids(dataset_keywords)
        results = (dataset_keywords, unique_keywords, dataset_newdate, dataset_newlength)
        print(len(dataset_keywords), len(unique_keywords), len(dataset_newdate))
    else:
        raise ValueError("dataset_name {:s} is not ready".format(dataset_name))
    with open(path_to_pre_dataset, 'wb') as f:
        pickle.dump(results, f)
    print(
        "Done pre-processing, saved {:s}, elapsed time ({:.0f} secs)".format(path_to_pre_dataset, time.time() - time0))
def process_pre_dataset(dataset_name, nkw=3000, force_recompute=False):
    global chosen_stemid, old_to_new_stemid
    path_to_pro_dataset = os.path.join(PRO_DATASET_FOLDER, dataset_name + '.pkl')
    if os.path.exists(path_to_pro_dataset) and not force_recompute:
        print("Processed dataset {:s} already existed, not processing again.".format(dataset_name))
        return
    time0 = time.time()
    print("Going to process {:s}".format(dataset_name))
    with open(os.path.join(PRE_DATASET_FOLDER, dataset_name + '.pkl'), 'rb') as f:
        dataset, vocabulary, dataset_date, dataset_length = pickle.load(f)
    english_stopwords = stopwords.words('english')
    ps = PorterStemmer()
    stem_vocabulary = [ps.stem(word) if (word.lower() not in english_stopwords and 2 < len(word) < 20
                                         and word.isalpha()) else None for word in vocabulary]
    stem_vocabulary_unique = sorted(list(set(stem_vocabulary) - {None}))
    kwid_to_stemid = {word: stem
                      for i, (word, stem) in enumerate(zip(vocabulary, stem_vocabulary)) if stem is not None}
    new_dataset = [sorted(list(set([kwid_to_stemid[kwid]
                                    for kwid in document if kwid in kwid_to_stemid]))) for document in dataset]
    print("Done stemming and trimming dataset, current size {:d} ({:.0f} secs)".format(
        len([id for document in new_dataset for id in document]), time.time() - time0))
    chosen_stems = []
    if dataset_name == "enron-full":
        chosen_stems = select_top_keyword(dataset_name, 2001, 6, nkw)
    if dataset_name == "lucene":
        chosen_stems = select_top_keyword(dataset_name, 2007, 12, nkw)
    stems_to_words = {stem: [] for stem in chosen_stems}
    for word, stem in zip(vocabulary, stem_vocabulary):
        if stem in stems_to_words:
            stems_to_words[stem].append(word)     
            print(stems_to_words)
    save_path = '../datasets_mon_alter/' + str(dataset_name) + '_stems_to_words.pkl'
    if os.path.exists(save_path) and not force_recompute:
        print("Processed dataset {:s} already existed, not processing again.".format(dataset_name))
        return
    with open(save_path, 'wb') as f:
        pickle.dump(stems_to_words, f)
def load_data_bymonth(dataset_name, year, month):
    save_path = '../datasets_mon_alter/' + dataset_name + '/' + str(year) + "_" + str(month) + ".pkl"
    with open(save_path, 'rb') as f:
        dataset, dataset_date, res_dataset_length = pickle.load(f)
    return dataset, dataset_date, res_dataset_length
def select_top_keyword(dataset_name, year, month, nkw):
    new_dataset, dataset_date, dataset_length = load_data_bymonth(dataset_name, year, month)
    stemid_counter = Counter([stemid for stemid_list in new_dataset for stemid in stemid_list])
    sorted_stemid = sorted(stemid_counter.keys(), key=lambda x: stemid_counter[x], reverse=True)
    chosen_stemid = sorted_stemid[:nkw]
    return chosen_stemid
def get_frequencies_from_google_trends(dataset_list):

    def load_keyword_trends():
        path_to_keyword_trends = 'trends.pkl'
        if os.path.exists(path_to_keyword_trends):
            with open(path_to_keyword_trends, "rb") as f:
                keyword_trends = pickle.load(f)
            print("Loaded keyword_trends, we have {:d} keywords".format(len(keyword_trends), flush=True))
        else:
            keyword_trends = {}
            print("Creating keyword_trends...", flush=True)
        return keyword_trends
    def save_keyword_trends(keyword_trends):
        path_to_keyword_trends = 'trends.pkl'
        with open(path_to_keyword_trends, "wb") as f:
            pickle.dump(keyword_trends, f)
    keyword_trends = load_keyword_trends()
    new_keywords = []
    for dataset_name in dataset_list:
        path_to_pro_dataset = os.path.join(PRO_DATASET_FOLDER, dataset_name + '_stems_to_words.pkl')
        with open(path_to_pro_dataset, "rb") as f:
            stems_to_words = pickle.load(f)
        keywords = [kw for kw_list in stems_to_words.values() for kw in kw_list]
        for kw in keywords:
            if kw not in keyword_trends and kw not in new_keywords:
                new_keywords.append(kw)
    print("Done scanning the datasets, we have {:d} new keywords".format(len(new_keywords)), flush=True)
    if len(new_keywords) > 0:
        t = gtab.GTAB()   
        t.set_options(pytrends_config={'timeframe': '2015-01-01 2020-12-31'}, gtab_config={'sleep': 2})
        for i, kw in enumerate(new_keywords):
            try:
                query = t.new_query(str(kw))
            except BaseException as e:
                print("There was an exception, so we are saving...")
                save_keyword_trends(keyword_trends)
                print("Saved!")
                raise
            keyword_trends[kw] = query
            print("Added '{:s}', {:d} left".format(kw, len(new_keywords) - i - 1), flush=True)
        save_keyword_trends(keyword_trends)
        print("Done adding all keywords!")
def add_frequency_trends_information_to_dataset(dataset_name):
    path_to_pro_dataset = '../datasets_mon_alter/' + str(dataset_name) + '_stems_to_words.pkl'
    with open(path_to_pro_dataset, "rb") as f:
        stems_to_words = pickle.load(f)
    print("Loaded", dataset_name)
    path_to_keyword_trends = '../datasets_mon_alter/trends.pkl'
    with open(path_to_keyword_trends, "rb") as f:
        keyword_trends = pickle.load(f)
    print("Loaded keyword_trends, we have {:d} keywords".format(len(keyword_trends), flush=True))
    trends_matrix = {}
    for key, value in stems_to_words.items():
        trends_matrix[key] = np.zeros(72)
        for word in value:
            if word in keyword_trends.keys() and isinstance(keyword_trends[word], pd.DataFrame):
                trends_matrix[key] += keyword_trends[word]['max_ratio'].values
    results = (stems_to_words, trends_matrix)
    path_to_pro_dataset = "../datasets_mon_alter/" + str(dataset_name) +"stems_trends.pkl"
    with open(path_to_pro_dataset, "wb") as f:
        pickle.dump(results, f)
    print("Written dataset to {}!".format(path_to_pro_dataset))
def save_stem_and_frequency_trends_information(dataset_name):
    path_to_pro_dataset = '../datasets_mon_alter/' + str(dataset_name) + '_stems_to_words.pkl'
    with open(path_to_pro_dataset, "rb") as f:
        stems_to_words = pickle.load(f)
    print("Loaded", dataset_name)
    path_to_keyword_trends = '../datasets_mon_alter/trends.pkl'
    with open(path_to_keyword_trends, "rb") as f:
        keyword_trends = pickle.load(f)
    print("Loaded keyword_trends, we have {:d} keywords".format(len(keyword_trends), flush=True))
    stems = sorted(stems_to_words.keys())
    print(len(stems))
    trends_matrix = np.zeros((len(stems), 72))
    for i_stem, stem in enumerate(stems):
        for word in stems_to_words[stem]:
            if word in keyword_trends.keys() and isinstance(keyword_trends[word], pd.DataFrame):
                trends_matrix[i_stem] += keyword_trends[word]['max_ratio'].values
    results = (stems, trends_matrix)
    path_to_pro_dataset = "../datasets_mon_alter/" + str(dataset_name) +"_sorted_stems_trends.pkl"
    with open(path_to_pro_dataset, "wb") as f:
        pickle.dump(results, f)
    print("Written dataset to {}!".format(path_to_pro_dataset))
def select_top_nkw_and_save(dataset_name, nkw):
    chosen_stems = []
    if dataset_name == "enron-full":
        chosen_stems = select_top_keyword(dataset_name, 2001, 6, nkw)
    if dataset_name == "lucene":
        chosen_stems = select_top_keyword(dataset_name, 2007, 12, nkw)
    path_to_pro_dataset = '../datasets_mon_alter/' + str(dataset_name) + '_stems_to_words.pkl'
    with open(path_to_pro_dataset, "rb") as f:
        stems_to_words = pickle.load(f)
    print("Loaded", dataset_name)
    path_to_keyword_trends = '../datasets_mon_alter/trends.pkl'
    with open(path_to_keyword_trends, "rb") as f:
        keyword_trends = pickle.load(f)
    stems = sorted(chosen_stems)
    print(len(stems))
    trends_matrix = np.zeros((len(stems), 72))
    for i_stem, stem in enumerate(stems):
        for word in stems_to_words[stem]:
            if word in keyword_trends.keys() and isinstance(keyword_trends[word], pd.DataFrame):
                trends_matrix[i_stem] += keyword_trends[word]['max_ratio'].values
    results = (stems, trends_matrix)
    path_to_pro_dataset = "../datasets_mon_alter/trends/" + str(dataset_name) + "_trends_" + str(nkw) + ".pkl"
    with open(path_to_pro_dataset, "wb") as f:
        pickle.dump(results, f)
    print("Written dataset to {}!".format(path_to_pro_dataset))
if __name__ == "__main__":
    dataset_list = ['enron-full', 'lucene']
    nkw_list = [100]
    for dataset_name in dataset_list:
        for nkw in nkw_list:
            select_top_nkw_and_save(dataset_name, nkw)
