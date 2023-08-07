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
from process_dict import get_size_frompkl, delete_get_size_frompkl
np.set_printoptions(suppress=True)

def load_sent_mail_contents(dataset_name, year=1999, month=1):
    pro_dataset_path = "../datasets/" + str(dataset_name) + str(year) + "_" + str(month) + ".pkl"
    if not os.path.exists(pro_dataset_path):
        raise ValueError("The file {} does not exist".format(pro_dataset_path))
    with open(pro_dataset_path, "rb") as f:
        dataset, res_dataset_length = pickle.load(f)
    return dataset, res_dataset_length

def load_stem_trends(dataset_name, nkw):
    pro_dataset_path = "../datasets/trends/" + str(dataset_name) + "_trends_" + str(nkw) + ".pkl"
    if not os.path.exists(pro_dataset_path):
        raise ValueError("The file {} does not exist".format(pro_dataset_path))
    with open(pro_dataset_path, "rb") as f:
        stem, stem_trends = pickle.load(f)
    return stem, stem_trends


def run_single_experiment(exp_params):
    def _generate_train_test_data(dataset_name, n_keywords):
        chosen_keywords, trend_matrix = load_stem_trends(dataset_name, n_keywords)
        # trend_matrix = np.array([keyword_dict[kw]['trend'] for kw in chosen_keywords])   
        trend_matrix_norm = trend_matrix.copy()
        for i_col in range(trend_matrix_norm.shape[1]):  
            if sum(trend_matrix_norm[:, i_col]) == 0:
                print("The {d}th column of the trend matrix adds up to zero, making it uniform!")
                trend_matrix_norm[:, i_col] = 1 / n_keywords
            else:
                trend_matrix_norm[:, i_col] = trend_matrix_norm[:, i_col] / sum(trend_matrix_norm[:, i_col])
        return trend_matrix_norm, chosen_keywords
    trend_real_norm, chosen_keywords = _generate_train_test_data(exp_params['dataset'], exp_params['nkw'])

    # real_queries = np.floor(exp_params["query_params"] * trend_real_norm)
    # print("trend_real_norm", trend_real_norm)
    return trend_real_norm


def static_enron_data_info(dataset_name, nkw, trend_norm, n_month, query_number, delete_rate):

    chosen_keywords, _ = load_stem_trends(dataset_name, nkw)

    statistic_trend_number = query_number * trend_norm


    year = 2007
    month = 1
    candidate_class = []   
    candidate = []         
    candidate_class_count = []  

    all_keyword_size = []
    delete_keyword_size = []
    for i in range(n_month):
        if i == 12:
            month = 1
            year += 1
        temp_all_size = get_size_frompkl(dataset_name, year, month, nkw)
        temp_del_size = delete_get_size_frompkl(dataset_name, year, month, nkw, delete_rate)

        for key in chosen_keywords:
            if key not in temp_all_size.keys():
                temp_all_size[key] = []
            if key not in temp_del_size.keys():
                temp_del_size[key] = []

        all_keyword_size.append(temp_all_size)             
        delete_keyword_size.append(temp_del_size)           

    # print("all_keyword_size", all_keyword_size[0]['abil'])

    word_to_id = {}
    id_to_word = {}
    for index, keyword in enumerate(chosen_keywords):
        word_to_id[keyword] = index
        id_to_word[index] = keyword

    candidate_keyword = {}
    for kv in chosen_keywords:
        candidate_keyword[kv] = []

    # print(all_keyword_size[0])
    # sum_key_size = []
    for month_count in range(n_month):
        temp_class = {}
        temp_class_count = {}
        temp_candidate = {}
        # class_count = 0
        # print(query_number * trend_norm[:, month_count])
        # print(month_count, np.floor(query_number * trend_norm[:, month_count]).astype(int))

        if month_count == 0:
            # keyword_size = get_size_frompkl(dataset_name, year, month, nkw)
            query_keyword = np.floor(query_number * trend_norm[:, month_count]).astype(int)
            # print(query_keyword)
            query_keyword_list = []
            for i in range(nkw):
                query_keyword_list.extend([i for npt in range(query_keyword[i])])
            # print(len(set(query_keyword_list)))
            # print(query_keyword_list)

            for q_id in query_keyword_list:
                in_flag = 0
                for k1, size1 in temp_class.items():
                    # print(all_keyword_size[month_count].keys())

                    if all_keyword_size[month_count][id_to_word[q_id]] == size1[0]:
                        in_flag = 1
                        temp_class_count[k1] = temp_class_count[k1] + 1
                if in_flag == 0:
                    temp_class[q_id] = []
                    temp_class[q_id].append(all_keyword_size[month_count][id_to_word[q_id]])
                    temp_class_count[q_id] = 1

            # print(temp_cand)
            for k2, v2 in temp_class.items():
                cand_np = np.abs(temp_class_count[k2] - statistic_trend_number[:, month_count].astype(int))
                cand_index = np.where(cand_np == cand_np.min())
                temp_candidate[k2] = set(cand_index[0].tolist())
            candidate_class.append(temp_class)
            candidate.append(temp_candidate)
            candidate_class_count.append(temp_class_count)
            month += 1
            # print(query_keyword[119])
            # print(temp_class_count[119])
            # print(temp_candidate[119])
        else:

            # last_month_size = all_keyword_size[month_count - 1]
            now_month_size = delete_keyword_size[month_count]
            keyword_size = {}
            for keyword in chosen_keywords:
                keyword_size[keyword] = []

            for keyword in chosen_keywords:
                for i_m in range(month_count):
                    if keyword in delete_keyword_size[i_m].keys():
                        keyword_size[keyword].append(delete_keyword_size[i_m][keyword])
                    else:
                        keyword_size[keyword].append([])

                if keyword in now_month_size.keys():
                    keyword_size[keyword].append(now_month_size[keyword])
                else:
                    keyword_size[keyword].append([])

            query_keyword = np.floor(query_number * trend_norm[:, month_count]).astype(int)
            query_keyword_list = []
            for i in range(nkw):
                query_keyword_list.extend([i for npt in range(query_keyword[i])])
            # print(len(set(query_keyword_list)))
            for q_id in query_keyword_list:
                in_flag = 0
                for k1, size1 in temp_class.items():
                    if keyword_size[id_to_word[q_id]] == size1:
                        in_flag = 1
                        temp_class_count[k1] = temp_class_count[k1] + 1
                if in_flag == 0:
                    temp_class[q_id] = keyword_size[id_to_word[q_id]]
                    temp_class_count[q_id] = 1
                    # class_count += 1
            # print(temp_cand)

            for k2, v2 in temp_class.items():
                cand_np = np.abs(temp_class_count[k2] - statistic_trend_number[:, month_count].astype(int))
                # print("cand_np", cand_np)
                cand_index = np.where(cand_np == cand_np.min())
                temp_candidate[k2] = set(cand_index[0].tolist())
                # print("cand_index", cand_index[0].tolist())
            candidate_class.append(temp_class)
            candidate.append(temp_candidate)
            candidate_class_count.append(temp_class_count)

        # print(month_count, temp_candidate[119])
        # if month_count == 0:
        #     break
    # print("candidate_class", candidate_class[1])
    # print("candidate", len(candidate))

    counter_size = []
    for i in range(n_month):
        temp_size = {}
        for k1, v1 in candidate_class[i].items():
            temp_size[k1] = []
            for v2 in v1:
                temp_size[k1].append(Counter(v2))
        counter_size.append(temp_size)
    # print("counter_size", counter_size[1])

    unique_keyword = set()
    for i in range(n_month):
        for k1, v1 in candidate[i].items():
            # print(v1)
            if len(v1) == 1:
                unique_keyword.add(k1)

    new_candidate_size = {}
    new_candidate = {}
    new_candidate_count = {}
    # for i in range(nkw):
    #     new_candidate_size[i] = []
    #     new_candidate[i] = set(i for i in range(nkw))
    ttc = {}
    for k1, v1 in counter_size[0].items():
        new_candidate_size[k1] = v1
        new_candidate[k1] = set(candidate[0][k1])
        new_candidate_count[k1] = candidate_class_count[0][k1]
        ttc[k1] = []
        ttc[k1].append((k1, candidate_class_count[0][k1]))
    # print("new_candidate_size", new_candidate_size)
    # print("new_candidate", new_candidate[119])


    for i in range(1, n_month):             
        for k1, v1 in counter_size[i].items():
            in_flag = 0
            max_key = 0
            max_sim_value = 0
            for k2 in new_candidate_size.keys():
                v2 = new_candidate_size[k2]
                v2_len = len(v2)            
                if v2_len == 0:
                    continue
                delete_size = v1[v2_len - 1]     
                all_size = v2[v2_len - 1]    
                inter_size = len(list(all_size & delete_size))
                union_size = len(list(all_size | delete_size))

                # if k1 == 2 and k2 == 2:
                #     print("inter_size", inter_size)
                #     print("union_size", union_size)

                # if union_size == 0 :
                #     in_flag = 1
                if union_size == 0 or inter_size / union_size > 0.6:
                    in_flag = 1
                    max_sim = 0 if union_size == 0 else inter_size / union_size
                    if max_sim_value > max_sim:
                        continue
                    else:
                        max_sim_value = max_sim
                        max_key = k2
                # if union_size == 0:
                #     print(k1, k2)
            #     if k1 == k2:
            #         print(k1, k2)
            #         print(max_sim_value)
            #         print(in_flag)
            #         print("inter_size", inter_size)
            #         print("union_size", union_size)
            # if k1 != max_key:
            #     print(i, "k1", k1, max_key, max_sim_value, in_flag)

            if in_flag == 1:
                new_candidate[max_key] = candidate[i][k1].intersection(new_candidate[max_key])
                new_candidate_size[max_key] = v1
                new_candidate_count[max_key] += candidate_class_count[i][k1]
                ttc[max_key].append((k1, candidate_class_count[i][k1]))
            # if in_flag == 0:            
            else:
                if k1 not in new_candidate_size.keys():
                    new_candidate_size[k1] = v1
                    new_candidate[k1] = candidate[i][k1]
                    new_candidate_count[k1] = candidate_class_count[i][k1]
                    ttc[k1] = []
                else:
                    # print("k1", i, k1, max_sim_value, in_flag)
                    # print()
                    new_candidate_size[k1 + 3000] = v1
                    new_candidate[k1 + 3000] = candidate[i][k1]
                    new_candidate_count[k1 + 3000] = candidate_class_count[i][k1]
                    ttc[k1 + 3000] = []

    ex_value = 0
    # print(ttc)
    for k3, v3 in new_candidate.items():
        if len(v3) == 1 and list(v3)[0] == k3:
            for kv4 in ttc[k3]:
                if kv4[0] == k3:
                    ex_value += kv4[1]
                # print(k3, kv4[0], kv4[1])
    print("correct query", ex_value)

    # print(new_candidate)
    return ex_value


if __name__ == "__main__":
    # nkw = 1000
    # query_number = 1000
    repeat_cnt = 1

    nkw_list = [500, 1000, 2000, 3000]
    # nkw_list = [500]
    query_list = [5000, 10000, 15000, 20000]
    query_month = [6, 12, 18, 24]
    delete_list = [0, 0.05, 0.1, 0.15, 0.2]
    for nkw in nkw_list:
        for query_number in query_list:
            for n_month in query_month:
                list_res = []
                # for i in range(repeat_cnt):
                parameter_dict = {'dataset': 'lucene', 'nkw': nkw, 'query_number_dist': 'poiss',
                                  'query_params': 24 * query_number / n_month, 'n_month': n_month}
                trend_norm = run_single_experiment(parameter_dict)
                ans = static_enron_data_info('lucene', nkw, trend_norm, n_month, query_number, delete_rate=0.10)
                    # list_res.append(ans * 1.0 / nkw)
                print("dataset name: lucene")
                print("nkw:", nkw, "query_number:", n_month, "*", 24 * query_number / n_month, "month:", n_month,
                      "delete_rate = 0.10")
                # print(list_res)
                print("average accuracy: ", ans * 1.0 / (24 * query_number))
        #         break
        #     break
        # break
    # nkw = 500
    # query_number = 1000
    # n_month = 6
    # parameter_dict = {'dataset': 'enron-full', 'nkw': nkw, 'query_number_dist': 'poiss',
    #                                       'query_params': query_number, 'n_month': n_month}
    # trend_norm = run_single_experiment(parameter_dict)
    # ans = static_enron_data_info('enron-full', nkw, trend_norm, n_month, query_number, delete_rate=0.00)
    # print(ans)

