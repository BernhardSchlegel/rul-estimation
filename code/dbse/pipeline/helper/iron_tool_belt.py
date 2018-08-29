'''
Created on 09.08.2017

@author: Q416435
'''
from itertools import groupby
import collections
import time
import random
from multiprocessing import Process, Queue, current_process, freeze_support
from scipy.stats import itemfreq
from scipy._lib.six import xrange
import multiprocessing
from openpyxl.styles.builtins import output
from copy import deepcopy
import os
import pandas as pd
global __DONE_NUMBER, __NEW_COL_THRSHLDs, __LOCK_1, __KNOWN_DICT_CNT, __KNOWN_DICT

def invertDictionary(orig_dict):
    return {v : k for k, v in orig_dict.items()}
    

def create_state_vector(df):
    col_names = df.columns.tolist()
    
    delete_rows = ['timestamp', 'validity_invalid_items', 'session_id', 'data_type', 'outlier', 'signal_name_short', 'Signalname']
    for el in delete_rows:
        try:
            col_names.remove(el)
        except:
            pass
    col_names.sort()
    
    df["state_vector"] = ""
    first = True
    for col_name in col_names:
        if first:
            first = False
        else:
            df["state_vector"] += ";"
        df["state_vector"] += col_name
        df["state_vector"] += "="
        df["state_vector"] += df[col_name].astype('str')
    return df

def _new_column(df):
    global __DONE_NUMBER, __NEW_COL_THRSHLD
    if df["diff"] > __NEW_COL_THRSHLD:  
        __DONE_NUMBER +=1 
    df["time_group"] = str(__DONE_NUMBER)
    return df
    
def segmentation_by_time(df, threshold_ns = 3000000000, start_col = "timestamp", end_col = "end_timestamp"):
    '''
    Where difference between end_col and start_col is bigger than threshold a seperation
    of time windows is performed and each window is assigned a number    
    '''
    global __DONE_NUMBER, __NEW_COL_THRSHLD
    __NEW_COL_THRSHLD = threshold_ns # in nanoseconds
    __DONE_NUMBER = 0
    
    df[end_col] = df[start_col].shift(-1)
    df["diff"] = df[end_col] - df[start_col]
    
    df = df.apply(_new_column, axis = 1)
    return df
    
def fuse(a):
    ''' Merge elements with same column value e.g. 
        a = data.groupby("timestamp")
        data = a.apply(fuse)'''
    
    return a.iloc[-1:]

def join_by_columns(df_1, df_2, join_col_1, join_col_2):
    return pd.merge(df_1, df_2,  how='left', left_on=[join_col_1, join_col_2], right_on = [join_col_1, join_col_2])
    

def state_vec(line):
    '''
    Aus a=1;b=3;c=5 mache mit auswahl Signalname z.B. b -> b=3
    wobei b in Spalte Signalname und a=1,... in Spalte state_vector steht
    '''
    line["value"] = line["Signalname"] + "=" + str(line[line["Signalname"]])#line["Signalname"] + line["state_vector"].split(line["Signalname"])[1].split(";")[0]
    return line

def aggregate_sequence(df):
    '''
    Aggregates sequence by index
    in value steht a=1 in Zeile 1, b=1 in Zeile 2... und time_group z.B. 20 sagt nach was zusammengefasst wird
    daraus mache ueber groupby und apply pro Gruppe -> [20, [(ts1, a=1), (ts2, b=1), ...]
    '''
    
    resulting = []
    for i in range(len(df)):
        resulting.append((df.iloc[i]["timestamp"],df.iloc[i]["value"]))
    resulting.sort(key=lambda x: x[0])    
    df["sequence_wo_timefuse"] =  str(resulting)
    df = df.head(1)
    df["sequence_index"] = df["time_group"]
    return df
    
def select_from_list(df, column, index_list, indices_support, value_support, idx):
    '''
    extracts all elements in df and column column
    selects a list of indices e.g. [293, 295, 395, 496, 305, 377]
    '''
    if isinstance(index_list, list):
        valid_indices = index_list
    else:
        valid_indices = eval(index_list)
    
    df1 = df[df[column].isin(valid_indices)]
    #df2 = df1.apply(state_vec, axis=1)
    df_out = df1.groupby(column).apply(aggregate_sequence)
    
    df_out = df_out[['sequence_index', 'sequence_wo_timefuse']]
    df_out["set_index"] = idx
    df_out["length_indices"] = indices_support
    df_out["length_content"] = value_support
    
    return df_out

def select_all_from_list(df_indices, df_original, ignore_idx = []):
    df_indices["aggregated"] = "FILL"
    df_original["time_group"] = pd.to_numeric(df_original["time_group"])
    
    # Parallel Version
    first = True
    res_df = None
    
    inputs = []
    for idx in range(len(df_indices)):
        if idx in ignore_idx: continue
        inputs.append([df_original, "time_group", df_indices.iloc[idx]["indices"], df_indices.iloc[idx]["length_indices"], df_indices.iloc[idx]["length_content"], idx])
    res_list = parallelize_stuff(inputs, select_from_list)
    
    for res in res_list:
        if first:
            res_df = res; first = False
        else:
            res_df = res_df.append(res)

    '''

    # Standard version    
    start = time.clock()
    first = True
    res_df = None
    for idx in range(len(df_indices)):
        if idx in ignore_idx: continue
        print("Processing Index: " + str(idx+1) + " | "+ str(len(df_indices)))        
        calc = select_from_list(df_original, "time_group", df_indices.iloc[idx]["indices"], df_indices.iloc[idx]["length_indices"], df_indices.iloc[idx]["length_content"], idx)
        calc["set_index"] = idx
        calc["length_indices"] = df_indices.iloc[idx]["length_indices"]
        calc["length_content"] = df_indices.iloc[idx]["length_content"]
        
        if first:
            res_df = calc; first = False
        else:
            res_df = res_df.append(calc)
        df_indices.ix[idx, "aggregated"]= str(select_from_list(df_original, "time_group", df_indices.iloc[idx]["indices"], df_indices.iloc[idx]["length_indices"], df_indices.iloc[idx]["length_content"], idx))
    
    print ("Sequential Version: " + str(time.clock() - start))
    #print ("Result DF: " + str(len(res_df)))
    ''' 
    return res_df.reset_index()[["sequence_index", "sequence_wo_timefuse", "set_index", "length_indices", "length_content"]]

def parallelize_stuff(list_input, method, simultaneous_processes = 10):
    '''
    The smarter way to loop - 
    list_input is a list of list of input arguments [[job1_arg1, job1_arg2, ...], [job2_arg1, job2_arg2, ...], [job3_arg1, job3_arg2, ...]] 
    and method is the method to compute from the input arguments
    the result is a list of output arguments from the given method [jobK_res, jobL_res, ...]

    Here: Lose order of in to output
    '''
    # Initialize
    process_number = len(list_input)
    split_number = simultaneous_processes # split by groups of 10
    task_queue = Queue()
    done_queue = Queue()
    
    cur_runs = 0

    # Submit tasks jedes Put hat: (methode, argumente_tuple) z.B. (multiply, (i, 7))
    for list_in in list_input:
        task_queue.put((method,  list_in))
    
    # Start worker processes
    jobs =[]        
    # Split tasks by defined number
    for i in range(process_number):
        print("Starting task "+str(i+1))
        p = Process(target=_worker, args=(task_queue, done_queue))
        jobs.append(p)

    # Get and print results
    output_list = []
    j = 0
    for i in range(len(jobs)):
        if cur_runs < split_number:
            print("Start job: "+str(i+1))
            jobs[i].start()
            cur_runs +=1
            if len(jobs) != split_number and (len(jobs) - i - 1) < split_number:# remaining_jobs = len(jobs) - i - 1
                j += 1
                print("Received results "+str(j) + " | " + str(len(list_input)))
                output_list.append(done_queue.get())
                #print("Got: "+ str(done_queue.get().head(1)))
            if len(jobs) == split_number and (i +1) == split_number:# remaining_jobs = len(jobs) - i - 1
                j += 1
                print("Received results "+str(j) + " | " + str(len(list_input)))
                output_list.append(done_queue.get())
                #print("Got: "+ str(done_queue.get().head(1)))     
            
        else:
            j += 1
            print("Received results "+str(j) + " | " + str(len(list_input)))
            output_list.append(done_queue.get())
    #print("FINAL: ")
    #print(output_list)
    while j != len(list_input):        
        res = done_queue.get()                
        j += 1
        print("Received results "+str(j) + " | " + str(len(list_input)))
        output_list.append(res)
        

    # End all 
    for i in range(process_number):
        task_queue.put('STOP')
    
    for job in jobs:
        try:
            job.shutdown()
        except: 
            pass
    return output_list

def all_aggregated_info_to_sequence(df):   
    '''
    from a database with sequence sets extract lists of sequences that
    can be processed
    '''
    
    # parallelize
    max_index = df["set_index"].max() 
    input_args = []
    for i in range(max_index):
        input_args.append([df, i])
    
    # jobs
    output_list = parallelize_stuff(input_args, aggregated_info_to_sequence)
    
    # resulting df
    first = True
    for res in output_list:
        if first:
            res_df = res; first = False
        else:
            res_df = res_df.append(res)
    return res_df

def sort_and_fuse_ts(line, col, res_col, remove_ts):
    '''
    for input list of type [(ts1, val), (ts2, val),...] which is in col of dataframe line
    fuse the list (i.e. items with same timestamp are grouped and sort by ts
    
    if remove_ts: False
        results in [{ts1:[val1, val2, ...]: (ts, ...}
    if remove_ts: True
        results in item set [[val1, val2, ...], [valx, valy, ...], ...]
    
    which is stored in column res_col
    '''
    lst = eval(line[col])
    known_ts = {}
    for el in lst:
        if el[0] in known_ts:
            known_ts[el[0]].append(el[1])
        else:
            known_ts[el[0]] = [el[1]]
    if not remove_ts:
        line[res_col] = collections.OrderedDict(sorted(known_ts.items()))
    else:
        line[res_col] = [a[1] for a in sorted(known_ts.items())]
    return line

def aggregated_info_to_sequence(data_frame, set_idx):
    '''        
    if they happen in identical timeslots they are grouped!
    Result is a sequence of itemsets [[a,n]...  
    
    '''
    cur_df = data_frame[data_frame["set_index"]==set_idx]
    
    # group and sort sequences by timestamp
    cur_df = cur_df.apply(sort_and_fuse_ts, args = ("sequence_wo_timefuse", "sequence", True), axis=1)
    
    return cur_df
    
def symbolize_values(df_set, to_number):
    '''
    In order to be processed more easy, strings will be named with short consecutive letters
    i.e. a unique shortname will be assigned to each unique string
    '''
    global _KNOWN_DICT, __KNOWN_DICT_CNT
    df_set["translator"] = str(_KNOWN_DICT)
    
    k = -1
    for sequence in df_set["sequence"].tolist():        
        k+=1
        for itemset in sequence: 
            for i in range(len(itemset)):
                if itemset[i] not in _KNOWN_DICT:
                    if to_number: _KNOWN_DICT[itemset[i]] = str(__KNOWN_DICT_CNT); __KNOWN_DICT_CNT += 1 
                    else: _KNOWN_DICT[itemset[i]] = "gen"+str(__KNOWN_DICT_CNT); __KNOWN_DICT_CNT += 1            
                itemset[i] = _KNOWN_DICT[itemset[i]]
    return df_set
    
def all_symbolize_values(df):
    global _KNOWN_DICT, __KNOWN_DICT_CNT
    _KNOWN_DICT = {}
    __KNOWN_DICT_CNT = 0
    
    start = time.clock()   
    df1 = deepcopy(df).groupby("set_index").apply(lambda x: symbolize_values(x, True))
    print ("Time for symbolization: " + str(time.clock() - start))
    df1["translator"] = str(_KNOWN_DICT)
    
    return df1
    
def desymbolize_sequential_pattern(level_dict, mapping_dict):
    mapping_dict = invertDictionary(mapping_dict)
    
    for el in level_dict:
        for i in range(len(level_dict[el])):
            for j in range(len(level_dict[el][i][0])):
                level_dict[el][i][0][j] = [mapping_dict[k1] for k1 in level_dict[el][i][0][j]]
            print("\nFound patterns: \n"+"\n".join(["|".join(o) for o in level_dict[el][i][0]]))
    return level_dict
    
def find_n_grams(n, single_sequence):
    ''' 
    Find n grams in Sequence, n=2 means pattern of length two, n=3 of length 3, ...
    n = pattern length
    e.g. in 
    3 6 1 2 7 3 8 9 7 2 2 0 2 7 2 8 4 8 9 7 2 4 1 0 3 2 7 2 0 3 8 9 7 2 0
    find
    3 6 1 [2 7] 3 [8 9 7 2] 2 0 [2 7] 2 8 4 [8 9 7 2] 4 1 0 3 [2 7] 2 0 3 [8 9 7 2] 0
    '''    
    
    grams = [single_sequence[i:i+n] for i in xrange(len(single_sequence)-n)]
    
    return itemfreq(grams)

def all_run_java_spmf(jar_path, sequence_df, algorithm, *args):
            
    #start = time.clock()   
    #df = sequence_df.groupby("set_index").apply(lambda x: run_java_spmf_seq(jar_path, x, deepcopy(_KNOWN_DICT), algorithm, *args))
    #print ("Time for Java: " + str(time.clock() - start))
    #return df
    
    # parallel - slower?
    start = time.clock() 
    # Job Input
    input_lst = [] 
    for set_index in list(sequence_df["set_index"].drop_duplicates()):
        input_lst.append([set_index, jar_path, sequence_df, deepcopy(_KNOWN_DICT), algorithm, *args])
    
    # Job execution
    output_lst = parallelize_stuff(input_lst, run_java_spmf_parallel, simultaneous_processes=60 )
    
    # job result
    res_df = _append_df_list(output_lst)        
        
    print ("Time for Java: " + str(time.clock() - start))
    return res_df

def _append_df_list(output_lst):
    first = True
    for res in output_lst:
        if first:
            res_df = res; first = False
        else:
            res_df = res_df.append(res)
    return res_df


def print_result_pattern_spmf(df, column, min_level = 0):
    i = 0
    for pattern in list(df[column]):
        i+=1
        cur_dict = eval(pattern)
        print("\n\n-------- Sequence " + str(i))
        for k in cur_dict.keys():
            if k < min_level: continue
            print("\nLevel "+str(k))
            print("\n\n".join(["Sup: " + str(a[1]) + " - Pattern" + str(a[0])  for a in cur_dict[k]]))
            #print("\n\nPattern: ".join([str(",".join(a[0])) for a in cur_dict[k]]))      

def run_java_spmf_parallel(set_index, jarPath, sequence_df, reverse_dict, algorithm, *args):
    '''
    runs the SPMF EntryPoint and reads out results    
    '''
    # ASSUME SEQUENCE SET ALREADY IN RIGHT FORMAT ONLY NUMBERS!
    sequence_sets = sequence_df[sequence_df["set_index"] == set_index]["sequence"].tolist()
    #print("Current set_index " + str(sequence_df[sequence_df["set_index"] == set_index]["set_index"].unique()))

    # 0. temporary path
    tmp_path =r"C:\repository\inputSequences"+str(set_index) + ".txt"
    
    java_jre8_path = r'C:\Program Files (x86)\JavaSoft\jre\1.8.0_121\bin\java'
    
    # 1. map to numbers
    #mapped_list, stored_map = symbolize_values(sequence_sets, True)
    stored_map = invertDictionary(reverse_dict)

    # 2. Write temporary file
    open(tmp_path, 'w').close()
    with open(tmp_path, 'a') as the_file:    
        first = True
        for sets in sequence_sets:      
            #sets = sets[:10]   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ACHTUNG 
            if first: 
                the_file.write(""+" -1 ".join([" ".join(a) for a in sets])+" -2")
                first = False
            else: 
                the_file.write("\n"+" -1 ".join([" ".join(a) for a in sets])+" -2")
    
    # 3. Run Java on this        
    from subprocess import Popen, PIPE, STDOUT
    p = Popen([java_jre8_path, '-jar', jarPath, algorithm, tmp_path]+ list(args), stdout=PIPE, stderr=STDOUT)

    # 4. read result as dictionary
    # dict[level] = [pattern_list, support]
    # lese auch eval raus: dict["pattern_number"] und dict[total_time] und dict["memory"]
    eval_dict = {}
    result_dict = {}
    result_dict_un = {}

    for line in p.stdout:
        #print(line)
        if line[:11] == b" Total time": eval_dict["total_time"] = str(line[12:]).replace("\\n", "").replace("b'", "").replace("'", "")
        if line[:28] == b" Frequent sequences count : ": eval_dict["patterns_found"] = str(line[28:]).replace("\\n", "").replace("b'", "").replace("'", "")
        if line[:17] == b" Max memory (mb):": eval_dict["max_memory_mb"] = str(line[17:]).replace("\\n", "").replace("b'", "").replace("'", "")
        
        if line[:5]==b"Level": 
            input = [el.split("#") for el in str(line[8:]).replace("\\n", "").replace("b'", "").replace("'", "").split(";")][:-1]
            all_patterns = []
            all_patterns_un = []
            for lev in input: 
                support = int(lev[1].replace("SUP: ",""))
                patterns_raw = [e.lstrip()[::-1].lstrip()[::-1].split(" ") for e in lev[0].split("-1") if e != " "]
                patterns_un = []
                patterns = []
                for p in patterns_raw:
                    patterns.append([stored_map[pi.replace(":", "")] for pi in p])
                    patterns_un.append([pi.replace(":", "")  for pi in p])
                all_patterns.append([patterns, float(support)/float(len(sequence_sets))])
                all_patterns_un.append([patterns_un, float(support)/float(len(sequence_sets))])
            result_dict[int(str(line[:7])[-2])] = all_patterns
            result_dict_un[int(str(line[:7])[-2])] = all_patterns_un
    os.remove(tmp_path)
    
    #sequence_df["all_sequences"] = str(sequence_df["sequence"].tolist())
    sequence_df["set_index"] = set_index
    sequence_df["sequence"] = str(sequence_sets)
    
    sequence_df["pattern_found"] = str(result_dict)
    sequence_df["pattern_found_un"] = str(result_dict_un)
    
    try:
        sequence_df["eval_total_time"] = eval_dict["total_time"]
    except:
        sequence_df["eval_total_time"] = "0"
    try:
        sequence_df["patterns_found"] = eval_dict["patterns_found"]
    except:
        sequence_df["patterns_found"] = 0
    try:
        sequence_df["max_memory_mb"] = eval_dict["max_memory_mb"]
    except: 
        sequence_df["max_memory_mb"] = 0
    return sequence_df.head(1)


# ------------- parallelization stuff ------------- 
  
    
def _worker(input, output): # Function run by worker processes
    for func, args in iter(input.get, 'STOP'):
        result = _calculate(func, args)
        output.put(result)
        
def _calculate(func, args): # Function used to calculate result
    result = func(*args)
    return result
       
       
       
       
        