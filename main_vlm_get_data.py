import pickle
import os
import random
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import base64
import requests
from collections import Counter
import pdb
from utils import *
import pandas as pd
from tqdm import tqdm
from natsort import natsorted
from huggingface_hub import hf_hub_download



parser = argparse.ArgumentParser(description="Causal inference with LLMs")
parser.add_argument("--task_type", type=str, default='base')
parser.add_argument("--data_path", type=str, default='shrec_empathic_debug.csv')
parser.add_argument("--data_name", type=str, default='shrec_empathic_debug')
parser.add_argument("--seed", type=str, default=0)
parser.add_argument("--transcript_level", type=str, default='turn')



args = parser.parse_args()
random.seed(args.seed)

data_path =str(args.data_path)
transcript_level = str(args.transcript_level)
task_type = args.task_type


def get_majority(items):
    item_counts = Counter(items)
    max_frequency = max(item_counts.values())

    candidates = [item for item, count in item_counts.items() if count == max_frequency]

    if candidates:
        return random.choice(candidates)
    else:
        return None

# Video-based sample_images_from_video / _in_sec / trim_video removed (unused
# in this file) — see download_hf_frames() in utils.py, imported above.

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')



df = pd.read_csv(data_path)

task_dataset_overlap = {}
task_dataset_no_annotation = {}


for index, row in df.iterrows():
    anots = []
    annotations_A = ast.literal_eval(row['Annotations_A'])
    annotations_B = ast.literal_eval(row['Annotations_B'])
    annotations_C = ast.literal_eval(row['Annotations_C'])

    if len(annotations_A) > 0: anots.append(annotations_A)
    if len(annotations_B) > 0 : anots.append(annotations_B)
    if len(annotations_C) > 0 : anots.append(annotations_C)
    
    if len(anots) < 2:
        continue 

    intervals1 = anots[0]
    intervals2 = anots[1]

    intervals1 = [current_interval for current_interval in intervals1 if current_interval['timestamp']['start'] != None and current_interval['timestamp']['end'] != None]
    intervals2 = [current_interval for current_interval in intervals2 if current_interval['timestamp']['start'] != None and current_interval['timestamp']['end'] != None]
    

    overlapping_groups = find_overlapping_interval_groups_pair(intervals1, intervals2)

    k = row['file_name']
    task_dataset_overlap[k] = {}
    
    count = 0
    max_end_time = 0
    for sublist in overlapping_groups:

        task_dataset_overlap[k][count] = [{**sub, 'video_id': k} for sub in sublist]

        if sublist[0]['timestamp']['end'] > max_end_time:
            max_end_time = sublist[0]['timestamp']['end']
        count += 1
    
    total_time_range = {'start': 0, 'end': max_end_time}

    intervals = []
    for sample in intervals1:intervals.append(sample['timestamp'])
    for sample in intervals2:intervals.append(sample['timestamp'])

    #Find no annotations regions for experiments
    try:
        task_dataset_no_annotation[k] = find_uncovered_float_intervals(intervals, total_time_range)
    except Exception:
        pdb.set_trace()



task_dataset_agreed_overlap = {}
for k, v in task_dataset_overlap.items():
    
    count = 0
    for groups, group_dict in v.items(): 
        
        if len(group_dict) == 1:
            pass

        if len(group_dict) == 2: 
            agree = False

            #change here if you want agreement in other places
            if task_type in ['debug', 'detection', 'rationale', 'correction', 'context']:
                agree = group_dict[0]['error'] == group_dict[1]['error']
                
            if task_type in ['detection_error_only']:
                agree = group_dict[0]['error'] == group_dict[1]['error']
                if agree:
                    if group_dict[0]['error'] == True and group_dict[1]['error'] == True:
                        agreed_dict = or_operation(group_dict[0]['attribute'], group_dict[1]['attribute'])
                        agree = True
                    else:
                        agree = False
            if task_type in ['attribute', 'attribute_disagree', 'attribute_agreed_multiple', 'attribute_agreed_multiple_subj']:
                agree = group_dict[0]['error'] == group_dict[1]['error']
                if agree: 
                    if task_type == 'attribute':
                        agreed_dict = or_operation(group_dict[0]['attribute'], group_dict[1]['attribute']) #can change this to or
                
                    if task_type == 'attribute_agreed_multiple':
                        agreed_dict = and_operation(group_dict[0]['attribute'], group_dict[1]['attribute'])
                        if sum(list(agreed_dict.values())) >= 2:
                            agree = True
                        else:
                            agree = False

                    if task_type == 'attribute_agreed_multiple_subj_test':
                        agreed_dict = and_operation(group_dict[0]['attribute'], group_dict[1]['attribute'])
                        if sum(list(agreed_dict.values())) >= 2:
                            agree = True
                        if sum(list(agreed_dict.values())) == 1:
                            agree = True 
                        else: 
                            agree = False
                    
                    if task_type == 'attribute_disagree': #chose but different tiers 
                        agreed_dict_or = or_operation(group_dict[0]['attribute'], group_dict[1]['attribute'])
                        agreed_dict_and = and_operation(group_dict[0]['attribute'], group_dict[1]['attribute'])
                        intersect_dict = and_operation(agreed_dict_or,agreed_dict_and)
                        if sum(list(intersect_dict.values())) == 0:
                            agree = True 
                            agreed_dict = agreed_dict_or
                        else:
                            agree = False 
                    
            if agree:
                if task_type in ['attribute', 'attribute_disagree', 'attribute_agreed_multiple', 'detection_error_only']:
                    group_dict[0]['tier2'] = agreed_dict
                    group_dict[1]['tier2'] = agreed_dict
                
                if not task_dataset_agreed_overlap.get(k):
                    task_dataset_agreed_overlap[k] = {}
                task_dataset_agreed_overlap[k][count] = group_dict
                count += 1
                


debug_count = 0
if task_type in ['debug' , 'detection']:
    task_dataset_final = []

    for k,v in tqdm(task_dataset_agreed_overlap.items()):
        transcriptions = df[df['file_name'] == k]['transcript'].item()
        
        task_dataset_curr = []
        for inner_k, inner_v in v.items():
            timestamp_dict = task_dataset_agreed_overlap[k][inner_k][0]['timestamp'] #get first annotations timestamp
            turn_convo = get_transcript(timestamp_dict, transcriptions, session_name = k, data_path = args.data_path) 

            if transcript_level == 'turn':
                conversation = turn_convo

            task_dataset_agreed_overlap[k][inner_k][0]['transcription'] = conversation
            task_dataset_curr.append(task_dataset_agreed_overlap[k][inner_k][0])
            

            len_true_annotations = max(task_dataset_agreed_overlap[k].keys()) 
            len_none_annotations = len(task_dataset_no_annotation[k]) 
            min_annotations = min(len_true_annotations,len_none_annotations)


            random.shuffle(task_dataset_curr)
            random.shuffle(task_dataset_no_annotation[k])

        
        count = 0
        for i in range(min_annotations):
            
            if len(task_dataset_no_annotation[k]) == 0:
                continue 

            else:
                try:
                    timestamp_dict = task_dataset_no_annotation[k][i]
                except Exception:
                    pdb.set_trace()


                turn_convo = get_transcript(timestamp_dict, transcriptions, session_name = k, data_path = args.data_path) 

                if transcript_level == 'turn':
                    transcript = turn_convo


                check_transcript_in_anot = False 
                for anot in task_dataset_curr:
                    if transcript in anot['transcription']:
                        check_transcript_in_anot = True
            
                #boilerplate None for matching
                if not check_transcript_in_anot:
                    task_dataset_final.append({'video_id': k, 'timestamp': timestamp_dict, 'error': None,'transcription' : transcript, 'rationale': 'None'})
                    count += 1 
        
        task_dataset_final.extend(task_dataset_curr[:count])

    processed_dataset = task_dataset_final



debug_count = 0
if task_type in ['detection_error_only']:
    task_dataset_final = []

    for k,v in tqdm(task_dataset_agreed_overlap.items()):
        
        transcriptions = df[df['file_name'] == k]['transcript'].item()

        task_dataset_curr = []
        for inner_k, inner_v in v.items():
           
            timestamp_dict = task_dataset_agreed_overlap[k][inner_k][0]['timestamp'] #get first annotations timestamp


            turn_convo = get_transcript(timestamp_dict, transcriptions, session_name = k, data_path = args.data_path) 
            if transcript_level == 'turn':
                conversation = turn_convo

            task_dataset_agreed_overlap[k][inner_k][0]['transcription'] = conversation
            task_dataset_curr.append(task_dataset_agreed_overlap[k][inner_k][0])
            task_dataset_final.extend(task_dataset_curr)
    
    processed_dataset = task_dataset_final



#Constructing datasets for different tastks 

if 'attribute' in task_type: 
    task_dataset_final = []

    for k,v in tqdm(task_dataset_agreed_overlap.items()):

        #get transcriptions 
        transcriptions = df[df['file_name'] == k]['transcript'].item()
        
        for inner_k, inner_v in v.items():
            task_dataset_curr = []
            timestamp_dict = task_dataset_agreed_overlap[k][inner_k][0]['timestamp'] #get first annotations timestamp
            turn_convo = get_transcript(timestamp_dict, transcriptions, session_name = k, data_path = args.data_path) 

            if transcript_level == 'turn':
                conversation = turn_convo


            task_dataset_agreed_overlap[k][inner_k][0]['transcription'] = conversation
            task_dataset_agreed_overlap[k][inner_k][0]['id'] = k
            task_dataset_curr.append(task_dataset_agreed_overlap[k][inner_k][0])
            task_dataset_final.extend(task_dataset_curr)
    processed_dataset = task_dataset_final

if task_type in ['rationale', 'context', 'correction']:

    task_dataset_final = []
    for k,v in tqdm(task_dataset_overlap.items()):

        transcriptions = df[df['file_name'] == k]['transcript'].item()
        
        for inner_k, inner_v in v.items():
            sample = task_dataset_overlap[k][inner_k][0]
            
            agree = False
            if task_type == 'rationale':
                if len(sample['rationale']) > 0: 
                    agree = True
            if task_type == 'correction':
                if sample['error'] == True and len(sample['correction']) > 0: 
                    agree = True
            if task_type == 'context':
                    agree = True
            if agree:
                timestamp_dict = sample['timestamp'] #get first annotations timestamp
                turn_convo = get_transcript(timestamp_dict, transcriptions, session_name = k, data_path = args.data_path) 
                if transcript_level == 'turn':
                    conversation = turn_convo
                
                reason = sample['rationale']
                sample['transcript'] = conversation
                sample['id'] = k
                task_dataset_final.append(sample)

    def check_all_false(dictionary):
        return all(value is False for value in dictionary.values())
    print('sampling')

    task_dataset_final_text = []
    for i in tqdm(range(len(task_dataset_final))): 
        sample = task_dataset_final[i]
        for tiers in ['attribute']:
            if check_all_false(sample[tiers]):
                pass
            else:
                sub_dict = sample[tiers]
                random_task_dataset_final_subset = random.choices(task_dataset_final, k = len(task_dataset_final))
                others = filter_dicts_without_subdict_tier2(random_task_dataset_final_subset, sub_dict)
                
                other_transcript_list = [other['transcript'] for other in others if other['error'] == sample['error']]
                other_recovery_list = [other['correction'] for other in others if other['error'] == sample['error']]
                other_reason_list = [other['rationale'] for other in others if other['error'] == sample['error']]
                other_id_list = [other['id'] for other in others if other['error'] == sample['error']]
                other_timestamp_list = [other['timestamp'] for other in others if other['error'] == sample['error']]

                if task_type in ['rationale']:

                    other_reason_list ,other_id_list, other_timestamp_list = remove_redundant_strings_id_timestamp(other_reason_list,other_id_list, other_timestamp_list) 
                
                if task_type in ['context']:

                    other_transcript_list ,other_id_list, other_timestamp_list = remove_redundant_strings_id_timestamp(other_transcript_list,other_id_list, other_timestamp_list) 

                if task_type in [ 'correction'] :

                    other_recovery_list ,other_id_list, other_timestamp_list = remove_redundant_strings_id_timestamp(other_recovery_list,other_id_list, other_timestamp_list) 

                if task_type == 'rationale' and len(other_reason_list) < 5: continue
                if task_type == 'correction' and len(other_recovery_list) < 5: continue
                if task_type == 'context' and len(other_transcript_list) < 5: continue

                sample['other_reason_list'] = other_reason_list
                sample['other_recovery_list'] = other_recovery_list
                sample['other_transcript_list'] = other_transcript_list
                sample['other_id_list'] = other_id_list
                sample['other_timestamp_list'] = other_timestamp_list
                
                task_dataset_final_text.append(sample)
                break


    processed_dataset = task_dataset_final_text

if task_type in ['pre', 'post']:

    task_dataset_final = []
    for k,v in tqdm(task_dataset_overlap.items()):

        # debug_count +=1  #DEBUG
        # if debug_count == 10:
        #     break 

        #get transcriptions 
        transcriptions = df[df['file_name'] == k]['transcript'].item()
        
        for inner_k, inner_v in v.items():
            sample = task_dataset_overlap[k][inner_k][0]
            
            agree = False
            agree = sample['error'] == False

            if agree:
                timestamp_dict = sample['timestamp'] #get first annotations timestamp
                turn_convo = get_transcript(timestamp_dict, transcriptions, session_name = k, data_path = args.data_path) 
                
                if transcript_level == 'turn':
                    conversation = turn_convo

                split_convo = turn_convo.split("\n")
                if len(split_convo) == 2 and 'User' in split_convo[0] and 'Agent' in split_convo[1]:
                    reason = sample['rationale']
                    sample['transcript'] = conversation
                    sample['transcript_user'] = split_convo[0]
                    sample['transcript_agent'] = split_convo[1]
                    sample['id'] = k
                    task_dataset_final.append(sample)

    def check_all_false(dictionary):
        return all(value is False for value in dictionary.values())
    print('sampling')

    task_dataset_final_text = []
    for i in tqdm(range(len(task_dataset_final))): 
        sample = task_dataset_final[i]
        for tiers in ['attribute']:
            if check_all_false(sample[tiers]):
                pass
            else:
                sub_dict = sample[tiers]
                others = random.choices(task_dataset_final, k =100)
                # others = filter_dicts_without_subdict_tier2(random_task_dataset_final_subset, sub_dict)
                other_transcript_agent_list = [other['transcript_agent'] for other in others]
                other_transcript_user_list = [other['transcript_user'] for other in others]
                other_id_list = [other['id'] for other in others]
                other_timestamp_list = [other['timestamp'] for other in others]

                # others = filter_dicts_with_any_true_in_subdict(random_task_dataset_final_subset, sub_dict)
                # other_reason_list = [other['rationale'] for other in others]


                if task_type in ['pre']:
                    other_transcript_user_list ,other_id_list, other_timestamp_list = remove_redundant_strings_id_timestamp(other_transcript_user_list,other_id_list, other_timestamp_list) 

                if task_type in ['post']:
                    other_transcript_agent_list ,other_id_list, other_timestamp_list = remove_redundant_strings_id_timestamp(other_transcript_agent_list,other_id_list, other_timestamp_list) 

                if task_type == 'pre' and len(other_transcript_user_list) < 5: continue 
                if task_type == 'post' and len(other_transcript_agent_list) < 5: continue

                sample['other_transcript_agent_list'] = other_transcript_agent_list
                sample['other_transcript_user_list'] = other_transcript_user_list
                sample['other_id_list'] = other_id_list
                sample['other_timestamp_list'] = other_timestamp_list
                
                task_dataset_final_text.append(sample)
                break

    
    processed_dataset = task_dataset_final_text


os.makedirs("./output_datasets", exist_ok=True)
file_path = './output_datasets/{}_{}.pickle'.format(args.data_name, args.task_type)
with open(file_path, 'wb') as handle: pickle.dump(processed_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
