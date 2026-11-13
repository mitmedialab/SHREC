
import pdb
import ast
import pandas as pd
import os
import cv2
import base64
from natsort import natsorted
from huggingface_hub import hf_hub_download

def find_overlapping_interval_groups_pair(intervals1, intervals2):
    # Sort intervals based on start value
    

    sorted_intervals1 = sorted(enumerate(intervals1), key=lambda x: x[1]['timestamp']['start'] )
    sorted_intervals2 = sorted(enumerate(intervals2), key=lambda x: x[1]['timestamp']['start'])

    overlapping_groups = []

    for current_index, current_interval in sorted_intervals1:

        # if current_interval['timestamp']['start'] != None and current_interval['timestamp']['end'] != None :
        #     continue

        current_group = [current_interval]
        added = False 
        
        need_rerank_group = []
        for index, interval in sorted_intervals2:
            # if interval['timestamp']['start'] == None or interval['timestamp']['end']:
            #     continue
            if interval['timestamp']['start'] < current_interval['timestamp']['start'] and interval['timestamp']['end'] > current_interval['timestamp']['end'] and abs(interval['timestamp']['end'] - current_interval['timestamp']['end']) < 10:
                need_rerank_group.append((index,interval))
            elif interval['timestamp']['start'] > current_interval['timestamp']['start'] and interval['timestamp']['end'] < current_interval['timestamp']['end'] and abs(interval['timestamp']['end'] - current_interval['timestamp']['end']) < 10:
                need_rerank_group.append((index,interval))
            elif abs(interval['timestamp']['start'] - current_interval['timestamp']['start']) < 3:
                need_rerank_group.append((index,interval))
            elif abs(interval['timestamp']['end'] - current_interval['timestamp']['end']) < 3:
                need_rerank_group.append((index,interval))
        
        #finished populating potential matches rank by closest to end and start
        for index, match_interval in need_rerank_group:
            if current_interval['error'] == match_interval['error']:
                current_group.append(match_interval)
                sorted_intervals2.remove((index, match_interval))
                overlapping_groups.append(current_group)
                added = True 
                break
        
        if len(need_rerank_group) > 0 and not added:
            current_group.append(match_interval)
            sorted_intervals2.remove((index, match_interval))
            overlapping_groups.append(current_group)

        if len(need_rerank_group) == 0 and not added:
            overlapping_groups.append(current_group)

        
    #add remaining from interval2
    for index, interval in sorted_intervals2:
        overlapping_groups.append([interval])
    

    return overlapping_groups

def merge_intervals(intervals):
    merged_intervals = []
    for interval in sorted(intervals, key=lambda x: x['start']):
        if not merged_intervals or interval['start'] > merged_intervals[-1]['end']:
            merged_intervals.append(interval)
        else:
            merged_intervals[-1]['end'] = max(merged_intervals[-1]['end'], interval['end'])
    return merged_intervals

def find_uncovered_float_intervals(intervals, total_time_range):
    # Merge overlapping intervals
    merged_intervals = merge_intervals(intervals)
    
    # Initialize the uncovered intervals list
    uncovered_intervals = []
    
    # Initialize variables for tracking current covered range
    current_start = total_time_range['start']
    current_end = total_time_range['start']

    for interval in merged_intervals:
        if interval['start'] > current_end:
            # Gaps exist between current_end and next interval's start
            uncovered_intervals.append({'start': current_end, 'end': interval['start']})
        
        # Update current_start and current_end
        current_start = interval['start']
        current_end = max(current_end, interval['end'])
    
    # Check for uncovered time at the end of the total time range
    if current_end < total_time_range['end']:
        uncovered_intervals.append({'start': current_end, 'end': total_time_range['end']})
    
    return uncovered_intervals


def flatten(xss):
    return [x for xs in xss for x in xs]



def filter_dicts_without_subdict_tier2(dicts_list, sub_dict):
    all_other_dict = []
    for other_dict in dicts_list:
        diff = True 
        for key in sub_dict:
            if sub_dict.get(key) is True:  # Check if the value in dict1 is True
                if other_dict['attribute'].get(key) is True:  # Check if the value in dict2 is also True
                    diff = False
                    # print(f"Both dictionaries have True for key '{key}'")
                else:
                    pass
                    # print(f"dict1 has True for key '{key}', but dict2 does not")
        if diff == True:
            all_other_dict.append(other_dict)
    return all_other_dict


def filter_dicts_without_subdict(dicts_list, sub_dict):
    filtered_dicts = [d for d in dicts_list if not all(item in d.items() for item in sub_dict.items())]
    return filtered_dicts

def filter_dicts_with_subdict(dicts_list, sub_dict):
    filtered_dicts = [d for d in dicts_list if all(item in d.items() for item in sub_dict.items())]
    return filtered_dicts

def filter_dicts_with_any_true_in_subdict(list_of_dicts, match_dict):
    filtered_list = [d for d in list_of_dicts if not any(d.get(key) and value for key, value in match_dict.items())]
    return filtered_list

def remove_redundant_strings(strings_list):
    unique_strings = list(set(strings_list))
    return unique_strings


def remove_redundant_strings_id_timestamp(strings_list,other_id_list, other_timestamp_list):
    unique_strings = []
    unique_id = [] 
    unique_timestamps = []

    for i in range(len(strings_list)):
        if strings_list[i] not in unique_strings:
            unique_strings.append(strings_list[i])
            unique_id.append(other_id_list[i])
            unique_timestamps.append(other_timestamp_list[i])

    return unique_strings, unique_id, unique_timestamps



def and_operation(dict1, dict2):
    result_dict = {}
    for key in dict1.keys() & dict2.keys():
        result_dict[key] = dict1[key] and dict2[key]
    return result_dict

def or_operation(dict1, dict2):
    result_dict = {}
    for key in dict1.keys() & dict2.keys():
        result_dict[key] = dict1[key] or dict2[key]
    return result_dict

def disagree_operation(dict1, dict2):
    result_dict = {}
    for key in dict1.keys() & dict2.keys():
        result_dict[key] = dict1[key] and dict2[key]
    return result_dict


def find_all_indexes(string, substring):
    return [index for index, _ in enumerate(string) if string[index:index + len(substring)] == substring]

def return_exact_convos(start_time,end_time,df, turn = False, more_context = True):

    agent_id = 'AI Agent:'


    conversation = ""
    if (df['timestamp'] < start_time)[df['timestamp'] < start_time].empty:
        start_index = 0
    else:
        start_index = (df['timestamp'] < start_time)[df['timestamp'] < start_time].index[-1]
        
    if (df['timestamp'] < start_time)[df['timestamp'] < start_time].empty:
        end_index = 0
    else:
        end_index = (~(df['timestamp'] > end_time))[(~(df['timestamp'] > end_time))].index[-1]


    if turn:
        if df['speaker'].iloc[start_index] == agent_id: #get what user did before 
            if start_index == 0: 
                curr_df = df.iloc[start_index:start_index + 1,:]
            else: 
                curr_df = df.iloc[start_index - 1 :start_index + 1,:]
        else: #get what agent does next
            if start_index == len(df) - 1: 
                curr_df =  df.iloc[start_index:start_index + 1,:] 
            else:
                curr_df = df.iloc[start_index :start_index + 2,:]

    if not turn:
        curr_df = df.iloc[start_index:end_index + 1,:] #if you don't want the next turn to be included just get rid of the +1 or add end_index +1 
    
    if start_index != 0 and start_index != len(df) - 1 and len(curr_df) <= 1:
        pdb.set_trace()

    conversation = "\n".join(f"{row['speaker']} {row['text']}" for _, row in curr_df.iterrows())
    return conversation



def get_transcript(timestamp_dict, transcriptions, session_name = None, data_path = None, time_type = None):
    
    start_time = timestamp_dict['start']
    end_time = timestamp_dict['end']

    context_transcriptions = []
    turn_context_transcriptions = []

    transcriptions = transcriptions.split('\n')

    for transcription in transcriptions:
        if 'User' in transcription or 'AI Agent' in transcription: 
            speaker = transcription.replace("\n", "")
        elif '(' in transcription and ')' in transcription and ':' in transcription:
            
            times_list = find_all_indexes(transcription, "(")
            all_words = ""
            for i, time in enumerate(times_list):
                
                start_index = time
                end_index = time + 10 #the time stamp is 10 characters so we just index 10 indices away
                time_str = transcription[start_index: end_index]

                if i != len(times_list)-1:
                    word = transcription[end_index: times_list[i+1]]
                else:
                    word = transcription[end_index:]

                all_words += " " + word.replace(" ", "")
                tmp = time_str.split('(')[-1].split(')')[0].split(":")
                _timestamp = float(tmp[0]) * 3600 + float(tmp[1]) * 60 + float(tmp[2])
                if i == 0:
                    start_all_timestamp = _timestamp
                    
                context_transcriptions.append({'timestamp':_timestamp, 'speaker':speaker, 'text': word})

            turn_context_transcriptions.append({'timestamp':start_all_timestamp, 'speaker':speaker, 'text': all_words})
    
    new_context_transcriptions = []
    for ct in context_transcriptions:
        if ct['text'] == '\n':
            pass
        else:
            new_context_transcriptions.append(ct)



    df = pd.DataFrame(new_context_transcriptions)
    df_turn = pd.DataFrame(turn_context_transcriptions)

    
    turn_convo = return_exact_convos(start_time,end_time,df_turn, turn = True)
    return turn_convo





def get_majority(items):
    item_counts = Counter(items)
    max_frequency = max(item_counts.values())

    candidates = [item for item, count in item_counts.items() if count == max_frequency]

    if candidates:
        return random.choice(candidates)
    else:
        return None

def download_hf_frames(repo_id, video_id, frame_paths, framerate, timestamp_range,
                        num_images, cache_dir=None, repo_type="dataset"):
    """Download the frames for a time range directly from the HF dataset repo
    (frame_index / framerate = seconds), instead of extracting from video."""
    start_time, end_time = timestamp_range
    start_frame = start_time * framerate
    end_frame = end_time * framerate

    def frame_index(path):
        return int(os.path.splitext(os.path.basename(path))[0])

    in_range = [p for p in frame_paths if start_frame <= frame_index(p) <= end_frame]
    if not in_range:
        print(f"VIDEO CORRUPT: no frames in range for {video_id} [{start_time}, {end_time}]")
        return []

    in_range = natsorted(in_range)
    step = max(1, len(in_range) // num_images)
    selected = in_range[::step][:num_images]

    img_path_list = []
    for frame_path in selected:
        local_path = hf_hub_download(
            repo_id=repo_id, repo_type=repo_type, filename=frame_path, cache_dir=cache_dir,
        )
        img_path_list.append(local_path)

    return img_path_list

def process_video(video_path, timestamp_range, num_images, output_dir):
    print("video_path:", video_path)
    
    os.makedirs(output_dir, exist_ok=True)
    base64Frames = []

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_time, end_time = timestamp_range
    start_frame = int(start_time * fps)
    end_frame = min(int(end_time * fps), total_frames - 1)
    frame_step = max(1, (end_frame - start_frame) // num_images)
    current_frame = start_frame

    for _ in range(num_images):
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        current_frame += frame_step
    cap.release()

    print(f"Extracted {len(base64Frames)} frames")
    return base64Frames

def process_video(img_paths, timestamp_range, num_images, output_dir):
    print("video_path:", video_path)
    
    os.makedirs(output_dir, exist_ok=True)
    base64Frames = []

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_time, end_time = timestamp_range
    start_frame = int(start_time * fps)
    end_frame = min(int(end_time * fps), total_frames - 1)
    frame_step = max(1, (end_frame - start_frame) // num_images)
    current_frame = start_frame

    for _ in range(num_images):
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        current_frame += frame_step
    cap.release()

    print(f"Extracted {len(base64Frames)} frames")
    return base64Frames

def process_images_base64(img_paths, output_dir=None):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    base64Frames = []

    for idx, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image at {img_path}")
            continue
        
        _, buffer = cv2.imencode(".jpg", img)
        base64_frame = base64.b64encode(buffer).decode("utf-8")
        base64Frames.append(base64_frame)

    print(f"Processed {len(base64Frames)} images")
    return base64Frames


# sample_images_from_video_in_sec and trim_video removed — see download_hf_frames() above.

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def return_None(a,b,c,d): return []

def get_frame_paths(frame_dir, start, end):
    all_files = os.listdir(frame_dir)
    frame_paths = []

    for fname in all_files:
        if fname.endswith('.png'):
            try:
                frame_idx = int(os.path.splitext(fname)[0])
                if start <= frame_idx <= end:
                    frame_paths.append(os.path.join(frame_dir, fname))
            except ValueError:
                continue  # Skip files that aren't integer-named

    frame_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_paths = natsorted(frame_paths)
    return frame_paths


import cv2
import os

def fill_and_write_video(frame_dir, output_dir, fps=24):
    os.makedirs(output_dir, exist_ok=True)

    # Get all .png files with integer-named stems
    frame_files = [
        f for f in os.listdir(frame_dir)
        if f.endswith('.png') and f[:-4].isdigit()
    ]
    
    if not frame_files:
        raise ValueError("No valid .png frames found in the directory.")

    # Map from index to full path, using actual file names
    frame_map = {
        int(f[:-4]): os.path.join(frame_dir, f)
        for f in frame_files
    }

    sorted_indices = sorted(frame_map.keys())

    # Load the first valid frame to get size
    first_frame = cv2.imread(frame_map[sorted_indices[0]])
    if first_frame is None:
        raise ValueError(f"Failed to read {frame_map[sorted_indices[0]]}")
    height, width, _ = first_frame.shape

    output_video_path = os.path.join(output_dir, 'trimmed_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write repeated frames for gaps
    for i, curr_idx in enumerate(sorted_indices):
        next_idx = sorted_indices[i + 1] if i + 1 < len(sorted_indices) else curr_idx + 1
        repeat_count = next_idx - curr_idx

        frame = cv2.imread(frame_map[curr_idx])
        if frame is None:
            print(f"Warning: Could not read {frame_map[curr_idx]}, skipping.")
            continue
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))

        for _ in range(repeat_count):
            out.write(frame)

    out.release()
    print(f"Video saved to {output_video_path}")
    return output_video_path
