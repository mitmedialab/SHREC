import pickle
import os
import random
import cv2
import openai
import google.generativeai as genai
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
from dotenv import load_dotenv

load_dotenv()
random.seed(42)
openai.api_key = os.environ.get("OPENAI_API_KEY", "")
genai.configure(api_key=os.environ.get("GOOGLE_GENAI_API_KEY") or os.environ.get("GEMINI_KEY", ""))


parser = argparse.ArgumentParser(description="Causal inference with LLMs")
parser.add_argument("--context_window", type=float, default=30)
parser.add_argument("--task_type", type=str, default='empathic_detection_100.pickle')
parser.add_argument("--model", type=str, default='gemini-pro-vision')
parser.add_argument("--data_path", type=str, default='./output_datasets')
parser.add_argument("--seed", type=str, default=0)
parser.add_argument("--video", action="store_true")
parser.add_argument("--csv_path", type=str, default='../shrec_empathic.csv')
parser.add_argument("--images_dir", type=str, default='../shrec_empathic')
parser.add_argument("--hf_repo_id", type=str, default='MIT-personal-robots/shrec_empathic',
                    help="HF dataset repo to pull frame images from, e.g. MIT-personal-robots/shrec_wellness_home")

args = parser.parse_args()

random.seed(args.seed)

task_type = args.task_type
data_path = args.data_path

file_path = './output/{}_{}.json'.format(args.model, args.task_type)


with open(os.path.join(data_path,task_type), 'rb') as pkl_file:
    processed_dataset = pickle.load(pkl_file)

random.shuffle(processed_dataset)

images_dir = args.images_dir 
df = pd.read_csv(args.csv_path)
model_name = args.model
results = []
count = 0
context_window = float(args.context_window)
video = bool(args.video)

#BUILD BASE PROMPT## 
if model_name in ['GPT4o_MINI_Lang', 'GPT4o_Lang', 'Llama-3.2-3B', 'Llama-3.2-3B-Instruct']:
    inputs = "Conversation History"
else:
    inputs = "Images and Conversation History"

wellness_dataset_prompt = "The social robotic agent is designed to be a social positive psychology coach that delivers interactive positive psychology interventions and provide other useful skills to build rapport with college students. "
empathic_dataset_prompt = "The social robotic agent is designed to be a social support companion that facilitates the exchange of emotionally relevant stories and employs narrative therapy techniques to enhance feelings of connection and belonging."

base_prompt = """You are given the {} between a social robotic agent (Jibo) and a participant. Answer the following questions about social interactions.""".format(inputs)


if 'wellness' in args.task_type:
    base_prompt = wellness_dataset_prompt + base_prompt
else: 
    base_prompt = empathic_dataset_prompt + base_prompt

print('Initializing Agent...')
##INITIALIZE AGENT## Uncomment later after debug
_VLMEVAL_MODELS = ['paligemma', 'llava_next_llama3', 'llava_video_qwen2_7b', 'InternVL2-8B',
                    'MiniCPM-V-2_6', 'Llama-3.2-3B', 'Llama-3.2-3B-Instruct', 'Llama-3.2-11B-Vision-Instruct']
if model_name in _VLMEVAL_MODELS:
    from vlmeval.config import supported_VLM

if model_name == 'paligemma':
    model = supported_VLM['paligemma-3b-mix-448']()
if model_name == 'llava_next_llama3':
    model = supported_VLM['llava_next_llama3']()
if model_name == 'Llama-3.2-11B-Vision-Instruct':
    model = supported_VLM['Llama-3.2-11B-Vision-Instruct']()
if model_name == 'llava_video_qwen2_7b':
    model = supported_VLM['llava_video_qwen2_7b']()
if model_name == 'InternVL2-8B':
    model = supported_VLM['InternVL2-8B']() 
if model_name == 'MiniCPM-V-2_6':
    model = supported_VLM['MiniCPM-V-2_6']()
if model_name == 'Llama-3.2-3B':
    model = supported_VLM['Llama-3.2-3B']()
if model_name == 'Llama-3.2-3B-Instruct':
    model = supported_VLM['Llama-3.2-3B-Instruct']()
if model_name == 'Llama-3.2-11B-Vision-Instruct':
    model = supported_VLM['Llama-3.2-11B-Vision-Instruct']()
if model_name == 'GPT4o_MINI_Image':
    from utils_gpt import * 
    model = Agent(base_prompt, model_info='gpt-4o-mini', inference_type='zero-shot', task_type=args.task_type)
if model_name == 'GPT4o_Image':
    from utils_gpt import * 
    model = Agent(base_prompt, model_info='gpt-4o', inference_type='zero-shot', task_type=args.task_type)
if model_name == 'GPT4o_MINI_Lang':
    from utils_gpt import * 
    model = Agent(base_prompt, model_info='gpt-4o-mini', inference_type='lang-only', task_type=args.task_type)
if model_name == 'GPT4o_Lang':
    from utils_gpt import * 
    model = Agent(base_prompt, model_info='gpt-4o', inference_type='lang-only', task_type=args.task_type)
if model_name in ['gemini-1.5-flash', 'gemini-2.0-flash-exp', 'gemini-1.5-flash-8b', 'gemini-1.5-pro', 'gemini-3.1-pro', 'gemini-3.1-pro-preview', 'gpt-realtime-2-preview', 'gpt-realtime-2', 'llava_video_next', 'llava_video_next_7b_dpo', 'o1', 'o1-mini']:
    from utils_gpt import * 
    model = Agent(base_prompt, model_info=model_name, inference_type=None, task_type=args.task_type)
if model_name in ['GPT4o_Image_few_shot']:
    from utils_gpt import * 
    model = Agent(base_prompt, model_info='gpt-4o', inference_type='few-shot', task_type=args.task_type)
if model_name in ['GPT4o_Image_cot']:
    from utils_gpt import * 
    model = Agent(base_prompt, model_info='gpt-4o', inference_type='cot', task_type=args.task_type)
if model_name in ['DeepSeek-R1-Distill-Qwen-32B']:
    from utils_deepseek import * 
    model = ModelLoader(model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    m_, t_ = model.load()
    

from transformers import pipeline


if model_name in ['GPT4o_MINI_Lang', 'GPT4o_Lang', 'Llama-3.2-3B', 'Llama-3.2-3B-Instruct', 'o1', 'o1-mini']:
    sample_images_from_video_in_sec = return_None


print('\n Dataset Size is: ', len(processed_dataset))
for i, sample in enumerate(processed_dataset):

    if video:
        video_id = sample["video_id"]
        print("[INFO] Processing {} ...".format(video_id.lower()))

        # sample images from the video
        num_images = 5
        tmp_dir = "tmp"
        folder_path = os.path.join(os.getcwd(), tmp_dir)

        center = round((sample['timestamp']['start'] + sample['timestamp']['end'])/2,2)
        

        frame_rate = df[df['file_name'] == video_id]['framerate'].item()
        row_frame_paths = ast.literal_eval(df[df['file_name'] == video_id]['frame_paths'].item())
        frames_dir = os.path.join(images_dir, video_id)

        # Pulled directly from HF on demand; falls back to local images_dir
        # if this row has no frame_paths (e.g. an older CSV export).
        if row_frame_paths:
            img_path_list = download_hf_frames(
                repo_id=args.hf_repo_id, video_id=video_id, frame_paths=row_frame_paths,
                framerate=frame_rate, timestamp_range=(sample['timestamp']['start'], sample['timestamp']['end']),
                num_images=num_images,
            )
        else:
            img_path_list = get_frame_paths(frames_dir, sample['timestamp']['start'] * frame_rate, sample['timestamp']['end']* frame_rate)



        if model_name in ['o1', 'o1-mini','gemini-1.5-flash', 'gemini-2.0-flash-exp', 'gemini-1.5-flash-8b', 'gemini-1.5-pro', 'gemini-3.1-pro', 'gemini-3.1-pro-preview', 'gpt-realtime-2-preview', 'gpt-realtime-2', 'llava_video_next', 'llava_video_next_7b_dpo', 'GPT4o_Image_cot', 'GPT4o_Image_few_shot', "GPT4o_MINI_Lang", "GPT4o_Lang", "GPT4o_MINI_Image", "GPT4o_Image"]:
            print('extract video for gemini/openai models')
            base64Frames = process_images_base64(img_path_list)

            vid_path = ""
            # gemini-3.1-pro uses frames directly, not a reconstructed video
            if 'gemini' in model_name and not model_name.startswith('gemini-3.1-pro'):
                vid_path = fill_and_write_video(frames_dir, './tmp', fps=frame_rate)
            
            if 'transcript' in sample.keys():
                transcription = sample['transcript']
            elif 'transcription' in sample.keys():
                transcription = sample['transcription']
   
    reason = sample['rationale']

    if  'debug' in task_type:
        context = sample['transcription'] #transcript 
        answer = sample['error'] # ['isCompotence'] #True, False, None 
        random_answer = random.choice(['(A) Social Competence', '(B) Social Error', '(C) None'])
       

        task = """ Now, given the {} between social agent (Jibo) and a participant, explain whether the agent exhibits Social Competence or Social Error or Nothing""".format(inputs)
        definitions = """We share the definitions here: Social Competence: Social competence is the ability to successfully conduct social interactions, which depends on the awareness and identification of social-emotional cues, the ability to process such cues, and the ability to decide on and express a normative response to these cues. Social Error: are errors that violate social norms and degrade a user's perception of a robot's socio-affective competence, such as interrupting a user at an inappropriate time during a conversation Simply put, socio-affective competence refers to skillful social and affective behavior that is aligned to the desired and/or normal behaviors expected by a user, thereby increasing trust, reliability, and overall perceived competence of the agent. Socio-affective error refers to a behavior exhibited by a robot that deviates from the desired or normal behaviors expected by a user, thereby degrading the overall perceived competence of the agent. None: neither a social error or competence."""
        context_prompt = "\n\nAnswer the above from the following Conversation History:\n{}".format(context)
        answer_choices = '\n\n Explain what this video demonstrates in at least 2 sentences'
    
        full_prompt = base_prompt + task + definitions + context_prompt + answer_choices


    if 'detection' in task_type:
        
        context = sample['transcription'] #transcript 
        answer = sample['error'] # ['isCompotence'] #True, False, None 
        random_answer = random.choice(['(A) Social Competence', '(B) Social Error', '(C) None'])
       

        task = """ Now, given the {} between social agent (Jibo) and a participant, return whether the agent exhibits (A) Social Competence or (B) Social Error or (C) None. """.format(inputs)
        definitions = """We share the definitions here: (A) Social Competence: Social competence is the ability to successfully conduct social interactions, which depends on the awareness and identification of social-emotional cues, the ability to process such cues, and the ability to decide on and express a normative response to these cues. (B) Social Error: are errors that violate social norms and degrade a user's perception of a robot's socio-affective competence, such as interrupting a user at an inappropriate time during a conversation Simply put, socio-affective competence refers to skillful social and affective behavior that is aligned to the desired and/or normal behaviors expected by a user, thereby increasing trust, reliability, and overall perceived competence of the agent. Socio-affective error refers to a behavior exhibited by a robot that deviates from the desired or normal behaviors expected by a user, thereby degrading the overall perceived competence of the agent. (C) None: neither a social error or competence."""
        example_answer = """The answer should be in following format: Answer: {} """.format(random_answer)
        context_prompt = "\n\nAnswer the above from the following Conversation History:\n{}".format(context)
        answer_choices = '\n\nChoose from the following:\n(A) Social Competence (B) Social Error, (C) None'
    
        full_prompt = base_prompt + task + definitions + context_prompt + answer_choices


    if 'attribute' in task_type:
        

        context = sample['transcription'] #transcript 
        answer = sample['attribute'] #True, False, None 
        definitions = ""

   
        if sample['error'] == False: 
            comp_or_error = 'Social Competence (competent social behavior)'      
        else:  
            comp_or_error = 'Social Error (a failure in social behavior)'  

        task = """ Now, we provide the {} between social agent (Jibo) and a participant which corresponds to an {} in socio-affective behavior, select which of the following categories this is related to:""".format(inputs, comp_or_error)
        
        task += """ (A) Emotions: The ability to identify and interpret emotional expressions in oneself and others, allowing for empathetic responses and social emotional awareness
                (B) Engagement: The skill to observe and assess levels of participation and involvement in social interactions, including cues that indicate interest or disinterest
                (C) Conversational Mechanics: Understanding the structure and flow of conversations, including turn-taking, interruptions, and cues for when to speak or listen
                (D) Knowledge State:  The ability to assess what others know or believe, as well as being aware of one's own knowledge in social situations
                (E) Intention: The capacity to infer the goals or purposes behind the actions and words of others, facilitating better responses in social interactions
                (F) Social Relationships: The ability to identify and understand the dynamics of social relationships and the context in which they occur, influencing behavior and expectation
                (G)Social Norms: The skill to identify accepted behaviors and attitudes within a social group, as well as recognizing negative or harmful interactions that violate these norms"""

        random_answer = random.choice(["(A) Emotions", "(B) Engagement", "(C) Conversational Mechanics", "(D) Understanding Knowledge State of Others and Self", "(E) Understanding Intention of Others", "(F) Social Relationships", "(G) Recognizing Social Norms including toxicity"])
        example_answer = """The answer should be in following format:\nAnswer: {} """.format(random_answer)
        context_prompt = "\n\nAnswer the above from the Conversation History:\n{}".format(context)

        full_prompt = base_prompt + task + context_prompt


    if 'attribute_disagree' in task_type or 'attribute_agreed_multiple' in task_type:
        context = sample['transcription'] #transcript 
        answer = sample['attribute'] #True, False, None 
        definitions = ""

           
        if sample['error'] == False: 
            comp_or_error = 'Social Competence (competent social behavior)'      
        else:  
            comp_or_error = 'Social Error (a failure in social behavior)' 


        random_answer = random.choice(["(A) Emotions", "(B) Engagement", "(C) Conversational Mechanics", "(D) Understanding Knowledge State of Others and Self", "(E) Understanding Intention of Others", "(F) Social Relationships", "(G) Recognizing Social Norms including toxicity"])
        random_answer2 = random.choice(["(A) Emotions", "(B) Engagement", "(C) Conversational Mechanics", "(D) Understanding Knowledge State of Others and Self", "(E) Understanding Intention of Others", "(F) Social Relationships", "(G) Recognizing Social Norms including toxicity"])
        
        
        task = """ Now, we provide the {} between social agent (Jibo) and a participant which corresponds to an {} in socio-affective behavior, select which of the following categories this is related to:""".format(inputs, comp_or_error)
        
        task += """ (A) Emotions: The ability to identify and interpret emotional expressions in oneself and others, allowing for empathetic responses and social emotional awareness
                (B) Engagement: The skill to observe and assess levels of participation and involvement in social interactions, including cues that indicate interest or disinterest
                (C) Conversational Mechanics: Understanding the structure and flow of conversations, including turn-taking, interruptions, and cues for when to speak or listen
                (D) Knowledge State:  The ability to assess what others know or believe, as well as being aware of one's own knowledge in social situations
                (E) Intention: The capacity to infer the goals or purposes behind the actions and words of others, facilitating better responses in social interactions
                (F) Social Relationships: The ability to identify and understand the dynamics of social relationships and the context in which they occur, influencing behavior and expectation
                (G)Social Norms: The skill to identify accepted behaviors and attitudes within a social group, as well as recognizing negative or harmful interactions that violate these norms"""

        answer_choices = "\n\nThere can be multiple answers, choose one of more from the following:\n (A) Emotions (B) Engagement (C) Conversational Mechanics (D) Understanding Knowledge State of Others and Self (E) Understanding Intention of Others (F) Social Relationships (G) Recognizing Social Norms including toxicity"
        
        context_prompt = "\n\nAnswer the above from the Conversation History:\n{}".format(context)

        full_prompt = base_prompt + task + context_prompt + answer_choices

    if 'attribute_agreed_multiple_subj' in task_type:
        ###NEEDS TO BE FIXED###
        context = sample['transcription'] #transcript 
        answer = sample['attribute'] #True, False, None 
        definitions = ""

        random_answer = random.choice(["True", "False"])



        if sample['error'] == False: 
            comp_or_error = 'Social Competence (competent social behavior)'      
        else:  
            comp_or_error = 'Social Error (a failure in social behavior)'  


        random_answer = random.choice(["(A) Emotions", "(B) Engagement", "(C) Conversational Mechanics", "(D) Understanding Knowledge State of Others and Self", "(E) Understanding Intention of Others", "(F) Social Relationships", "(G) Recognizing Social Norms including toxicity"])
        random_answer2 = random.choice(["(A) Emotions", "(B) Engagement", "(C) Conversational Mechanics", "(D) Understanding Knowledge State of Others and Self", "(E) Understanding Intention of Others", "(F) Social Relationships", "(G) Recognizing Social Norms including toxicity"])
        
        # task = """ Now, we provide the {} between social agent (Jibo) and a participant which corresponds to an {} in socio-affective behavior, Consider the following categories: (A) Emotions, (B) Engagement, (C) Conversational Mechanics, (D) Understanding Knowledge State of Others and Self, (E) Understanding Intention of Others, (F) Social Relationships, (G) Recognizing Social Norms including toxicity. Are there be multiple attributes in how the agent portrays {} behavior? Answer with True, if there are multiple attributes, or False if there is only a single attribute.""".format(inputs, comp_or_error)        
        task = '''We provide a {} of an interaction between the social agent (Jibo) and a user. The agent's behavior in this interaction corresponds to {}.'''.format(inputs, comp_or_error)    

        task += ''' Consider the following seven social attributes:
                (A) Emotions: The ability to identify and interpret emotional expressions in oneself and others, allowing for empathetic responses and social emotional awareness
                (B) Engagement: The skill to observe and assess levels of participation and involvement in social interactions, including cues that indicate interest or disinterest
                (C) Conversational Mechanics: Understanding the structure and flow of conversations, including turn-taking, interruptions, and cues for when to speak or listen
                (D) Knowledge State:  The ability to assess what others know or believe, as well as being aware of one's own knowledge in social situations
                (E) Intention: The capacity to infer the goals or purposes behind the actions and words of others, facilitating better responses in social interactions
                (F) Social Relationships: The ability to identify and understand the dynamics of social relationships and the context in which they occur, influencing behavior and expectation
                (G)Social Norms: The skill to identify accepted behaviors and attitudes within a social group, as well as recognizing negative or harmful interactions that violate these norms 

                Based on the transcript, determine whether the agent's behavior involves multiple social attributes.Respond with "True" if the behavior demonstrates more than one social attribute. Respond with "False" if the behavior is based on only a single attribute.'''
                        
        # example_answer = """The answer should be in following format:\nAnswer: {}""".format(random_answer)
        context_prompt = "\n\nAnswer the above from the Conversation History:\n{}".format(context)

        full_prompt = base_prompt + task + context_prompt 


    if 'rationale' in task_type:

        context = sample['transcript']
        answer = sample['rationale']
        all_answers = sample['other_reason_list'][:4]

        #append
        all_answers.append(answer)
        enum_wrong_answers = list(enumerate(all_answers))
        random.shuffle(enum_wrong_answers)
        indices, l = zip(*enum_wrong_answers)
        answer_index = indices.index(4)
        answer_value = "({})".format(str(answer_index + 1))
        answer = answer_value

        if sample['error'] == False: 
            comp_or_error = 'Social Competence'      
        else:  
            comp_or_error = 'Social Error'     

        task = """ Now, we provide the {} between social agent (Jibo) and a participant which corresponds to an {} in behavior, select which is the correct reason behind the {}.""".format(inputs, comp_or_error, comp_or_error)
        
        context_prompt = "\n\nConversation History:{}".format(context)
        answer_choices = "\n\nReasons: (1) {} (2) {} (3) {} (4) {} (5) {}".format(l[0], l[1], l[2], l[3], l[4])
        
        
        random_answer = random.choice(['(1)', '(2)', '(3)', '(4)', '(5)'])
        example_answer = """The answer should be in following format: Answer: {} """.format(random_answer)

        full_prompt = base_prompt + task + example_answer + context_prompt + answer_choices
        

    if 'correction' in task_type: #aka forward action 
        

        context = sample['transcript']
        answer = sample['correction']
        all_answers = sample['other_recovery_list'][:4]

        all_answers.append(answer)
        enum_wrong_answers = list(enumerate(all_answers))
        random.shuffle(enum_wrong_answers)
        indices, l = zip(*enum_wrong_answers)
        answer_index = indices.index(4)
        answer_value = "({})".format(str(answer_index + 1))
        answer = answer_value

        task = """ Now, we provide the {} between social agent (Jibo) and a participant which corresponds to an error in socio-affective behavior, from the following numerical choices select which behavior Jibo (social agent) should have done instead."""
        
        context_prompt = "\n\nConversation History:{}".format(context)
        answer_choices = "\n\nBehaviors: (1) {} (2) {} (3) {} (4) {} (5) {}".format(l[0], l[1], l[2], l[3], l[4])
        
        
        random_answer = random.choice(['(1)', '(2)', '(3)', '(4)', '(5)'])
        example_answer = """The answer should be in following format: Answer: {} """.format(random_answer)

        full_prompt = base_prompt + task + context_prompt + answer_choices


    if 'pre' in task_type or 'post' in task_type: #aka backward belief
        
        if 'post' in task_type:
            context = sample['transcript_user']
            answer = sample['transcript_agent']
        if  'pre' in task_type:
            context = sample['transcript_agent']
            answer = sample['transcript_user']

        all_id = sample['other_id_list']
        all_answers = []

        if 'post' in task_type:
            
            all_answers = sample['other_transcript_agent_list'][:4]
        if 'pre' in task_type:
            
            all_answers = sample['other_transcript_user_list'][:4]
        all_answers.append(answer)
        enum_wrong_answers = list(enumerate(all_answers))
        random.shuffle(enum_wrong_answers)
        indices, l = zip(*enum_wrong_answers)

        answer_index = indices.index(4)
        answer_value = "({})".format(str(answer_index + 1))
        answer = answer_value

        if 'post' in task_type:
            context_prompt = """\n\n User Behavior: {} """.format(context)
            task = """ Now, we provide the user’s behavior. From the following ordered choices of the social agent’s behaviors: (1), (2), (3), (4), (5), select which agent’s behavior was the appropriate response from the user’s action. """

        if 'pre' in task_type:
            context_prompt = """\n\n Agent Behavior: {} """.format(context)
            task = """Now, we provide what the social agent did in response to a user behavior. From the following ordered choices of users' behaviors: (1), (2), (3), (4), (5), select which user’s behavior was the appropriate pre-condition from the agent’s action."""

        
        answer_choices = "\n\nContext (Conversational History): (1) {} (2) {} (3) {} (4) {} (5) {}".format(l[0], l[1], l[2], l[3], l[4])
        
        random_answer = random.choice(['(1)', '(2)', '(3)', '(4)', '(5)'])
        example_answer = """The answer should be in following format: Answer: {} """.format(random_answer)
        
        full_prompt = base_prompt + task + context_prompt + answer_choices


    # Some Logic Implemented for Our Set of Experiments
    if model_name in ['GPT4o_MINI_Lang', 'GPT4o_Lang', 'o1', 'o1-mini' 'Llama-3.2-3B', 'Llama-3.2-3B-Instruct', 'DeepSeek-R1-Distill-Qwen-32B', 'DeepSeek-R1-Distill-Qwen-32B']:
        img_path_list = full_prompt
    else:
        img_path_list.append(full_prompt)

    if model_name in ['o1', 'o1-mini', 'gemini-1.5-flash', 'gemini-2.0-flash-exp', 'gemini-1.5-flash-8b', 'gemini-1.5-pro', 'gemini-3.1-pro', 'gemini-3.1-pro-preview', 'gpt-realtime-2-preview', 'gpt-realtime-2', 'llava_video_next', 'llava_video_next_7b_dpo', 'GPT4o_Image_cot', "GPT4o_MINI_Lang", "GPT4o_Lang", "GPT4o_MINI_Image", "GPT4o_Image"]:
        response = model.chat(full_prompt, img_frames=base64Frames, transcription=transcription, vid_path=vid_path, examples=None)
    elif model_name == 'GPT4o_Image_few_shot':
        examples = get_few_shot_examples(processed_dataset, sample)
        response = model.chat(full_prompt, img_frames=base64Frames, transcription=transcription, vid_path=vid_path, examples=examples)
    else:
        response = model.generate(img_path_list)

    count += 1
    
    print("="*100)
    print()
    print("[prompt]")
    print(full_prompt)
    print("="*100)
    print()
    print("[response]")
    print(response.replace('\n',''))
    print("="*100)
    print()
    print("[label]")
    print(answer)


    print("="*100)
    print()
    
    results.append({'question': full_prompt, 'response': response, 'label': answer, 'rationale': reason})
 
    print("[INFO] Total number of processed results:", "{}/{}".format(str(count),str(len(processed_dataset))))

folder_path = os.path.join(os.getcwd(), 'output')
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

with open(file_path, 'w') as json_file:
    json.dump(results, json_file, indent=4)
