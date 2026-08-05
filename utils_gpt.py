
import pdb
import ast
import os
import pandas as pd
import numpy as np
import time
import asyncio
import base64
import io
import json
import websockets
import openai
from openai import OpenAI
import google.generativeai as genai
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
from transformers import AutoProcessor, LlavaForConditionalGeneration
import av
import cv2
from huggingface_hub import hf_hub_download
from PIL import Image
import random
from transformers import pipeline


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
genai.configure(api_key=os.environ.get("GOOGLE_GENAI_API_KEY") or os.environ.get("GEMINI_KEY", ""))
openai.api_key=os.environ["OPENAI_API_KEY"]


from transformers import AutoModel

import torch
from transformers import StoppingCriteria, StoppingCriteriaList
import numpy as np
from PIL import Image
from torchvision.transforms import PILToTensor
import matplotlib.pyplot as plt
from easydict import EasyDict
from torchvision import transforms
import time


def get_few_shot_examples(processed_dataset, current_sample, num_examples=3):
    """
    Get few-shot examples excluding the current test sample.
    
    Args:
        processed_dataset: List of all samples
        current_sample: The current test sample to exclude
        num_examples: Number of examples to include
    
    Returns:
        List of example dictionaries
    """
    # Filter out the current sample
    available_samples = [sample for sample in processed_dataset 
                        if sample != current_sample]
    
    # Randomly select examples
    selected_examples = random.sample(available_samples, min(num_examples, len(available_samples)))
    return selected_examples

def format_few_shot_prompt(task_type, examples, base_prompt):
    """
    Format the few-shot prompt based on task type.
    """
    few_shot_text = "\nHere are some examples:\n"
    
    for i, example in enumerate(examples, 1):
        if 'detection' in task_type:
            label = '(A) Social Competence' if example['selections']['isCompotence'] else '(B) Social Error'
            few_shot_text += f"\nExample {i}:\nConversation: {example['transcription']}\nAnswer: {label}\n"
            
        elif 'what' in task_type:
            # Convert tier2 boolean dict to category label
            categories = {
                'Recognizing Emotions': '(A) Recognizing Emotions',
                'Recongizing Engamenet': '(B) Recognizing Engagement',
                'Recognition of Conversational Mechanics': '(C) Recognition of Conversational Mechanics',
                'Understanding Knowledge State of Others and Self': '(D) Understanding Knowledge State of Others and Self',
                'Understanding Intention of Others': '(E) Understanding Intention of Others',
                'Recognizing Social and Context Relationships': '(F) Recognizing Social Relationships',
                'Recognizing Social Norms (Toxicity)': '(G) Recognizing Social Norms including toxicity'
            }

            cand_label =[k for k,v in example['tier2'].items() if v == True]
            
            label = ""
            for k in cand_label: label += categories[k] + ","
            few_shot_text += f"\nExample {i}:\nConversation: {example['transcription']}\nAnswer: {label}\n"
            
        elif 'reason' in task_type or 'context'in task_type or 'recovery' in task_type:
            if 'reason' in task_type:
                answer = example['reason']
                context = example['transcript']
            elif 'context'in task_type:
                answer = example['transcript']
                context = example['recoveryBehavior']
            elif 'recovery' in task_type:
                answer = example['recoveryBehavior']
                context = example['transcript']
                
            few_shot_text += f"\nExample {i}:\nConversation: {context}\nAnswer: {answer}\n"

        elif 'pre' in task_type or 'context'in task_type or 'post' in task_type:
            if 'pre' in task_type:
                context = example['transcript_agent']
                answer = example['transcript_user']
            elif 'post' in task_type:
                context = example['transcript_user']
                answer = example['transcript_agent']
                
            few_shot_text += f"\nExample {i}:\nConversation: {context}\nAnswer: {answer}\n"
    
    return base_prompt + few_shot_text

def get_prompt(conv):
    ret = conv.system + conv.sep
    for role, message in conv.messages:
        if message:
            ret += role + " " + message + " " + conv.sep
        else:
            ret += role
    return ret

def get_prompt2(conv):
    ret = conv.system + conv.sep
    count = 0
    for role, message in conv.messages:
        count += 1
        if count == len(conv.messages):
            ret += role + " " + message
        else:
            if message:
                ret += role + " " + message + " " + conv.sep
            else:
                ret += role
    return ret

def get_context_emb(conv, model, img_list, answer_prompt=None, print_res=False):
    if answer_prompt:
        prompt = get_prompt2(conv)
    else:
        prompt = get_prompt(conv)
    if print_res:
        print(prompt)
    if '<VideoHere>' in prompt:
        prompt_segs = prompt.split('<VideoHere>')
    else:
        prompt_segs = prompt.split('<ImageHere>')
    
    # assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
    
    with torch.no_grad():
        seg_tokens = [
            model.mistral_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to("cuda:0").input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        # seg_embs = [model.mistral_model.base_model.model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        seg_embs = [model.mistral_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
    mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
    mixed_embs = torch.cat(mixed_embs, dim=1)
    return mixed_embs

def ask(text, conv):
    conv.messages.append([conv.roles[0], text])
        
class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False
    
def answer(conv, model, img_list, do_sample=True, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, answer_prompt=None, print_res=False):
    stop_words_ids = [
        torch.tensor([2]).to("cuda:0"),
        torch.tensor([29871, 2]).to("cuda:0")]  # '</s>' can be encoded in two different ways.
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    
    conv.messages.append([conv.roles[1], answer_prompt])
    embs = get_context_emb(conv, model, img_list, answer_prompt=answer_prompt, print_res=print_res)
    with torch.no_grad():
        outputs = model.mistral_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
    output_token = outputs[0]
    if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
    if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
    output_text = model.mistral_tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('</s>')[0]  # remove the stop sign </s>
#     output_text = output_text.split('[/INST]')[-1].strip()
    conv.messages[-1][1] = output_text + '</s>'
    return output_text, output_token.cpu().numpy()

# from dataset.hd_utils import HD_transform_padding, HD_transform_no_padding

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def load_video(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=3, padding=False):
    import decord  # macOS has no decord wheel; only needed here
    from decord import VideoReader, cpu
    decord.bridge.set_bridge("torch")
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Normalize(mean, std)
    ])

    frames = vr.get_batch(frame_indices)
    frames = frames.permute(0, 3, 1, 2)

    if padding:
        frames = HD_transform_padding(frames.float(), image_size=resolution, hd_num=hd_num)
    else:
        frames = HD_transform_no_padding(frames.float(), image_size=resolution, hd_num=hd_num)

    frames = transform(frames)
    print(frames.shape)
    
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return frames, msg
    else:
        return frames

def get_sinusoid_encoding_table(n_position=784, d_hid=1024, cur_frame=8, ckpt_num_frame=4, pre_n_position=784): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 
    
    # generate checkpoint position embedding
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(pre_n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 
    sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)
    
    print(f"n_position: {n_position}")
    print(f"pre_n_position: {pre_n_position}")
    
    if n_position != pre_n_position:
        T = ckpt_num_frame # checkpoint frame
        P = 14 # checkpoint size
        C = d_hid
        new_P = int((n_position // cur_frame) ** 0.5) # testing size
        if new_P != 14:
            print(f'Pretraining uses 14x14, but current version is {new_P}x{new_P}')
            print(f'Interpolate the position embedding')
            sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
            sinusoid_table = sinusoid_table.reshape(-1, P, P, C).permute(0, 3, 1, 2)
            sinusoid_table = torch.nn.functional.interpolate(
                sinusoid_table, size=(new_P, new_P), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            sinusoid_table = sinusoid_table.permute(0, 2, 3, 1).reshape(-1, T, new_P, new_P, C)
            sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
    
    if cur_frame != ckpt_num_frame:
        print(f'Pretraining uses 4 frames, but current frame is {cur_frame}')
        print(f'Interpolate the position embedding')
        T = ckpt_num_frame # checkpoint frame
        new_T = cur_frame # testing frame
        # interpolate
        P = int((n_position // cur_frame) ** 0.5) # testing size
        C = d_hid
        sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
        sinusoid_table = sinusoid_table.permute(0, 2, 3, 4, 1).reshape(-1, C, T)  # BHW, C, T
        sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=new_T, mode='linear')
        sinusoid_table = sinusoid_table.reshape(1, P, P, C, new_T).permute(0, 4, 1, 2, 3) # B, T, H, W, C
        sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
        
    return sinusoid_table

def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

async def _gpt_realtime_ws_call(model_name, prompt, img_frames):
    """Call an OpenAI Realtime model over WebSocket with images + text (no audio)."""
    ws_url = f"wss://api.openai.com/v1/realtime?model={model_name}"
    headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}

    async with websockets.connect(ws_url, additional_headers=headers) as ws:
        await ws.recv()  # session.created

        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "output_modalities": ["text"],
                "instructions": "You are a socially intelligent agent. Be concise.",
            }
        }))
        await ws.recv()  # session.updated

        content = []
        if img_frames:
            content.append({"type": "input_text", "text": "These are the frames from the video."})
        for frame_b64 in img_frames:
            content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{frame_b64}",
            })
        content.append({"type": "input_text", "text": prompt})

        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user", "content": content},
        }))
        await ws.recv()  # conversation.item.created
        await ws.send(json.dumps({"type": "response.create"}))

        parts = []
        async for raw in ws:
            event = json.loads(raw)
            etype = event.get("type", "")
            if etype == "response.output_text.delta":
                parts.append(event.get("delta", ""))
            elif etype == "response.done":
                break
            elif etype == "error":
                print(f"  [API ERROR] {event.get('error')}")
                break
        return "".join(parts)


class Agent:
    def __init__(self, instruction, model_info=None, inference_type='zero-shot', task_type='detection'):
        self.instruction = instruction
        self.model_info = model_info
        self.inference_type = inference_type
        self.task_type = task_type

        if 'gemini' in self.model_info:
            self.model = genai.GenerativeModel(self.model_info)
            self._chat = self.model.start_chat(history=[])

        elif self.model_info.startswith('gpt-realtime'):
            pass  # WebSocket opened per call in chat()

        elif self.model_info in ['DeepSeek-R1']:
            self.model = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1", trust_remote_code=True)

        elif self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4v', 'gpt-4o', 'gpt-4o-mini', 'o1', 'o1-mini']:
            self.messages = [
                {"role": "system", "content": f"You are a socially intelligent agent."},
            ]

        elif self.model_info == 'video-chat2':
            self.model = AutoModel.from_pretrained("OpenGVLab/VideoChat2_HD_stage4_Mistral_7B_hf", trust_remote_code=True).to("cuda:0")
            num_frame = 1
            resolution = 224
            new_pos_emb = get_sinusoid_encoding_table(n_position=(resolution//16)**2*num_frame, cur_frame=num_frame)
            self.model.vision_encoder.encoder.pos_embed = new_pos_emb

        elif self.model_info == 'kangaroo':
            self.tokenizer = AutoTokenizer.from_pretrained("KangarooGroup/kangaroo")
            self.model = AutoModelForCausalLM.from_pretrained(
                "KangarooGroup/kangaroo",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            self.model = self.model.to("cuda")
            self.terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

        elif 'llava_video_next' in self.model_info:

            if self.model_info == 'llava_video_next':
                model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"
            elif self.model_info == 'llava_video_next-34b':
                model_id = "llava-hf/LLaVA-NeXT-Video-34B-hf"
            elif self.model_info == 'llava_video_next-7b-dpo':
                model_id = "llava-hf/LLaVA-NeXT-Video-7B-DPO-hf"

            self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                model_id, 
                torch_dtype=torch.float16, 
                low_cpu_mem_usage=True, 
            ).to(0)

            self.processor = LlavaNextVideoProcessor.from_pretrained(model_id)

        # In your __init__ method
        elif self.model_info == 'llava_interleave_qwen':
            self.model = LlavaForConditionalGeneration.from_pretrained(
                "llava-hf/llava-interleave-qwen-0.5b-hf",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to('cuda')  
            self.processor = AutoProcessor.from_pretrained("llava-hf/llava-interleave-qwen-0.5b-hf")


    def chat(self, message, img_frames=None, transcription=None, vid_path=None, examples=None):
        if self.inference_type == 'few-shot' and examples:
            # Add few-shot examples to the prompt
            message = format_few_shot_prompt(
                task_type=self.task_type,
                examples=examples,
                base_prompt=message
            )
        
        print('\nformatted message\n')
        print(message)
        # pdb.set_trace()
        if self.model_info in ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-2.0-flash-exp', 'gemini-1.5-flash-8b']:
            try:
                # CoT
                if self.inference_type == 'cot':
                    message += " Let's think step-by-step."

                # Upload the video file
                video_file = genai.upload_file(path=vid_path)
                pdb.set_trace()
                while video_file.state.name == "PROCESSING":
                    print('.', end='')
                    time.sleep(10)
                    video_file = genai.get_file(video_file.name)

                if video_file.state.name == "FAILED":
                    raise ValueError(video_file.state.name)

                # Initialize the generative model
                model = genai.GenerativeModel(model_name=self.model_info)

                # Make the LLM request
                print("Making LLM inference request...")
                response = model.generate_content([video_file, message], request_options={"timeout": 600})

                # Extract token usage metadata
                usage_metadata = response.usage_metadata
                if usage_metadata:
                    prompt_token_count = usage_metadata.prompt_token_count
                    output_token_count = usage_metadata.candidates_token_count
                    total_token_count = usage_metadata.total_token_count

                    # Pricing details for the Gemini models
                    if self.model_info == 'gemini-1.5-pro':
                        input_price_per_1M = 2.50  # $2.50 per 1M input tokens
                        output_price_per_1M = 10.00  # $10.00 per 1M output tokens
                    elif self.model_info == 'gemini-1.5-flash':
                        input_price_per_1M = 0.15  # $1.50 per 1M input tokens
                        output_price_per_1M = 0.6  # $6.00 per 1M output tokens
                    elif self.model_info == 'gemini-1.5-flash-8b':
                        input_price_per_1M = 0.075  # $0.80 per 1M input tokens
                        output_price_per_1M = 0.30  # $4.00 per 1M output tokens
                    elif self.model_info == 'gemini-2.0-flash-exp':
                        input_price_per_1M = 0.15  # $1.50 per 1M input tokens
                        output_price_per_1M = 0.6  # $6.00 per 1M output tokens
                    # Calculate costs
                    input_cost = (prompt_token_count / 1_000_000) * input_price_per_1M
                    output_cost = (output_token_count / 1_000_000) * output_price_per_1M
                    total_cost = input_cost + output_cost

                    # Print token usage and costs
                    print(f"Prompt Tokens: {prompt_token_count}, Output Tokens: {output_token_count}, Total Tokens: {total_token_count}")
                    print(f"Input Cost: ${input_cost:.6f}, Output Cost: ${output_cost:.6f}, Total Cost: ${total_cost:.6f}")

                return response.text
            except Exception as e:
                print(e)

                return "[ERROR]"

        elif self.model_info.startswith('gemini-3.1-pro'):
            # Frames + text directly, no video upload. Uses the current
            # google.genai SDK (google.generativeai is fully deprecated).
            try:
                if self.inference_type == 'cot':
                    message += " Let's think step-by-step."

                from google import genai as genai_new
                from google.genai import types

                gemini_client = genai_new.Client(api_key=os.environ.get("GEMINI_KEY", ""))
                parts = [
                    types.Part.from_bytes(data=base64.b64decode(frame), mime_type="image/jpeg")
                    for frame in (img_frames or [])
                ]
                parts.append(types.Part.from_text(text=message))

                response = gemini_client.models.generate_content(
                    model=self.model_info,
                    contents=[types.Content(role="user", parts=parts)],
                    config=types.GenerateContentConfig(
                        system_instruction="You are a socially intelligent agent. Be concise.",
                    ),
                )
                return response.text or ""
            except Exception as e:
                print(e)
                return "[ERROR]"

        elif self.model_info.startswith('gpt-realtime'):
            # Frames + text over the WebSocket, no audio.
            try:
                if self.inference_type == 'cot':
                    message += " Let's think step-by-step."

                return asyncio.run(
                    _gpt_realtime_ws_call(self.model_info, message, img_frames or [])
                )
            except Exception as e:
                print(e)
                return "[ERROR]"

        elif self.model_info in ['gpt-4v', 'gpt-4o', 'gpt-4o-mini', 'o1', 'o1-mini']:
            try:
                # CoT
                if self.inference_type == 'cot':
                    message += " Let's think step-by-step."

                # self.messages.append({"role": "user", "content": message})

                if self.inference_type != 'lang-only':
                    response = client.chat.completions.create(
                        model=self.model_info,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "These are the frames from the video."
                                    },
                                    *[
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f'data:image/jpg;base64,{frame}',
                                                "detail": "low"
                                            }
                                        } for frame in img_frames
                                    ],
                                    # {
                                    #     "type": "text",
                                    #     "text": f"The audio transcription is: {transcription}"
                                    # },
                                    {
                                        "type": "text",
                                        "text": message
                                    }
                                ]
                            }
                        ],
                        # temperature=0,
                    )

                if self.inference_type == 'lang-only':
                    response = client.chat.completions.create(
                        model=self.model_info,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": message
                                    }
                                ]
                            }
                        ],
                        # temperature=0,
                    )

                # Append the assistant's response
                self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
                
                # Token usage details
                token_usage = response.usage
                input_tokens = token_usage.prompt_tokens
                output_tokens = token_usage.completion_tokens
                total_tokens = token_usage.total_tokens
                
                # Pricing based on model
                if self.model_info == 'gpt-4o':
                    input_cost_per_token = 0.0025 / 1000  # $2.50 per 1M tokens
                    output_cost_per_token = 0.01 / 1000   # $10 per 1M tokens
                elif self.model_info == 'gpt-4o-mini':
                    input_cost_per_token = 0.00015 / 1000  # $0.15 per 1M tokens
                    output_cost_per_token = 0.0006 / 1000  # $0.60 per 1M tokens
                else:
                    input_cost_per_token = 0  # Define for other models if needed
                    output_cost_per_token = 0

                # Calculate costs
                input_cost = input_tokens * input_cost_per_token
                output_cost = output_tokens * output_cost_per_token
                total_cost = input_cost + output_cost

                print(f"Input Tokens: {input_tokens}, Output Tokens: {output_tokens}, Total Tokens: {total_tokens}")
                print(f"Estimated Cost: ${total_cost:.4f}")

                return response.choices[0].message.content
            
            except Exception as e:
                print(e)
                pdb.set_trace()
                return "[ERROR]"

        elif self.model_info in ['DeepSeek-R1']:
            try:
                # # CoT
                # if self.inference_type == 'cot':
                #     message += " Let's think step-by-step."

                # messages = [
                #     {
                #         "role": "user",
                #         "content": message
                #     }
                # ]

                # response = self.model.chat.completions.create(
                #     model="deepseek-ai/DeepSeek-R1", 
                #     messages=messages, 
                #     max_tokens=500
                # )

                # print(response.choices[0].message)

                messages = [
                    {
                        "role": "user",
                        "content": message
                    }
                ]

                response = self.model(messages)

                pdb.set_trace()
                return response.choices[0].message.content
            
            except Exception as e:
                print(e)
                pdb.set_trace()
                return "[ERROR]"
        elif self.model_info == 'video-chat2':
            num_frame = 1
            resolution = 224
            hd_num = 1
            padding = False
            vid, msg = load_video(
                vid_path, num_segments=num_frame, return_msg=True, resolution=resolution,
                hd_num=hd_num, padding=padding
            )

            T_, C, H, W = vid.shape
            video = vid.reshape(1, T_, C, H, W).to("cuda:0")

            img_list = []
            with torch.no_grad():
                image_emb, _, _ = self.model.encode_img(video, message)

            chat = EasyDict({
                "system": "",
                "roles": ("[INST]", "[/INST]"),
                "messages": [],
                "sep": ""
            })

            # chat.messages.append([chat.roles[0], "<Video><VideoHere></Video> [/INST]"])
            chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video> {message} [/INST]"])
            # ask("Describe the video in details.", chat)

            llm_message = answer(conv=chat, model=self.model, do_sample=False, img_list=img_list, max_new_tokens=32, print_res=True)[0]
            return llm_message
        
        elif self.model_info == 'kangaroo':
            try:
                # Ensure attention mask is properly handled
                out, history = self.model.chat(
                    video_path=vid_path,
                    query=message,
                    tokenizer=self.tokenizer,
                    max_new_tokens=256,
                    eos_token_id=self.terminators,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    # Add attention_mask handling if needed
                    attention_mask=None,  # Will be automatically created by the model
                )
                return out
            
            except Exception as e:
                print(f"Error in chat method: {str(e)}")
                return "[ERROR]"

        elif 'llava_video_next' in self.model_info:

            try:
                conversation = [
                    {

                        "role": "user",
                        "content": [
                            {"type": "text", "text": message},
                            {"type": "video"},
                            ],
                    },
                ]

                prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                container = av.open(vid_path)

                # Sample uniformly 8 frames from the video
                total_frames = container.streams.video[0].frames
                indices = np.arange(0, total_frames, total_frames / 8).astype(int)
                clip = read_video_pyav(container, indices)

                # Process the video and text inputs
                inputs_video = self.processor(
                    text=prompt, 
                    videos=clip, 
                    padding=True, 
                    return_tensors="pt"
                ).to(self.model.device)

                # Generate the output
                output = self.model.generate(**inputs_video, max_new_tokens=100, do_sample=False)
            
                return self.processor.decode(output[0][2:], skip_special_tokens=True)
            except:
                return "[ERROR]"

        
        elif self.model_info == 'llava_interleave_qwen':
            try:
                # Sample frames from video
                cap = cv2.VideoCapture(vid_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_indices = np.linspace(0, total_frames-1, 5, dtype=int)  # Sample 5 frames
                
                frames = []
                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                        frames.append(pil_image)
                
                cap.release()

                # Prepare conversation format
                conversation = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": message},
                        *[{"type": "image"} for _ in range(len(frames))]
                    ],
                }]
                
                # Get prompt
                prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                
                # Process frames
                inputs = self.processor(
                    images=frames, 
                    text=prompt, 
                    return_tensors='pt'
                ).to(self.model.device)
                
                # Generate response
                output = self.model.generate(
                    **inputs, 
                    max_new_tokens=200, 
                    do_sample=False
                )
                
                return self.processor.decode(output[0][2:], skip_special_tokens=True)
            
            except Exception as e:
                print(e)
                return "[ERROR]"


