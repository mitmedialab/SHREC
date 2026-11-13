# Social Human Robot Embodied Conversation (SHREC) Dataset

<p align="center">
  <img src="https://github.com/mitmedialab/SHREC/blob/main/images/shrec_side.png?raw=true" width="500"/>
</p>

- **Authors**: Dong Won Lee, Yubin Kim, Sooyeon Jeong, Denison Guvenoz, Parker Malachowsky, Louis-Philippe Morency, Cynthia Breazeal, Hae Won Park
- **Collaborators**: Rianna Santra
- **Institutions**: MIT, Purdue University, Carnegie Mellon University
- **License**: [Pending final approval]

## Updates

- Frames for benchmark experiments are now pulled directly from the HF dataset repos (`download_hf_frames()` in `utils.py`) instead of requiring a local video file or a manually pre-populated images directory.
- Added support for two new models: `gpt-realtime-2` and `gemini-3.1-pro-preview`.

## 🧠 SHREC Dataset Summary

The Social Human Robot Embodied Conversation (SHREC) Dataset is a unique, one-of-a-kind large-scale, real-world benchmark designed to evaluate **social reasoning** in **language** and **vision-language models** through physically embodied human-robot interactions (HRI). It contains:

- **~400 real-world interaction videos**
- **~10,000+ trained human annotations**
- Labels for **social errors**, **competencies**, **rationales**, and **corrections**
- Coverage of **seven social attributes** critical for social intelligence

<p align="center">
  <img src="https://github.com/mitmedialab/SHREC/blob/main/images/first_fig_fin.png?raw=true" width="400"/>
</p>

The dataset is split into 3 subsets:

- The **SHREC Wellness Home** subset contains real-world human-robot interaction video data, longitudinal from [Jeong et al. (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11094612/) recordings from an 8-week in-home study with adult participants aged 18–83. Participants engaged with a **socially assistive robot designed to improve psychological well-being**, affect, and readiness for change through evidence-based positive psychology interventions (PPIs).
- The **SHREC Wellness Dorm** subset contains longitudinal, real-world human-robot interaction video data data from [Jeong et al. (2020)](https://ieeexplore.ieee.org/document/9206085), where a **robotic positive psychology coach** was deployed in **MIT student dormitories**. Participants engaged in daily wellbeing sessions with the robot over the course of 1–4 weeks.
- The **SHREC Empathic** subset contains real-world human-robot interaction video data from [Shen et al. (2024)](https://aclanthology.org/2024.findings-acl.268.pdf), collected over a month-long deployment of social robots in participants’ homes, as participants engage in natural, **empathic storytelling interactions with a social robot**.

It supports research in rapport-building, mental health intervention, and social reasoning in intimate, longitudinal HRI settings.

## 💾 Download Dataset from HuggingFace

To downloadload this dataset from HuggingFace, then load into a pandas df:

#### For Wellness Home

<pre>
import pandas as pd
from datasets import load_dataset
import pandas as pd
import glob
import os
from huggingface_hub import snapshot_download

repo_id = "MIT-personal-robots/shrec_wellness_home"
local_dir = "shrec_wellness_home"

snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir, token = True)

parquet_files = glob.glob(f"{local_dir}/**/*.parquet", recursive=True)

df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

</pre>

Change repo_id (and local_dir) accordingly for different subsets:

- **HF Repo ID:** "MIT-personal-robots/shrec_wellness_home"
- **HF Repo ID:** "MIT-personal-robots/shrec_wellness_dorm",
- **HF Repo ID:** "MIT-personal-robots/shrec_empathic"

### Building the CSV for Steps 1 and 2 below

`main_vlm_get_data.py` and `main_vlm_exp.py` read a `.csv`, not the parquet directly. Build it from the downloaded `df` above:

<pre>
import ast

csv_df = df.rename(columns={"ID": "file_name"})
for col in ["Annotations_A", "Annotations_B", "Annotations_C", "frame_paths"]:
    csv_df[col] = csv_df[col].apply(lambda x: str([ast.literal_eval(e) for e in x]) if col.startswith("Annotations") else str(list(x)))

csv_df.to_csv("shrec_wellness_home.csv", index=False)
</pre>

## 🧪 Running SHREC Benchmark Experiments

### 🔑 Environment Setup

Before running experiments, install all necessary dependencies:

```bash
pip install -r requirements.txt
```

If you'd like to use OpenAI or Google Gemini models, ensure these environment variables are set (a `.env` file in the repo root also works, via `python-dotenv`):

```bash
export OPENAI_API_KEY="your-openai-key"
export GEMINI_KEY="your-google-api-key"   # also accepted as GOOGLE_GENAI_API_KEY
```

[VLMEvalKit](https://github.com/open-compass/VLMEvalKit) is only needed for the open-source VLM entries in the Supported Models table below (paligemma, llava_next_llama3, llava_video_qwen2_7b, InternVL2-8B, MiniCPM-V-2_6, Llama-3.2-3B\*, Llama-3.2-11B-Vision-Instruct). Skip this if you're only using the OpenAI/Gemini models.

To install it:

```bash
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .
```

More details and configuration options can be found in their [Quickstart guide](https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Quickstart.md).

---

Then, you can evaluate **LLMs and VLMs** on SHREC tasks by following these steps:

### 🔧 Step 1: Preprocess the Dataset

Run the following script to extract task-specific data from the raw HuggingFace dataset (in `.csv` format). This will create `.pickle` files under `./output_datasets` for each supported task.

```bash
python main_vlm_get_data.py --data_path ../shrec_empathic.csv --data_name shrec_empathic --task_type pre
```

**Arguments**:

- `--data_path`: Path to the HuggingFace-downloaded CSV file.
- `--data_name`: Dataset identifier (e.g., `shrec_empathic`, `shrec_wellness_home`, etc.)
- `--task_type`: Task to extract. Options:
  - `detection`, `attribute`, `rationale`, `correction`
  - `post`, `pre`
  - `attribute_agreed_multiple_subj`, `detection_error_only`

---

### 🚀 Step 2: Run the Model Evaluation

After preprocessing, run the benchmark with the following command:

```bash
python main_vlm_exp.py \
  --context_window 15 \
  --model GPT4o_MINI_Image \
  --data_path ./output_datasets \
  --task_type shrec_empathic_pre.pickle \
  --video \
  --csv_path ../shrec_empathic.csv \
  --hf_repo_id MIT-personal-robots/shrec_empathic
```

**Key Flags**:

- `--context_window`: Number of utterances for context (e.g., 15).
- `--model`: Model to evaluate (see list below).
- `--task_type`: Preprocessed `.pickle` file generated in Step 1.
- `--video`: Include frame-based input (set this for vision-language models).
- `--hf_repo_id`: HF dataset repo to pull frame images from (e.g. `MIT-personal-robots/shrec_empathic`, `MIT-personal-robots/shrec_wellness_home`, `MIT-personal-robots/shrec_wellness_dorm`). Frames for each sample's time window are downloaded on demand directly from the dataset — no local video file or manually-extracted image folder needed.
- `--images_dir`: No longer required for normal use. Kept only as a fallback for older CSV exports that don't have a `frame_paths` column.

---

### 📊 Step 3: Evaluate Model Performance

After running inference, evaluate model predictions using the following steps:

**(a) Parse Model Outputs:**

```bash
python eval_pydantic.py
```

- This extracts predicted answer choices from LLM output files located in `./output/`.
- Outputs are saved into `./output_pydantic/`.

**(b) Compute Accuracy Metrics:**

```bash
python eval.py
```

- This script computes task-specific performance metrics across all models.

---

### 🧠 Supported Models

Below are the models currently supported in the SHREC benchmark pipeline:

| Category         | Model Identifier                                                                                                                                                               |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Open-source VLMs | `paligemma-3b-mix-448`, `llava_next_llama3`, `llava_video_qwen2_7b`, `InternVL2-8B`, `MiniCPM-V-2_6`, `Llama-3.2-3B`, `Llama-3.2-3B-Instruct`, `Llama-3.2-11B-Vision-Instruct` |
| GPT-4o Variants  | `GPT4o_Image`, `GPT4o_MINI_Image`, `GPT4o_Lang`, `GPT4o_MINI_Lang`, `GPT4o_Image_few_shot`, `GPT4o_Image_cot`                                                                  |
| OpenAI Realtime  | `gpt-realtime-2`                                                                                                                                                               |
| Google Gemini    | `gemini-1.5-flash`, `gemini-2.0-flash-exp`, `gemini-1.5-pro`, `gemini-1.5-flash-8b`, `gemini-3.1-pro-preview`                                                                  |
| Others           | `o1`, `o1-mini`, `llava_video_next`, `llava_video_next_7b_dpo`, `DeepSeek-R1-Distill-Qwen-32B`                                                                                 |

Each model is loaded via a unified interface. For GPT models and Gemini, `utils_gpt.py` provides consistent handling of prompt strategies (`zero-shot`, `few-shot`, `cot`, etc.).

## 📦 Dataset Structure

Each interaction sample includes:

- `video_id`: Identifier for the interaction session
- `frame_paths`: List of image paths (15 selected frames from the video)
- `transcript`: Multi-turn dialogue between user and robot
- `label`: `"competence"`, `"error"`, or `"none"`
- `social_attributes`: List of relevant attributes from 7 core categories
- `rationale`: Explanation for the error or competence
- `correction`: Suggested repair if the segment is an error

### 🧪 SHREC Task Overview

<p align="center">
  <img src="https://github.com/mitmedialab/SHREC/blob/main/images/tasks_fig.png?raw=true" width="1000"/>
</p>

#### 1. Detecting Social Behavior

| Task                                    | Description                                                              | `task_type` argument   |
| --------------------------------------- | ------------------------------------------------------------------------ | ---------------------- |
| **Error / Competence / None Detection** | Classify the robot’s behavior as a social error, competence, or neither. | `detection`            |
| **Error Detection**                     | Determine whether a given behavior constitutes a social error.           | `detection_error_only` |

#### 2. Identifying Social Attributes

| Task                                | Description                                                                     | `task_type` argument             |
| ----------------------------------- | ------------------------------------------------------------------------------- | -------------------------------- |
| **Social Attribute Identification** | Identify which of the seven social attributes are relevant to a given behavior. | `attribute`                      |
| **Multiple Attribute Detection**    | Determine whether multiple social attributes are present in the behavior.       | `attribute_agreed_multiple_subj` |

**Seven Social Attributes**:

- **Emotions** – Identifying and responding to emotional expressions
- **Engagement** – Monitoring user interest and presence
- **Conversational Mechanics** – Managing turn-taking, timing, and pauses
- **Knowledge State** – Tracking shared knowledge and references
- **Intention** – Inferring the goals or motives behind actions
- **Social Context & Relationships** – Acting appropriately based on context and social role
- **Social Norms & Routines** – Following culturally appropriate social conventions

#### 3. Understanding Interaction Flow

| Task                         | Description                                                                       | `task_type` argument |
| ---------------------------- | --------------------------------------------------------------------------------- | -------------------- |
| **Pre-Condition Reasoning**  | Given the robot’s utterance, choose the plausible user behavior that came before. | `pre`                |
| **Post-Condition Reasoning** | Given the user’s utterance, select the robot’s likely follow-up behavior.         | `post`               |

These tasks are structured as **multiple-choice questions**, with distractors sampled from real robot-user interactions.

#### 4. Rationalizing & Correcting Social Errors

| Task                      | Description                                                                        | `task_type` argument |
| ------------------------- | ---------------------------------------------------------------------------------- | -------------------- |
| **Rationale Selection**   | Choose the correct explanation for why the robot’s behavior was an error.          | `rationale`          |
| **Correction Suggestion** | Select the most appropriate corrective action the robot should have taken instead. | `correction`         |

These tasks evaluate both **diagnostic** (understanding what went wrong) and **prescriptive** (knowing how to fix it) reasoning abilities.

## 🔍 Example Sample

```json
{
  "ID": "P15_s002-006",
  "sample_frame": "P15_s002-006/0000.png",
  "transcript": "AI Agent: (00:00:02) Hey there. How was your day today?\nUser A: (00:00:04) Good. How was yours?\n...\nAI Agent: (00:10:42) ... brighten our days.",
  "Annotations_A": [
    {
      "timestamp": {"start": 7.21, "end": 20.23},
      "error": true,
      "source": {"Verbal": true, "Non-Verbal": false},
      "attribute": {
        "Conversational Mechanics": true,
        "Intention": false,
        "Emotions": false,
        "Engagement": false,
        "Knowledge State": false,
        "Social Context &  Relationships",
        "Social Norms & Routines"
      },
      "rationale": "Delayed response and failure to understand participant.",
      "correction": "Should have responded within 2–3 seconds."
    }
  ],
  "Annotations_B": [
    {
      "..."
    }
  ],
  "Annotations_C": [
    {
      "..."
    }
  ],
  "framerate": 15.0,
  "frame_paths": [
    "P15_s002-006/0000.png",
    "P15_s002-006/0013.png",
    "P15_s002-006/0034.png",
    "..."
  ]
}

```
