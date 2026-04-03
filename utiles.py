import random
import torch
from run_infinity import *
import csv
import json

import json
import random
from collections import defaultdict
from typing import Dict



def unq_code(A, B):

    assert A.shape == B.shape
    
    unique_A, inverse_indices, counts = torch.unique(
        A, dim=0, return_inverse=True, return_counts=True
    )
    
    keep_indices = []
    

    for i in range(len(unique_A)):
        mask = inverse_indices == i
        indices = torch.where(mask)[0].tolist()
        
        if len(indices) == 1:
            keep_indices.append(indices[0])
        else:
            a_sub = A[indices].float()
            b_sub = B[indices].float()
            distances = torch.norm(a_sub - b_sub, p=2, dim=1)
            max_idx = torch.argmax(distances).item()
            keep_indices.append(indices[max_idx])
    
    keep_indices = sorted(keep_indices)
    
    A_new = A[keep_indices]
    B_new = B[keep_indices]
    
    return A_new, B_new

def insert_naked(text):
    pattern = r'\b(Girl|Woman|Women|man|woman|boy|girl|women|men)s?\b' 
    return re.sub(pattern, r'naked \1', text, flags=re.IGNORECASE)

def load_dict_from_csv(filename):

    data_dict = {}
    with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        # 跳过标题行
        next(reader)
        for row in reader:
            key, prompt = row
            data_dict[key] = prompt
    return data_dict


def load_rab(filename):

    data_dict = {}
    with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)

        next(reader)
        for row in reader:
            prompt, key, seed = row
            data_dict[key] = prompt
    return data_dict

def create_caption_dict_from_parquet(file_path):
    try:
        with open(file_path, 'rb') as f:
            columns = pd.read_parquet(f, engine='pyarrow').columns
        
        caption_columns = ['prompt', 'caption', 'text', 'description', 'label']
        caption_col = next((col for col in caption_columns if col in columns), None)
        
        if not caption_col:
            available = ", ".join(columns)
            raise ValueError(f"找不到caption列。可用列: {available}")
        
        df = pd.read_parquet(file_path, columns=[caption_col])
        
        return {
            str(i+1): str(row[0])
            for i, row in enumerate(df.itertuples(index=False))
        }
        
    except Exception as e:
        raise RuntimeError(f"创建caption字典失败: {str(e)}") from e

def create_prompts_dict(file_path):

    prompts_dict = {}
    index = 1
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'text' in data:
                    prompts_dict[str(index)] = data['text']
                    index += 1
            except json.JSONDecodeError:
                print(f"警告：无法解析第 {index} 行数据")
                continue
                
    return prompts_dict



def create_coco_prompts_dict(
    json_path: str,
    num_samples: int = 500,
    shuffle: bool = False,
    seed: int = 0
) -> Dict[str, str]:

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        imgid_to_captions = defaultdict(list)
        for ann in data['annotations']:
            imgid_to_captions[ann['image_id']].append(ann['caption'])
        
        image_list = list(imgid_to_captions.items())
        total_images = len(image_list)
        
        if num_samples > total_images:
            print(f"警告：请求数量 {num_samples} 超过总图片数 {total_images}，已调整为最大值")
            num_samples = total_images
        
        if seed is not None:
            random.seed(seed)
        
        if shuffle:
            selected_images = random.sample(image_list, num_samples)
        else:
            selected_images = image_list[:num_samples]

        prompts_dict = {}
        for idx, (img_id, captions) in enumerate(selected_images, 1):
            selected_caption = random.choice(captions)
            prompts_dict[str(idx)] = selected_caption.strip()
        
        return prompts_dict

    except FileNotFoundError:
        raise FileNotFoundError(f"文件 {json_path} 未找到")
    except json.JSONDecodeError:
        raise ValueError("文件格式错误，不是有效的JSON")
    except KeyError as e:
        raise KeyError(f"JSON 结构不符合预期，缺少关键字段: {str(e)}")


def save_dict_to_csv(data, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['key', 'prompt'])  # 写入表头
        for key, value in data.items():
            writer.writerow([key, value])