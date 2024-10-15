from huggingface_hub import login
from sentence_transformers import SentenceTransformer
import os

def flat_to_string(str_list):
    flattened_data = [item for sublist in str_list for item in sublist]
    result_string = ''.join(flattened_data)
    result_string = result_string.replace('\\n', '\n')

    return result_string

oldEmbedModelID = "sentence-transformers/stsb-xlm-r-multilingual"
def get_embedModel(model_id="jhgan/ko-sroberta-multitask"):
    
    embedModel = SentenceTransformer(model_id, local_files_only=True)

    return embedModel

def get_only_filename(file_path, getForm=False):
    file_name_tmp = os.path.basename(file_path)
    file_name = str(os.path.splitext(file_name_tmp)[0])

    if getForm:
        file_name += str(os.path.splitext(file_name_tmp)[1])
    
    return file_name

def print_distance(D):
    for i in range(len(D[0])):
        print(f"Vec{i}'s distance: {D[0][i]}")

# 해당하는 확장자만 파일리스트에서 포함시킴
def filter_by_extension(file_list, allowed_extensions):
    filtered_files = []
    for file in file_list:
        if file.lower().endswith(tuple(allowed_extensions)):
            filtered_files.append(file)
    return filtered_files

def max_min_value(values:list):
    max = -100000
    min = 10000000
    for value in values:
        if max < value:
            max = value
        if min > value:
            min = value
    
    return max, min

def get_mean(values:list):
    sum = 0
    for value in values:
        sum += value

    return sum / len(values)