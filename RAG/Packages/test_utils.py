import pandas as pd
from datasets import Dataset
import matplotlib.pyplot as plt
import os
import numpy as np
from Packages.basic_utils import max_min_value, get_mean

# 쿼리 셋 설정, 정답 셋 설정, 라벨링
def set_testset():
    dataset = Dataset.from_pandas(pd.read_csv("/home/gangguri/World/RAG_World/FAISS/testset.csv")).map()
    querys = []
    answers = []
    labels = []
    label_set = ["CLEAR", "DARK", "COMPLEX"]
    """
    LABEL_TYPES:
        CLEAR: 문서를 대표하거나, 특정할 수 있을 만한 정확한 질의
        DARK: 문서를 특정하기 모호하고 애매한 질의
        COMPLEX: 두가지 이상의 문서가 필요한 질의로 해당 문서들을 모두 답해야 함
    """

    for data in dataset:
        querys.append(data["query"])
        answers.append(data["answer"])
        labels.append(label_set[int(data["label"])])

    return dataset

# top-k에 얼마나 포함되는지
def include_top_k(files, answer, label):

    return

def is_include(outputs, answer):
    for output in outputs:
        if output == answer:
            return True
    return False

def is_top(outputs, answer):
    if outputs[0] == answer:
        return True
    return False

# k수에 따라 포함되는 확률
# k수에 따른 탐색 시간?
# top1에 포함되는지


# 그래프 그리기
def paint_graph(title, save_dir, xname, yname, x, y, x2, y2):
    print(y)
    print(y2)

    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-')
    plt.plot(x2, y2, marker='o', linestyle='-', color="forestgreen")
    # plt.bar(x, y, width=0.5)

    # 그래프 제목 및 축 레이블 설정
    plt.title(title, fontsize=14)
    plt.xlabel(xname, fontsize=14)
    plt.ylabel(yname, fontsize=14)

    max1, min1 = max_min_value(y)
    
    # y축 범위 설정
    plt.ylim(get_mean(y) - (max1 - min1)*3, get_mean(y) + (max1 - min1)*3)
    # plt.ylim(0.04, 0.16)
    plt.xticks(x)

    # plt.tight_layout()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + title + ".png")

    print(f"{title}.png painted.")
    print("MEAN: ", get_mean(y))
