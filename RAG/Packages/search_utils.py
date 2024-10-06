import faiss
import numpy as np
import sys
import warnings
from scipy import spatial
sys.path.append("/home/gangguri/World/RAG_World/FAISS/embed")
sys.path.append("/home/gangguri/World/RAG_World")
from sentence_transformers import SentenceTransformer

# FutureWarning을 무시
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

filepath = "../embed/"

def collect_texts(filename, idxs, bound, text_sources):
    # size of chunk = 500 > size of string = 250
    # pdf 1page maximum 1200 string
    # upside 1000 string, downside 1000 string > upside 4 chunk, downsize 4 chunk
    # >> collect until 2000 string or 4000 chunk size
    newChunks = [[] for _ in range(len(idxs))]
    texts = np.load(filepath+filename+"_texts.npy")
    halfBound = int(bound)//2

    for index, _ in enumerate(newChunks):
        idx = idxs[index]
        doc_name = text_sources[idx]
        upside = ""
        while len(upside) <= halfBound:
            if doc_name != text_sources[idx]:
                break
            upside+=texts[idx]
            idx+=1

        idx = idxs[index] - 1
        downside = ""
        while len(downside) <= halfBound:
            if doc_name != text_sources[idx]:
                break
            downside+=texts[idx]
            idx-=1
        
        newChunks[index]+=downside
        newChunks[index]+=upside

    return newChunks

def collect_chunks(filename, idxs, bound = 2000):
    # size of chunk = 500 > size of string = 250
    # pdf 1page maximum 1200 string
    # upside 1000 string, downside 1000 string > upside 4 chunk, downsize 4 chunk
    # >> collect until 2000 string or 4000 chunk size
    newChunks = [[] for _ in range(len(idxs))]
    chunks = np.load(filepath+filename+"_chunks.npy")
    halfBound = int(bound)//2

    for index, _ in enumerate(newChunks):
        idx = idxs[index]
        upside = ""
        while True:
            if idx >= len(chunks) or (len(upside)+len(chunks[idx])) >= halfBound:
                break
            upside+=chunks[idx]
            idx+=1

        idx = idxs[index] - 1
        downside = ""
        while True:
            if idx < 0 or (len(downside)+len(chunks[idx])) >= halfBound:
                break
            downside+=chunks[idx]
            idx-=1
        
        newChunks[index]+=downside
        newChunks[index]+=upside

    return newChunks

def collect_chunks_(chunks, idxs, bound = 2000):
    # size of chunk = 500 > size of string = 250
    # pdf 1page maximum 1200 string
    # upside 1000 string, downside 1000 string > upside 4 chunk, downsize 4 chunk
    # >> collect until 2000 string or 4000 chunk size
    newChunks = [[] for _ in range(len(idxs))]
    halfBound = int(bound)//2

    for index, _ in enumerate(newChunks):
        idx = idxs[index]
        upside = ""
        while True:
            if idx >= len(chunks) or (len(upside)+len(chunks[idx])) >= halfBound:
                break
            upside+=chunks[idx]
            idx+=1

        idx = idxs[index] - 1
        downside = ""
        while True:
            if idx < 0 or (len(downside)+len(chunks[idx])) >= halfBound:
                break
            downside+=chunks[idx]
            idx-=1
        
        newChunks[index]+=downside
        newChunks[index]+=upside

    return newChunks


def search_vector(filename, query):
    filepath = "./embed/"
    # pltname = str(sys.argv[2])
    print("Queries:", query)

    # Sentence-BERT 모델 로드
    embedModel = SentenceTransformer("sentence-transformers/stsb-xlm-r-multilingual")

    # 저장된 FAISS 인덱스 불러오기
    index = faiss.read_index(filepath+filename+".faiss")
    index.hnsw.edConstruction = 40
    index.hnsw.efSearch = 64
    index.hnsw.search_bounded_queue = True

    # 저장된 임베딩 벡터와 출처 불러오기
    text_sources = np.load(filepath+filename+"_text_sources.npy")

    # 각 쿼리에 대해 반복 실행
    print(f"\nProcessing Query: {query}")
    # 검색할 쿼리 벡터 생성
    query_fv = embedModel.encode(query).reshape(1, -1)
    # FAISS를 사용한 검색
    k = 6
    D, I = index.search(query_fv, k)

    # chunking newly
    idxs=[]
    for idx in I[0]:
        pdf_name = text_sources[idx]
        print(f"Vec{idx} 출처 PDF 파일: {pdf_name}")
        idxs.append(idx)

    newChunks = collect_chunks(filename, idxs, 2000, text_sources)

    return newChunks

def search_vector_index(index, chunks, query):
    print("Queries:", query)

    # Sentence-BERT 모델 로드
    embedModel = SentenceTransformer("sentence-transformers/stsb-xlm-r-multilingual")
    query_fv = embedModel.encode(query).reshape(1, -1)
    # FAISS를 사용한 검색
    k = 6
    D, I = index.search(query_fv, k)

    # idx가 index안에 있는 벡터중 top-k인 벡터의 인덱스이고
    # embeddings랑 인덱스 순서가 같음
    result = []
    for idx in I[0]:
        result.append(chunks[idx])

    return result

# 목적: 잘의에서 문서를 찾기
# 문서 안에서 질의 검색을 하기 위해 먼저 질의와 가장 연관된 문서를 먼저 찾기 위함
# 문서 인덱스들 반환
def search_doc(embedModel, query, index):
    print("finding doc~")
    query_embedded = embedModel.encode(query).reshape(1, -1)

    k = 6
    D, I = index.search(query_embedded, k)

    idxs = set()
    for idx in I[0]:
        idxs.add(idx)

    idxs_list = list(idxs)
    return idxs_list, query_embedded

# 각 문서에서 top-k를 추리고 
# 전체 후보벡터에서 다시 유사도로 top-k를 추림
# 최종 추려진 벡터 인덱스 반환
# 반환된 인덱스로 text_sources에서 해당 청크를 찾아가면 됨
def search_vec_in_doc(query_embedded, index_paths):
    # index_paths: 문서들의 index가 있는 주소들을 알려줌
    # 각 문서마다 다른 인덱스를 유지하고 있고 그 안에서만 찾음

    k = 6
    d = [[] for _ in range(len(index_paths))]
    where = [[] for _ in range(len(index_paths))]
    for path_idx, index_path in enumerate(index_paths):
        index = faiss.read_index(index_path)
        D, I = index.search(query_embedded, k)

        for idx, dis in enumerate(D[0]):
            d[path_idx].append(dis)
            where[path_idx].append(I[0][idx])
    
    # 어디 인덱스의 어디 위치에 있는지
    result = []
    for few in range(k):
        # 가장 가까운 벡터 찾기
        here = [] # (어디 인덱스, 어느 위치)
        close = d[0][0]
        for i, tmp_d in enumerate(d):
            for n in range(len(tmp_d)):
                if close <= tmp_d[n]:
                     continue
                close = tmp_d[n]
                here = []
                here.append(i)
                here.append(n)
        # 매우 큰 값을 줘서 다음에 다시 찾기 방지
        d[here[0]][here[1]] = 100000 
        tmp = []
        tmp.append(here[0])
        tmp.append(here[1])
        result.append(tmp)
    
    # result = [[어느 인덱스, 어느 위치], ..., [어느 인덱스, 어느 위치]]
    # len(result) = k
    return result

def search_vec_in_doc_v2(query_embedded, index_paths):
    # index_paths: 문서들의 index가 있는 주소들을 알려줌
    # 각 문서마다 다른 인덱스를 유지하고 있고 그 안에서만 찾음

    k = 6
    candidates = []  # 전체 후보 벡터를 저장할 리스트

    for path_idx, index_path in enumerate(index_paths):
        index = faiss.read_index(index_path)
        D, I = index.search(query_embedded, k)

        for distance, idx in zip(D[0], I[0]):
            candidates.append((distance, path_idx, idx))

    # 전체 후보 벡터를 거리 기준으로 정렬
    candidates.sort(key=lambda x: x[0])

    # 상위 k개의 벡터 선택
    result = []
    for i in range(min(k, len(candidates))):
        distance, idx_num, idx_pos = candidates[i]
        result.append([idx_num, idx_pos])

    return result

def get_index(filename):
    index = faiss.read_index(filename)
    return index
        
def search_vec_in_doc_v3(query_embedded, index_paths):
    # index_paths: 문서들의 index가 있는 주소들을 알려줌
    # 각 문서마다 다른 인덱스를 유지하고 있고 그 안에서만 찾음

    k = 6
    candidates = []  # 전체 후보 벡터를 저장할 리스트

    for path_idx, index_path in enumerate(index_paths):
        index = faiss.read_index(index_path)
        D, I = index.search(query_embedded, k)

        # 각 인덱스에서 검색한 결과를 출력
        print(f"인덱스 {path_idx}에서 검색한 결과:")
        for idx_in_k in range(len(D[0])):
            distance = D[0][idx_in_k]
            idx = I[0][idx_in_k]
            print(f"  순위 {idx_in_k + 1}: 거리 = {distance}, 인덱스 내 위치 = {idx}")

            # 후보 벡터 리스트에 추가
            candidates.append((distance, path_idx, idx))

    # 전체 후보 벡터를 거리 기준으로 정렬
    candidates.sort(key=lambda x: x[0])

    # 정렬된 전체 후보 벡터 출력
    print("\n전체 후보 벡터를 거리 기준으로 정렬한 결과:")
    for rank, (distance, idx_num, idx_pos) in enumerate(candidates, start=1):
        print(f"  전체 순위 {rank}: 거리 = {distance}, 인덱스 = {idx_num}, 인덱스 내 위치 = {idx_pos}")

    # 상위 k개의 벡터 선택
    result = []
    for i in range(min(k, len(candidates))):
        distance, idx_num, idx_pos = candidates[i]
        result.append([idx_num, idx_pos])

    return result


import requests
def infer_to_api(query, url="http://172.21.3.60:30050/generate", max_tokens=50, min_tokens=10, temperature=0, top_k=5, use_beam_search=False, best_of=5):
    data = {
        "prompt": query,
        "max_tokens": max_tokens,
        "min_tokens": min_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "use_beam_search": use_beam_search,
        "best_of": best_of
    }

    response = requests.post(url, json=data)
    return response.json()