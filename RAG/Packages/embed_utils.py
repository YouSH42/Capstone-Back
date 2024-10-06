# python embedding.py 10 "no_sum_1" 0
import numpy as np
import faiss
import os
import warnings
# FutureWarning을 무시
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from unstructured.partition.pdf import partition_pdf
from vllm import LLM, SamplingParams
import glob
from Packages.load_utils import load_to_text
import sys
sys.path.append("/home/gangguri/World/RAG_World/FAISS")

embed_path = "./embed/"
model_id = "jungyuko/DAVinCI-42dot_LLM-PLM-1.3B-v1.5.3"

from semantic_text_splitter import TextSplitter
def chunk_semantic(text, max_size=200):
    splitter = TextSplitter(max_size)
    chunks = splitter.chunks(text)

    return chunks

from kiwipiepy import Kiwi
def chunk_semantic_kiwi(text):
    kiwi = Kiwi()
    chunks_tmp = kiwi.split_into_sents(text)

    chunks = list()
    for chunk in chunks_tmp:
        chunks.append(chunk.text)

    return chunks

ffpath = "/home/gangguri/World/RAG_World/FAISS/pdfs4/"
def extract_file_list(fpath=ffpath):
    file_list = []
    pdf_files = glob.glob(os.path.join(fpath, "*.pdf"))
    hwp_files = glob.glob(os.path.join(fpath, "*.hwp"))
    hwpx_files = glob.glob(os.path.join(fpath, "*.hwpx"))
    xlsx_files = glob.glob(os.path.join(fpath, "*.xlsx"))

    file_list.extend(pdf_files)
    file_list.extend(hwp_files)
    file_list.extend(hwpx_files)
    file_list.extend(xlsx_files)

    return file_list

# 요소를 유형별로 분류
def categorize_elements(raw_pdf_elements):
    """
    PDF에서 추출된 요소를 테이블과 텍스트로 분류합니다.
    raw_pdf_elements: unstructured.documents.elements의 리스트
    """
    texts = []  # 텍스트 저장 리스트
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))  # 텍스트 요소 추가
    return texts


from huggingface_hub import login
import torch

def chunk_text(text, chunk_size=2000):
    """
    text: 입력 텍스트
    chunk_size: 청크로 나눌 문자열 길이 (기본 2000자)
    """
    # 2000자 단위로 텍스트 나누기
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def infer_sum_vllm(text_chunks):
    prompts = []
    prefix = """You are an assistant tasked with summarizing tables and text for retrieval.\
Keep the summary very concise and ensure it does not exceed 100 characters.\
These summaries will be embedded and used to retrieve the raw text or table elements. Table or text:"""

    for chunk in text_chunks:
        prompts.append(prefix+chunk)

    samplingParams = SamplingParams(temperature=0.5, top_p=0.95)
    llm = LLM(
        model=model_id, 
        tokenizer=model_id,
        swap_space=30, 
        gpu_memory_utilization=0.6,
    )
    torch.cuda.empty_cache()
    outputs = llm.generate(prompts, samplingParams)

    prompts = []
    sumsum = ""
    for output in outputs:
        sum = output.outputs[0].text
        sumsum += sum
        torch.cuda.empty_cache()
        if len(sumsum) >= 7500:
            break
        # print("**********************")
        # print(sum)
    print(f"SumSum len: {len(sumsum)}")
    
    prompts.append(prefix+sumsum)
    outputsEnd = llm.generate(prompts, samplingParams)
    print(f"Doc Sum Len: {len(outputsEnd[0].outputs[0].text)}")

    del llm
    torch.cuda.empty_cache()

    return outputsEnd[0].outputs[0].text

def sum_vllm_by_chunk(text_chunks):
    prompts = []
    prefix = """You are an assistant tasked with summarizing tables and text for retrieval.\
Keep the summary very concise and ensure it does not exceed 100 characters.\
These summaries will be embedded and used to retrieve the raw text or table elements. Table or text:"""

    for chunk in text_chunks:
        prompts.append(prefix+chunk)

    samplingParams = SamplingParams(temperature=0.5, top_p=0.95)
    llm = LLM(
        model=model_id, 
        tokenizer=model_id,
        swap_space=30, 
        gpu_memory_utilization=0.6,
    )
    torch.cuda.empty_cache()
    outputs = llm.generate(prompts, samplingParams)

    sum_chunks = []
    for output in outputs:
        sum_chunks.append(output.outputs[0].text)

    return sum_chunks

def summarize_doc(pdf_list, texts, max_cnt):
    final_summary = []

    for idx, pdf in enumerate(pdf_list):
        if idx >= max_cnt:
            break
        print(f"{pdf} split start.")
        text = texts[idx]
        
        text_chunks = chunk_text(text, chunk_size=2000)
        docSum = infer_sum_vllm(text_chunks)

        final_summary.append(docSum)

    return final_summary

def summarize_and_embed_by_chunk(chunks):
    sumChunks = sum_vllm_by_chunk(chunks)

    embedModel = SentenceTransformer("sentence-transformers/stsb-xlm-r-multilingual") 
    embeddings = embedModel.encode(sumChunks)

    index = faiss.IndexHNSWFlat(embeddings.shape[1], 32)
    index.hnsw.efConstruction = 40
    index.add(embeddings)
    index.hnsw.efSearch = 32
    index.hnsw.search_bounded_queue = True

    del embedModel
    torch.cuda.empty_cache()

    return index, embeddings

def extract(pdf_list, max_cnt):
    # 요소 추출
    texts = []
    text_source = []
    cnt = 0
    for pdf in pdf_list:
        if cnt >= max_cnt:
            break
        print(pdf," split start.")
        text = load_to_text(pdf)
        texts.append(text)
        text_source.append(pdf)
        cnt+=1

        print(f"{pdf} split done. len: {len(text)}")
    
    return texts, text_source

from sentence_transformers import SentenceTransformer
def embedding_and_save(chunks, chunk_sources, filename, embed_id = "sentence-transformers/stsb-xlm-r-multilingual", save_embed = False):
    embedModel = SentenceTransformer(embed_id, local_files_only=True) 
    embeddings = embedModel.encode(chunks)
    # print(embeddings)
    M = 32

    print("BERT DIMENSIONS: ", embeddings.shape)
    index = faiss.IndexHNSWFlat(embeddings.shape[1], M)
    index.hnsw.efConstruction = 40
    # index = faiss.IndexFlatL2(d)

    index.add(embeddings)
    index.hnsw.efSearch = 32
    index.hnsw.search_bounded_queue = True

    if not os.path.exists(embed_path):
        os.makedirs(embed_path)

    # FAISS 인덱스와 벡터를 파일로 저장
    faiss.write_index(index, embed_path+filename+".faiss")

    # 임베딩 벡터와 출처를 Numpy로 저장
    np.save(embed_path+filename+"_chunk_sources.npy", np.array(chunk_sources))
    np.save(embed_path+filename+"_chunks.npy", np.array(chunks))
    
    if save_embed:
        np.save(embed_path+filename+"_embbedings.npy", np.array(embeddings))

oldEmbedModelID = "sentence-transformers/stsb-xlm-r-multilingual"
def get_embedModel(model_id="jhgan/ko-sroberta-multitask"):
    login("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")
    embedModel = SentenceTransformer(model_id)

    return embedModel

def get_only_filename(file_path, getForm=False):
    file_name_tmp = os.path.basename(file_path)
    file_name = str(os.path.splitext(file_name_tmp)[0])

    if getForm:
        file_name += str(os.path.splitext(file_name_tmp)[1])
    
    return file_name

# 문서 하나당 하나의 인덱스를 생성
def embed_per_doc(file_paths, embed_dirname, embedModel, chunk_size = 100, useKiwi = False):
    base_path="/home/gangguri/World/RAG_World/FAISS/embed/"
    embed_dir = base_path + embed_dirname
    if not os.path.exists(embed_dir):
        os.makedirs(embed_dir)
    
    # embed_dirname: most include "/" to end.
    for file_path in file_paths:
        text = load_to_text(file_path)
        chunks = list()
        if useKiwi:
            chunks_tmp = chunk_semantic_kiwi(text)
            for chunk_tmp in chunks_tmp:
                chunk_chunk_tmp = chunk_semantic(chunk_tmp, chunk_size)
                chunks.extend(chunk_chunk_tmp)
        else:
            chunks = chunk_semantic(text, chunk_size)

        embeddings = embedModel.encode(chunks)
        M = 32

        print("BERT DIMENSIONS: ", embeddings.shape)
        index = faiss.IndexHNSWFlat(embeddings.shape[1], M)
        index.hnsw.efConstruction = 40
        index.hnsw.efSearch = 32
        index.hnsw.search_bounded_queue = True
        index.add(embeddings)

        # FAISS 인덱스와 벡터를 파일로 저장
        file_name = get_only_filename(file_path)
        faiss.write_index(index, embed_dir+"/"+file_name+".faiss")
        np.save(embed_dir+"/"+file_name+"_chunks.npy", np.array(chunks))
        print(f"{file_name} is embedded all chunks.")

    print(f"Successful embed {len(file_paths)} docs.")

def semantic_embed_query(query, embedModel):
    qChunks = chunk_semantic(query)
    embededQueries = embedModel.encode(qChunks)
    print(len(qChunks))

    return embededQueries

def embed(file_list, save_name, chunk_size = 200):
    chunks = []
    chunk_sources = []
    
    # 모든 텍스트를 청크로 나누고(chunks) 출처를 표시(chunk_sources)
    for file_name in file_list:
        text = load_to_text(file_name)
        chunks_tmp = chunk_semantic(text, chunk_size)
        chunks.extend(chunks_tmp)
        chunk_sources.extend(file_name for i in range(len(chunks_tmp)))

    embedding_and_save(chunks, chunk_sources, save_name, "jhgan/ko-sroberta-multitask", False)
    return