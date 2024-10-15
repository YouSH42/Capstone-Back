import faiss
import json
import os
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from Packages.basic_utils import get_embedModel

# 벡터 db에 담기 위한 인덱싱 작업

# Vector DB 구조
# index
# - embeddings = "위치한 곳:" + areaCode(지역으로 변환) + "카테고리:" + contenttypeid(카테고리로 변환) + summary(overview)
# - areaCode로 인덱스를 나누고, 그 안에서 다시 contenttypeid로 나눔
# - contenttypeid로만 질의가 왔으면, 각 areaCode에서 해당 contenttypeid 인덱스들을 합쳐서 search
#   - index를 미리 사용에 따라 로드 해놓고, api 요청이 오면 맞는 인덱스에서 찾아줘야 할 듯
#   - 인덱스가 areaCode/contenttypeid로 나뉠텐데 모든 인덱스를 각자 로드하고, 필요에 따라 add_with_ids()로 합쳐서 사용해야할 듯
#   - 우려되는 것은 add하는 타임이 얼마나 걸리는지 확인할 필요있음.
#   - 하지만 메모리 측면에서는 가장 최적임. (add하면서 복사가 일어날 수도 있긴함)
# - 인덱스 상세2
#   - index.add로 각 인덱스를 구성하고, 스토리지에 각 인덱스의 index_record 파일을 만들고 id:contentid 로 맵핑해둔다

# TODO
# 1. func. Collector에서 POST 요청하면 받아서 vector DB에 저장
# 2. func. areaCode, contenttypeid에 따라 해당되는 index파일에 저장 (CLEAR)
# 3. func. areaCode, contenttypeid 각각 지역, 카테고리로 번역해주는 함수 필요함 (CLEAR)

embedModel = get_embedModel()

def update_index(jsondata):
    global embedModel
    embeddings = embedding_data(jsondata, embedModel)
    
    areaCode = jsondata["areaCode"]
    contenttypeid = jsondata["conttenttypeid"]
    embed_dir = "./EMBED"
    if not os.path.exists(embed_dir):
        os.makedirs(embed_dir)

    base_name = f"{embed_dir}/{areaCode}_{contenttypeid}_"
    
    # index update
    try:
        index = faiss.read_index(base_name+"index.faiss")
    except:
        index = faiss.IndexHNSWFlat(embeddings.shape[1], 32)
    
    index.add(embeddings)
    last_id = index.ntotal
    faiss.write_index(index, base_name+"index.faiss")
    
    # index_record update
    try:
        with open(base_name+"record.json", "r") as file:
            index_record = json.load(file)
        index_record[last_id] = jsondata["contentid"]
        
        with open(base_name+"record.json", "w") as file:
            json.dump(index_record, file)
    except:
        index_record_init = { last_id : int(jsondata["contentid"]) }
        with open(base_name+"record.json", "w") as file:
            json.dump(index_record_init, file)

def embedding_data(jsondata, embedModel:SentenceTransformer):
    areaCode = convert_areaCode(int(jsondata["areaCode"]))
    contenttype = convert_contenttypeid(int(jsondata["contenttypeid"]))
    before_vec = "위치한 곳:" + areaCode + ", 카테고리:" + contenttype + "\n" + jsondata["summary"]
    
    embeddings = embedModel.encode(before_vec)
    return embeddings, before_vec

def convert_areaCode(areaCode:int) -> str:
    if areaCode == 1:
        return "서울특별시"
    elif areaCode == 2:
        return "인천광역시"
    elif areaCode == 3:
        return "대전광역시"
    elif areaCode == 4:
        return "대구광역시"
    elif areaCode == 5:
        return "광주광역시"
    elif areaCode == 6:
        return "부산광역시"
    elif areaCode == 7:
        return "울산광역시"
    elif areaCode == 8:
        return "세종특별자치시"
    elif areaCode == 31:
        return "경기도"
    elif areaCode == 32:
        return "강원도"
    elif areaCode == 33:
        return "충청북도"
    elif areaCode == 34:
        return "충청남도"
    elif areaCode == 35:
        return "경상북도"
    elif areaCode == 36:
        return "경상남도"
    elif areaCode == 37:
        return "전라북도"
    elif areaCode == 38:
        return "전라남도"
    elif areaCode == 39:
        return "제주특별자치시"

def convert_contenttypeid(contenttypeid:int) -> str:
    if contenttypeid == 12:
        return "관광지"
    elif contenttypeid == 14:
        return "문화시설"
    elif contenttypeid == 15:
        return "축제/공연/행사"
    elif contenttypeid == 25:
        return "여행코스"
    elif contenttypeid == 28:
        return "레저/스포츠"
    elif contenttypeid == 32:
        return "숙박"
    elif contenttypeid == 38:
        return "쇼핑"
    elif contenttypeid == 39:
        return "음식"
