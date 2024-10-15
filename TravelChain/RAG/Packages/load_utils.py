import os
from langchain_community.document_loaders import UnstructuredExcelLoader, UnstructuredPDFLoader
from langchain_teddynote.document_loaders import HWPLoader
import gethwp
import xml.etree.ElementTree as ET

def hwp_to_text_v2(filename):
    loader = HWPLoader(filename)
    text = loader.load()

    return text[0].page_content

def hwpx_to_text(filename):
    hwp = gethwp.read_hwpx(filename)
    return hwp

def hwp_to_text(file_path):
    try:
        # XML 파서 초기화 (엔티티 처리 설정)
        parser = ET.XMLParser()
        parser.entity.update({'nbsp': ' '})

        # XML 파일 파싱
        tree = ET.parse(file_path, parser=parser)
        root = tree.getroot()

        # 모든 CHAR 요소에서 텍스트 추출
        content = []
        for char in root.iter('CHAR'):
            if char.text:
                content.append(char.text)

        # 추출한 텍스트를 하나의 문자열로 결합
        full_text = ''.join(content)
        return full_text

    except Exception as e:
        text = hwp_to_text_v2(file_path)
        return text

def pdf_to_text(filename):
    loader = UnstructuredPDFLoader(filename)
    text = loader.load()

    return text[0].page_content

def excel_to_text(filename):
    loader = UnstructuredExcelLoader(filename)
    text = loader.load()

    return text[0].page_content

def load_to_text(filename):
    file_extension = str(os.path.splitext(filename)[1])

    text = ""
    if file_extension == ".hwp":
        text = hwp_to_text(filename)
    elif file_extension == ".hwpx":
        text = hwpx_to_text(filename)
    elif file_extension == ".pdf":
        text = pdf_to_text(filename)
    elif file_extension == ".xlsx":
        text = excel_to_text(filename)

    return text

def extract_all_file(path):
    file_paths = []
    for root, directories, files in os.walk(path):
         for filename in files:
            try:
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)
            except PermissionError:
                print(f"권한이 없어 접근할 수 없습니다: {os.path.join(root, filename)}")
    # print(file_paths)
    return file_paths

