import pickle
from bs4 import BeautifulSoup
import json
import wandb
from langchain_text_splitters import HTMLHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer
import numpy as np
import sys


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")


MIN_TOKENS = 32
MAX_TOKENS = 1024
version = "v0"

with wandb.init(project="dataSynthesis", job_type="extract") as run:


    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
        ("h4", "Header 4"),
    ]

    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_TOKENS,
                                                    chunk_overlap=0)

    artifact = run.use_artifact(f'LaplaceDecoder/dataSynthesis/unimelb_gradhub_rawhtml_binary:{version}', type='data')
    artifact_dir = artifact.download()

    with open(f"./artifacts/unimelb_gradhub_rawhtml_binary:{version}/" + "content_full.pkl", "rb") as f:
        content_full = pickle.load(f)
    
    sys.exit()

    max_toks = 0 
    texts = []
    for label, url, content_html in content_full:
        chunks = html_splitter.split_text(content_html)
        for i in range(len(chunks)):
            chunk = chunks[i]
            ntoks = len(tokenizer.tokenize(chunk.page_content))
            if ntoks > max_toks:
                max_toks = ntoks


            if ntoks >= MIN_TOKENS:
                if ntoks > MAX_TOKENS:
                    subchunks = text_splitter.split_documents([chunk])
                    for subchunk in subchunks:
                        ntoks = len(tokenizer.tokenize(subchunk.page_content))
                        subchunk.metadata.update( {"url": url, "label": label, "ntoks": ntoks} )
                        texts.append(subchunk.to_json())
                else:
                    chunk.metadata.update( {"url": url, "label": label, "ntoks": ntoks} )
                    texts.append(chunk.to_json())
    
    
    with open("text.json", "w") as f:
        json.dump(texts, f, indent=4)
    
    #print(f"Max tokens: {max_toks}")
    median_chunk_size = np.median(list(map(lambda x:x['kwargs']['metadata']['ntoks'], texts)))
    mean_chunk_size = np.mean(list(map(lambda x:x['kwargs']['metadata']['ntoks'], texts)))
    min_chunk_size = np.min(list(map(lambda x:x['kwargs']['metadata']['ntoks'], texts)))
    max_chunk_size = np.max(list(map(lambda x:x['kwargs']['metadata']['ntoks'], texts)))
    print(f"Median chunk size: {median_chunk_size}")
    print(f"Mean chunk size: {mean_chunk_size}")
    print(f"Min chunk size: {min_chunk_size}")
    print(f"Max chunk size: {max_chunk_size}")




    data_artifact = wandb.Artifact("unimelb_gradhub_text_json",
                                    type="data",
                                    description="Text extracted from raw html")

    data_artifact.add_file("text.json")
    data_artifact.save()

