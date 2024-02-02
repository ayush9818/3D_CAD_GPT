import open_clip
import numpy as np 
import os 
from tqdm import tqdm
import torch
import sys
import faiss
import json


class TextTo2dRetrieval:
    """
    Works on Batches
    Input: Batch of Text 
    Output: Batch of Topk Image Paths
    """
    def __init__(self, db_path, image_path, topk=5, device='cuda:2'):
        print(f"Loading Faiss Index")
        self.index = faiss.read_index(db_path)
        self.image_list = np.array(json.load(open(image_path)))
        self.topk = topk
        self.device = device
        self.open_clip_model, _ = self.load_open_clip()
        
    @torch.no_grad()
    def extract_text_feat(self, texts):
        text_tokens = open_clip.tokenizer.tokenize(texts).to(self.device)
        return self.open_clip_model.encode_text(text_tokens).detach().cpu()

    def load_open_clip(self):
        print("loading OpenCLIP model...")
        open_clip_model, _, open_clip_preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', 
                                                                                         pretrained='laion2b_s39b_b160k',
                                                                 cache_dir="/home/aagarwal/kaiming-fast-vol/workspace/open_clip_model/")
        open_clip_model = open_clip_model.to(self.device).eval()
        return open_clip_model, open_clip_preprocess

    def fetch_topk_images(self,texts):
        """
        Input: Text in batch
        Output: similarity_scores, topk Image Paths
        """
        text_feats = self.extract_text_feat(texts)
        query_embedding = np.array(text_feats, dtype=np.float32)
        distances, indices = self.index.search(query_embedding, self.topk)

        return distances, self.image_list[indices]



if __name__ == "__main__":
    ttr = TextTo2dRetrieval(
        db_path = '/home/aagarwal/openShape/data/imagenet/db/imagenet_emb.index',
        image_path = '/home/aagarwal/openShape/data/imagenet/db/imagenet_emb_meta.json'
    )

    texts = [
        'a white vase and a glass on a black background',
        'a white vase and a glass on a black background'
    ]
    sim_scores, topk_paths = ttr.fetch_topk_images(texts)