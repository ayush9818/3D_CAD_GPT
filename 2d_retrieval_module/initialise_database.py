'''
Ayush Agarwal
'''
import os 
from tqdm import tqdm
import argparse
import torch 
import numpy as np 
import json 
import open_clip
from PIL import Image
import faiss

def load_clip_model(device):
    print("loading OpenCLIP model...")
    open_clip_model, _, open_clip_preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k',
                                                         cache_dir="/home/aagarwal/kaiming-fast-vol/workspace/open_clip_model/")
    open_clip_model = open_clip_model.to(device).eval()
    return open_clip_model, open_clip_preprocess

def _extract_class_embeddings(image_list, model, batch_size):
    embedding_list = []
    for idx in range(0, len(image_list), batch_size):
        start_index = idx 
        end_index = min(start_index + batch_size, len(image_list))
        image_batch = torch.tensor(np.array(image_list[start_index : end_index]))
        if torch.cuda.is_available():
            image_batch = image_batch.to('cuda:2')
        with torch.no_grad():
            image_features = model.encode_image(image_batch)

        image_features = image_features.detach().cpu().numpy().tolist()
        embedding_list.extend(image_features)
    return np.array(embedding_list)

    
def extract_embeddings(data_dir, model, open_clip_preprocess, batch_size = 64):
    db_embeddings = []
    path_list = []
    for class_name in tqdm(os.listdir(data_dir)):
        print(f"Extracting Class Embeddings for {class_name}")
        class_dir = os.path.join(data_dir, class_name)
        class_images = []
        for idx, image_name in enumerate(os.listdir(class_dir)):
            image_path = os.path.join(class_dir, image_name)
            image = open_clip_preprocess(Image.open(image_path))
            class_images.append(image)
            path_list.append(image_path)


        class_embeddings = _extract_class_embeddings(class_images, model, batch_size)
        db_embeddings.extend(class_embeddings)
        break
    db_embeddings = np.array(db_embeddings)
    db_embeddings = db_embeddings / np.linalg.norm(db_embeddings, axis=1, keepdims=True)
    return db_embeddings, path_list


def init_database(emb_list, image_list, save_dir, db_name):
    index = faiss.IndexFlatIP(db_embeddings.shape[1])  # L2 distance index
    # Add your embeddings to the Faiss index
    index.add(db_embeddings)

    index_path = os.path.join(save_dir, db_name)
    print(f"Saving Faiss Index at {index_path}")
    faiss.write_index(index, index_path)

    meta_data_name = f"{db_name.split('.')[0]}_meta.json"
    meta_data_path = os.path.join(save_dir,meta_data_name )
    print(f"Saving Image Paths at {meta_data_path}")
    with open(meta_data_path,'w') as f:
        json.dump(image_list, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='path to directory where imagenet images are stored')
    parser.add_argument('db_dir',  help='path to save database index')
    parser.add_argument('--db-name', default='imagenet_emb.index', help='name to save faiss index')
    parser.add_argument('--device', default='cuda:0', help='cuda device to use to extract embeddings')
    parser.add_argument('--batch-size', default=64)

    args = parser.parse_args()

    data_dir = args.data_dir
    db_dir = args.db_dir
    device = args.device
    batch_size = args.batch_size
    db_name = args.db_name
    assert os.path.exists(data_dir), f"Path does not Exist: {data_dir}"
    os.makedirs(db_dir, exist_ok=True)
    

    model, open_clip_preprocess = load_clip_model(device)
    db_embeddings, image_path_list = extract_embeddings(data_dir, model, open_clip_preprocess, batch_size = batch_size)
    init_database(emb_list=db_embeddings, 
                  image_list=image_path_list,
                  save_dir=db_dir,
                  db_name=db_name)
    