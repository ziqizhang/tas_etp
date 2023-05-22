'''
Uses BERT sentence transformer to encode texts
'''
import sys
from sentence_transformers import SentenceTransformer, util

def embed_sentences(model, sentences:list):
    embeddings = model.encode(sentences)
    return embeddings

def load_model(model_name_or_path):
    model = SentenceTransformer(model_name_or_path)
    return model

if __name__ == "__main__":
    sentences=["Facebook Ireland Ltd","Google Inc"]
    mod=load_model(sys.argv[1])
    print("model loaded")
    embeddings=embed_sentences(mod, sentences)
    cosine_scores = util.cos_sim(embeddings, embeddings)
    print(cosine_scores)
    print("done")