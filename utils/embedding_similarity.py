# embedding_similarity.py

from sentence_transformers import SentenceTransformer, util

def generate_embedding_similarity(text1, text2):

    # Load the pretrained embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Get embeddings
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)

    # Compute cosine similarity
    cosine_score = util.cos_sim(embedding1, embedding2)

    # Print result
    return cosine_score
