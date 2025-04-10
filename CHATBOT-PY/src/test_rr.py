
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np

load_dotenv()

def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df

def embed_question(question: str, embedder: str = "text-embedding-3-small"):
    embeddings = OpenAIEmbeddings(model=embedder)
    vector_from_query = embeddings.embed_query(question)
    return vector_from_query

def cosine_similarity(vec1, vec2):
    # Calculate dot product
    dot_product = np.dot(vec1, vec2)
    
    # Calculate norm (magnitude) of each vector
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Avoid division by zero
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    
    # Calculate cosine similarity
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity

def calculate_relevancy(df: pd.DataFrame, result_name: str):
    data = {
        "P0": [],
        "P1": [],
        "P2": [],
        "P3": [],
        "Vector P0": [],
        "Vector P1": [],
        "Vector P2": [],
        "Vector P3": [],
        "eg1": [],
        "eg2": [],
        "eg3": [],
        "RR Score": []
    }
    
    # Set threshold for high-quality matches
    score_threshold = 0.5

    for index, row in df.iterrows():
        embedding_question = embed_question(row['P0'])
        embedding1 = embed_question(row['P1'])
        embedding2 = embed_question(row['P2'])
        embedding3 = embed_question(row['P3'])

        data["Vector P0"].append(embedding_question)
        data["Vector P1"].append(embedding1)
        data["Vector P2"].append(embedding2)
        data["Vector P3"].append(embedding3)

        eg1 = cosine_similarity(embedding_question, embedding1)
        eg2 = cosine_similarity(embedding_question, embedding2)
        eg3 = cosine_similarity(embedding_question, embedding3)

        data["P0"].append(row['P0'])
        data["P1"].append(row['P1'])
        data["P2"].append(row['P2'])
        data["P3"].append(row['P3'])

        data["eg1"].append(eg1)
        data["eg2"].append(eg2)
        data["eg3"].append(eg3)

        # Apply threshold - only count scores above the threshold
        valid_scores = [score for score in [eg1, eg2, eg3] if score > score_threshold]
        if valid_scores:
            rr_score = sum(valid_scores) / len(valid_scores)
        else:
            rr_score = 0.0  # If no score exceeds threshold, set RR score to 0
            
        data["RR Score"].append(rr_score)

        print("\nIndex: ", index)
        print("Original Question: ", row['P0'])


    df = pd.DataFrame(data)
    df.to_excel(f"eval/result/{result_name}.xlsx", index=False, engine="openpyxl")
    print(df)

if __name__ == "__main__":
    df = read_csv("eval/Virtual Assistant RAGAS Evaluation - RR TEST.csv")
    print(df.columns)
    calculate_relevancy(df, "RR_Result")
