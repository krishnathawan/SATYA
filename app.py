import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import json
import streamlit as st

# paste your model path below for generating embeddings 
model_path = r"D:\new_model_directory\content\new_fine_tuned"
model = SentenceTransformer(model_path)

# paste dataset path (csv)
bhagwat_gita_df = pd.read_csv(r"D:\Bhagwad_Gita_Verses_English.csv")
yoga_sutras_df = pd.read_csv(r"D:\Patanjali_Yoga_Sutras_Verses_English.csv")

sanskrit_shlokas = list(bhagwat_gita_df['Sanskrit ']) + list(yoga_sutras_df['Sanskrit '])
bg_translations = bhagwat_gita_df['Swami Adidevananda'].tolist()
bg_chapters = bhagwat_gita_df['Chapter'].tolist()
bg_verse_no = bhagwat_gita_df['Verse'].tolist()
pys_translations = yoga_sutras_df['Translation '].tolist()
pys_chapters = yoga_sutras_df['Chapter'].tolist()
pys_verse_no = yoga_sutras_df['Verse'].tolist()

verse = bg_translations + pys_translations
chapter_no = bg_chapters + pys_chapters
verse_no = bg_verse_no + pys_verse_no

# paste path of saved embeddings of verses 
index = faiss.read_index(r"D:\fine_tuned_embeddings.faiss")

# paste your api key below : 
client = InferenceClient(api_key="...")


st.title("SATYA : Sanatana Atman Tapas Yoga Ananda")
st.write("Ask your question about Bhagavad Gita or Patanjali Yoga Sutras.")


user_query = st.text_input("Enter your query:", "")

if user_query:
    query_embedding = model.encode([user_query]).astype('float32')
    k = 5
    D, I = index.search(query_embedding, k)
    retrieved_shlokas = [sanskrit_shlokas[i] for i in I[0]]
    retrieved_verses = [verse[i] for i in I[0]]
    
    prompt1 = (
        f"Answer in '0' or '1': '1' if any of these verses among {retrieved_verses} are relevant to the query '{user_query}'. "
        "'0' if none of them are related."
    )
    
    messages = [{"role": "user", "content": prompt1}]

    completion = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        messages=messages,
        max_tokens=3,
    )

    response1 = completion.choices[0].message["content"].strip()
    
    if response1 == "'0'":
        response2 = "Sorry! I don't know."
    else:
        prompt2 = (
            f"Query: {user_query}\n"
            f"Give answer of query in long by deeply using most related {retrieved_verses} and your own knowledge too and summarise the answer."
        )
        
        messages = [{"role": "user", "content": prompt2}]

        completion = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            messages=messages,
            max_tokens=600,
        )

        response2 = completion.choices[0].message["content"].strip()
    
    if I[0][0] < 700:
        output_source = "Srimad Bhagavad Gita"
    else:
        output_source = "Patanjali Yoga Sutra"
    
    if response1 == "'0'":
        output_data = {
            "Your Answer": response2
        }

    else:
        output_data = {
        "Book": output_source,
        "Chapter number": chapter_no[I[0][0]],
        "Verse number": verse_no[I[0][0]],
        "Top Matched Shloka": sanskrit_shlokas[I[0][0]],
        "Translation": verse[I[0][0]],
        "References": retrieved_verses,
        "Your Answer": response2

    }
        
    st.json(output_data)
