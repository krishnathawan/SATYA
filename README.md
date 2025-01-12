# SATYA: Sanatana Atman Tapas Yoga Ananda

SATYA is a chatbot designed to answer spiritual queries based on the **Bhagavad Gita** and **Patanjali Yoga Sutras**. It combines modern AI with ancient wisdom, enabling users to retrieve relevant shlokas and detailed answers to their queries.

---

## Features
- Retrieval-Augmented Generation (RAG) pipeline for answering queries.
- Uses a fine-tuned SentenceTransformer model for embedding generation of dataset (shlokas).
- Employs **FAISS** for storing these embeddings and search top retrieval embeddings.
- Integration with Hugging Face's inference API for text response generation using model="mistralai/Mistral-7B-Instruct-v0.3"
- Supports two major spiritual texts:
  - Bhagavad Gita
  - Patanjali Yoga Sutras

---

## Workflow
- When user ask the query then the query text is first converted to embeddings using fine_tuned sbert model. 
- index.search helps in searching top 5 retrieval shloka from saved index.(relevancy with respect to user query)
- These retrieved shlokas are passed to model="mistralai/Mistral-7B-Instruct-v0.3" as a prompt.(It helps in augumenting the knowledge).
- Firstly, this model is given instruction to find whether the retrieved shlokas are relevant to query or not. If not, then we output: "Sorry! I don't know." 
- If relevant then we instruct model to ignore other irrelevant shlokas out of five shlokas and prompt the model. Prompt: f"Query: {user_query}\n"  f"Give answer of query in long by deeply using most related {retrieved_verses} and your own knowledge too and summarise the answer." 
- Finally, output all these in json format . 

---

## Instructions for running
  ** Comments are added only above that lines where we are required to paste the path.(This is avoid confusion and mess) **

  INITIAL DATASETS : Bhagwad_Gita_Verses_English.csv , Patanjali_Yoga_Sutras_Verses_English.csv , Bhagwad_Gita_Verses_English_Questions.csv , Patanjali_Yoga_Sutras_Verses_English_Questions.csv
  FORMED DATASETS(Through below codes) : new_fine_tuned link , fine_tuned_embeddings.faiss.
(1) Fine_tuning.ipynb : This notebook fine tunes sbert model = 'nomic-ai/modernbert-embed-base' and save it in PC. So that after saving it in google drive, we can use it later.
    (Paste csv datasets in this code and you will require api key of hugging face before training on running the last block.)
    
(2) Preprocess_verses.ipynb : Now convert shlokas csv into embeddings using above model and save .faiss file in your pc .
    (Paste shlokas csv path of both books and paste path of model saved in previous code.)
    
(3) Check_accuracy.ipynb : For checking accuracy of provided dataset.
    (Paste path of two csv of evaluation , paste .faiss path which we saved in previous code , paste fine tuned model path.)

(4) app.py : save this code in visual studio ( install requirements using these commands in terminal ) (Paste paths , Comments are added only above that lines where path and api key of Hugging Face is to be pasted.)
    commnand : pip install faiss-cpu  
               pip install sentence-transformers
               pip install pandas
               pip install numpy
               pip uninstall -y torch torchvision torchaudio
               pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
               pip install git+https://github.com/huggingface/transformers.git
               pip install safetensors
               pip install streamlit

