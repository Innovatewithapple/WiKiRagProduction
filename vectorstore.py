from langchain_community.vectorstores import FAISS
import os


def get_vector_store(embeddings,all_chunks):
    os.makedirs("data", exist_ok=True)
    if os.path.exists("data/faiss_index"):
        print("Loading existing FAISS index...")

        vector_Store = FAISS.load_local("data/faiss_index",embeddings,allow_dangerous_deserialization=True)
        return vector_Store

    else:
        print("Creating FAISS index...")
        vector_Store = FAISS.from_documents(documents=all_chunks,embedding=embeddings)
        vector_Store.save_local("data/faiss_index")
        return vector_Store