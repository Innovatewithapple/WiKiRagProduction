from nltk.tokenize import sent_tokenize
import nltk
nltk.download("punkt_tab")
from langchain_core.documents import Document
from models import decoder_tokenizer,encoder_tokenizer
from langchain_community.document_loaders import WikipediaLoader
import wikipedia
wikipedia.set_user_agent("MyRAGProject/1.0 (mihirvyasapple@email.com)")
import os
import pickle

# -------Load the wikipedia articles--------!
articles = [
    "Apollo program",
    "James Webb Space Telescope",
    "Voyager program",
    "International Space Station",
    "Mars rover",
    "Hubble Space Telescope",
    "SpaceX",
    "ISRO",
    "Black hole",
    "Milky Way",
]

def get_chunks():
    os.makedirs("data", exist_ok=True)
    if os.path.exists("data/chunks.pkl"):
        print("Loading saved chunks...")

        with open("data/chunks.pkl", "rb") as f:
            return pickle.load(f)
        
    print("Creating chunks...")
    docs = load_wikipedia_Articles()
    chunks = createParent_child_chunks(docs=docs,parent_chunk_size=400,parent_overlap=2,child_chunk_size=150,child_overlap=1)

    with open("data/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    return chunks

def load_wikipedia_Articles():
    docs = []
    for article in articles:
        loader = WikipediaLoader(query=article, load_max_docs=1)
        docs.extend(loader.load())
    return docs


def createParent_child_chunks(
    docs, parent_chunk_size, parent_overlap, child_chunk_size, child_overlap
):
    chunk_data = []
    for doc in docs:

        # ---------Parent Chunking------!
        parent_chunks = []
        parent_current_chunk = []
        parent_current_chunk_len_list = []
        parent_current_chunk_len = 0

        parent_sentences = sent_tokenize(doc.page_content)
        parent_chunk_token_len = [
            len(decoder_tokenizer.encode(parent_sent, add_special_tokens=False))
            for parent_sent in parent_sentences
        ]

        for parent_sent, parent_token in zip(parent_sentences, parent_chunk_token_len):
            if parent_current_chunk_len + parent_token <= parent_chunk_size:
                parent_current_chunk.append(parent_sent)
                parent_current_chunk_len_list.append(parent_token)
                parent_current_chunk_len += parent_token
            else:
                if parent_current_chunk:
                    parent_chunks.append(" ".join(parent_current_chunk))
                parent_current_chunk = parent_current_chunk[-parent_overlap:] + [
                    parent_sent
                ]
                parent_current_chunk_len_list = parent_current_chunk_len_list[
                    -parent_overlap:
                ] + [parent_token]
                parent_current_chunk_len = sum(parent_current_chunk_len_list)
        if parent_current_chunk:
            parent_chunks.append(" ".join(parent_current_chunk))

        # -------Child Chunking--------!
        for p_idx, parent_chunk in enumerate(parent_chunks):
            parent_chunk_sentences = sent_tokenize(parent_chunk)
            parent_chunk_sentence_token_len = [
                len(
                    encoder_tokenizer.encode(
                        parent_chunk_sentence, add_special_tokens=False
                    )
                )
                for parent_chunk_sentence in parent_chunk_sentences
            ]

            child_chunks = []
            child_current_chunk = []
            child_current_chunk_len_list = []
            child_current_chunk_len = 0

            for child_sent, child_token in zip(
                parent_chunk_sentences, parent_chunk_sentence_token_len
            ):
                if child_current_chunk_len + child_token <= child_chunk_size:
                    child_current_chunk.append(child_sent)
                    child_current_chunk_len_list.append(child_token)
                    child_current_chunk_len += child_token

                else:
                    if child_current_chunk:
                        child_chunks.append(" ".join(child_current_chunk))
                    child_current_chunk = child_current_chunk[-child_overlap:] + [
                        child_sent
                    ]
                    child_current_chunk_len_list = child_current_chunk_len_list[
                        -child_overlap:
                    ] + [child_token]
                    child_current_chunk_len = sum(child_current_chunk_len_list)
            if child_current_chunk:
                child_chunks.append(" ".join(child_current_chunk))

            # -------chunk data---------!
            for c_idx, child in enumerate(child_chunks):
                chunk_data.append(
                    Document(
                        page_content=child,
                        metadata={
                            "title": doc.metadata["title"],
                            "c_id": f"{p_idx}_{c_idx}",
                            "p_id": p_idx,
                            "parent_text": parent_chunk,
                        },
                    )
                )
    return chunk_data