from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer
from config import DEVICE, HF_TOKEN, NVIDIA_API_KEY
from huggingface_hub import login
from langchain_core.embeddings import Embeddings
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA

model = "mistralai/ministral-14b-instruct-2512"  # "mistralai/mistral-medium-3.5-128b"

chat_Model = ChatNVIDIA(
    model=model,
    nvidia_api_key=NVIDIA_API_KEY,
    temperature=0.1,
    max_completion_tokens=100,
    top_p=0.9,
)

# ---------Encoder Model-------!
encoder_tokenizer = AutoTokenizer.from_pretrained(
    "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
)
encoder_Model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
)

# ----------LLm Tokenizer------!
login(token=HF_TOKEN)
decoder_tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct", trust_remote_code=True
)
decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

# ---------Reranking Model-----!
reranking_model = CrossEncoder("BAAI/bge-reranker-large", device=DEVICE)


class PreloadingSentenceEmbeddings(Embeddings):
    def __init__(self, encoder_Model):
        self.encoder_Model = encoder_Model

    def embed_documents(self, text: List[str]) -> List[List[float]]:
        return self.encoder_Model.encode(
            text, normalize_embeddings=True, batch_size=32, show_progress_bar=True
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.encoder_Model.encode(text, normalize_embeddings=True).tolist()


def PromptTemplate():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an experienced wikipedia expert, You are bound to answer the questions from the provided context only. Speak in natural and respective human accent like this is a debate between two humans.
                You MUST answer using only the supplied context. If any fact is not explicitly present in the context, do not include it.

                Do not add historical facts,background knowledge,or famous quotes that are not present in the context.
                Rules:
                1. Do NOT use outside knowledge or trained knowledge from model.
                2. Copy numerical and decimal values exactly as written in the context.
                3. Do NOT round, estimate, modify, or infer numbers.
                4. Do NOT explain or expand abbreviations unless explicitly written in the context.
                5. Preserve important entities, task names, benchmark names, and relationships exactly as written in the context.
                6. Include the specific task or subject associated with the answer if present in the context.
                7. Give a clear and natural answer in 2-3 sentences.
                8. Keep the answer grounded strictly in the provided context.
                9. If the exact answer is not explicitly stated but can be reasonably inferred from the context, provide the most likely answer based ONLY on the context.
                10.If the context does not contain enough information, say:
                   "I don't have enough information about this query."
                11.Do not say:
                   "Based on the context"
                   "According to the context"
                   "The context states"
                """,
            ),
            (
                "human",
                """
                Context:
                {context}

                Question:
                {question}
                """,
            ),
        ]
    )
    return prompt
