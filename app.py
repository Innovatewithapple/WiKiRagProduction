from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from chunking import get_chunks
from models import (
    encoder_Model,
    PreloadingSentenceEmbeddings,
    reranking_model,
    PromptTemplate,
    chat_Model,
)
from retrieval import Reranking, AskQuestion
from vectorstore import get_vector_store
import time

app_start = time.time()

all_chunks = get_chunks()
embeddings = PreloadingSentenceEmbeddings(encoder_Model=encoder_Model)

vector_Store = get_vector_store(embeddings, all_chunks)

retriever = vector_Store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

reranker = RunnableLambda(Reranking(reranking_model, top_k=3))

parser = StrOutputParser()

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | reranker
    | PromptTemplate()
    | chat_Model
    | parser
)

if __name__ == "__main__":
    answer = AskQuestion("What does JWST observe?",rag_chain=rag_chain)
    print(answer)
    print(f"Total App Time: {time.time()-app_start:.2f}s")