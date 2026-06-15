import redis
import hashlib
import time

class Reranking:
    def __init__(self, model, top_k):
        self.model = model
        self.top_k = top_k

    def __call__(self, inputs):
        start = time.time()
        question = inputs["question"]
        docs = inputs["context"]

        pairs = [(question, doc.page_content) for doc in docs]
        scores = self.model.predict(inputs=pairs)

        for i, doc in enumerate(docs):
            doc.metadata["rerank_score"] = scores[i]

        reranked = sorted(docs, key=lambda x: x.metadata["rerank_score"], reverse=True)
        top_docs = reranked[: self.top_k]

        unique_parents = []
        seen = set()

        for doc in top_docs:
            parent_id = doc.metadata["p_id"]

            if parent_id not in seen:
                seen.add(parent_id)
                unique_parents.append(doc.metadata["parent_text"])

        print(f"Reranker Time: {time.time()-start:.2f}s")
        return {"question": question, "context": "\n\n".join(unique_parents)}
    

redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)


def AskQuestion(question,rag_chain):
    # cache_key = hashlib.md5(question.lower().strip().encode()).hexdigest()
    # cached_Answer = redis_client.get(cache_key) # or use question string directly
    # if cached_Answer:
    #   print('Cache Hit!!!')
    #   return cached_Answer
    question_start = time.time()
    print("Cache Miss")
    answer = rag_chain.invoke(question)
    print(f"Question Time: {time.time()-question_start:.2f}s")
    # redis_client.set(cache_key,answer,ex=3600)
    return answer
