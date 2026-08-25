import httpx

from src.core.config import settings


def extractive_answer(question, results):
    if not results:
        return "I could not find relevant information in the indexed documents."
    evidence = []
    for item in results[:3]:
        evidence.append(f"[{item['document']}, page {item['page']}] {item['text']}")
    return "No external LLM is configured. Relevant evidence:\n\n" + "\n\n".join(evidence)

def generate_answer(question, results):
    if not (settings.llm_base_url and settings.llm_api_key and settings.llm_model):
        return extractive_answer(question, results)
    context = "\n\n".join(f"[{x['document']}, page {x['page']}]\n{x['text']}" for x in results)
    r = httpx.post(
        settings.llm_base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {settings.llm_api_key}"},
        json={
            "model": settings.llm_model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": "Answer only from context. Cite claims with the supplied [document, page N] labels."},
                {"role": "user", "content": f"Question:\n{question}\n\nContext:\n{context}"},
            ],
        },
        timeout=60.0,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
