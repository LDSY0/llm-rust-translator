import chromadb
from unixcoder.embedding import encode

chroma_client = chromadb.PersistentClient("vector_db/chroma_db")
collection = chroma_client.get_collection(name="function_pair")

def retrieve_translation_pairs(query_source, top_k=1):
    query_embedding = encode(query_source).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas"]
    )

    function_pairs = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        pair = {
            "source": doc,
            "translation": meta.get("translation", "")
        }
        function_pairs.append(pair)
    return function_pairs


p=retrieve_translation_pairs("pub(crate) fn encoding_unicode_range(iana_name: &str) -> Result<Vec<&str>, String> {}")
print(p)
