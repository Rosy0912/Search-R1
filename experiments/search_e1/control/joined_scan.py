"""Preserve encoder batches while sharing one exact full-index vector scan."""
import numpy as np
from types import MethodType

def joined_batch_search(self, query_list, num=None, return_score=False, *, load_docs):
    if isinstance(query_list, str):
        query_list = [query_list]
    if num is None:
        num = self.topk
    if not query_list:
        return ([], []) if return_score else []
    # Identical original encode calls: batch128, same boundaries and precision.
    pieces = [self.encoder.encode(query_list[start:start+self.batch_size])
              for start in range(0, len(query_list), self.batch_size)]
    embeddings = np.concatenate(pieces, axis=0)
    scores, ids = self.index.search(embeddings, k=num)
    documents = load_docs(self.corpus, ids.reshape(-1).tolist())
    results = [documents[start:start+num] for start in range(0, len(documents), num)]
    return (results, scores.tolist()) if return_score else results

def install_joined_scan(retriever, load_docs):
    def search(self, query_list, num=None, return_score=False):
        return joined_batch_search(self, query_list, num, return_score, load_docs=load_docs)
    retriever._batch_search = MethodType(search, retriever)
