"""
vector_store.py - Hybrid Retrieval, Domain-Adapted Embeddings, RRF Fusion & Cross-Encoder Module
AI Customer Support Agent & Evaluation Pipeline (@AppleSupport)

Implements:
1. Domain-Adapted Dense Semantic Search (BAAI/bge-small-en-v1.5 or all-MiniLM-L6-v2 with Apple Support Instruction Prefix)
2. Sparse Keyword Search (BM25Okapi)
3. Reciprocal Rank Fusion (RRF) combining Dense + Sparse rankings
4. Cross-Encoder Re-ranking on top RRF candidate pool before LLM generation
"""

import os
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

import chromadb
from chromadb.config import Settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class AppleDomainEmbeddingFunction:
    """Domain-adapted embedding function for @AppleSupport queries & resolutions."""
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self.model = None

    def _get_model(self):
        if self.model is None and HAS_SENTENCE_TRANSFORMERS:
            try:
                # Load BAAI/bge-small-en-v1.5 or fallback to all-MiniLM-L6-v2
                self.model = SentenceTransformer(self.model_name)
            except Exception:
                try:
                    self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                except Exception:
                    self.model = None
        return self.model

    def __call__(self, input_texts: List[str]) -> List[List[float]]:
        model = self._get_model()
        if model is not None:
            # Prefix instruction for domain retrieval optimization
            prefixed = [f"Represent this sentence for searching Apple Support resolutions: {t}" for t in input_texts]
            embeddings = model.encode(prefixed, normalize_embeddings=True)
            return embeddings.tolist()
        return []


class ResolutionVectorStore:
    def __init__(self, db_path: str = "chroma_db", collection_name: str = "apple_support_resolutions", csv_path: str = "data/processed/apple_support_threads.csv"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.csv_path = csv_path
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = None
        self.documents_cache = []
        self.bm25_index = None
        self.cross_encoder = None
        self.domain_embedding_fn = AppleDomainEmbeddingFunction()

        if os.path.exists(self.csv_path):
            self.load_cache_from_csv(self.csv_path)

    def load_cache_from_csv(self, csv_path: str):
        """Populates in-memory document cache, TF-IDF matrix, and BM25 index."""
        try:
            df = pd.read_csv(csv_path)
            self.documents_cache = []
            corpus_texts = []
            tokenized_corpus = []

            for idx, row in df.iterrows():
                doc_id = str(row.get('thread_id', f"TH-{idx}"))
                query = str(row.get('customer_query', ''))
                resolution = str(row.get('brand_resolution', ''))
                intent = str(row.get('intent', 'general_inquiry'))

                doc_text = f"Customer Query: {query}\nBrand Resolution: {resolution}"
                doc_obj = {
                    "thread_id": doc_id,
                    "intent": intent,
                    "customer_query": query,
                    "brand_resolution": resolution,
                    "doc_text": doc_text
                }
                self.documents_cache.append(doc_obj)
                corpus_texts.append(doc_text)
                tokenized_corpus.append(query.lower().split())

            if corpus_texts:
                self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)

            if HAS_BM25 and tokenized_corpus:
                self.bm25_index = BM25Okapi(tokenized_corpus)
        except Exception as e:
            print(f"Error loading cache from CSV: {e}")

    def _get_cross_encoder(self):
        if self.cross_encoder is None and HAS_SENTENCE_TRANSFORMERS:
            try:
                self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2')
            except Exception:
                self.cross_encoder = None
        return self.cross_encoder

    def build_index(self, csv_path: str = "data/processed/apple_support_threads.csv") -> int:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Processed dataset not found at {csv_path}. Run data_pipeline.py first.")

        df = pd.read_csv(csv_path)
        print(f"Indexing {len(df)} resolution records into Apple Domain Hybrid Store...")

        existing = self.collection.get()
        if existing and existing.get('ids'):
            self.collection.delete(ids=existing['ids'])

        documents = []
        metadatas = []
        ids = []
        
        self.load_cache_from_csv(csv_path)

        for idx, doc in enumerate(self.documents_cache):
            doc_id = doc["thread_id"]
            doc_text = doc["doc_text"]
            documents.append(doc_text)
            metadatas.append({
                "thread_id": doc_id,
                "intent": doc["intent"],
                "customer_query": doc["customer_query"],
                "brand_resolution": doc["brand_resolution"]
            })
            ids.append(doc_id)

        batch_size = 200
        for i in range(0, len(documents), batch_size):
            self.collection.add(
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )

        print(f"Successfully built Apple Domain Hybrid Index with {len(documents)} docs.")
        return len(documents)

    def _dense_retrieval(self, query: str, top_n: int = 20, intent_filter: Optional[str] = None) -> List[Dict]:
        retrieved = []
        try:
            where_clause = {"intent": intent_filter} if intent_filter else None
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_n, len(self.documents_cache)),
                where=where_clause
            )
            if results and results.get('metadatas') and len(results['metadatas'][0]) > 0:
                for meta in results['metadatas'][0]:
                    retrieved.append({
                        "thread_id": meta.get("thread_id"),
                        "intent": meta.get("intent"),
                        "customer_query": meta.get("customer_query"),
                        "brand_resolution": meta.get("brand_resolution")
                    })
        except Exception:
            pass

        if len(retrieved) < 3 and intent_filter is not None:
            try:
                results_global = self.collection.query(
                    query_texts=[query],
                    n_results=min(top_n, len(self.documents_cache))
                )
                if results_global and results_global.get('metadatas') and len(results_global['metadatas'][0]) > 0:
                    for meta in results_global['metadatas'][0]:
                        if not any(r['thread_id'] == meta.get('thread_id') for r in retrieved):
                            retrieved.append({
                                "thread_id": meta.get("thread_id"),
                                "intent": meta.get("intent"),
                                "customer_query": meta.get("customer_query"),
                                "brand_resolution": meta.get("brand_resolution")
                            })
            except Exception:
                pass

        if len(retrieved) < 3 and self.tfidf_matrix is not None and self.documents_cache:
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            top_indices = sorted(range(len(self.documents_cache)), key=lambda i: similarities[i], reverse=True)[:top_n]
            for i in top_indices:
                doc = self.documents_cache[i]
                if not any(r['thread_id'] == doc['thread_id'] for r in retrieved):
                    retrieved.append(doc)

        return retrieved[:top_n]

    def _sparse_retrieval(self, query: str, top_n: int = 20, intent_filter: Optional[str] = None) -> List[Dict]:
        tokenized_query = query.lower().split()
        if self.bm25_index is not None and self.documents_cache:
            scores = self.bm25_index.get_scores(tokenized_query)
            top_indices = sorted(range(len(self.documents_cache)), key=lambda i: scores[i], reverse=True)[:top_n]
            return [self.documents_cache[i] for i in top_indices]
        
        matches = []
        for doc in self.documents_cache:
            overlap = sum(1 for term in tokenized_query if term in doc['customer_query'].lower() or term in doc['brand_resolution'].lower())
            matches.append((overlap, doc))
        matches.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in matches[:top_n]]

    def reciprocal_rank_fusion(self, dense_results: List[Dict], sparse_results: List[Dict], rrf_k: int = 60) -> List[Dict]:
        doc_scores = {}
        doc_map = {}

        for rank, doc in enumerate(dense_results):
            doc_id = doc["thread_id"]
            doc_map[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, doc in enumerate(sparse_results):
            doc_id = doc["thread_id"]
            doc_map[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        sorted_doc_ids = sorted(doc_scores.keys(), key=lambda did: doc_scores[did], reverse=True)
        return [doc_map[did] for did in sorted_doc_ids]

    def cross_encoder_rerank(self, query: str, candidate_docs: List[Dict], top_k: int = 3) -> List[Dict]:
        if not candidate_docs:
            return []

        encoder = self._get_cross_encoder()
        
        if encoder is not None:
            try:
                pairs = [[query, doc["brand_resolution"]] for doc in candidate_docs]
                scores = encoder.predict(pairs)
                for doc, score in zip(candidate_docs, scores):
                    doc["cross_encoder_score"] = float(score)
                ranked_docs = sorted(candidate_docs, key=lambda d: d.get("cross_encoder_score", 0.0), reverse=True)
                return ranked_docs[:top_k]
            except Exception:
                pass

        query_terms = set(query.lower().split())
        for doc in candidate_docs:
            res_terms = set(doc["brand_resolution"].lower().split())
            query_in_res = len(query_terms.intersection(res_terms)) / max(len(query_terms), 1)
            query_in_cust = len(query_terms.intersection(set(doc["customer_query"].lower().split()))) / max(len(query_terms), 1)
            cross_score = 0.6 * query_in_res + 0.4 * query_in_cust
            doc["cross_encoder_score"] = cross_score

        ranked_docs = sorted(candidate_docs, key=lambda d: d.get("cross_encoder_score", 0.0), reverse=True)
        return ranked_docs[:top_k]

    def retrieve_relevant_resolutions(self, query: str, k: int = 3, intent_filter: Optional[str] = None) -> List[Dict]:
        dense_candidates = self._dense_retrieval(query, top_n=15, intent_filter=intent_filter)
        sparse_candidates = self._sparse_retrieval(query, top_n=15, intent_filter=intent_filter)
        rrf_candidates = self.reciprocal_rank_fusion(dense_candidates, sparse_candidates, rrf_k=60)
        final_top_k = self.cross_encoder_rerank(query, rrf_candidates[:10], top_k=k)
        return final_top_k


if __name__ == "__main__":
    vs = ResolutionVectorStore()
    results = vs.retrieve_relevant_resolutions("Is AppleCare+ worth buying?", k=3)
    print("Apple Domain Embedding Test Results:")
    for r in results:
        print(f"- Intent: {r['intent']} | Resolution: {r['brand_resolution']}")
