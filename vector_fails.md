
## When Vector Search Fails

Vector search is powerful for semantic retrieval, but it is not reliable for every type of query. In enterprise RAG systems, certain queries require exact matching rather than semantic similarity.

### 1. Product Codes

**Example:** `SKU-7742X specs`

- Product codes are unique identifiers, not natural-language concepts.
- Embedding models may not understand the relationship between an SKU and its corresponding product.
- A semantically similar product may be retrieved instead of the exact product.

**Problem:** Product codes often have no meaningful semantic information.

**Better approach:** Use exact keyword matching, metadata filtering, or hybrid search.

### 2. Error Codes

**Example:** `E_CONN_REFUSED`

- Error codes are often treated as character sequences by embedding models.
- The model may not understand the technical meaning of a specific error code.
- A search may retrieve general troubleshooting documents instead of documentation for that exact error.

**Problem:** The embedding model may not capture the significance of the exact error-code string.

**Better approach:** Combine vector search with BM25 or exact-match retrieval.

### 3. Acronyms

**Example:** `WCAG compliance`

- Acronyms can have multiple meanings depending on the domain.
- An embedding model may interpret an abbreviation incorrectly.
- The retrieved results may be semantically related but contextually irrelevant.

**Problem:** The model may not understand domain-specific abbreviations.

**Better approach:** Expand acronyms when possible, maintain an acronym dictionary, and use metadata or keyword filtering.

### 4. Exact Names

**Example:** `John Smith accounting`

- Vector search prioritizes semantic similarity.
- It may retrieve documents about accounting principles, other people named Smith, or unrelated businesses.
- The exact name may receive less importance than the surrounding semantic context.

**Problem:** Semantic similarity can override the importance of exact identifiers and names.

**Better approach:** Use exact-name matching, structured metadata filters, or hybrid search.

---

### Key Takeaway

These are not rare edge cases. They are common challenges in enterprise RAG systems.

**Vector search works well for semantic queries, but exact identifiers require lexical or structured retrieval.**

### Recommended Solution: Hybrid Search

Hybrid search combines:

1. **Dense retrieval:** Uses embeddings to find semantically similar content.
2. **Sparse retrieval:** Uses keyword-based methods such as BM25 to find exact terms and identifiers.
3. **Reranking:** Reorders retrieved results based on their relevance to the original query.

This combination improves retrieval for both natural-language questions and exact-match queries.

**Remember:** Do not rely exclusively on vector similarity when building production RAG systems.
