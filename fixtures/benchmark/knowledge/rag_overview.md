# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is a framework that combines external knowledge retrieval with large language model generation.
When a user submits a query, an embedding model encodes the query to perform semantic similarity search against an external vector database.
The top-k relevant text chunks are extracted and concatenated into the prompt context for the LLM.
This grounds model responses in up-to-date, domain-specific evidence while significantly reducing hallucinations.
