"""
Retriever: queries the unified `rag_chunks` table in Postgres + pgvector
for the top-k most similar chunks to a given embedding.

Same as RAG_AI. Works with both Ollama and VLLM embeddings.
"""

from typing import List, Optional
import psycopg2
import psycopg2.extras

from .response_types import Source


class PgVectorRetriever:
    def __init__(self, conn_string: str, table: str = "rag_ai.rag_chunks"):
        self.conn_string = conn_string
        self.table = table

    def _connect(self):
        return psycopg2.connect(self.conn_string)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        doc_type: Optional[str] = None,
    ) -> List[Source]:
        """
        Cosine-similarity search against pgvector.
        `doc_type` optionally filters retrieval (e.g. only 'schema' chunks).
        """
        vec_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"

        where_clause = "WHERE doc_type = %s" if doc_type else ""
        params = [vec_literal] + ([doc_type] if doc_type else []) + [vec_literal, top_k]

        sql = f"""
            SELECT id, content, source, doc_type, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM {self.table}
            {where_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """

        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute("SET search_path TO rag_ai, vector_core, public")
                cur.execute(sql, params)
                rows = cur.fetchall()

        return [
            Source(
                chunk_id=row["id"],
                source=row["source"],
                doc_type=row["doc_type"],
                score=float(row["score"]),
                text=row["content"],
                metadata=row["metadata"] or {},
            )
            for row in rows
        ]
