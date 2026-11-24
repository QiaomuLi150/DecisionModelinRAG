import sqlite3

from scripts.v11_track_b_index import bm25_hits


def test_track_b_bm25_recovers_ids_from_contentless_fts():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE chunks (
            chunk_id TEXT, internal_doc_id TEXT, original_doc_id TEXT,
            archive_path TEXT, source_type TEXT, title TEXT,
            chunk_index INTEGER, text TEXT
        );
        CREATE VIRTUAL TABLE chunks_fts USING fts5(search_text, chunk_id, content='');
        INSERT INTO chunks VALUES
          ('chunk-a', 'doc-a', 'dsid_a', 'a.txt', 'slack', 'Title A', 0, 'alpha evidence');
        INSERT INTO chunks_fts(rowid, search_text, chunk_id)
          VALUES (1, 'title a alpha evidence', 'chunk-a');
        """
    )

    hits = bm25_hits(conn, "alpha")

    assert hits[0]["chunk_id"] == "chunk-a"
    assert hits[0]["doc_id"] == "doc-a"
    conn.close()
