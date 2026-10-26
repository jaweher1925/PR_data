CREATE_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS perf_pr (
        pr_id           INTEGER PRIMARY KEY,
        pr_url          VARCHAR,
        agent           VARCHAR,
        pattern         VARCHAR,
        sub_pattern     VARCHAR,
        label_source    VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS commit_details (
        sha                     VARCHAR NOT NULL,
        pr_id                   INTEGER NOT NULL REFERENCES perf_pr(pr_id),
        pr_number               INTEGER,
        commit_stats_total      INTEGER,
        commit_stats_additions  INTEGER,
        commit_stats_deletions  INTEGER,
        filename                VARCHAR NOT NULL,
        status                  VARCHAR,
        additions               INTEGER,
        deletions               INTEGER,
        changes                 INTEGER,
        patch                   TEXT,
        repo_owner              VARCHAR,
        repo_name               VARCHAR,
        commit_sha              VARCHAR,
        PRIMARY KEY (sha, filename)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS patch_classification (
        pr_id           INTEGER NOT NULL REFERENCES perf_pr(pr_id),
        sha             VARCHAR NOT NULL,
        filename        VARCHAR NOT NULL,
        is_performance  INTEGER NOT NULL,
        categories      TEXT,
        confidence      REAL,
        reasoning       TEXT,
        classified_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (pr_id, sha, filename)
    )
    """,
]


def create_all(conn):
    cur = conn.cursor()
    for stmt in CREATE_STATEMENTS:
        cur.execute(stmt)
    conn.commit()
