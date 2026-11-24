"""Tiny scripted DB-API fakes shared by the round-3 Postgres-bound tests (not a test module)."""


class Col:
    def __init__(self, name):
        self.name = name


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.description = []
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.conn.log.append((" ".join(str(sql).split()), params))
        res = self.conn.responder(self.conn.log[-1][0], params)
        if isinstance(res, Exception):
            raise res
        cols, rows = res if res else ([], [])
        self.description = [Col(c) for c in cols]
        self._rows = list(rows)

    def fetchone(self):
        return self._rows.pop(0) if self._rows else None

    def fetchall(self):
        rows, self._rows = self._rows, []
        return rows

    def __iter__(self):
        return iter(self.fetchall())


class FakeConn:
    """responder(sql, params) -> (columns, rows) | Exception | None"""

    def __init__(self, responder=None):
        self.responder = responder or (lambda sql, params: None)
        self.log = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True

    def sqls(self):
        return [s for s, _ in self.log]
