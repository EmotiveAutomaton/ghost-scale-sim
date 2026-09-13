"""Transactional compressed constructor records. One database, bounded file count."""
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import zlib
from ..v16.records import canonical,digest
from .continuation_analysis import unit_statistics,paired_value,group_surface

def utc(): return datetime.now(timezone.utc).isoformat()
def packed(value): return zlib.compress(canonical(value),3)
def unpacked(value): return json.loads(zlib.decompress(value))

class Store:
    def __init__(self,path):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.path,timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS totals(singleton INTEGER PRIMARY KEY,units INTEGER,cases INTEGER,rows INTEGER,cpu REAL,wall REAL,issues INTEGER);
        INSERT OR IGNORE INTO totals VALUES(1,0,0,0,0,0,0);
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS units(stage TEXT,branch TEXT,ci INTEGER,digest TEXT,body BLOB,stats BLOB,paired REAL,cases INTEGER,rows INTEGER,cpu REAL,wall REAL,PRIMARY KEY(stage,branch,ci));
        CREATE TABLE IF NOT EXISTS identities(stage TEXT,branch TEXT,kind TEXT,identity TEXT,PRIMARY KEY(stage,branch,kind,identity)) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS dependencies(stage TEXT,branch TEXT,ci INTEGER,body BLOB,digest TEXT,PRIMARY KEY(stage,branch,ci));
        CREATE TABLE IF NOT EXISTS failures(id TEXT PRIMARY KEY,stage TEXT,branch TEXT,ci INTEGER,at TEXT,reason TEXT);
        CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY,kind TEXT,payload TEXT,at TEXT);
        CREATE TABLE IF NOT EXISTS snapshots(name TEXT PRIMARY KEY,body BLOB,digest TEXT);
        """)
        self.db.commit()
    def close(self):
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.db.close()
    def lock(self,key,value):
        text=canonical(value).decode()
        old=self.db.execute("SELECT value FROM metadata WHERE key=?",(key,)).fetchone()
        if old and old[0]!=text: raise ValueError("immutable database metadata changed: "+key)
        if not old:
            self.db.execute("INSERT INTO metadata VALUES(?,?)",(key,text));self.db.commit()
    def metadata(self,key):
        row=self.db.execute("SELECT value FROM metadata WHERE key=?",(key,)).fetchone()
        return json.loads(row[0]) if row else None
    def has(self,stage,branch,ci):
        return self.db.execute("SELECT 1 FROM units WHERE stage=? AND branch=? AND ci=?",(stage,branch,ci)).fetchone() is not None
    def failed(self,stage,branch):
        return self.db.execute("SELECT 1 FROM failures WHERE stage=? AND branch=? LIMIT 1",(stage,branch)).fetchone() is not None
    def save(self,stage,design,ci,items,*,cpu,wall,dependency_files=None,contrast=None):
        stats=unit_statistics(items)
        payload=canonical(items);hashvalue=hashlib.sha256(payload).hexdigest()
        key=(stage,design["id"],ci)
        old=self.db.execute("SELECT digest FROM units WHERE stage=? AND branch=? AND ci=?",key).fetchone()
        if old:
            if old[0]!=hashvalue: raise ValueError("attempt to replace retained constructor records")
            return False
        value=paired_value(items,contrast) if contrast else None
        with self.db:
            self.db.execute("INSERT INTO units VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (*key,hashvalue,zlib.compress(payload,3),packed(stats),value,len(items),sum(len(i["rows"]) for i in items),cpu,wall))
            self.db.execute("UPDATE totals SET units=units+1,cases=cases+?,rows=rows+?,cpu=cpu+?,wall=wall+?,issues=issues+? WHERE singleton=1",
                (len(items),sum(len(i["rows"]) for i in items),cpu,wall,sum(bool(r.get("apparatus_failure")) for i in items for r in i["rows"])))
            for item in items:
                case=item["case"]
                for kind in ("public_problem_sha256","private_construction_sha256","structural_family"):
                    self.db.execute("INSERT OR IGNORE INTO identities VALUES(?,?,?,?)",(stage,design["id"],kind,case[kind]))
            if dependency_files:
                body=canonical(dependency_files)
                self.db.execute("INSERT INTO dependencies VALUES(?,?,?,?,?)",(*key,zlib.compress(body,3),hashlib.sha256(body).hexdigest()))
        return True
    def failure(self,stage,branch,ci,reason):
        identity=digest([stage,branch,ci,reason])
        with self.db:self.db.execute("INSERT OR IGNORE INTO failures VALUES(?,?,?,?,?,?)",(identity,stage,branch,ci,utc(),reason))
        return identity
    def event(self,kind,payload):
        identity=digest([kind,payload])
        with self.db:self.db.execute("INSERT OR IGNORE INTO events VALUES(?,?,?,?)",(identity,kind,canonical(payload).decode(),utc()))
        return identity
    def snapshot(self,name,value):
        body=canonical(value);hashvalue=hashlib.sha256(body).hexdigest()
        old=self.db.execute("SELECT digest FROM snapshots WHERE name=?",(name,)).fetchone()
        if old and old[0]!=hashvalue: raise ValueError("scientific checkpoint changed")
        if not old:
            with self.db:self.db.execute("INSERT INTO snapshots VALUES(?,?,?)",(name,zlib.compress(body,3),hashvalue))
        return value
    def get_snapshot(self,name):
        row=self.db.execute("SELECT body FROM snapshots WHERE name=?",(name,)).fetchone()
        return unpacked(row[0]) if row else None
    def counts(self):
        n,c,r,cpu,wall,issues=self.db.execute("SELECT units,cases,rows,cpu,wall,issues FROM totals WHERE singleton=1").fetchone()
        return dict(constructor_units=n,cases=c,rows=r,worker_cpu_seconds=cpu,unit_wall_seconds=wall,apparatus_issue_rows=issues,
            failed_units=self.db.execute("SELECT COUNT(*) FROM failures").fetchone()[0])
    def surface(self,stage,design):
        def stats():
            for (body,) in self.db.execute("SELECT stats FROM units WHERE stage=? AND branch=? ORDER BY ci",(stage,design["id"])):
                yield from unpacked(body)
        unique={k:n for k,n in self.db.execute("SELECT kind,COUNT(*) FROM identities WHERE stage=? AND branch=? GROUP BY kind",(stage,design["id"]))}
        return dict(id=design["id"],family=design["family"],regime=design["regime"],stage=stage,
            comparisons=group_surface(stats()),uniqueness=unique,failed=self.failed(stage,design["id"]))
    def pairs(self,branch):
        return [x[0] for x in self.db.execute("SELECT paired FROM units WHERE stage='confirmation' AND branch=? ORDER BY ci",(branch,))]
    def iter_items(self,stage=None,branch=None):
        where=[];args=[]
        if stage: where.append("stage=?");args.append(stage)
        if branch: where.append("branch=?");args.append(branch)
        sql="SELECT digest,body FROM units"+(" WHERE "+" AND ".join(where) if where else "")+" ORDER BY stage,branch,ci"
        for expected,body in self.db.execute(sql,args):
            payload=zlib.decompress(body)
            if hashlib.sha256(payload).hexdigest()!=expected: raise ValueError("compressed raw unit hash mismatch")
            yield from json.loads(payload)
    def verify_all(self):
        integrity=self.db.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity!="ok": raise ValueError(integrity)
        n=0
        for item in self.iter_items():
            from .continuation_analysis import verify_item
            verify_item(item);n+=1
        return dict(passed=True,raw_cases_verified=n,sqlite_integrity=integrity)
