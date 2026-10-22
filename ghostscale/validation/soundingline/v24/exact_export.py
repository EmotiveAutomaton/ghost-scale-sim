"""Public-ledger export with source-bound invariant caching.

Every bundle still passes the original canonical validation and truth rejection.
Only immutable engine fingerprints and repeated numeric content conversions are
cached. The emitted row order, identities, payloads and reader hash stay exact.
"""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from ..v23 import ledger
from ..v23.common import canonical, digest
from ..v23.engine_bridge import numeric_features


class ExactExporter:
    def __init__(self, schema, engine_files, expected_implementation):
        from tompathy_engine import admission
        from tompathy_engine.ledger.schema import schema_hash
        self.schema = schema
        self.root = Path(admission.__file__).resolve().parent
        self.engine_files = dict(engine_files)
        self.verify_sources()
        self.implementation = admission.implementation_hash()
        if self.implementation != expected_implementation:
            raise ValueError("loaded engine implementation differs")
        self.schema_hash = schema_hash()
        if self.schema_hash != ledger.schema_hash(schema):
            raise ValueError("native and engine schemas differ")
        self.content_features = lru_cache(maxsize=4096)(self._features)

    @staticmethod
    def _features(text):
        return numeric_features(json.loads(text))

    def verify_sources(self):
        actual = {p.relative_to(self.root).as_posix() for p in self.root.rglob("*")
                  if p.is_file() and p.suffix in (".py", ".yaml")}
        if actual != set(self.engine_files):
            raise ValueError("engine fingerprint source roster changed")
        for name, expected in self.engine_files.items():
            if digest(self.root/name) != expected:
                raise ValueError("engine source changed; invalidate exporter")

    def export(self, bundle, *, canaries=()):
        from tompathy_engine.ledger.ids import digest as engine_digest
        from tompathy_engine.ghost.bridge import reject_truth_fields
        ledger.validate(bundle,self.schema,canaries=canaries)
        if bundle["schema_hash"] != self.schema_hash:
            raise ValueError("schema mismatch")
        rows=deepcopy(bundle["rows"]);names=set()
        for decision in rows["Decision"]:
            decision["context"]={"observed_context":decision["context"]}
            for option in decision["options"]:
                # Copy before adding position; one cached mapping serves many options.
                features=dict(self.content_features(bundle["blobs"][option["content_ref"]]))
                features["menu_position_control"]=float(option["position"])
                option["features"]=features;names.update(features)
        order=sorted(names)
        for decision in rows["Decision"]:
            for option in decision["options"]:
                option["features"]={name:option["features"].get(name,0.) for name in order}
        for event in rows["ContextEvent"]:
            event["payload"]={"observation":event["payload"]}
        by_id={(kind,row[self.schema.ID_FIELDS[kind]]):row for kind,table in rows.items()
               if kind!="Provenance" for row in table}
        for provenance in rows["Provenance"]:
            source=by_id[(provenance["record_type"],provenance["record_id"])]
            provenance["source_file_sha256"]=hashlib.sha256(canonical(source)).hexdigest()
        result=dict(ledger_schema_hash=self.schema_hash,implementation_hash=self.implementation,
                    records=[dict(type=kind,record=row) for kind,table in rows.items() for row in table],
                    texts=deepcopy(bundle["blobs"]))
        reject_truth_fields(result)
        if any(token.encode() in canonical(result) for token in canaries):
            raise ValueError("truth canary reached export")
        result["reader_hash"]=engine_digest(result)
        return result

    def verify_export(self, root, variant, bundle, *, canaries=()):
        """Write the ordinary bridge artifact using one source-verified cache.

        This is intentionally the same public contract as V23's per-maker
        ``verify_export``.  The caller owns the exporter for a complete world,
        so source fingerprints and immutable content conversions are reused
        without changing row identity, order, or the canonical output.
        """
        from ..v23 import engine_bridge
        from ..v23.common import canonical, write

        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        exported = self.export(bundle, canaries=canaries)
        write(root / "reader.json", exported)

        from tompathy_engine.ghost.bridge import load_reader, reader_choices
        rows, texts = load_reader(root / "reader.json")
        choices = reader_choices(rows)
        native = engine_bridge.native_view(exported, self.schema)
        if variant["family"].startswith("T"):
            from ..v23.twin_ledger import decode
        else:
            decode = ledger.decode
        if canonical(decode(variant, native, self.schema)) != canonical(
                decode(variant, bundle, self.schema)):
            raise ValueError("cached engine projection changed native observations")
        ledger.export(root / "canonical", native)
        receipt = dict(
            state="ENGINE_TRANSPORT_VERIFIED",
            implementation_hash=exported["implementation_hash"],
            ledger_schema_hash=exported["ledger_schema_hash"],
            reader_hash=exported["reader_hash"],
            records=len(rows),
            resolved_baseline_choices=len(choices),
            blob_count=len(texts),
            native_observations_recovered=True,
            scientific_engine_admission=False,
            limitation=("Observable option-feature baseline only; native exact "
                        "inference and hidden labels are not engine inputs."),
        )
        write(root / "CONFORMANCE.json", receipt)
        return receipt
