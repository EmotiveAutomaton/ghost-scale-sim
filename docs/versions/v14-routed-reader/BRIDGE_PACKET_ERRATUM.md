# V14 bridge packet — erratum

*Additive. `BRIDGE_PACKET.md` is unchanged and its hash is recorded below. This file exists because the packet was generated from a snapshot of the record taken before the cards it describes had closed, and the evidence that this happened is the stale file itself.*

- original `BRIDGE_PACKET.md` sha256: `207fa353db597d3475842396f22e78cb0ee3b068b87c44111df63b4e0824ec1d`
- erratum written: 2026-08-31 by V15 card I04

## What the packet says, and what the committed record says

| card | packet | committed state | criterion | verdict sha256 |
|---|---|---|---|---|
| F04 | as recorded | LANDED | failed | `315376e8ffab59b1` |
| F05 | as recorded | LANDED | failed | `977942fc05b5badb` |
| F06 | as recorded | LANDED | held | `63fe6ebe3386ec89` |
| F08 | UNRUN | LANDED | held | `8c9062aa64665805` |

## The rule this changes

An export whose source verdict hash predates the closure of the cards it describes cannot pass validation. `runners/validate_v15_program.py` checks it for V15, and the check is what makes this class of error visible rather than a matter of remembering to regenerate.
