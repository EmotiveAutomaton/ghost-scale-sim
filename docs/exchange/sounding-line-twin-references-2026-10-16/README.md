# Measured source references for Ghost twins

T1 and T4 are complete aggregate inputs from the frozen exposed development
cohorts. `MANIFEST.json` binds each reference to its exact retained source-summary
bytes. They contain no raw text or participant identifiers. See the
[whole foundation verification](FOUNDATION.json).

- T1: 61 writers, 1,399 sessions, 16,855 decisions. The true menu histogram is
  1:956, 2:1259, 3:1913, 4:3739, 5:8988. The contract histogram excludes size 1,
  as required by the twin; the total cohort and handling marginals still include it.
- Timing: 16,719 selection/close endpoints and 136 right-censored ignores.
  These are recorded/proxy endpoints, not observed internal decision times.
  Typed counts are Unicode codepoints in user deltas between decision endpoints,
  not individual physical keystrokes. Final documents are canonically replayed.
- T1 handling order is accept, edit, dismiss, ignore. The twin calls its fourth
  category `other`; interpret the source value as logged ignore handling.
- T4: complete 123-by-350 DICES panel, target `Q_overall`, nominal order Yes,
  No, Unsure. `T4_104.json` is unavailable because the actual quality-exclusion
  identities/criteria are absent; no arbitrary subset was substituted.
- T3 is delivered in [its separate supplement](T3_README.md), pending API-native source-identity review. T2 remains pending its bounded download and parsing.

These inputs remove source-shape dependencies only. They do not heal failed
profile controls, promote an engine module or prove a human mechanism. The Ghost
owner remains responsible for source verification and scientific/resource admission.
