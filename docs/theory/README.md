# Theory: the hypothesis store

**The dense channel.** Every claim this project makes, organised under umbrella hypotheses, with
the evidence tabled beneath each, in one file: [READING_INTENT.md](READING_INTENT.md). The wide
channel (how each test was run, in what order, and what its numbers mean exactly) is
[FINDINGS.md](../../FINDINGS.md) and the version documents. **If a result is in one channel it
must be in both**; the maintenance rule at the bottom of the hypothesis store says how.

The format is adopted from the sibling project's theory folder
(`../../../../SoundingLine/sounding-line/docs/theory/`), adapted where this repository differs.

**The theory itself lives outside this repository**, in the preprint and the artist-facing essay:

- *Art as an Algorithmic Virus: Unifying the Generative Crash and AI Value Convergence via
  Cognitive Affordances*: Zenodo, DOI
  [`10.5281/zenodo.19407789`](https://doi.org/10.5281/zenodo.19407789)
- The plain-language version: <https://abrahamhaskins.org/art>

---

## Format for the hypothesis store

    TOP OF FILE   the theory in the curator's words, then two or three lines of what it claims,
                  then the one empirical commitment. A visitor gets the shape from the first screen
    EACH SECTION  1. the date of the claim and whose words state it
                  2. THE CLAIM, as a blockquote
                  3. WHAT IT SAYS, short and plain
                  4. the hypothesis table, with status and evidence
                  5. WHAT THESE ADD UP TO, a first pass at combining the rows into a claim

**Sections run in decreasing load-bearing order**, the core commitment first, then whatever
explains the theory most naturally. Not the order things were run in.

**Blockquotes are the curator's words only.** In this repository those come from the essay and
the preprint (his text, 2026-04) and from the front page above the marker (◐ Curator tier,
2026-08, flagged with ◐ when quoted, since that tier is human-and-machine synthesis rather than
untouched prose). Machine text is never blockquoted as his.

**Corrections are folded in, never appended. Superseded material takes a bold status** (RETIRED,
SUPERSEDED, WITHDRAWN) and stays in the section it belongs to. A hypothesis with a history gets
its history in the notables column, one line. **Identifiers are stable**, they are the
experiment IDs (E, A, D, R, H, N, T, S, MIN), never reused, never renumbered.

**A disconfirmed thing gets one row and no elaboration.** State what was checked and what came
back. The reasoning behind a useful dead measurement is worth a clause; the reasoning behind a
claim that simply failed is not.

**Under every table, the summary paragraph is revisited in the same edit that changes the
table.** A stale conclusion under a fresh table is worse than no conclusion, because it reads as
current.

## Status legend

    SUPPORTED        the committed verdict backs the claim as stated
    REJECTED         the committed verdict came back against it
    WITHDRAWN        the project retracted the claim itself
    WITHHELD         the instrument failed its own control; not a null, a refusal to report
    RETIRED          did not survive a later, stricter pass (exact inference, audit)
    SUPERSEDED       replaced by a better-specified construct; kept as a record
    SPLIT            the pre-registered form fails while a stated other form holds; both reported
    QUALIFIED        holds with a named limitation that changes how it may be quoted
    CONTESTED        the published literature says otherwise; the difference is the contribution
    INSTRUMENT DEAD  the measure died, not the idea
    OPEN             not yet run, with the blocker named

## Source legend, because they are not equally strong

    (run)   a committed verdict file in this repository. The strongest thing we have, and every
            (run) row is a METHOD result about a constructed world. Whether the mechanism holds in
            people is what a simulation cannot say, and rows resting on a real-world fact say so
    (lit)   published work, from the retrospective search in EVIDENCE.md. READ if fetched and
            read; SNIPPET if not. It informed no design
    (SL)    the sibling project's result on real text, where one bears on a claim here

## Files

| | |
|---|---|
| **[READING_INTENT.md](READING_INTENT.md)** | **the hypothesis store**: ten umbrella claims, every result under the one it bears on |
| [Art as an Algorithmic Virus (PDF)](Art%20as%20an%20Algorithmic%20Virus%20-%20Unifying%20the%20Generative%20Crash%20and%20AI%20Value%20Convergence%20via%20Cognitive%20Affordances-1.pdf) | the preprint. **The authority for anything mathematical** |
| [Art_a_unifying_model.md](Art_a_unifying_model.md) | the artist-facing essay ([public home](https://abrahamhaskins.org/art)), transcribed in full, the rawest form of the intent and the source of most section quotes. **Kept deliberately as material for occasional entropy injection**: sampling the unsharpened original when drafting or curation needs variation the polished record no longer supplies. The technique earns less as the paper's claims sharpen, and the transcript stays so the move is cheap when it is still appropriate |
| [Art_as_an_Algorithmic_Virus_preprint.md](Art_as_an_Algorithmic_Virus_preprint.md) | machine transcription of the preprint, kept greppable and diffable; equations survive the two-column extraction badly, so read those against the PDF |

## The vocabulary, and how it maps onto the code

The simulation's variable names are not the theory's words. This is the crosswalk.

| the theory says | the model calls it | what it governs |
|---|---|---|
| compressed intent, artfulness | **depth** (μ) | how many levels of the maker's decision hierarchy reach the surface |
| appreciation | **engagement** | whether the reader spends the effort to look closely |
| epistemic trust | **trust in the label** (κ) | how much the reader weights what it is told about origin against what it can see |
| the disgust firewall | **the acceptance gate** (θ, λ) | whether what was recovered is allowed to change the reader |
| precision weighting (ω) | **the attention decision** | how much the reader is currently crediting what it sees |
| metabolic reserve (E) | **the reserve** | how worn down the reader is, carried across encounters |
| the generative crash | **the crash signature** | unresolved, and no longer looking |
| fabrication | **the fabrication index** | confident *and* mutually contradictory: invention, as against honest confusion |
| the Ghost Scale tiers | **provenance**, and its opacity α | how much of the maker's intent survives into the object |

**One term is used differently in the two places and it is worth flagging.** In the preprint, ω is
the reader's own precision weighting, an *output*, which collapses when the inference fails. In
the code, ω is feature overlap: a fixed property of the content, an *input*. The code's analogue
of the preprint's ω is the attention decision. They share a letter and are different objects.
(The E41 row in the store is the measured consequence: the paper's mechanism and the code's are
not the same object, and the difference is testable.)

## The maintenance rule

**A result goes into the hypothesis store's table in the same pass that records it in
[FINDINGS.md](../../FINDINGS.md).** FINDINGS is organised by what was run; the store by what is
believed. A result recorded only in FINDINGS gets lost; a claim recorded only in the store loses
its provenance. Literature found later is harvested into the store as (lit) rows in the same pass
that adds it to [EVIDENCE.md](../../EVIDENCE.md).
