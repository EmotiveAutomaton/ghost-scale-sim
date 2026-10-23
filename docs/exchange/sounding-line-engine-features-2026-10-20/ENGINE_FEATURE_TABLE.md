# Engine feature table: prepared for the Tuesday return

**Hypothesis.** The completed channel measurements can be returned with their exact definitions and limitations without turning missing measurements or failed controls into validated badges.
**METHOD.** Map the complete aggregate PMAP, V3, D17, D29 and EV1–3 receipts into one source-hashed schema; preserve distinct estimands, every control warning and reasons for unavailable fields.
**Found.** All 21 requested channel rows are filed. **Means.** The local handoff is prepared; Tuesday delivery and literal dictionary release review remain open. This filing adds no scientific result.

The table gives observed real and mean fake identification fractions where measured, the median defined featurewise Spearman-Brown split-half reliability where measured, and V3's conditional projected record count for 0.8 reliability. A dash means that quantity was not measured, not zero. Other reliability estimates and exact definitions appear below and in the JSON. Control flags travel with the observed numbers.

| Channel | Real identification | Fake mean | Median split-half | Projected records | Control status |
|---|---:|---:|---:|---:|---|
| PMAP.lean: Handling rates | 0.155556 | 0.033 | 0.8124 | — | biased fake reference |
| PMAP.style: Typed words and punctuation | 0.466667 | 0.007 | 0.475518 | — | biased fake reference |
| PMAP.rhythm: Logged production rhythm | 0.977778 | 0.0341111 | 0.985157 | — | biased fake reference |
| PMAP.revision: Revision rates | 0.311111 | 0.0341111 | 0.952677 | — | biased fake reference |
| PMAP.requests: Request frequency | 0.111111 | 0.0276667 | 0.981581 | — | reference passes; paired contrast marginal |
| PMAP.length: Prompt-relative final length | 0.2 | 0.0303333 | 0.950827 | — | biased fake reference |
| PMAP.content_taste: Accepted-minus-dismissed content | 0.047619 | 0.0149425 | 0.0443138 | — | biased fake reference, incomplete fake reference |
| V3.accept: Acceptance | — | — | — | 39 | single outcome shuffle only, paired identity control/curve unmeasured |
| V3.edit: Editing | — | — | — | 32 | single outcome shuffle only, paired identity control/curve unmeasured |
| V3.dismiss: Dismissal | — | — | — | 218 | single outcome shuffle only, paired identity control/curve unmeasured |
| V3.ignore: Ignoring | — | — | — | 766 | single outcome shuffle only, paired identity control/curve unmeasured |
| V3.handling: Combined handling | — | — | — | 51 | single outcome shuffle only, paired identity control/curve unmeasured |
| V3.hover_events_log1p: Logged hover count | — | — | — | 27 | single outcome shuffle only, paired identity control/curve unmeasured |
| V3.typed_since_previous_log1p: Typed text between decision resolutions | — | — | — | 13 | single outcome shuffle only, paired identity control/curve unmeasured |
| V3.observed_menu_latency_log1p: Observed menu latency | — | — | — | 13 | single outcome shuffle only, paired identity control/curve unmeasured |
| D17.slow_pause: Slow-pause mixture share | — | — | — | — | single writer shuffle only, paired identity reference pending |
| D29.length: Final length across sessions | — | — | — | — | original shuffle; paired reference pending |
| EV1.departure_pause: Departure and preceding pause | — | — | — | — | reference passes; scoped route |
| EV2.signatures: Cross-work signature recurrence | — | — | — | — | reference passes; scoped route |
| EV3.final: Engine habits: final document | 0.377778 | 0.0143333 | 0.714264 | — | biased fake reference |
| EV3.typed: Engine habits: surviving typed fragments | 0.511111 | 0.03 | 0.747341 | — | biased fake reference |

PMAP's rhythm-minus-handling contrast separately passes its paired reference; this does not erase either channel's chance-centered warning. PMAP content remains unsupported for admission. Both EV3 variants fail their fixed bias controls and license no habit badge. EV1 supports the conditional pause association; EV2 supports aggregate cross-work recurrence. No new probabilities are quoted.

V3 estimates model intraclass correlations, not identification or split-half medians. Its record counts assume independent exchangeable records and are not empirical learning curves. D17 estimates a mixture-conditioned writer correlation; D29 estimates one odd/even Pearson correlation. None is substituted into the median split-half column.

## Definitions, inputs and evidence

### PMAP.lean: Handling rates

**Input:** suggestion log.
Half-profile fractions of accept, edit, dismiss and ignore across usable menus.
Fixed gallery; minimum four sessions and two prompts; prompt-alternating split; B-only feature scaling. Median feature reliability is not a reliability of the whole channel.
Evidence: [PMAP.json](PMAP.json); FINDINGS: S15-DRIVE-PMAP-1018.

### PMAP.style: Typed words and punctuation

**Input:** revision history.
Counts of 100 corpus-most-frequent casefolded produced words and 22 punctuation tokens, divided by all produced word plus punctuation tokens in each half. Includes subsequently deleted text; no part-of-speech filter.
Fixed gallery; minimum four sessions and two prompts; prompt-alternating split; B-only feature scaling. Median feature reliability is not a reliability of the whole channel. Transductive corpus-selected dictionary. The observed 47% is only Sounding Line's definition, with its bias warning; it does not transfer to another word list.
Evidence: [PMAP.json](PMAP.json); FINDINGS: S15-DRIVE-PMAP-1018.

### PMAP.rhythm: Logged production rhythm

**Input:** timing log, revision history.
Half-profile medians of pooled within-word intervals, word-onset intervals, and latencies for 20 corpus-most-frequent adjacent typed codepoint pairs; distinct single-codepoint delta ordinals only.
Fixed gallery; minimum four sessions and two prompts; prompt-alternating split; B-only feature scaling. Median feature reliability is not a reliability of the whole channel. Logged delta timing carries device, recording and task context; not isolated keypress speed or automaticity. Owner-capture use only for ToMpathy rhythm routing.
Evidence: [PMAP.json](PMAP.json); FINDINGS: S15-DRIVE-PMAP-1018, S15-DRIVE-PMAP-RHYTHM-1018.

### PMAP.revision: Revision rates

**Input:** revision history, suggestion log.
Content and housekeeping edits per 100 produced typed words; deleted share of original characters from supported accepted suggestions.
Fixed gallery; minimum four sessions and two prompts; prompt-alternating split; B-only feature scaling. Median feature reliability is not a reliability of the whole channel.
Evidence: [PMAP.json](PMAP.json); FINDINGS: S15-DRIVE-PMAP-1018.

### PMAP.requests: Request frequency

**Input:** suggestion log, revision history.
Suggestion requests per 100 produced typed words, pooled within each half.
Fixed gallery; minimum four sessions and two prompts; prompt-alternating split; B-only feature scaling. Median feature reliability is not a reliability of the whole channel.
Evidence: [PMAP.json](PMAP.json); FINDINGS: S15-DRIVE-PMAP-1018.

### PMAP.length: Prompt-relative final length

**Input:** final text, prompt metadata.
Mean within-half final nonprompt word count divided by the pooled median for the corresponding prompt.
Fixed gallery; minimum four sessions and two prompts; prompt-alternating split; B-only feature scaling. Median feature reliability is not a reliability of the whole channel.
Evidence: [PMAP.json](PMAP.json); FINDINGS: S15-DRIVE-PMAP-1018.

### PMAP.content_taste: Accepted-minus-dismissed content

**Input:** suggestion log, frozen content embeddings.
384-dimensional difference between mean MiniLM vectors of accepted and declined options, pooled within each half.
Fixed gallery; minimum four sessions and two prompts; prompt-alternating split; B-only feature scaling. Median feature reliability is not a reliability of the whole channel. 55 of 200 fake assignments are unsupported. The content channel is uninterpretable for admission; this does not establish absence of taste.
Evidence: [PMAP.json](PMAP.json); FINDINGS: S15-DRIVE-PMAP-1018.

### V3.accept: Acceptance

**Input:** suggestion log, prompt metadata, decoding setting.
One-hot acceptance response. Observed-response ICC over writers conditional on these exact prompts. Prompt main-effect variance excluded; setting, writer-by-prompt and residual variance retained. Handling primary is the trace ratio across four one-hot response coordinates.
Gaussian observed-response moment model, not latent multinomial reliability. Prompt main variance excluded; setting, writer-by-prompt and residual variance retained. D17 slow-pause feature was absent from V3; its later completion was not retroactively fitted.
Evidence: [V3.json](V3.json); FINDINGS: OPS-S15-D14-PILOT-1018.

Other reliability: response-scale model intraclass correlation, not split-half; estimate 0.0933379, 95% interval [0.0286712, 0.135921].

### V3.edit: Editing

**Input:** suggestion log, prompt metadata, decoding setting.
One-hot edited-acceptance response. Observed-response ICC over writers conditional on these exact prompts. Prompt main-effect variance excluded; setting, writer-by-prompt and residual variance retained. Handling primary is the trace ratio across four one-hot response coordinates.
Gaussian observed-response moment model, not latent multinomial reliability. Prompt main variance excluded; setting, writer-by-prompt and residual variance retained. D17 slow-pause feature was absent from V3; its later completion was not retroactively fitted.
Evidence: [V3.json](V3.json); FINDINGS: OPS-S15-D14-PILOT-1018.

Other reliability: response-scale model intraclass correlation, not split-half; estimate 0.113061, 95% interval [0.0336545, 0.164567].

### V3.dismiss: Dismissal

**Input:** suggestion log, prompt metadata, decoding setting.
One-hot dismissal response. Observed-response ICC over writers conditional on these exact prompts. Prompt main-effect variance excluded; setting, writer-by-prompt and residual variance retained. Handling primary is the trace ratio across four one-hot response coordinates.
Gaussian observed-response moment model, not latent multinomial reliability. Prompt main variance excluded; setting, writer-by-prompt and residual variance retained. D17 slow-pause feature was absent from V3; its later completion was not retroactively fitted.
Evidence: [V3.json](V3.json); FINDINGS: OPS-S15-D14-PILOT-1018.

Other reliability: response-scale model intraclass correlation, not split-half; estimate 0.0180285, 95% interval [0.00494996, 0.0457727].

### V3.ignore: Ignoring

**Input:** suggestion log, prompt metadata, decoding setting.
One-hot ignore response. Observed-response ICC over writers conditional on these exact prompts. Prompt main-effect variance excluded; setting, writer-by-prompt and residual variance retained. Handling primary is the trace ratio across four one-hot response coordinates.
Gaussian observed-response moment model, not latent multinomial reliability. Prompt main variance excluded; setting, writer-by-prompt and residual variance retained. D17 slow-pause feature was absent from V3; its later completion was not retroactively fitted.
Evidence: [V3.json](V3.json); FINDINGS: OPS-S15-D14-PILOT-1018.

Other reliability: response-scale model intraclass correlation, not split-half; estimate 0.00519642, 95% interval [0.00118657, 0.0101039].

### V3.handling: Combined handling

**Input:** suggestion log, prompt metadata, decoding setting.
Trace ratio across the four one-hot handling coordinates. Observed-response ICC over writers conditional on these exact prompts. Prompt main-effect variance excluded; setting, writer-by-prompt and residual variance retained. Handling primary is the trace ratio across four one-hot response coordinates.
Gaussian observed-response moment model, not latent multinomial reliability. Prompt main variance excluded; setting, writer-by-prompt and residual variance retained. D17 slow-pause feature was absent from V3; its later completion was not retroactively fitted.
Evidence: [V3.json](V3.json); FINDINGS: OPS-S15-D14-PILOT-1018.

Other reliability: response-scale model intraclass correlation, not split-half; estimate 0.0739271, 95% interval [0.0268312, 0.105969].

### V3.hover_events_log1p: Logged hover count

**Input:** suggestion log, prompt metadata, decoding setting.
log(1 + number of logged hover events per menu). Observed-response ICC over writers conditional on these exact prompts. Prompt main-effect variance excluded; setting, writer-by-prompt and residual variance retained. Handling primary is the trace ratio across four one-hot response coordinates.
Gaussian observed-response moment model, not latent multinomial reliability. Prompt main variance excluded; setting, writer-by-prompt and residual variance retained. D17 slow-pause feature was absent from V3; its later completion was not retroactively fitted.
Evidence: [V3.json](V3.json); FINDINGS: OPS-S15-D14-PILOT-1018.

Other reliability: response-scale model intraclass correlation, not split-half; estimate 0.13016, 95% interval [0.0441191, 0.313072].

### V3.typed_since_previous_log1p: Typed text between decision resolutions

**Input:** suggestion log, prompt metadata, decoding setting, revision history.
log(1 + user-inserted Unicode codepoints after the previous decision resolution through the current decision resolution), using the frozen D15 row definition. Observed-response ICC over writers conditional on these exact prompts. Prompt main-effect variance excluded; setting, writer-by-prompt and residual variance retained. Handling primary is the trace ratio across four one-hot response coordinates.
Gaussian observed-response moment model, not latent multinomial reliability. Prompt main variance excluded; setting, writer-by-prompt and residual variance retained. D17 slow-pause feature was absent from V3; its later completion was not retroactively fitted.
Evidence: [V3.json](V3.json); FINDINGS: OPS-S15-D14-PILOT-1018.

Other reliability: response-scale model intraclass correlation, not split-half; estimate 0.246182, 95% interval [0.0799958, 0.418158].

### V3.observed_menu_latency_log1p: Observed menu latency

**Input:** suggestion log, prompt metadata, decoding setting.
log(1 + observed selection/close latency); right-censored menu latencies excluded. Observed-response ICC over writers conditional on these exact prompts. Prompt main-effect variance excluded; setting, writer-by-prompt and residual variance retained. Handling primary is the trace ratio across four one-hot response coordinates.
Gaussian observed-response moment model, not latent multinomial reliability. Prompt main variance excluded; setting, writer-by-prompt and residual variance retained. D17 slow-pause feature was absent from V3; its later completion was not retroactively fitted. 16,719 observed latencies; 136 right-censored records excluded.
Evidence: [V3.json](V3.json); FINDINGS: OPS-S15-D14-PILOT-1018.

Other reliability: response-scale model intraclass correlation, not split-half; estimate 0.238091, 95% interval [0.0880566, 0.403804].

### D17.slow_pause: Slow-pause mixture share

**Input:** timing log, revision history.
Take the natural log of each between-word interval divided by that session's median within-word interval, excluding system waits; choose a pooled Gaussian mixture with 1-5 components by BIC; estimate writer ICC of each session's slowest-component weight.
Three fitted components are not identified psychological processes. Interval conditions on selected mixture. Logged timing carries device and recording context; multi-character deltas and zero intervals are explicitly excluded.
Evidence: [D17-complete.json](D17-complete.json); FINDINGS: S15-DRIVE-MIDRUN-LANDINGS-1018.

Other reliability: writer ICC conditional on selected pooled mixture; estimate 0.724086, 95% interval [0.538672, 0.798951].

### D29.length: Final length across sessions

**Input:** final text, revision history.
Pearson correlation of each writer's mean final nonprompt word count across odd/even chronological sessions; at least two sessions per half.
Shared prompts and minimum-time expectations can contribute. Histograms and the separate timer extension do not establish a personal stopping mechanism.
Evidence: [D29.json](D29.json); FINDINGS: S15-DRIVE-W2-1018.

Other reliability: single Pearson split-half correlation, not Spearman-Brown median; estimate 0.908123, 95% interval [0.839516, 0.958549].

### EV1.departure_pause: Departure and preceding pause

**Input:** final text, revision history, timing log, pinned engine reader export.
Log pre-word pause on page-normalized departure, session-demeaned with the exact original D33 nuisance/history terms. Fixed session intercepts span the nested writer and session intercept space. Covariates: population_surprisal, length, log_frequency, sentence_start, after_punctuation, personal_surprise.
Conditional within-session association on 44,494 surviving engine-scored original words; does not establish causal attention or explain rejected/vanished words. Departure is page-normalized allocated token surprisal, not a complete-word probability.
Evidence: [EV1_PAUSES_V1.json](EV1_PAUSES_V1.json); FINDINGS: S15-EV1-PAUSES-V1.

### EV2.signatures: Cross-work signature recurrence

**Input:** final text, prompt metadata, writer grouping, pinned engine reader export.
Lowercase literal phrase recurrence per 10,000 words in the same writer's other-prompt documents minus other writers on those same prompts; equal-writer means of equal-source-document means.
S1 common supported document-signatures across real and all200 prompt-stratified session reassignments Aggregate recurrence only; not a guarantee about each signature, private preferences or exclusively human wording.
Evidence: [EV2_SIGNATURES_V1.json](EV2_SIGNATURES_V1.json); FINDINGS: S15-EV2-SIGNATURES-V1.

### EV3.final: Engine habits: final document

**Input:** final text.
Unweighted per-session engine-vector averages; PMAP prompt-alternating halves, B-only scaling and fixed gallery; paired real-probe bootstrap; whole-session fake labels preserve counts. 17 features: eight connectives and six punctuation rates using unquoted-word denominator, sentence-length mean/spread, quoted-word share.
INVALID for habit-identification admission despite a positive observed contrast; no habit badge. Per-session averages do not pool word counts; distinct feature population from PMAP.
Evidence: [EV3_IDENTIFICATION_V1.json](EV3_IDENTIFICATION_V1.json); FINDINGS: S15-EV3-IDENTIFICATION-V1.

### EV3.typed: Engine habits: surviving typed fragments

**Input:** final text, revision history.
Unweighted per-session engine-vector averages; PMAP prompt-alternating halves, B-only scaling and fixed gallery; paired real-probe bootstrap; whole-session fake labels preserve counts. 17 features: eight connectives and six punctuation rates using unquoted-word denominator, sentence-length mean/spread, quoted-word share.
INVALID for habit-identification admission despite a positive observed contrast; no habit badge. Per-session averages do not pool word counts; distinct feature population from PMAP. Surviving typed fragments with synthetic gap boundaries, not historical continuous prose
Evidence: [EV3_IDENTIFICATION_V1.json](EV3_IDENTIFICATION_V1.json); FINDINGS: S15-EV3-IDENTIFICATION-V1.

## Aggregate dictionaries and delivery

The exact 100-word, 22-punctuation and 20-bigram arrays remain local, with ordered-array hashes in the JSON. The [official CoAuthor page](https://coauthor.stanford.edu/) and local extracted release did not provide explicit dataset licence terms in this inspection. Under the reversible S5 interpretation, this preparation uses the packet's hash form while release terms are resolved; it does not assert that aggregate definitions are prohibited. Source text, participant identities and profiles are absent. The engine's own list must remain labelled as its own; the earlier observed 47% belongs only to Sounding Line's definition and retains its control warning.

Delivery is scheduled by the packet for Tuesday October 20 at 20:00 PDT. This file is prepared locally, not delivered or deployed. The receiving copies and hashes still require a receipt. Original result reports and the Sunday final-packet policy remain authoritative.

[Machine-readable table and source hashes](ENGINE_FEATURE_TABLE.json).
