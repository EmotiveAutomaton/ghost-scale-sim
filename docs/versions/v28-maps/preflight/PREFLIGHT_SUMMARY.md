# Ghost V28 pre-flight summary (analyst, 11 November 2026)

Scripts and raw outputs are in this folder. numpy only, fixed seeds. These numbers size the study and seed its known-answer checks. They are not results.

Files: `g1_joint_history.py` (spec as first written), `g1_ext.py` (adds the empirical-habit loop and disjoint high overlap), `g1_ext2.py` (adds typical-habit readers), `g1_scale.py` and `g1_scale_b.py` (larger world), `g2_setpoints.py`, `g2_signature.py`, `g3_sacrifice.py`, each with `*_out.txt`.

## G1 small world (V 12, H 8, G 4, T 64), 4,000 worlds per cell, gap to the joint reader J in nats

Columns are K = 0 / 1 / 4 / 16 earlier documents. "high" keeps goal triples disjoint.

| Reader | low, rho .2 | low, rho .3 | low, rho .4 | high, rho .2 | high, rho .3 | high, rho .4 |
|---|---|---|---|---|---|---|
| Ps (library, passes) | .08/.07/.02/.01 | .26/.22/.11/.06 | .31/.27/.19/.14 | .09/.09/.05/.04 | .53/.32/.17/.08 | 1.02/.79/.43/.27 |
| I(1) (library, one loop) | .00/.00/.00/.00 | .00/.00/.00/.00 | .00/.00/.00/.00 | .00/.00/.00/.00 | .05/.00/.00/.00 | .13/.03/.00/.00 |
| Pe (own tokens, one pass) | .51/.25/.14/.06 | .25/.13/.05/.02 | .07/.07/.02/.00 | .90/.43/.21/.10 | .60/.27/.11/.04 | .26/.14/.04/.01 |
| PeL(4) (own tokens, loop) | 1.05/.33/.05/.01 | .96/.18/.01/.00 | .41/.11/.00/.00 | 1.93/.64/.08/.01 | 2.39/.63/.03/.00 | 1.70/.30/.01/.00 |
| Pp (one typical habit) | 2.90/2.71/2.86/2.62 | not run | .55/.42/.47/.45 | 4.03/4.13/3.85/4.19 | not run | 1.05/.89/1.06/1.04 |

- J's own mean log-score is between -0.04 and 0.00 in every cell: the joint reader nearly always finds the goal.
- Tail share (true goal given under 1%): PeL(4) at K = 0 is .02 to .27; Pe at K = 0 is .002 to .024; Pp is .035 to .36 at every K; J is at most .001.
- Paired standard error of J minus PeL(4): .04 to .09 at K = 0, under .01 from K = 4.
- A prior of 8 or 32 pseudo-tokens centred on the typical habit (PpS) behaves like Pe; with the loop (PpSL) like PeL.
- From the first run (400 worlds): beam with one habit type is worse than Ps and overconfident; J's departure-mark AUC is .90 to .93 at every K (within .02 of an oracle that knows the true habit and goal); Pe's rises from .58 to .68 at K = 0 to .87 to .90 at K = 16. At rho .1 nothing separates (all gaps under .02) so rho .1 is dropped for the small world.

## G1 larger world (V 200, H 32, G 8, 10 target moves, T 128), 800 worlds per cell

Gap to J in nats, K = 0 / 1 / 4 / 16; then departure-mark AUC at K = 0 and K = 16.

| Reader | low, rho .05 | low, rho .10 | high, rho .05 | high, rho .10 |
|---|---|---|---|---|
| Ps | .00 throughout | .00 | .00 | .00 |
| Pp | 1.81/1.75/1.88/1.53 | .74/.80/.58/.59 | 4.90/5.03/4.82/4.94 | 6.39/5.36/4.99/6.12 |
| Pe | .83/.58/.33/.15 | .16/.11/.03/.02 | 1.42/.95/.44/.21 | 1.52/.55/.14/.07 |
| PeL(4) | .74/.46/.23/.08 | .17/.10/.03/.01 | 1.67/1.04/.34/.10 | 3.06/1.19/.13/.03 |

- Mark AUC: J .97 to .98 everywhere. Pe .81 to .95 at K = 0, .95 to .97 at K = 16. Pp .61 to .89.
- At T 256 with rho .2 or .4 this world is trivial (every gap .00 except high overlap, rho .2), so those settings are not used.

## G2 (visible stock)

- At action cost 1 (as first written): a restore is chosen 27% of the time with no deficit, the stock sits near 20, deficits (s below theta minus 2) occur on .1 to .4% of ONE's steps. DULL and SHARP never differ from ONE under N. A spend is always available at value -1, so POINT and BAND separate from every no-spend type in 2 to 5 records. W resolves nothing that N or D does not.
- At action cost 3: deficits on 20% of ONE's steps; DULL against ONE separates under N in a median 141 records (21% censored), SHARP against ONE in 85 (0% censored); truth ONE(12, 1) gets posterior .95 on theta 12 and .95 on pi 1 by step 256. Only D separates ONE from DULL as truth; SHARP at kap .02 against ONE is resolved by nothing.
- Nesting: DULL, SHARP and MOVE contain ONE as their rate goes to 0, and their .02 grid points nearly equal ONE, so ln p(ONE)/p(rival) is capped near ln 2 = .69.
- Centre setting, cost 1, regime N, median records (rows truth, columns rival): POINT against BAND 24; BAND against POINT 48; MOVE against ONE 95; ONE against MOVE 125.
- Signature table (`g2_signature.py`; cost 3, regime N, 256 steps, 200 worlds per truth; every truth has reference 12 and precision 1). A POINT reader fitted by maximum likelihood (theta 4 to 22, pi in .25, .5, 1, 2, 4) returned: POINT truth, theta 12 and pi 1 in 99% of worlds; ONE, median theta 14 (10th to 90th percentile 12 to 14), pi .5 in 74%, loss .061 nats per step; BAND, theta 12 (12 to 14), pi 1 in 58%, loss .017; MOVE, theta 9 (7 to 11), pi .25 in 100%, loss .113, mean stock 4.2; DULL, theta 13 (4 to 14), pi .25 in 59%, loss .125; SHARP, theta 15 (13 to 15), pi .5 in 63%, loss .102.
- CPU: about 1.8 s per 200-world cell of 256 steps.

## G3 (sacrifice moments)

- Steps with no unmet need change no likelihood (max increment 0.0). CAPTURED(3) equals a SAVER whose reserve is never reached. A reader that scores only steps with an unmet need keeps exactly 100% of the evidence. CAPTURED(3) against SAVER(20) settles after a median 1 to 4 sacrifice moments wherever it settles.
- Truth CAPTURED(3) against SAVER(10) / SAVER(15) / SAVER(20), median records (censored share), M0 = 5:
  - q_in .3: q_need .05: 39/63/89 (0); .15: 101 (.15) / 331 (.39) / aliased (.59); .30: aliased (.99 to 1.0).
  - q_in .5: .05: 27/40/54 (0); .15: 26/52/72 (0); .30: 328 (.39) / aliased (.70) / aliased (.89).
  - q_in .8: .05: 19/24/31; .15: 13/23/31; .30: 17/35/54 (all 0).
- Share of worlds where M ever exceeds 10 / 20: 100% / 100% when q_in is .8, or .5 with q_need at most .15; (.3, .15): 77 to 98% / 29 to 81%; (.5, .3): 30 to 92% / 1 to 69%; (.3, .3): at most 17% / at most 1%.
- Weights (3,1) against (1,3) separate in 5 to 37 records, never censored; the share of that evidence from steps with both needs unmet tracks q_need.
- TOOL(2) against TOOL(4) is aliased (85% censored) at the centre: all the evidence comes from the one climb from M0 = 5.
- CPU: .76 s per 200-world cell of 512 steps.

## Independent check (11 November)

A second implementation, written from the packet's text alone before these files were opened (`independent_check/` in this bundle). Gap to J in nats, small world, rho .3, 4,000 worlds; "check / pre-flight".

| Cell | K = 0 | K = 4 |
|---|---|---|
| low overlap, I(1) | .01 / .00 | .00 / .00 |
| low, Ps | .28 / .26 | .17 / .11 |
| low, Pe | .22 / .25 | .05 / .05 |
| low, PeL(4) | .81 / .96 | .01 / .01 |
| high overlap, I(1) | .01 / .05 | .00 / .00 |
| high, Ps | .42 / .53 | .18 / .17 |
| high, Pe | .58 / .60 | .09 / .11 |
| high, PeL(4) | 2.25 / 2.39 | .02 / .03 |

- PeL(4) at low overlap, K = 0 differed by .15 at 4,000 worlds; at 40,000 worlds the check gave .88. Standard errors over blocks are 1.4 to 1.7 times those over worlds (PeL(4) gap: .058 against .034).
- Pe minus PeL(4) in log-score at K = 0, 40,000 worlds: .64 (low overlap) and 1.76 (high).
- G2, cost 3, truth ONE(12, 1): share of steps with the stock below 10, .196 (pre-flight .20); posterior on theta 12 at step 256 over ONE's grid, mean .949 with 88% of worlds at or above .9 (pre-flight .95); POINT truth recovered at (12, 1) in 99% (99%); ONE read by POINT as theta 14, pi .5 in 72% (14, .5 in 74%). ONE against MOVE separated in 87 records (1% never); ONE against SHARP in 221 (32% never).
- G3, CAPTURED(3) against SAVER(10), M starting at 5: (.3, .15) 106.5 records with .14 never separated (101, .15); (.5, .3) 322 (328); (.5, .15) 23 (26). Steps with no unmet need changed no likelihood.
- Not implemented in the check: G1's mark AUC and larger world, G2's tables 1 to 3 in full, the 1,024-step cell.
