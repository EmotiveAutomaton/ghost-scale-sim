# U1-X02 independent-review resource failure

Corrections first: this packet produced no scientific result and changes no prior
claim. Its complete original and replay passes remain retained but do not support
interpretation until independent reconstruction finishes.

The 128-world-by-four-setting original pass completed in 297.515625 process CPU
seconds. Replay completed in 293.265625 seconds and wrote the full replay receipt.
The prospective independent-review reservation was 133.7734375 seconds. At the
next resource check the packet had accumulated 732.0625 seconds: 590.78125 for
the two complete execution passes and 141.28125 in the review process tree. The
runner therefore stopped review at its frozen pass boundary. The packet is
`FAILED`; it has no independent-review receipt, completion receipt, aggregate
export, report or claim. Its 939.2265625-second failed reservation remains charged.

This is an engineering resource-estimation failure, not a control failure or a
scientific null. U1-X03 and U1-X04 are held during this separate review rather
than used as blind retries.

A first verification-only healing packet was frozen before execution. It bound and
copied the failed packet's complete original raw index, aggregate and cost record;
it did not redraw or repeat the scientific original pass. Against its separate
963.546875-second shared-pool reservation, deterministic replay used 297.828125
process CPU seconds and independent reconstruction used 162.734375 seconds.

The healing is complete under hash
`54e8f73444d3a6149b405ac266edd0cafe3912c30e6128cdf7b5481c12ead93f`.
All 512 retained worlds replay byte-identically, and separate scalar-posterior and
decimal whole-world reconstruction pass. The retained result now supports claim
`G-V26-FR-U1X02-1`; the original failed reservation and this 462.59375-second
healing charge remain separate. U1-X03 and U1-X04 stay held rather than becoming
automatic descendants or retries.

[Failed packet status](../../../../results/v26/friday-20261104/CURRENT_STATUS.json).
