# Fresh-reader reliability weighting

The inherited conditional fitter resets its world counter inside each **256-label**
fresh minibatch. With twelve settings, all sixteen reliability laws are represented,
but the early laws receive more labels. N4b and N4c retain those training weights;
they are not exactly uniform over laws within each fresh minibatch. Fixed training,
validation and complete scientific law blocks remain unchanged.

An initial operator note incorrectly used the **128-case cost-profile sample**
as the fitter's minibatch size and claimed missing law support. The actual fitter
and retained N4c draw records both show 256 labels per fresh minibatch. That
missing-support claim is withdrawn before publication of a scientific conclusion.
No N4b/N4c result was invalidated by it. The original scientific records are unchanged.

N14 and N9 make fresh minibatch weights uniform **in expectation** by independently
randomizing each minibatch's law offset. A control enumerates all sixteen equally
likely offsets and verifies exactly equal total law counts, along with unchanged
fixed-training samples. The original N14 pilot and first successor are retained;
a second successor binds the corrected explanation and complete source. No fit
or production batch was drawn under the earlier new-study pilots.

The original N4b/N4c exact mechanism posteriors, primary gains, aliases and six
saved forecasts remain as reported. Their fresh-reader diagnostics should be
read with the actual unequal law weighting, not relabelled as uniform-law
training or silently replaced with newly fitted predictions.

The frozen N9 protocol calls fixed training/validation allocations "balanced".
That wording is imprecise: each cycles across all sixteen laws, but the final
incomplete cycle gives slightly unequal counts. The 2,048-label set has
120–132 labels per law, the 8,192-label set 504–516, and validation 252–264.
This qualification preserves the running source, actual allocations and stored
forecasts. Scientific batches have exactly 64 cases per law and setting.
