# October 11 review-service repair

The scientific supervisor survived, but the notifier failed on October 10 at
14:00 Pacific. Its next reviewer exited before reading stdin because its detached
environment could not resolve the user home. Writing to that closed pipe then
terminated the notifier. Four scheduled health inspections were missed. No
successful inspection or acknowledgement is backfilled for those missed checks.

The final queued family replay applied its declared verification hold at 15:18;
the last in-flight scientific worker completed at 15:39. The October 11 native
inspection found the supervisor alive, no scientific workers, 108 complete items
and 160 waiting. No scientific worker crashed and no failure was silently erased.
This is an operational outage, not evidence that the whole queue finished.

The frozen notifier replacement supplies the existing user profile to the child
and uses file-backed stdin. An early child exit is retained as a failed delivery,
with bounded retries. Routine completions accumulate until a health checkpoint;
failures, a held queue or a queue needing refill may request earlier attention.
One review owner remains the rule. Scientific capsules and acceptance are unchanged.

The owner replaced six-hour checks with twelve-hour checks. The new schedule is
02:00 and 14:00 Pacific, beginning October 12 at 02:00. After one final report and
verified owned-science closure, a hash-bound closeout receipt retires the notifier;
it stops generating checkpoints or requesting model reviews.

Ten focused controls passed, including early child failure, profile restoration,
twelve-hour checkpoints, batching and closed-campaign retirement. A real installed
CLI reviewer acknowledged the exact fixture event and exited successfully. The
replacement production owner loaded the frozen source and twelve-hour configuration;
it stays disarmed while the operating review is active. Final arming is a separate
native handoff receipt. [Validation](../../../results/v21/notifier-repair-20261011/VALIDITY.json).
The tested delivery uses the [documented noninteractive CLI](https://learn.chatgpt.com/docs/non-interactive-mode).

Completed scientific records undergo a separate all-packet integrity and score
check, plus the five required full family replay comparisons. That check does not
by itself accept every scientific contrast or promote a confirmation candidate.
The current health report records its actual outcome and the queue resumption.
