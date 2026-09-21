# V19 local-goal readout review

Does a predictive bank help recover local goals and operations? Its advantage reverses with evidence: across four training budgets, the bank lowers logarithmic loss by 0.13193 nats with final artifacts alone, but raises it by 0.83956 nats when every operation is witnessed, relative to direct history. With artifacts alone it reaches the exact-reference loss plus 0.02 nats at 512 labels; with full witnesses direct history reaches that target at 2,048 labels and the bank does not. This is an exploratory constructed-method result, miniature — architecture untested; joint process correspondence and human intent are not established.

Goal and operation losses are separated; complete-witness operation prediction includes copying supplied evidence. The exact fresh-episode bank remains a structural rival for preceding-process recovery. [Full evidence and limits](../versions/v19-local-maker/LOCAL_PRIMARY_REPORT.md).
