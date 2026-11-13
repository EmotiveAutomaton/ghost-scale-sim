"""X1's reader (no frozen sums in the grid): is a rule told from a live sum by its buying side?
Same world as x_exchange.py. Reader: LIVE with w in 1..256, RULE with buying weight 1, 2, 4, HUB with h in 2, 4."""
import numpy as np, x_exchange as X
X.HY = [('LIVE', w) for w in (1, 2, 4, 8, 16, 32, 64, 128, 256)] + [('RULE', v) for v in (1, 2, 4)] + [('HUB', h) for h in (2, 4)]
print('truth RULE(2), reader without frozen sums: regime, environment, share of buy offers -> separated at 256 / 1024 | median steps | unresolved')
for reg in ('LOW', 'MIXED'):
    for env in ('plenty', 'scarce'):
        for qbuy in (0.0, 0.1, 0.5):   # at 0.0 the rule must stay unseparated from large live sums
            o, sep, dr, ns = X.run(('RULE', 2), reg, env, seed=7, qbuy=qbuy)
            med = np.median(sep) if np.mean(np.isfinite(sep)) > .5 else float('inf')
            print('  %-5s %-6s buy share %.1f -> %.2f / %.2f | %6s | %.2f' % (reg, env, qbuy, o[256][0], o[1024][0], med, np.mean(~np.isfinite(sep))))
print('truth HUB(4), same reader, MIXED: environment, share of buy offers -> median steps | unresolved')
for env in ('plenty', 'scarce'):
    for qbuy in (0.0, 0.1, 0.5):
        o, sep, dr, ns = X.run(('HUB', 4), 'MIXED', env, seed=11, qbuy=qbuy)
        med = np.median(sep) if np.mean(np.isfinite(sep)) > .5 else float('inf')
        print('  %-6s buy share %.1f -> %6s | %.2f' % (env, qbuy, med, np.mean(~np.isfinite(sep))))
