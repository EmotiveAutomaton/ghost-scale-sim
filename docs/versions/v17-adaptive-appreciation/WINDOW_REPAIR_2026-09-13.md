# Background window repair and restored-file inspection

The two blank desktop terminals came from the initial scheduled-task actions:
both entered the console Python executable before child-process window suppression
could apply. Both confirmed Ghost windows were hidden without closing their
consoles. The original worker, supervisor and notification process retained their
identifiers and creation times.

Future task actions now enter a pinned copy of
[`launch_background.py`](../../../runners/launch_background.py) through the same
environment's GUI interpreter. It starts the original explicit scientific Python
command in module form with `CREATE_NO_WINDOW`, file-backed output and error logs,
and propagates its exit code. The live tasks were not restarted. Their triggers,
principals and settings were preserved, with original and replacement task XML
retained locally. Updating definitions without affecting running instances follows
the [documented Task Scheduler behavior](https://learn.microsoft.com/en-us/powershell/module/scheduledtasks/set-scheduledtask).

Two isolated Windows regression tests passed. A temporary native scheduled task
also executed the complete launch path, reported no visible console, captured
both output streams and returned the fixture's intentional exit code 17. The
completed fixture task was removed. AGENTS.md now requires consoleless initial
background entry points and ownership checks before touching existing consoles.

The restored-file inspection found no evidence of inappropriate project files:

- No unexpected tracked code changes or non-cache ignored files in source,
  runner, test, configuration, documentation or workflow directories.
- All 37 worker and 39 notifier manifest-bound files matched. The plan, admission
  and controller bindings also matched.
- One extra packet-test helper in the worker checkout matches tracked source and
  is referenced by retained tests. It remains outside the continuation manifest;
  the active continuation worker does not import that CLI.
- Of 184 historically deleted Git paths, 139 still exist locally. All are V15
  runtime or smoke records, deliberately untracked by earlier commits and last
  written on 7 September. They were retained as historical evidence.
- The outer workspace retains the organized project, source-document and theory
  folders; the previously filed handoff clutter has not reappeared at its root.

Existing pooled-acquisition failure-review edits were legitimate event-review
work. No project files were deleted or quarantined. Detailed desktop, task and
file-audit receipts remain machine-local. This inspection checked the working
tree and active source integrity; it did not reconstruct NTFS restore history or
rehash every retained raw simulation file.
