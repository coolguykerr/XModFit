<!-- Template for handoff/NN-<topic>.md. Keep the section order and headings. -->

# <Topic> — XModJointFit change record

**Category:** <one line>
**Status:** <Production / Working-but-unvalidated-live / Experimental / Research harness>
**Owner chat(s):** <which memory files / sessions produced it>
**Last touched:** <date from git/memory>

## 1. Why this exists

What scientific/analysis problem it solves, and which XModFit models or workflows it is
meant to be used with. 3-10 sentences. No code.

## 2. Files

| File | Lines | State | Role |
|---|---|---|---|
| `path` | ~N | new / modified / new-untracked | one line |

State legend: `new` = added in a local commit; `modified` = upstream file we changed;
`new-untracked` = exists only in the working tree (not committed anywhere).

## 3. How it works

Design and the load-bearing decisions, each with *why*. Cite `file.py:LINE` for
anything a reader would otherwise have to grep for. Include data/control flow for the
main path (e.g. GUI click -> engine call -> result).

## 4. How to use it

GUI path, parameters, env vars, file formats, example invocations. Enough that someone
can reproduce a run without reading the code.

## 5. Contact surface with stock XModFit

The merge-risk section. Every *upstream* file/function this work modifies, what the
change is, and whether it is additive or a rewrite of existing behaviour.

| Upstream file | Function / region | Change | Additive? |
|---|---|---|---|

## 6. Known limitations, gotchas and TODOs

Including anything the notes flag as deliberately deferred, and traps that cost time.

## 7. Claims to verify (audit hooks)

Statements that were asserted by the chat that built this but which THIS document did
not independently confirm. The audit pass will work from these tables.

| # | Claim | Asserted by | Suggested check |
|---|---|---|---|

## 8. Consolidation candidates

Files/branches/params that duplicate each other or look dead, with a recommendation
(keep / merge / drop / move) and the risk of acting on it.
