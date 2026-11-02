"""Fast global search for protein-pose reflectivity models (XLayersPDB_XPR).

A two-stage seeded multistart that finds the deepest basin of a badly multimodal
pose problem in ~2 minutes, and reports a ranked, curvature-certified shortlist of
the alternatives it found.

Read :mod:`fastscan.modes` first. Its docstring states the one thing that governs
every design decision here: the method is a good optimiser and a poor enumerator,
both measured, so the shortlist is a LOWER BOUND on ambiguity and the code is
arranged so that it cannot quietly be presented as anything else.

Layout
------
``modes``      the ``Mode``/``ModeSet``/``Provenance`` records, dedup, certification,
               Laplace weights. Qt-free and Julia-free, hence fully unit-testable.
``seeds``      membrane seed screening, pooling and file IO.
``core``       the search itself: the scan box, and stages 1 and 2.
``homometry``  the stage-0 enumerator (Patterson -> turnpike -> sign partners ->
               basin scan -> spectral family). Deterministic and pure numpy.
``worker``     ``spawn``-child entry point, so a run cannot be raced by the GUI.
``io``         the ``.xmodes`` sidecar.
``solvers.jl`` the Ipopt bridge, loaded through ``Functions.xpr_backend.get_jl()``.

Nothing in this package may import Qt at module level, and nothing but ``worker``
and ``core`` may touch Julia -- that is what keeps the rules testable without a
display or a Julia depot.
"""
__all__ = ['modes', 'seeds', 'core', 'homometry', 'worker', 'io']
