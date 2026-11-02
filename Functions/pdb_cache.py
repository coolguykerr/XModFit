"""
pdb_cache.py -- shared, input-keyed caches for the XLayersPDB* model family.

Every XLayersPDB* variant (XLayersPDB, XLayersPDB_G/_H/_box, and any future
XLayersPDB_<name>) reads the same PDB file and builds the same atom table before
running its own (per-variant) electron-density physics.  Historically each variant
carried its OWN ``pdbread`` as an ``@lru_cache``-decorated METHOD, so the cache was
keyed on ``self`` (object identity) with ``maxsize=2``.  Consequences:

  * two model instances -- e.g. two datasets, or a model and its ``deepcopy`` in the
    joint-fit path -- never shared a cached read even for the SAME .pdb file;
  * with more than two live PDB-model instances the tiny cache thrashed and every
    dataset switch re-parsed the structure (Bio.PDB) AND re-fetched the mendeleev
    element table.

This module lives in ``Functions/`` (NOT inside a category folder like
``Functions/XRR/``) on purpose: XModFit treats every ``.py`` in a category folder
as a fittable model class named after the file, so a helper module there would be
mis-listed as a function and break the function list.

This module lifts the byte-identical, ``self``-independent parts out to module-level
functions keyed on their real inputs, so the read is shared across instances,
``deepcopy``s, datasets and variants, and fetched from mendeleev only once per
process.  The per-variant physics (``ED_profrot`` / ``calcProfComb`` and their
bespoke kernels) stays in each model file -- only the common PDB scaffolding lives
here.
"""

from functools import lru_cache

import numpy as np
import Bio.PDB

try:
    from mendeleev.fetch import fetch_table
except Exception:  # pragma: no cover - depends on mendeleev version
    fetch_table = None
    from mendeleev import get_table


@lru_cache(maxsize=1)
def _element_table():
    """Return the ``{symbol: [atomic_number, vdw_radius, atomic_weight]}`` map.

    Fetched from mendeleev once per process (the fetch is the slow part of a PDB
    read on a cold cache).  Mirrors the original inline fetch in XLayersPDB.pdbread.
    """
    if fetch_table is not None:
        ptable = fetch_table('elements')[['symbol', 'atomic_number', 'vdw_radius', 'atomic_weight']]
    else:
        ptable = get_table('elements')[['symbol', 'atomic_number', 'vdw_radius', 'atomic_weight']]
    return ptable.set_index('symbol').T.to_dict('list')


def _lookup(atominfo, element, atom_name, fname):
    """``atominfo`` row for a Bio.PDB element symbol, tolerating its casing.

    mendeleev keys the table by proper-case symbols (``Nd``, ``Zn``, ``Ca``) while
    Bio.PDB upper-cases what it reads from the element columns and what it guesses
    from an atom name, so ANY two-letter element missed the table and raised a bare
    ``KeyError: 'ND'`` from inside the cache -- reproduced on ``Data/Nd9.pdb``.
    Single-letter elements (the whole of a plain protein: C, N, O, S, H) were
    unaffected, which is why it went unnoticed; selenomethionine, a bound Zn/Mg/Fe,
    or a chloride would each have hit it.

    Exact key first, so nothing that already worked can change; then the
    capitalised form; then a message that names the file and the atom.
    """
    for key in (element, str(element).capitalize()):
        if key in atominfo:
            return atominfo[key]
    raise KeyError(
        "%s: no element data for atom %r (Bio.PDB element %r). If the PDB leaves "
        "columns 77-78 blank, Bio.PDB guesses the element from the atom NAME, and "
        "for a non-standard atom that guess is often wrong -- set the element "
        "column, or add a name override next to the CAL one in pdb_cache.pdbread."
        % (fname, atom_name, element))


@lru_cache(maxsize=32)
def pdbread(fname):
    """Parse ``fname`` and return per-atom ``(x, y, z, electrons, radius, mass)``.

    Keyed on ``fname`` alone, so all XLayersPDB* instances reading the same file
    share one parse.  Body is the byte-for-byte equivalent of the original
    ``XLayersPDB.pdbread`` method (element table now via ``_element_table()``).

    **``mass`` is load-bearing geometry for XLayersPDB_XPR**, not bookkeeping. Since
    XPR's 2026-09-08 revision it re-centres the coordinates on the MASS centroid and
    measures ``protins`` from there, so a mis-typed atom moves the zero of the
    insertion parameter. The ``CAL`` override below therefore has to set the mass as
    well as the electron count and the radius -- upstream ``XPRsim._import_pdb``
    overrides only the latter two and takes the mass from ``atom.element``, which on
    a blank element column Bio.PDB guesses as ``'C'``; that is worth 0.0345 A of
    centroid shift on ``XPR-main/test_data/protein.pdb``. Audited 2026-09-08 over
    the 18 structures in ``Data/`` and ``../xpr-seeding-study``: every ``CAL`` is
    correctly given calcium's 40.078 here.
    """
    atominfo = _element_table()
    pdbcode = 'struct'

    pdbparser = Bio.PDB.PDBParser(QUIET=True)
    struct = pdbparser.get_structure(pdbcode, fname)

    radius = np.zeros(len(list(struct.get_atoms())))
    electrons = np.zeros(len(list(struct.get_atoms())))
    mass = np.zeros(len(list(struct.get_atoms())))
    coord = np.zeros((len(list(struct.get_atoms())), 3))

    ind = 0

    for atom in struct.get_atoms():
        coord[ind, :] = atom.coord
        if atom.name == 'CAL':
            electrons[ind] = atominfo['Ca'][0]
            radius[ind] = atominfo['Ca'][1] / 100
            mass[ind] = atominfo['Ca'][2]
        else:
            row = _lookup(atominfo, atom.element, atom.name, fname)
            electrons[ind] = row[0]
            radius[ind] = row[1] / 100
            mass[ind] = row[2]
        ind += 1

    xcoord = np.squeeze(coord[:, 0])
    ycoord = np.squeeze(coord[:, 1])
    zcoord = np.squeeze(coord[:, 2])

    return xcoord, ycoord, zcoord, electrons, radius, mass
