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


@lru_cache(maxsize=32)
def pdbread(fname):
    """Parse ``fname`` and return per-atom ``(x, y, z, electrons, radius, mass)``.

    Keyed on ``fname`` alone, so all XLayersPDB* instances reading the same file
    share one parse.  Body is the byte-for-byte equivalent of the original
    ``XLayersPDB.pdbread`` method (element table now via ``_element_table()``).
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
            electrons[ind] = atominfo[atom.element][0]
            radius[ind] = atominfo[atom.element][1] / 100
            mass[ind] = atominfo[atom.element][2]
        ind += 1

    xcoord = np.squeeze(coord[:, 0])
    ycoord = np.squeeze(coord[:, 1])
    zcoord = np.squeeze(coord[:, 2])

    return xcoord, ycoord, zcoord, electrons, radius, mass
