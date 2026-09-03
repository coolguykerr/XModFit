#!/usr/bin/env python
"""Regenerate ``Data/1ubq.pdb``, the default template for the XLayersPDB family.

This is the provenance of the shipped default, and the seed for the general PDB
preparation step wanted in ``handoff/10-fix-list.md`` B13.

Usage::

    # fetch the source structure once (any method; not done by this script)
    curl -o 1UBQ.pdb https://files.rcsb.org/download/1UBQ.pdb
    python scripts/make_1ubq_default.py 1UBQ.pdb Data/1ubq.pdb

Two preparation steps are applied, and neither is optional -- each compensates
for something the model does not do for itself:

1. **Strip waters / heteroatoms.**  ``Functions.pdb_cache.pdbread`` iterates every
   atom ``Bio.PDB`` returns with no filter, so 1UBQ's 58 crystallographic HETATM
   waters would be counted as protein: electrons, van der Waals radii and mass all
   inflate, and that propagates into ``protvol``, the coverage normalisation and the
   electron-density profile.
2. **Centre on the mass-weighted centre of mass.**  ``XLayersPDB.ED_profrot``
   rotates the raw coordinates about the origin (its own CoM step is currently
   commented out, deliberately -- see fix-list A1.14), so an off-centre template
   swings through an arc as theta/phi vary.  Once A1.14 is reverted this step
   becomes redundant rather than wrong.

Deliberately stdlib-only: it must run in any interpreter, without Bio/numpy.

Ubiquitin is a small, standard, published structure chosen as a working default
and smoke test.  It is NOT a membrane protein and carries no physical meaning for
these fits.
"""

import sys

# Elements occurring in a protein-only PDB.  Anything else is a signal that the
# input was not what we think it is, so raise rather than guess.
MASS = {'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'S': 32.06, 'P': 30.974}


def load_protein_atoms(path):
    """Return [(line, element, x, y, z)] for ATOM records only."""
    atoms = []
    for line in open(path):
        if not line.startswith('ATOM  '):
            continue
        element = line[76:78].strip() or line[12:16].strip()[0]
        if element not in MASS:
            raise SystemExit('unknown element %r in %r' % (element, line[:30]))
        atoms.append((line, element,
                      float(line[30:38]), float(line[38:46]), float(line[46:54])))
    if not atoms:
        raise SystemExit('no ATOM records found in %s' % path)
    return atoms


def centre_of_mass(atoms):
    masses = [MASS[a[1]] for a in atoms]
    total = sum(masses)
    return tuple(sum(m * a[2 + i] for m, a in zip(masses, atoms)) / total
                 for i in range(3))


def write(path, atoms, com, n_dropped):
    cx, cy, cz = com
    with open(path, 'w', newline='\n') as out:
        out.write('HEADER    UBIQUITIN (1UBQ), WATERS REMOVED, MASS-CENTRED AT ORIGIN\n')
        out.write('REMARK   1 Source: RCSB PDB entry 1UBQ, Vijay-Kumar/Bugg/Cook 1987,\n')
        out.write('REMARK   1 "Structure of ubiquitin refined at 1.8 A resolution".\n')
        out.write('REMARK   2 Default template for the XLayersPDB model family. Chosen as a\n')
        out.write('REMARK   2 small, standard, published structure -- it is NOT a membrane\n')
        out.write('REMARK   2 protein and is intended only as a working default / smoke test.\n')
        out.write('REMARK   3 %d HETATM water records removed; coordinates translated so the\n'
                  % n_dropped)
        out.write('REMARK   3 mass-weighted centre of mass sits at the origin, because\n')
        out.write('REMARK   3 XLayersPDB.ED_profrot rotates about the origin without centring.\n')
        out.write('REMARK   4 Regenerate with scripts/make_1ubq_default.py\n')
        for line, _element, x, y, z in atoms:
            out.write('%s%8.3f%8.3f%8.3f%s' % (line[:30], x - cx, y - cy, z - cz, line[54:]))
        out.write('END\n')


def main(argv):
    if len(argv) != 3:
        raise SystemExit(__doc__)
    src, dest = argv[1], argv[2]
    n_dropped = sum(1 for line in open(src) if line.startswith('HETATM'))
    atoms = load_protein_atoms(src)
    com = centre_of_mass(atoms)
    write(dest, atoms, com, n_dropped)

    # Re-read what we wrote and confirm the invariant, rather than trusting it.
    check = load_protein_atoms(dest)
    ncx, ncy, ncz = centre_of_mass(check)
    span = [max(a[2 + i] for a in check) - min(a[2 + i] for a in check) for i in range(3)]
    print('atoms kept        : %d' % len(check))
    print('heteroatoms dropped: %d' % n_dropped)
    print('CoM removed       : %.3f %.3f %.3f' % com)
    print('new CoM           : %.4f %.4f %.4f  (must be ~0)' % (ncx, ncy, ncz))
    print('extent x/y/z (A)  : %.1f %.1f %.1f' % tuple(span))
    if max(abs(ncx), abs(ncy), abs(ncz)) > 1e-3:
        raise SystemExit('FAIL: output is not centred')
    print('OK')


if __name__ == '__main__':
    main(sys.argv)
