# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:15:13 2019

@author: pce
"""
import os
import numpy as np
from pymatgen import Structure, Composition, DummySpecie, Element
# from pymatgen.transformations.standard_transformations import DiscretizeOccupanciesTransformation
# from Bio import PDB  ## can't read .cif, only mmcif! Why is our structure community so fractured on computing syntax!?!?!??!?!?!?
# from pdbparser import pdbParser  # futurized fork !

class Discretizomator(object):
    """
    Given a structure with partial site occupancies, create a discrete structure with unit
    occupancy of each site selected by random choice weighted on the normalized partial
    occupancy (probability).

    Useful for turning unitcell models into large explicit supercell models.
    """

    def __init__(self):
        """  """

    def probabilize(self, site):
        """
        Turn partial occupancies into probabilities, inserting vacancies if occupancy is less than 1.

        Note:
            PeriodicSite : collection of species
                -- species [Composition] mapping of Elements and occupancies
                    -- keys [Element(s)]
                    -- values [float|occupancy]

        Returns:
            dict(zip([pymatgen.Element, ], [probability, ])
        """
        k = list(site.species.keys())  # set of elements
        v = np.array(list(site.species.values())) # corresponding occupancies

        if len(k) == 1:  # single atom type or vacancy
            if v < 1.:  # partial occupancy
                p = np.array((v[0], 1.-v[0]))
                k.append(DummySpecie(''))  # Vacancy(self.structure, site)) ~! these seem to create trouble later on

            elif v >= 1.: # single occupancy
                p = v / v
            else:
                print('last site:')
                print(site)
                raise(Exception('something went wrong... couldn\'t probabilize'))

        elif len(k) > 1:  # multiple atom types
            if np.round(v.sum(), 5) < 1.: # partial vacancy
                v = np.squeeze((v, 1.-v.sum()))
                p = v / v.sum()
                k.append(DummySpecie(''))  # Vacancy(self.structure, site))

            elif np.round(v.sum(), 5) >= 1.: # fully site mixed
                p = v / v.sum()
                
            else:
                print('last site:')
                print(site)
                raise(Exception('something went wrong... couldn\'t probabilize'))

        return dict(zip(k, p))

    def apply_transformation(self, structure):
        """
        For each site in a structure, choose a single occupant weighted by the normalized partial
        occupancy (probability) of each element in the composition.

        pop DummySpecie instances (vacant)

        Returns:
            pymatgen.Structure (new instance)
        """

        for site in structure:
            rd = self.probabilize(site)
            k = np.random.choice(list(rd.keys()), p=list(rd.values()))
            if type(k) is DummySpecie:
                # print ('removing %s' % site.__str__())
                structure.remove(site)
            elif type(k) is Element:
                site.species = Composition({k: 1.0})
                # print (site.__str__())
            else:
                raise(Exception('Need Element or DummySpecie as key to manipulate site Composition'))

        return Structure(structure.lattice, structure.species_and_occu, structure.frac_coords ) # return new instance


class PDB():
    """
    Protein Data Bank (.pdb) file format is a punch-card syntax for storing atomic structure information.
    The syntax is thus constrained by hard character limits and alignments.[`1]
    <https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html>`_.

    Note:
        The line writer coerces these character limits regardless of user input!

    This parser is written specifically to support interchange between pymatgen.Structure
    objects and fullRMC compatible input files.

    fullRMC treats a molecule as a collection of atoms sharing the same residue name (e.g. CO2)
    and sequence and segment identifier numbers (cols 18-20, 23-26, and 73-76, respectively)
    `[Bachir] <bachiraoun.github.io/fullrmc/fullrmcFAQ/pdbFile.html>`_.

    The additional information (atom serial, atom name, residue name, sequence number,
    and segment number) must be created as site attributes by the user, else default
    values are supplied (otherwise, no atoms in the structure are grouped).

    Example:
       |          0        1         2         3         4         5         6         7         8
       |          12345678901234567890123456789012345678901234567890123456789012345678901234567890
       |
       |     0    REMARK    this file is generated using 'pdbParser' package
       |     1    REMARK    Boundary Conditions: 54.0  0.0  0.0  0.0  54.0  0.0  0.0  0.0  54.0
       |     2    ATOM      1 C    CO2     1     -19.317   2.636  -9.568  1.00  0.00         1C
       |     3    ATOM      2 O1   CO2     1     -20.330   3.198  -9.506  1.00  0.00         1O
       |     4    ATOM      3 O2   CO2     1     -18.304   2.075  -9.630  1.00  0.00         1O
       |     5    ATOM      1 C    CO2     2     -23.556  16.509  -9.132  1.00  0.00         1C
       |     6    ATOM      2 O1   CO2     2     -23.275  17.063 -10.112  1.00  0.00         1O
       |     7    ATOM      3 O2   CO2     2     -23.837  15.956  -8.152  1.00  0.00         1O
       |     8    ATOM      1 C    CO2     3      12.969 -23.431   3.920  1.00  0.00         1C
       |     9    ATOM      2 O1   CO2     3      12.989 -24.310   3.163  1.00  0.00         1O
       |     10   ATOM      3 O2   CO2     3      12.949 -22.552   4.676  1.00  0.00         1O
       |     11   ATOM      1 C    CO2     4      16.999 -19.836   8.272  1.00  0.00         1C
       |     12   ATOM      2 O1   CO2     4      17.957 -20.165   7.706  1.00  0.00         1O
       |     ...

    """
    @staticmethod
    def _msg(silent, message):
        """ toggle verbose """
        if silent:
            pass
        else:
            print (message)

    def __str__(self):
        """ """
        return self._str_head() + '\n' + '\n'.join(filter(None, [self._str_atom(site) for site in self._structure]))

    def __init__(self, structure):
        """ """
        self._count = 0

        # if self._check_single(structure) is True: ~! this isn't necessary, but 
        # currently there is no write method for multiple occupancy
        self._structure = structure
        self._preprocess(self._structure)
        # else:
        #     raise(Exception('pdb sites must be singlets'))

    def _check_single(self, structure):
        """ require no site mixing """
        return all([len(site.species) == 1 or len(site.species) ==0 for site in structure])

    def _check_attr(self, site, silent=True):
        """
        enter dummy values for serial, name, residue, sequence, and segment attributes
        if not supplied by user

        ~! refactor these into Site.Properties ?
        """
        keys = list(site.properties.keys())
        if not 'serial' in keys:  # hasattr(site, 'serial'):
            site.properties['serial'] = self.count
            self._msg (silent, 'setting  ' + site.__str__() + ' serial to %s ' % self.count)

        if not 'name' in keys:  # hasattr(site, 'name'):
            site.properties['name'] = site.species.elements[0].__str__() + str(site.properties['serial'])
            self._msg (silent, 'setting  ' + site.__str__() + ' name to default == %s ' % site.properties['name'])

        if not 'residue' in keys:  # hasattr(site, 'residue'):
            site.properties['residue'] = 'RRR'
            self._msg (silent, 'setting  ' + site.__str__() + ' residue to default == RRR ')

        if not 'sequence' in keys:  # hasattr(site, 'sequence'):
            site.properties['sequence'] = self.count
            self._msg (silent, 'setting  ' + site.__str__() + ' sequence to %s ' % self.count)

        if not 'segment' in keys:  # hasattr(site, 'segment'):
            site.properties['segment'] = ' '
            self._msg (silent, 'setting  ' + site.__str__() + " segment to default == ' ' ")

        return

    def _get_vectors(self, structure):
        """ get basis vectors in direct space """
        return np.ravel(structure.lattice.matrix)

    def _get_abc_angles(self, structure):
        """ get unit cell """
        return list(np.ravel(structure.lattice.lengths_and_angles))

    def _preprocess(self, structure, silent=True):
        """ fill default values if absent """
        self.count = 0
        print('\n>>> PDB._preprocess checking site has PDB attributes (serial, name, residue, sequence, segment).\n')
        for site in structure:
            self._check_attr(site)
            self.count += 1
        print('\n>>> PDB._preprocess complete')
        return

    def _str_atom(self, site):
        u"""
        Formatted str for site per .pdb syntax [`1] <https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html>`_.

        .. table:: Protein Data Bank Format: Coordinate Section (* used in this representation).
            :width: auto
            :align: left
        ===========   =======   ===============================   =============   ==========
        Record Type   Columns   Data                              Justification   Data Type
        ===========   =======   ===============================   =============   ==========
        ATOM          1-4  *   “ATOM”                                            character
                      7-11 *   Atom serial number                right           integer
                      13-16*   Atom name                         left            character
                      17        Alternate location indicator                      character
                      18-20*   Residue name                      right           character
                      22        Chain identifier                                  character
                      23-26*   Residue sequence number           right           integer
                      27        Code for insertions of residues                   character
                      31-38*   X orthogonal Å coordinate         right           real (8.3)
                      39-46*   Y orthogonal Å coordinate         right           real (8.3)
                      47-54*   Z orthogonal Å coordinate         right           real (8.3)
                      55-60*   Occupancy                         right           real (6.2)
                      61-66*   Temperature factor                right           real (6.2)
                      73-76*   Segment identifier                left            character
                      77-78*   Element symbol                    right           character
                      79-80     Charge                                            character
        ===========   =======   ===============================   =============   ==========

        """
        rv = []
        for sp, occu in site.species.items():
            strmap = dict(zip(['record',
                               'serial',
                               'atom_name',
                               'residue_name',
                               'sequence',
                               'x',
                               'y',
                               'z',
                               'occupancy',
                               'adp',
                               'segment',
                               'element'
                                      ],
                              ['ATOM  '.ljust(6)[:6], # 1-6 left
                               '{:d}'.format(site.properties['serial']).rjust(5)[:5], # 7-11 right
                               '{}'.format(site.properties['name']).ljust(4)[:4], # 13-16 left
                               '{}'.format(site.properties['residue']).rjust(3)[:3], # 18-20 right
                               '{}'.format(site.properties['sequence']).rjust(4)[:4], # 23-26 right
                               '{:.3f}'.format(site.x).rjust(8)[:8], # 31-38 right
                               '{:.3f}'.format(site.y).rjust(8)[:8], # 39-46 right
                               '{:.3f}'.format(site.z).rjust(8)[:8], # 47-54 right
                               '{:.3f}'.format(occu).rjust(6)[:6], # 55-60 right
                               '{:.3f}'.format(0.0).rjust(6)[:6], # 61-66 right
                               '{}'.format(site.properties['segment']).ljust(4)[:4], # 73-76 left
                               '{}'.format(sp).rjust(2)[:2] # 77-78  right
                                      ]))
            rv.append('{record}{serial} {atom_name} {residue_name}  {sequence}    {x}{y}{z}{occupancy}{adp}      {segment}{element}  '.format(**strmap))
        return '\n'.join(rv)

    def _str_cryst1(self):
        """ for crystal in P1 symmetry """
        fmt = ('{:9.3f}',) * 3 + ('{:7.2f}',) * 3 + (' {:11}',) + ('{:4d}',)
        fill = self._get_abc_angles(self._structure) + ['P1',] + [1,]
        fmtstr = [s1.format(s2) for (s1, s2) in list(zip(fmt, fill))]
        return "CRYST1" + ''.join(fmtstr)

    def _str_head(self):
        """ """
        s1 = "REMARK    This file was generated using pymatgen."
        s2 = "REMARK    Boundary Conditions: %s" % ' '.join(['{:.2f}'.format(s) for s in self._get_vectors(self._structure)])
        s3 = self._str_cryst1()
        s4 = self._str_origxn()
        s5 = self._str_scalen()

        return '\n'.join([s1, s2, s3, s4, s5])  # s6])

    def _str_origxn(self):
        """
        ~! dummy unit transformation
        transformation orthogonal space and the crystal lattice basis
        ~! doesn't implement a translation vector at present

        ref: `zhanglab.ccmb.med.umich.edu <https://zhanglab.ccmb.med.umich.edu/COFACTOR/pdb_atom_format.html#ORIGXn>`_.
         1 -  6       Record name     "ORIGXn" (n=1, 2, or 3)

        11 - 20       Real(10.6)      o[n][1]

        21 - 30       Real(10.6)      o[n][2]

        31 - 40       Real(10.6)      o[n][3]

        46 - 55       Real(10.5)      t[n]

        """
        fmt = ''.join(('ORIGX{idx}    ',) + ('{:10.6f}',) * 3 + ('     {:10.5f}',))
        fill = np.zeros((3,4))
        _ = np.fill_diagonal(fill, 1)
        fmtstr = []
        for idx, line in enumerate(fill):
            fmtstr.append(fmt.format(*list(line), idx=idx+1))
        return  '\n'.join(fmtstr)

    def _str_scalen(self):
        """
        transformation between orthogonal coordinates and unit cell (Fractional) coordinates


        ref: `zhanglab.ccmb.med.umich.edu <https://zhanglab.ccmb.med.umich.edu/COFACTOR/pdb_atom_format.html#ORIGXn>`_.
                COLUMNS       DATA TYPE      CONTENTS
        --------------------------------------------------------------------------------
         1 -  6       Record name    "SCALEn" (n=1, 2, or 3)

        11 - 20       Real(10.6)     s[n][1]

        21 - 30       Real(10.6)     s[n][2]

        31 - 40       Real(10.6)     s[n][3]

        46 - 55       Real(10.5)     u[n]
        """
        fmt = ('SCALE{}    ',) + ('{:10.6f}',) * 3 + ('     {:10.5f}',)  # formats for composition
        just = ['ljust', 'rjust', 'rjust', 'rjust', 'rjust']                         # justification with len == len(fmt)
        char = [10, 10, 10, 10, 10]
        fill = [list(np.append(a, (0.0))) for a in np.linalg.inv(self._structure.lattice.matrix)]  # contents of composition
        fmtstr = []
        for idx, line in enumerate(fill):
            line = [idx+1,] + line
            line = [s1.format(s2) for (s1, s2) in list(zip(fmt, line))]
            line = [getattr(s1, s2)(d1) for (s1, s2, d1) in list(zip(line, just, char))]
            fmtstr.append(''.join(line))
            # fmtstr.append(fmt.format(*list(line), idx=idx+1))
        return '\n'.join(fmtstr)

    def write(self, fname):
        """ """
        with open(fname, 'wt') as f:
            f.write(self.__str__())


