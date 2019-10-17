# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 13:28:16 2019

@author: pce
"""
import re
import os
import numpy as np
from pymatgen import Structure, Lattice, PeriodicSite, Composition, DummySpecie, Element


def fold(structure, supercell):
    """
    return folded single unit cell
    
    Args:
      * structure (pymatgen.Structure) : pymatgen structure instance. Expects supercell dimensions are integer multiples of some flavor.
      * supercell (list, np.array) : (m, n, o ) [integers] or matrix with shape (3,3)
    
    Essentially:
       A = inv(supercell) (3,3)
       lattice = A * superlattice  (3,3)
       B = inv(lattice)
       coords = (B dot coord ) % 1 for coord in superstructure
    
    Returns:
      pymatgen.Structure instance with each site folded into (1,1,1) lattice ( 0 < frac_coord < 1 )
    """
    if len(supercell) == 3:
        supercell = np.identity(3) * np.array(supercell, dtype=float)
    elif np.array(supercell).shape == (3,3):
        supercell = np.array(supercell, dtype=float)
    else:
        raise(Exception('supercell must be (m, n, o) or (3x3) matrix'))
    
    transform = np.linalg.inv(supercell)
    lat = Lattice(transform * structure.lattice.matrix)
    invlat = np.linalg.inv(lat.matrix)
    coords = [np.dot(invlat, coord) % 1 for coord in structure.cart_coords]
    comp = structure.species_and_occu
    return Structure(lat, comp, coords)


class PMG():
    """ """

    def __reload__(self):
        self.__init__(self.filepath)

    def __init__(self, filepath=None):
        """ """
        self.patterns = {'remark': re.compile('.*REMARK .*'),
                         'cryst1': re.compile('.*CRYST1 .*'),
                         'origxn': re.compile('.*ORIGX\d+ .*'),
                         'scalen': re.compile('.*SCALE\d+ .*'),
                         'atom': re.compile('.*ATOM .*'),
                         }
        self.filepath = filepath
        self.lines = self._read_file(filepath)
        self._get_blocks(self.lines)
        self._make_structure()
    
    def _read_file(self, filename):
        """ return list of raw strings """
        with open(filename, 'r') as f:
            return f.readlines()

    def _get_masked(self, block, lines):
        """ """
        rv = list(filter(None, [self.patterns[block].match(line) for line in lines]))
        return [group[0] for group in rv]
    
    def _get_blocks(self, lines):
        """ """
        self._get_remarks(lines)
        self._get_cryst1(lines)
        self._get_origxn(lines)
        self._get_scalen(lines)
        self._get_atoms(lines)

    def _get_remarks(self, lines):
        """ """
        self.remarks = self._get_masked('remark', lines)
        return
    
    def _get_cryst1(self, lines):
        """ """
        self.cryst1 = self._get_masked('cryst1', lines)[0]
        self._make_lattice()
        return
        
    def _get_origxn(self, lines):
        """ """
        self.origxn = self._get_masked('origxn', lines)
        self._make_transformation()
        return
        
    def _get_scalen(self, lines):
        """ """
        self.scalen = self._get_masked('scalen', lines)
        return
        
    def _get_atoms(self, lines):
        """ """
        self.atoms = self._get_masked('atom', lines)
        self._make_sites()
        return
    
    def _make_transformation(self):
        """ """
        raw = [line.split()[1:] for line in self.origxn]
        raw = np.array(raw).astype(float)
        self.cart_to_lat = raw[:, :3]
        self.translate = raw[:, -1]
        return
    
    def _make_lattice(self):
        """ """
        raw = self.cryst1
        raw = 'CRYST1' + re.split('CRYST1', raw)[-1] # sanitize raw
        proc = [s.strip() for s in (raw[6:15], raw[15:24], raw[24:33], raw[33:40], raw[40:47], raw[47:54], raw[55:66], raw[66:70])]
        self.lattice = Lattice.from_lengths_and_angles(np.array(proc[:3], dtype=float), np.array(proc[3:6], dtype=float))
        self.spacegroup = proc[-2]
        self.Z = proc[-1]
        return
    
    def _make_sites(self):
        """ """
        self.sites = []
        self.coords = []
        for raw in self.atoms:
            proc = [s.strip() for s in (raw[:6], raw[6:11],  raw[12:16], raw[16], raw[17:20],
                                        raw[21], raw[22:26], raw[26], raw[30:38], raw[38:46],
                                        raw[46:54], raw[54:60], raw[60:66], raw[72:76], 
                                        raw[76:78], raw[78:])
                    ]
            coord = np.dot(self.cart_to_lat, np.array(proc[8:11], dtype=float) ) + self.translate
            props = dict(serial=proc[1], name=proc[2], altloc=proc[3], residue=proc[4],
                         chain=proc[5], sequence=proc[6], segment=proc[13])
            self.coords.append(coord)
            self.sites.append(PeriodicSite(Composition({proc[-2]: float(proc[11])}),
                                           coord, self.lattice, coords_are_cartesian=True,
                                           properties=props))
        return
    
    def _make_structure(self):
        """ """
        self.structure = Structure.from_sites(self.sites)
        return

    