"""
This module provides functions to identify and visualize pharmacophoric features
located on the surface of a molecule.
"""

from __future__ import annotations

import io
from typing import List, Sequence, Optional, Tuple

import numpy as np
import py3Dmol
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial import cKDTree
from src.mol_processing.features import Feature


def select_surface_features(
    mol: Chem.Mol,
    features: Sequence[Feature],
    ligand: Optional[Chem.Mol] = None,
    default_probe_radius: float = 1.4,
    min_sasa_threshold: float = 0.1,
    distance_threshold: float = 3.0,
) -> List[Feature]:
    """
    Filters a sequence of features, returning only those near the molecule's surface.
    This function uses Biopython in the backend for robust SASA calculation.

    Args:
        mol: RDKit molecule with a 3D conformer.
        features: A sequence of Feature objects to be filtered.
        ligand: Optional RDKit molecule used to dynamically define the probe radius
                based on its estimated size.
        default_probe_radius: Probe radius in Angstroms to use if no ligand is provided.
        min_sasa_threshold: Minimum SASA value (in Å²) for an atom to be considered
                            a surface atom.
        distance_threshold: The maximum allowed distance from a surface atom for a
                            feature to be kept.

    Returns:
        A list of Feature objects located near the molecular surface.
    """
    if mol.GetNumConformers() < 1:
        raise ValueError("The input molecule 'mol' must have a 3D conformer.")

    probe_radius = default_probe_radius
    if ligand:
        if ligand.GetNumConformers() < 1:
            raise ValueError("The input 'ligand' molecule must have a 3D conformer.")
        ligand_volume = AllChem.ComputeMolVolume(ligand, confId=-1, gridSpacing=0.2)
        if ligand_volume > 0:
            probe_radius = (0.75 * ligand_volume / np.pi) ** (1 / 6.0)
            print(f"Probe radius calculated from ligand: {probe_radius:.2f} Å")

    # --- RDKIT -> BIOPYTHON BRIDGE ---
    # 1. Convert the RDKit molecule to a text block in PDB format.
    pdb_block = Chem.MolToPDBBlock(mol)

    # 2. Read this text block with Biopython's PDBParser.
    #    io.StringIO treats a string as a file for the parser.
    parser = PDBParser(QUIET=True)  # Suppresses PDB parsing warnings.
    structure = parser.get_structure("internal_mol", io.StringIO(pdb_block))
    # --- END OF BRIDGE ---

    # 3. Compute the Solvent Accessible Surface Area (SASA) using Biopython.
    sr = ShrakeRupley(probe_radius=probe_radius, n_points=100)
    sr.compute(structure, level="A")  # "A" for per-atom calculation.

    # 4. Retrieve the coordinates of all atoms considered to be on the surface.
    surface_atom_coords = []
    for atom in structure.get_atoms():
        # The 'sasa' attribute is attached to the atom object by the calculator.
        if hasattr(atom, "sasa") and atom.sasa > min_sasa_threshold:
            surface_atom_coords.append(atom.get_coord())

    if not surface_atom_coords:
        print("Warning: No surface atoms found with the current settings.")
        return []

    # 5. Build a k-d tree for efficient distance lookup and filter the features.
    surface_tree = cKDTree(np.array(surface_atom_coords))
    surface_features: List[Feature] = []
    for feat in features:
        distance, _ = surface_tree.query(feat.position, k=1, workers=-1)
        if distance <= distance_threshold:
            surface_features.append(feat)

    return surface_features


import py3Dmol
from rdkit import Chem
from typing import List, Tuple, Optional


def visualize_selection(
    molecule: Chem.Mol,
    all_points: List[Feature],
    selected_points: List[Feature],
    ligand: Optional[Chem.Mol] = None,
) -> py3Dmol.view:
    """
    Visualizes the molecule, pharmacophore points, and an optional ligand.

    If a ligand is provided, it is highlighted, the camera is focused on it,
    and the main molecule (receptor) is made transparent.

    Args:
        molecule: RDKit molecule (receptor) with a 3D conformer.
        all_points: A list of all Feature objects to display.
        selected_points: A subset of Feature objects to highlight.
        ligand: An optional RDKit ligand molecule with a 3D conformer.

    Returns:
        A py3Dmol.view instance containing the visualization.
    """
    selected_positions: set[Tuple[float, float, float]] = {
        tuple(p.position) for p in selected_points
    }

    view = py3Dmol.view(width=1200, height=900)
    view.setBackgroundColor("white")

    for p in all_points:
        pos = tuple(p.position)
        is_selected = pos in selected_positions
        color = "green" if is_selected else "red"
        radius = 0.5 if is_selected else 0.3
        view.addSphere(
            {
                "center": {"x": pos[0], "y": pos[1], "z": pos[2]},
                "radius": radius,
                "color": color,
                "alpha": 0.8,
            }
        )

    if ligand is not None:
        receptor_block = Chem.MolToMolBlock(molecule)
        view.addModel(receptor_block, "mol")
        # view.setStyle({"model": 0}, {"cartoon": {"color": "lightgrey", "opacity": 0.6}})

        ligand_block = Chem.MolToMolBlock(ligand)
        view.addModel(ligand_block, "mol")
        view.setStyle({"model": 1}, {"stick": {}})

        view.zoomTo({"model": 1})

    else:
        mol_block = Chem.MolToMolBlock(molecule)
        view.addModel(mol_block, "mol")
        view.setStyle({}, {"stick": {}, "sphere": {"scale": 0.2}})
        view.zoomTo()

    return view
