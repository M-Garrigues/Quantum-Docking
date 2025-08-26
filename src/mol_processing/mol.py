"""Module to deal with the Rdkit molecules and their features."""

import copy
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

from Bio.PDB import PDBIO, PDBParser, Select
from rdkit import Chem
from rdkit.Chem import AllChem, ChemicalFeatures, Mol

from src.config.general import SELECTED_FEATURES
from src.mol_processing.features import Feature


class NoWaterSelect(Select):
    """
    Select class that excludes water molecules from PDB output.
    """

    def accept_residue(self, residue) -> bool:
        return residue.get_resname() not in ("HOH", "WAT", "H2O")


def remove_water_from_rdkit_mol(mol: Chem.Mol) -> Chem.Mol:
    """
    Removes water molecules from an RDKit molecule by converting it to a PDB file,
    filtering out waters using Biopython, and reloading the cleaned molecule.

    Args:
        mol (Chem.Mol): The RDKit molecule containing a protein structure, potentially with water molecules.

    Returns:
        Chem.Mol: A new RDKit molecule with water molecules removed.

    Raises:
        ValueError: If the intermediate molecule could not be parsed by RDKit.
    """
    # Generate 3D coordinates if not present
    if mol.GetNumConformers() == 0:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)

    with (
        NamedTemporaryFile(suffix=".pdb", delete=False) as tmp_in,
        NamedTemporaryFile(suffix=".pdb", delete=False) as tmp_out,
    ):
        tmp_in_path = Path(tmp_in.name)
        tmp_out_path = Path(tmp_out.name)

        # Write RDKit Mol to PDB
        Chem.MolToPDBFile(mol, str(tmp_in_path))

        # Load and filter with Biopython
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("mol", str(tmp_in_path))
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(tmp_out_path), select=NoWaterSelect())

        # Reload cleaned PDB as RDKit Mol
        cleaned_mol = Chem.MolFromPDBFile(str(tmp_out_path), sanitize=True, removeHs=False)

    # Clean up files manually if needed
    tmp_in_path.unlink(missing_ok=True)
    tmp_out_path.unlink(missing_ok=True)

    if cleaned_mol is None:
        raise ValueError("Failed to parse the cleaned PDB file into an RDKit Mol.")

    return cast(Chem.Mol, cleaned_mol)


def get_protein_from_pdb_file(pdb_path: str) -> Mol:
    """
    Loads a PDB file and returns an RDKit Mol object containing only the
    protein chains by filtering for 'ATOM' records at the text level.
    This is the most robust method.

    Args:
        pdb_path: The file path to the PDB file.

    Returns:
        An RDKit Mol object of the protein.
    """
    with open(pdb_path) as f:
        # Read all lines and filter for those starting with 'ATOM'
        protein_lines = [line for line in f if line.startswith("ATOM")]

    # Join the filtered lines back into a single string block
    pdb_block = "".join(protein_lines)

    # Load the molecule from the filtered PDB block
    protein_mol = Chem.MolFromPDBBlock(pdb_block, sanitize=True, removeHs=False)

    if protein_mol is None:
        raise ValueError(f"Could not load protein from PDB file: {pdb_path}")

    return protein_mol


def generate_aligned_conformers(
    molecule: Chem.Mol,
    num_confs: int = 100,
    core_smarts: str = "c1ccccc1",
) -> Chem.Mol:
    """
    Generates multiple conformers for a molecule and aligns them to a common core.

    Args:
        molecule: The input RDKit molecule (with hydrogens added).
        num_confs: The number of conformers to generate.
        core_smarts: A SMARTS string representing the rigid core for alignment.
                     Defaults to a simple benzene ring.

    Returns:
        The same molecule object, now containing the generated and aligned conformers.
        The operation is done in-place.
    """
    copied_mol = copy.deepcopy(molecule)
    # Generate the conformers
    AllChem.EmbedMultipleConfs(copied_mol, numConfs=num_confs, randomSeed=42)

    # Optimize their geometry
    AllChem.MMFFOptimizeMoleculeConfs(copied_mol)

    # Find the atom indices of the core structure
    core_pattern = Chem.MolFromSmarts(core_smarts)
    core_match = molecule.GetSubstructMatch(core_pattern)

    # Align all conformers to the core of the first conformer
    if core_match:
        AllChem.AlignMolConformers(copied_mol, atomIds=core_match)
        print(
            f"Generated and aligned {copied_mol.GetNumConformers()} conformers on the specified core.",
        )
    else:
        print(f"Warning: Core pattern '{core_smarts}' not found. Conformers were not aligned.")

    return copied_mol


def get_features(mol: Mol, mol_id: str) -> list[Feature]:
    """Get features from a molecule, from the configuration's selected families.

    Args:
        mol (Mol): Rdkit molecule.
        mol_id (str): The id of the molecule.

    Returns:
        list[Feature]: List of all features selected.
    """
    FACTORY = ChemicalFeatures.BuildFeatureFactory("data/BaseFeatures.fdef")
    features = []
    positions = set()
    for family in SELECTED_FEATURES.keys():
        for rd_feat in FACTORY.GetFeaturesForMol(mol, includeOnly=family):
            pos = tuple(rd_feat.GetPos())
            if pos in positions:
                # Here we ignore if a feature belongs to 2 families
                continue
            features.append(Feature(rd_feat, mol_id))
            positions.add(pos)
    return features
