import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.transform import Rotation

from src.mol_processing.features import Feature, extract_pharmacophores


def place_ligand_from_contacts(
    ligand_with_confs: Chem.Mol,
    receptor_features: list[Feature],
    contacts: list[tuple[str, str]],
) -> Chem.Mol | None:
    """
    Finds the best conformation and 3D placement for a ligand that satisfies
    a list of pharmacophore contacts with a receptor.

    Args:
        ligand_with_confs: An RDKit molecule of the ligand with multiple conformers.
        receptor_features: A list of pre-calculated Feature objects for the receptor.
        contacts: A list of interactions, where each item is a tuple of
                  feature names: (ligand_feature_name, receptor_feature_name).

    Returns:
        A new RDKit Mol object of the best-placed ligand conformer, or None.
    """
    if len(contacts) < 3:
        print("Warning: At least 3 contacts are recommended for a stable 3D alignment.")
        if not contacts:
            return None

    # Prepare receptor target coordinates
    receptor_points_map = {f.name: f.position for f in receptor_features}
    target_receptor_coords = np.array([receptor_points_map[rec_name] for _, rec_name in contacts])

    best_solution = {"rmsd": float("inf"), "conformer_id": -1, "transform_matrix": None}

    # Iterate through each ligand conformer to find the best fit
    for conf_id in range(ligand_with_confs.GetNumConformers()):

        # This part requires a function to get feature positions for a specific conformer
        lig_conf_features = extract_pharmacophores(ligand_with_confs, "l", confId=conf_id)
        ligand_points_map = {f.name: f.position for f in lig_conf_features}

        # Build the source coordinate array for this conformer
        current_ligand_coords = []
        for lig_name, _ in contacts:
            if lig_name not in ligand_points_map:
                current_ligand_coords = []  # Invalidate if a feature is missing
                break
            current_ligand_coords.append(ligand_points_map[lig_name])

        if len(current_ligand_coords) != len(contacts):
            continue  # Skip conformers that don't match all required features

        current_ligand_coords = np.array(current_ligand_coords)

        # Calculate optimal alignment using the Kabsch algorithm
        centroid_lig = current_ligand_coords.mean(axis=0)
        centroid_rec = target_receptor_coords.mean(axis=0)

        centered_lig = current_ligand_coords - centroid_lig
        centered_rec = target_receptor_coords - centroid_rec

        rotation, rmsd = Rotation.align_vectors(centered_rec, centered_lig)

        # Store if this is the best solution so far
        if rmsd < best_solution["rmsd"]:
            best_solution["rmsd"] = rmsd
            best_solution["conformer_id"] = conf_id

            # Build the 4x4 transformation matrix
            transform = np.eye(4)
            transform[:3, :3] = rotation.as_matrix()
            transform[:3, 3] = centroid_rec - rotation.apply(centroid_lig)
            best_solution["transform_matrix"] = transform

    # Apply the best transformation found
    if best_solution["conformer_id"] == -1:
        print("Could not find a valid alignment for any conformer.")
        return None

    print(
        f"Best solution: RMSD = {best_solution['rmsd']:.3f} Å for conformer {best_solution['conformer_id']}",
    )

    best_conformer = ligand_with_confs.GetConformer(best_solution["conformer_id"])
    placed_ligand = Chem.Mol(ligand_with_confs)
    placed_ligand.RemoveAllConformers()
    placed_ligand.AddConformer(best_conformer, assignId=True)

    AllChem.TransformMol(placed_ligand, best_solution["transform_matrix"])

    return placed_ligand
