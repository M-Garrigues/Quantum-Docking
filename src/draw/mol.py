# --- Start of the function ---

import os

import numpy as np
import py3Dmol
from rdkit import Chem

from src.mol_processing.features import Feature, MinMaxDistance


def show_3d(
    molecules: Chem.Mol | list[Chem.Mol],
    style: str = "stick",
    width: int = 800,
    height: int = 500,
) -> py3Dmol.view:
    """
    Displays one or more RDKit molecules in a 3D viewer.

    Args:
        molecules: A single RDKit molecule or a list of molecules.
                   Each molecule must have a 3D conformation.
        style: The representation style ('stick', 'line', 'sphere', etc.).
        width: The viewer width in pixels.
        height: The viewer height in pixels.

    Returns:
        A py3Dmol.view object for interactive display in a notebook.
    """
    if not isinstance(molecules, list):
        molecules = [molecules]

    view = py3Dmol.view(width=width, height=height)

    for mol in molecules:
        if mol.GetNumConformers() == 0:
            print(
                f"Warning: Molecule {Chem.MolToSmiles(mol)} has no 3D conformer and will be skipped.",
            )
            continue

        mol_block = Chem.MolToMolBlock(mol, confId=-1)  # -1 includes all conformers
        view.addModels(mol_block, "mol")

    view.setStyle({}, {style: {}})
    view.zoomTo()

    return view


def visualize_flexibility_ensemble(molecule: Chem.Mol, num_to_display: int = 15):
    """Creates a visualisation of different conformers for a molecule, centered around"""
    view = py3Dmol.view(width=1400, height=1000)
    total_confs = molecule.GetNumConformers()
    if total_confs == 0:
        return None

    num_to_show = min(num_to_display, total_confs)
    print(f"Plotting {num_to_show} of {total_confs} aligned conformers.")

    mb = Chem.MolToMolBlock(molecule, confId=0)
    writer = Chem.SDWriter("temp_conformers.sdf")
    for conf_id in range(num_to_show):
        writer.write(molecule, confId=conf_id)
    writer.close()

    with open("temp_conformers.sdf") as f:
        sdf_block = f.read()
    os.remove("temp_conformers.sdf")

    view.addModels(sdf_block, "sdf")

    view.setStyle({}, {"stick": {"opacity": 0.6, "radius": 0.12}})

    view.zoomTo()
    return view


def visualize_min_max_distance_pair(
    molecule: Chem.Mol,
    distance_data: dict[tuple[Feature, Feature], MinMaxDistance],
) -> py3Dmol.view:
    """
    Finds the feature pair with the largest distance variation and displays
    the two corresponding conformers side-by-side.
    """
    if not distance_data:
        print("No distance data to visualize.")
        return None

    # Find the pair with the largest difference (max_dist - min_dist)
    target_pair = max(
        distance_data.keys(),
        key=lambda pair: distance_data[pair].max_dist - distance_data[pair].min_dist,
    )

    dist_info = distance_data[target_pair]
    feat1, feat2 = target_pair

    print(f"Pair with largest variation: ({feat1.name}, {feat2.name})")
    print(f"  Min distance: {dist_info.min_dist:.2f} Å (Conformer ID: {dist_info.min_conf_id})")
    print(f"  Max distance: {dist_info.max_dist:.2f} Å (Conformer ID: {dist_info.max_conf_id})")

    # Create a 1x2 grid for the side-by-side view
    view = py3Dmol.view(width=1400, height=700, linked=False, viewergrid=(1, 2))

    # --- LEFT PANE: Minimum Distance Conformer ---
    min_conf = molecule.GetConformer(dist_info.min_conf_id)
    pos1_min = np.mean([min_conf.GetAtomPosition(i) for i in feat1.atom_ids], axis=0)
    pos2_min = np.mean([min_conf.GetAtomPosition(i) for i in feat2.atom_ids], axis=0)

    view.addModel(Chem.MolToMolBlock(molecule, confId=dist_info.min_conf_id), "mol", viewer=(0, 0))
    view.setStyle({"stick": {}}, viewer=(0, 0))

    view.addSphere(
        {
            "center": {"x": pos1_min[0], "y": pos1_min[1], "z": pos1_min[2]},
            "radius": 0.4,
            "color": feat1.family.color,
        },
        viewer=(0, 0),
    )
    view.addSphere(
        {
            "center": {"x": pos2_min[0], "y": pos2_min[1], "z": pos2_min[2]},
            "radius": 0.4,
            "color": feat2.family.color,
        },
        viewer=(0, 0),
    )
    view.addCylinder(
        {
            "start": {"x": pos1_min[0], "y": pos1_min[1], "z": pos1_min[2]},
            "end": {"x": pos2_min[0], "y": pos2_min[1], "z": pos2_min[2]},
            "color": "black",
            "radius": 0.05,
            "dashed": True,
        },
        viewer=(0, 0),
    )
    mid_point_min = (pos1_min + pos2_min) / 2
    view.addLabel(
        f"{dist_info.min_dist:.2f} Å",
        {
            "position": {"x": mid_point_min[0], "y": mid_point_min[1], "z": mid_point_min[2]},
            "fontColor": "black",
            "backgroundColor": "white",
            "backgroundOpacity": 0.6,
        },
        viewer=(0, 0),
    )

    # --- RIGHT PANE: Maximum Distance Conformer ---
    max_conf = molecule.GetConformer(dist_info.max_conf_id)
    pos1_max = np.mean([max_conf.GetAtomPosition(i) for i in feat1.atom_ids], axis=0)
    pos2_max = np.mean([max_conf.GetAtomPosition(i) for i in feat2.atom_ids], axis=0)

    view.addModel(Chem.MolToMolBlock(molecule, confId=dist_info.max_conf_id), "mol", viewer=(0, 1))
    view.setStyle({"stick": {}}, viewer=(0, 1))

    view.addSphere(
        {
            "center": {"x": pos1_max[0], "y": pos1_max[1], "z": pos1_max[2]},
            "radius": 0.4,
            "color": feat1.family.color,
        },
        viewer=(0, 1),
    )
    view.addSphere(
        {
            "center": {"x": pos2_max[0], "y": pos2_max[1], "z": pos2_max[2]},
            "radius": 0.4,
            "color": feat2.family.color,
        },
        viewer=(0, 1),
    )
    view.addCylinder(
        {
            "start": {"x": pos1_max[0], "y": pos1_max[1], "z": pos1_max[2]},
            "end": {"x": pos2_max[0], "y": pos2_max[1], "z": pos2_max[2]},
            "color": "black",
            "radius": 0.05,
            "dashed": True,
        },
        viewer=(0, 1),
    )
    mid_point_max = (pos1_max + pos2_max) / 2
    view.addLabel(
        f"{dist_info.max_dist:.2f} Å",
        {
            "position": {"x": mid_point_max[0], "y": mid_point_max[1], "z": mid_point_max[2]},
            "fontColor": "black",
            "backgroundColor": "white",
            "backgroundOpacity": 0.6,
        },
        viewer=(0, 1),
    )

    view.zoomTo(viewer=(0, 0))
    view.zoomTo(viewer=(0, 1))

    return view
