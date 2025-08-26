"""Script used to visualise pharmacophore points in PyMol"""

import json

from pymol import cmd


def load_pharmacophores_from_file(
    ligand_file: str,
    receptor_file: str,
    ligand_group: str = "ligand_pharmacophores",
    receptor_group: str = "receptor_pharmacophores",
    connection_group: str = "connections",
    linked_receptor_group: str = "linked_receptor_points",
):
    """
    Load and display pharmacophore points from ligand and receptor files, connecting compatible pairs.

    Args:
        ligand_file (str): Path to the ligand pharmacophore file.
        receptor_file (str): Path to the receptor pharmacophore file.
        ligand_group (str): Group name for ligand pharmacophores.
        receptor_group (str): Group name for receptor pharmacophores.
        connection_group (str): Group name for pharmacophore connections.
        linked_receptor_group (str): Group name for receptor points linked to ligand points.
    """
    color_map = {
        "A": "red",  # Acceptor
        "D": "blue",  # Donor
        "H": "yellow",  # Hydrophobic
        "R": "gray",  # Aromatic (for ligand)
        "P": "green",  # Positive (charge) or "P" point
        "a": "red",  # Acceptor (for receptor)
        "d": "blue",  # Donor (for receptor)
        "h": "yellow",  # Hydrophobic (for receptor)
        "r": "gray",  # Aromatic (for receptor)
        "p": "green",  # Positive (charge) or "p" point (for receptor)
    }

    # Load pharmacophores from JSON file
    def load_pharmacophores(file_path, group_name, is_ligand=True):
        with open(file_path) as f:
            data = json.load(f)
        features = data["feature_coords"]
        points = {}

        for index, (feature_type, coords) in enumerate(features):
            if feature_type == "a":  # Convert lowercase 'a' to 'r' for aromatic
                feature_type = "r"
            feature_type = (
                feature_type.upper() if is_ligand else feature_type.lower()
            )  # Distinguish between ligand and receptor

            # Handle potential overlapping points with 'a' and 'd'
            key = tuple(coords)
            if key in points and points[key][2] == "d":
                continue  # Keep 'd' and skip 'a'
            points[key] = (f"{feature_type}_{index+1}_{group_name}", coords, feature_type)

        # Create and display pseudoatoms
        for sphere_name, coords, feature_type in points.values():
            color = color_map.get(feature_type, "gray")
            cmd.pseudoatom(sphere_name, pos=coords)
            cmd.set("sphere_scale", 1.0, sphere_name)
            cmd.color(color, sphere_name)
            cmd.group(group_name, sphere_name)

        cmd.show("spheres", group_name)
        cmd.set("sphere_transparency", 0.5, group_name)
        return list(points.values())

    # Calculate distance between two 3D points
    def distance(coord1, coord2):
        return (
            (coord1[0] - coord2[0]) ** 2
            + (coord1[1] - coord2[1]) ** 2
            + (coord1[2] - coord2[2]) ** 2
        ) ** 0.5

    # Load pharmacophore points
    ligand_points = load_pharmacophores(ligand_file, ligand_group, is_ligand=True)
    receptor_points = load_pharmacophores(receptor_file, receptor_group, is_ligand=False)

    # Filter receptor points by distance to ligand points
    filtered_receptor_points = []
    linked_receptor_points = []
    for r_name, r_coords, r_type in receptor_points:
        min_distance = min([distance(r_coords, l_coords) for _, l_coords, _ in ligand_points])
        if 0.5 < min_distance <= 4.5:
            filtered_receptor_points.append((r_name, r_coords, r_type))
            # Check if receptor point is within interaction distance of ligand point
            for l_name, l_coords, l_type in ligand_points:
                if (l_type, r_type) in {
                    ("D", "a"),
                    ("A", "d"),
                    ("H", "h"),
                    ("R", "r"),
                } and distance(l_coords, r_coords) <= 4:
                    linked_receptor_points.append((r_name, r_coords, r_type))
                    break

    # Create a new group for filtered receptor points
    cmd.group(f"{receptor_group}_filtered")
    for r_name, r_coords, r_type in filtered_receptor_points:
        cmd.group(f"{receptor_group}_filtered", r_name)

    # Create a group for linked receptor points
    cmd.group(linked_receptor_group)
    for r_name, r_coords, r_type in linked_receptor_points:
        cmd.group(linked_receptor_group, r_name)

    # Display receptor in stick within 4.5 Å and in ribbon otherwise
    cmd.hide("everything", receptor_group)
    cmd.show("ribbon", receptor_group)
    cmd.show("sticks", f"{receptor_group}_filtered")

    # Draw connections for specific pharmacophore pairs within 4 Å
    cmd.group(connection_group)  # Create a group for all connections
    for l_name, l_coords, l_type in ligand_points:
        for r_name, r_coords, r_type in linked_receptor_points:
            # Ensure that only specific pairs and group differences are connected
            if (l_type, r_type) in {("D", "a"), ("A", "d"), ("H", "h"), ("R", "r")} and distance(
                l_coords,
                r_coords,
            ) <= 4:
                # Create dashed connection between ligand and receptor points
                cmd.distance(f"connection_{l_name}_{r_name}", l_name, r_name)
                cmd.set("dash_color", "green", f"connection_{l_name}_{r_name}")
                cmd.set("dash_gap", 0.4, f"connection_{l_name}_{r_name}")
                cmd.set("dash_radius", 0.1, f"connection_{l_name}_{r_name}")
                cmd.group(connection_group, f"connection_{l_name}_{r_name}")


# Example usage
load_pharmacophores_from_file(
    "ligand.pma",
    "receptor.pma",
    "ligand_group",
    "receptor_group",
    "pharmacophore_connections",
    "linked_receptor_group",
)
