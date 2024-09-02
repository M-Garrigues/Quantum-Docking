import json
from pymol import cmd


def load_pharmacophores_from_json(data: dict, is_ligand: bool):
    """
    Load pharmacophore points from a JSON string and display them with different representations in PyMOL.

    Args:
        json_data (str): JSON string containing pharmacophore coordinates and types.

    The JSON should have the following format:
    {
        "bin_step": 1,
        "feature_coords": [
            ["a", [x1, y1, z1]],
            ["D", [x2, y2, z2]],
            ["A", [x3, y3, z3]],
            ...
        ]
    }
    """
    color_map = {
        "R": "orange",  # Aromatic ring, assumed to be "a"
        "D": "blue",  # Donor
        "A": "red",  # Acceptor
        "H": "yellow",  # Hydrophobic
        "P": "green",  # PosIonisable
    }

    features = data["feature_coords"]

    for index, (feature_type, coords) in enumerate(features):
        x, y, z = coords
        feature_type = feature_type.upper()  # Normalize type to uppercase
        color = color_map.get(feature_type, "gray")  # Default to gray if type is not recognized

        if is_ligand:
            name = feature_type
        else:
            name = feature_type.lower()

        sphere_name = f"pharma_{name}_{index+1}"

        # Create the sphere at the specified coordinates
        cmd.pseudoatom(sphere_name, pos=[x, y, z])
        cmd.set("sphere_scale", 1.0, sphere_name)  # Radius can be adjusted if needed
        cmd.color(color, sphere_name)
        cmd.label(sphere_name, f'"{name}_{index+1}"')

        cmd.group("pharmacophore", sphere_name)

    # Show all the spheres and optionally set transparency
    cmd.show("spheres", "pharma_*")
    cmd.set("sphere_transparency", 0.3, "pharma_*")  # Optional transparency


# Example usage
# json_data = '{"bin_step": 1, "feature_coords": [["R", [44.71616666666667, 28.498166666666666, 2.8508333333333336]], ["D", [41.158, 31.976, 4.586]], ["A", [42.995, 28.767, -0.494]], ["A", [41.153, 28.787, 1.199]], ["A", [46.472, 27.817, 4.923]], ["H", [44.71616666666667, 28.498166666666666, 2.8508333333333336]], ["H", [48.19833333333333, 26.479666666666663, 4.2139999999999995]], ["H", [48.4985, 24.634, 3.3645]]]}'
with open("./pharma.pma") as f:
    json_data = json.loads(f.read())

is_ligand = True
load_pharmacophores_from_json(json_data, is_ligand)
