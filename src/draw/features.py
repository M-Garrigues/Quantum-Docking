import copy
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import py3Dmol
from rdkit import Chem

from src.graph.interaction_graph import InteractionNode
from src.graph.mapping import results_to_interaction_graph_cliques
from src.mol_processing.features import Feature
from src.utils.dataclasses import OrderedTupleDict


def show_molecule_with_features(molecule: Chem.Mol, features: list[Feature]) -> py3Dmol.view:
    """
    Displays a molecule with its pharmacophore features as colored spheres.
    The color is taken directly from the FeatureFamily object.
    """
    view = py3Dmol.view(width=600, height=500)
    view.addModel(Chem.MolToMolBlock(molecule), "mol")
    view.setStyle({}, {"stick": {}})

    for feature in features:
        # Get the color directly from the feature's family attribute
        color = feature.family.color
        pos = feature.position

        view.addSphere(
            {
                "center": {"x": pos[0], "y": pos[1], "z": pos[2]},
                "radius": 0.5,
                "color": color,
                "alpha": 0.8,
            },
        )
        view.addLabel(
            feature.name,
            {
                "position": {"x": pos[0], "y": pos[1], "z": pos[2]},
                "fontColor": color,
                "backgroundOpacity": 0.33,
                "backgroundColor": "white",
            },
        )

    view.zoomTo()
    return view


def visualize_docking_site(
    receptor: Chem.Mol,
    ligand: Chem.Mol,
    receptor_features: Sequence[Feature],
    ligand_features: Sequence[Feature],
) -> None:
    """
    Visualizes a docking site with py3Dmol.

    Displays receptor (cartoon style) and ligand (stick style),
    shows chemical features as spheres colored by FeatureFamily.color,
    fades receptor regions >20Å from ligand center, and centers zoom on ligand.

    Args:
        receptor, ligand: RDKit Mol objects with 3D conformers.
        receptor_features: chemical features on receptor.
        ligand_features: chemical features on ligand.

    Raises:
        ValueError: if either molecule lacks a 3D conformer.
    """
    if ligand.GetNumConformers() == 0 or receptor.GetNumConformers() == 0:
        raise ValueError("Both receptor and ligand must have 3D conformers.")

    # Compute ligand centroid
    lconf = ligand.GetConformer()
    l_coords = np.array([lconf.GetAtomPosition(i) for i in range(ligand.GetNumAtoms())])
    l_center = np.mean(l_coords, axis=0)

    # Distances of receptor atoms
    rconf = receptor.GetConformer()
    r_coords = np.array([rconf.GetAtomPosition(i) for i in range(receptor.GetNumAtoms())])
    distances = np.linalg.norm(r_coords - l_center, axis=1)
    close_serials = [i + 1 for i, d in enumerate(distances) if d <= 20.0]

    # Initialize viewer
    viewer = py3Dmol.view(width=1400, height=1000)

    receptor_pdb = Chem.MolToPDBBlock(receptor)
    ligand_pdb = Chem.MolToPDBBlock(ligand)

    viewer.addModel(receptor_pdb, "pdb")  # Model 0
    viewer.addModel(ligand_pdb, "pdb")  # Model 1

    # Apply styles to each model by its index
    viewer.setStyle({"model": 0}, {"cartoon": {"color": "lightgray", "opacity": 0.6}})
    viewer.setStyle({"model": 1}, {"stick": {"colorscheme": "magentaCarbon"}})

    # Add feature spheres
    def add_spheres(features, model_idx):
        for feat in features:
            x, y, z = feat.position
            viewer.addSphere(
                {
                    "center": {"x": x, "y": y, "z": z},
                    "radius": 0.5,
                    "color": feat.family.color,
                    "alpha": 0.8,
                    "model": model_idx,
                },
            )
            viewer.addLabel(
                feat.name,
                {
                    "position": {"x": x, "y": y, "z": z},
                    "fontColor": feat.family.color,
                    "backgroundOpacity": 0.2,
                    "backgroundColor": "white",
                },
            )

    add_spheres(receptor_features, 0)
    add_spheres(ligand_features, 1)

    # Center and zoom to ligand
    viewer.zoomTo({"model": 1})
    viewer.render()
    viewer.show()


def get_skeleton_edges(features: list[Feature], distance_matrix: OrderedTupleDict) -> list[tuple]:
    def _find_group(list_of_lists, element):
        found_index = None
        for index, sublist in enumerate(list_of_lists):
            if element in sublist:
                found_index = index
                break
        return found_index

    edges = set()
    features_remaining = copy.copy(features)
    ordered_distances = sorted(distance_matrix.items(), key=lambda x: x[1])
    groups = [[feature] for feature in features_remaining]

    while len(features_remaining) > 0 or len(groups) > 1:
        smallest_edge = ordered_distances[0]
        ordered_distances.pop(0)
        new_edge = smallest_edge[0]
        edges.add(new_edge)

        for feat in new_edge:
            try:
                features_remaining.remove(feat)
            except:
                pass

        first_group_index = _find_group(groups, new_edge[0])
        second_group_index = _find_group(groups, new_edge[1])

        if first_group_index != second_group_index:
            groups[first_group_index] += groups[second_group_index]
            groups.pop(second_group_index)

    return [*(zip(node_a.position, node_b.position) for node_a, node_b in edges)]  # type: ignore


def plot_feat_and_edges(
    features: list[Feature],
    edges: list,
    ax,
    vertical_offset: float = 0,
) -> None:
    for feat in features:
        x, y, z = feat.position
        ax.scatter(x, y, z + vertical_offset, c=feat.family.color, s=50)
        ax.text(x, y, z + vertical_offset + 0.3, feat.name, fontsize=8, color="black")

    for edge in edges:
        x_vector, y_vector, z_vector = edge
        z_vector = tuple(z + vertical_offset for z in z_vector)
        ax.plot(x_vector, y_vector, z_vector, color="tab:gray", linewidth=1)


def _format_axes(ax):
    """Visualization options for the 3D axes."""
    # Turn gridlines off
    ax.grid(False)
    ax.set_axis_off()

    # Suppress tick labels
    for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
        dim.set_ticks([])


def draw_feature_list(features: list[Feature], distance_matrix: OrderedTupleDict) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    edges = get_skeleton_edges(features, distance_matrix)

    plot_feat_and_edges(features, edges, ax)

    _format_axes(ax)
    fig.tight_layout()
    plt.show()


def draw_docking(
    L_features: list[Feature],
    L_distance_matrix: OrderedTupleDict,
    R_features: list[Feature],
    R_distance_matrix: OrderedTupleDict,
    interacting_nodes: list[InteractionNode],
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    vertical_offset = 20

    R_edges = get_skeleton_edges(R_features, R_distance_matrix)
    plot_feat_and_edges(R_features, R_edges, ax)

    L_edges = get_skeleton_edges(L_features, L_distance_matrix)
    plot_feat_and_edges(L_features, L_edges, ax, vertical_offset=vertical_offset)

    for interaction in interacting_nodes:
        x_vector, y_vector, z_vector = zip(
            interaction.R_feature.position,
            interaction.L_feature.position,
        )
        R_z_vector, L_z_vector = z_vector
        L_z_vector += vertical_offset
        z_vector = (R_z_vector, L_z_vector)
        ax.plot(
            x_vector,
            y_vector,
            z_vector,
            color="tab:green",
            linewidth=3,
            linestyle="dashed",
        )

    _format_axes(ax)
    fig.tight_layout()
    plt.show()


def draw_multiple_dockings(
    L_features: list[Feature],
    L_distance_matrix: OrderedTupleDict,
    R_features: list[Feature],
    R_distance_matrix: OrderedTupleDict,
    cliques_list: list[list[str]],
) -> None:
    nodes_cliques = results_to_interaction_graph_cliques(cliques_list, L_features, R_features)

    for clique in nodes_cliques:
        draw_docking(L_features, L_distance_matrix, R_features, R_distance_matrix, clique)


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import proj3d


class Interactive3DPlotter:
    def __init__(self, ligand_points, receptor_points, contact_list):
        self.ligand_points_data = ligand_points
        self.receptor_points_data = receptor_points
        self.contact_list = contact_list

        self.fig = plt.figure(figsize=(14, 12))
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.lig_scatter_artist = None
        self.rec_scatter_artist = None
        self.contact_circles_scatter_artist = None

        self.lig_coords_3d = None
        self.rec_coords_3d = None
        self.contacted_rec_coords_3d = None

        self.S_MIN_SIZE = 20.0
        self.S_MAX_SIZE = 150.0  # Increased max size for better visual effect
        self.TEXT_OFFSET_FACTOR = 0.03  # Slightly increased for larger points

        self._is_updating_sizes = False
        self._plot_all_elements()
        self.event_connection_id = self.fig.canvas.mpl_connect("draw_event", self._on_draw_event)

        # Debug: Print initial camera state
        # print(f"Initial view: elev={self.ax.elev:.1f}, azim={self.ax.azim:.1f}, dist={self.ax.dist:.1f}")

    def _get_text_offset(self):
        all_coords = []
        if self.ligand_points_data:
            all_coords.extend(p.position for p in self.ligand_points_data)
        if self.receptor_points_data:
            all_coords.extend(p.position for p in self.receptor_points_data)
        if not all_coords:
            return 0.1

        all_coords_np = np.array(all_coords)
        ranges = np.ptp(all_coords_np, axis=0)
        avg_range = np.mean(ranges[ranges > 0]) if np.any(ranges > 0) else 1.0
        return self.TEXT_OFFSET_FACTOR * avg_range

    def _calculate_initial_sizes(self, points_3d_coords):
        if points_3d_coords is None or points_3d_coords.shape[0] == 0:
            return np.array([])

        # Initial sizing uses data Z, assuming larger data Z might be "further" initially
        # This is mostly a placeholder until the first dynamic update.
        z_coords_data = points_3d_coords[:, 2]
        if not z_coords_data.size:
            return np.array([(self.S_MIN_SIZE + self.S_MAX_SIZE) / 2.0] * len(points_3d_coords))

        z_min_data, z_max_data = np.min(z_coords_data), np.max(z_coords_data)
        z_range_data = z_max_data - z_min_data

        sizes = []
        for z_val in z_coords_data:
            if z_range_data > 1e-6:
                normalized_z_data = (z_val - z_min_data) / z_range_data
                size = self.S_MIN_SIZE + normalized_z_data * (self.S_MAX_SIZE - self.S_MIN_SIZE)
            else:
                size = (self.S_MIN_SIZE + self.S_MAX_SIZE) / 2.0
            sizes.append(size)
        return np.array(sizes)

    def _plot_all_elements(self):
        text_offset = self._get_text_offset()
        ligand_points_map = {p.name: p for p in self.ligand_points_data}
        receptor_points_map = {p.name: p for p in self.receptor_points_data}

        if self.ligand_points_data:
            self.lig_coords_3d = np.array([p.position for p in self.ligand_points_data])
            initial_lig_sizes = self._calculate_initial_sizes(self.lig_coords_3d)  # Temporary
            self.lig_scatter_artist = self.ax.scatter(
                self.lig_coords_3d[:, 0],
                self.lig_coords_3d[:, 1],
                self.lig_coords_3d[:, 2],
                c="blue",
                s=initial_lig_sizes,
                label="Ligand Points",
                depthshade=True,
            )
            default_lig_label_color = "black"
            for i, p_obj in enumerate(self.ligand_points_data):
                label_color = default_lig_label_color
                if p_obj.family and hasattr(p_obj.family, "color"):
                    label_color = p_obj.family.color
                self.ax.text(
                    self.lig_coords_3d[i, 0] + text_offset,
                    self.lig_coords_3d[i, 1] + text_offset,
                    self.lig_coords_3d[i, 2],
                    p_obj.name,
                    color=label_color,
                    fontsize=9,
                )

        if self.receptor_points_data:
            self.rec_coords_3d = np.array([p.position for p in self.receptor_points_data])
            initial_rec_sizes = self._calculate_initial_sizes(self.rec_coords_3d)  # Temporary
            self.rec_scatter_artist = self.ax.scatter(
                self.rec_coords_3d[:, 0],
                self.rec_coords_3d[:, 1],
                self.rec_coords_3d[:, 2],
                c="red",
                s=initial_rec_sizes,
                alpha=0.7,
                label="Receptor Points",
                depthshade=True,
            )
            for i, p_obj in enumerate(self.receptor_points_data):
                self.ax.text(
                    self.rec_coords_3d[i, 0] + text_offset,
                    self.rec_coords_3d[i, 1] + text_offset,
                    self.rec_coords_3d[i, 2],
                    p_obj.name,
                    color=p_obj.family.color,
                    fontsize=9,
                )

        contacted_receptor_point_objects = set()
        if self.contact_list:
            for contact_str in self.contact_list:
                try:
                    lig_name, rec_name = contact_str.split("-", 1)
                except ValueError:
                    continue
                lig_point = ligand_points_map.get(lig_name)
                rec_point = receptor_points_map.get(rec_name)
                if lig_point and rec_point:
                    self.ax.plot(
                        [lig_point.position[0], rec_point.position[0]],
                        [lig_point.position[1], rec_point.position[1]],
                        [lig_point.position[2], rec_point.position[2]],
                        linestyle="--",
                        color="green",
                        linewidth=1.5,
                        label="_nolegend_",
                    )
                    contacted_receptor_point_objects.add(rec_point)

        if contacted_receptor_point_objects:
            self.contacted_rec_coords_3d = np.array(
                [p.position for p in contacted_receptor_point_objects],
            )
            if self.contacted_rec_coords_3d.shape[0] > 0:
                initial_contact_point_sizes = self._calculate_initial_sizes(
                    self.contacted_rec_coords_3d,
                )  # Temporary
                initial_ring_sizes = [
                    max(self.S_MAX_SIZE * 1.5, s_pt * 2.0) for s_pt in initial_contact_point_sizes
                ]
                self.contact_circles_scatter_artist = self.ax.scatter(
                    self.contacted_rec_coords_3d[:, 0],
                    self.contacted_rec_coords_3d[:, 1],
                    self.contacted_rec_coords_3d[:, 2],
                    s=np.array(initial_ring_sizes),
                    facecolors="none",
                    edgecolors="green",
                    linewidth=2,
                    alpha=0.9,
                    label="_nolegend_",
                )

        self.ax.set_xlabel("X coordinate")
        self.ax.set_ylabel("Y coordinate")
        self.ax.set_zlabel("Z coordinate")
        self.ax.set_title("Interactive 3D Visualization (Sizes Update with View)")
        self._update_legend()
        # Force an initial draw to trigger _on_draw_event for correct initial sizes
        self.fig.canvas.draw_idle()

    def _update_legend(self):
        handles, labels = self.ax.get_legend_handles_labels()
        filtered_handles_labels = [(h, l) for h, l in zip(handles, labels) if l != "_nolegend_"]
        by_label = dict(filtered_handles_labels)
        has_contacts = any(
            isinstance(c, str)
            and "-" in c
            and {p.name: p for p in self.ligand_points_data}.get(c.split("-", 1)[0])
            and {p.name: p for p in self.receptor_points_data}.get(
                c.split("-", 1)[1] if len(c.split("-", 1)) > 1 else None,
            )
            for c in self.contact_list
        )
        if has_contacts and "Contact Line" not in by_label:
            by_label["Contact Line"] = Line2D(
                [0],
                [0],
                linestyle="--",
                color="green",
                linewidth=1.5,
            )
        if (
            self.contacted_rec_coords_3d is not None
            and self.contacted_rec_coords_3d.shape[0] > 0
            and "Contacted Receptor" not in by_label
        ):
            by_label["Contacted Receptor"] = Line2D(
                [0],
                [0],
                linestyle="none",
                marker="o",
                markersize=10,
                markerfacecolor="none",
                markeredgecolor="green",
                markeredgewidth=2,
            )
        current_legend = self.ax.get_legend()
        if current_legend:
            current_legend.remove()
        if by_label:
            self.ax.legend(by_label.values(), by_label.keys())

    def _get_dynamic_sizes_for_view(self, points_3d_coords_original):
        if (
            self.ax.M is None
            or points_3d_coords_original is None
            or points_3d_coords_original.shape[0] == 0
        ):
            return np.array([])

        try:
            x_disp, y_disp, z_view = proj3d.proj_transform(
                points_3d_coords_original[:, 0],
                points_3d_coords_original[:, 1],
                points_3d_coords_original[:, 2],
                self.ax.M,
            )
        except Exception as e:
            # print(f"Error in proj_transform: {e}") # Debug
            return self._calculate_initial_sizes(points_3d_coords_original)  # Fallback

        if not z_view.size:
            return np.array([])

        # --- Key change in interpretation ---
        # Hypothesis: LARGER z_view values are CLOSER to the camera.
        #             SMALLER z_view values are FURTHER from the camera.

        z_view_min, z_view_max = np.min(z_view), np.max(z_view)
        z_view_range = z_view_max - z_view_min

        # --- Diagnostic prints (uncomment to debug) ---
        # if points_3d_coords_original is self.lig_coords_3d: # Print only for one set of points
        #     print(f"View: elev={self.ax.elev:.1f}, azim={self.ax.azim:.1f}, dist={self.ax.dist:.1f}")
        #     print(f"z_view range: {z_view_min:.3f} to {z_view_max:.3f} (range: {z_view_range:.3f})")
        #     if z_view.size > 0:
        #         idx_closest_visual = np.argmax(z_view) # Index of point with largest z_view
        #         idx_furthest_visual = np.argmin(z_view) # Index of point with smallest z_view
        #         print(f"  Point with max z_view ({z_view[idx_closest_visual]:.3f}) should be largest.")
        #         print(f"  Point with min z_view ({z_view[idx_furthest_visual]:.3f}) should be smallest.")

        new_sizes = []
        for i, z_v in enumerate(z_view):
            if z_view_range > 1e-9:
                # Normalized_z_v: 0 for z_view_min (furthest), 1 for z_view_max (closest)
                normalized_z_v = (z_v - z_view_min) / z_view_range

                # Size: S_MIN_SIZE for norm=0 (furthest), S_MAX_SIZE for norm=1 (closest)
                size = self.S_MIN_SIZE + normalized_z_v * (self.S_MAX_SIZE - self.S_MIN_SIZE)
            else:
                size = (self.S_MIN_SIZE + self.S_MAX_SIZE) / 2.0
            new_sizes.append(max(1.0, size))  # Ensure a minimum positive size

            # --- Diagnostic print for a specific point (e.g., first ligand point) ---
            # if points_3d_coords_original is self.lig_coords_3d and i == 0:
            #     print(f"  LIG0: z_orig={points_3d_coords_original[i,2]:.2f}, z_view={z_v:.3f}, norm_z_v={normalized_z_v:.3f}, size={size:.1f}")

        return np.array(new_sizes)

    def _on_draw_event(self, event):
        if self._is_updating_sizes:
            return
        self._is_updating_sizes = True
        try:
            if self.lig_scatter_artist and self.lig_coords_3d is not None:
                new_lig_sizes = self._get_dynamic_sizes_for_view(self.lig_coords_3d)
                if new_lig_sizes.size > 0:
                    self.lig_scatter_artist.set_sizes(new_lig_sizes)

            if self.rec_scatter_artist and self.rec_coords_3d is not None:
                new_rec_sizes = self._get_dynamic_sizes_for_view(self.rec_coords_3d)
                if new_rec_sizes.size > 0:
                    self.rec_scatter_artist.set_sizes(new_rec_sizes)

            if self.contact_circles_scatter_artist and self.contacted_rec_coords_3d is not None:
                underlying_point_dynamic_sizes = self._get_dynamic_sizes_for_view(
                    self.contacted_rec_coords_3d,
                )
                if underlying_point_dynamic_sizes.size > 0:
                    new_ring_sizes = [
                        max(self.S_MAX_SIZE * 1.2, s_pt * 1.8)
                        for s_pt in underlying_point_dynamic_sizes
                    ]  # Adjusted ring size logic
                    self.contact_circles_scatter_artist.set_sizes(np.array(new_ring_sizes))
        finally:
            self._is_updating_sizes = False

    def show(self):
        plt.tight_layout()
        plt.show()

    def __del__(self):
        if (
            hasattr(self, "event_connection_id")
            and self.event_connection_id is not None
            and self.fig.canvas
        ):
            try:
                self.fig.canvas.mpl_disconnect(self.event_connection_id)
            except Exception:
                pass


def plot_3d_features_with_contacts(ligand_points, receptor_points, contact_list):
    """
    Main function to create and show the interactive 3D plot.
    The plotter object is returned in case it needs to be kept alive (e.g., in Jupyter).
    """
    plotter = Interactive3DPlotter(ligand_points, receptor_points, contact_list)
    # plotter.show() # show() is blocking, so Interactive3DPlotter should manage this
    # The show() is called by the main script or Jupyter. The object must persist.
    return plotter
