# @title  Temporal Perturbation Scanning - LOOP VERSION (for batching)
'''
Temporal Perturbation Scanning (TPS) - LOOP version
By Fodil Azzaz, PhD, All right reserved
Copyright (c) 2025 Fodil Azzaz

ACADEMIC LICENSE
================
✅ PERMITTED:
- Academic research and teaching
- Non-commercial scientific use
- Integration into research pipelines
- Publication using this software

🚫 COMMERCIAL USE:
- Commercial use requires authorization
- Contact: azzaz.fodil@gmail.com

CITATION:
If you use this software in your research, please cite:
[Your Paper Reference - Coming Soon]

This license ensures open academic collaboration while protecting commercial rights.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy.spatial import cKDTree
from torch_geometric.nn import GCNConv
from scipy.stats import norm
import itertools
import os
import glob

print(" UNIVERSAL PROTEIN INTERFACE ANALYZER - SUM Δ VERSION (BIOLOGICALLY CORRECT)")
print("=" * 85)

# ============================================================
#  GLOBAL FILE PATHS - SET THESE AT THE END
# ============================================================
GRAPH_FILE_PATH = None
MODEL_FILE_PATH = None

def set_file_paths(graph_path, model_path):
    """Set the file paths once at the beginning"""
    global GRAPH_FILE_PATH, MODEL_FILE_PATH
    GRAPH_FILE_PATH = graph_path
    MODEL_FILE_PATH = model_path
    print(f"✅ File paths set:")
    print(f"   • Graph data: {GRAPH_FILE_PATH}")
    print(f"   • Model: {MODEL_FILE_PATH}")

# ============================================================
#  RESIDUE PROPERTY DATABASE
# ============================================================

def create_enhanced_residue_features():
    """Create detailed residue property database for biological perturbations"""
    residue_properties = {
        'ALA': {'charge': 0, 'hydrophobicity': 1.8, 'aromatic': 0, 'polar': 0, 'size': 1.0, 'h_bond_donor': 0, 'h_bond_acceptor': 0},
        'ARG': {'charge': 1, 'hydrophobicity': -4.5, 'aromatic': 0, 'polar': 1, 'size': 3.0, 'h_bond_donor': 4, 'h_bond_acceptor': 6},
        'ASN': {'charge': 0, 'hydrophobicity': -3.5, 'aromatic': 0, 'polar': 1, 'size': 1.5, 'h_bond_donor': 2, 'h_bond_acceptor': 4},
        'ASP': {'charge': -1, 'hydrophobicity': -3.5, 'aromatic': 0, 'polar': 1, 'size': 1.5, 'h_bond_donor': 1, 'h_bond_acceptor': 4},
        'CYS': {'charge': 0, 'hydrophobicity': 2.5, 'aromatic': 0, 'polar': 1, 'size': 1.5, 'h_bond_donor': 1, 'h_bond_acceptor': 1},
        'GLN': {'charge': 0, 'hydrophobicity': -3.5, 'aromatic': 0, 'polar': 1, 'size': 2.0, 'h_bond_donor': 2, 'h_bond_acceptor': 4},
        'GLU': {'charge': -1, 'hydrophobicity': -3.5, 'aromatic': 0, 'polar': 1, 'size': 2.0, 'h_bond_donor': 1, 'h_bond_acceptor': 4},
        'GLY': {'charge': 0, 'hydrophobicity': -0.4, 'aromatic': 0, 'polar': 0, 'size': 0.5, 'h_bond_donor': 2, 'h_bond_acceptor': 2},
        'HIS': {'charge': 0.5, 'hydrophobicity': -3.2, 'aromatic': 1, 'polar': 1, 'size': 2.5, 'h_bond_donor': 2, 'h_bond_acceptor': 4},
        'ILE': {'charge': 0, 'hydrophobicity': 4.5, 'aromatic': 0, 'polar': 0, 'size': 2.5, 'h_bond_donor': 1, 'h_bond_acceptor': 1},
        'LEU': {'charge': 0, 'hydrophobicity': 3.8, 'aromatic': 0, 'polar': 0, 'size': 2.5, 'h_bond_donor': 1, 'h_bond_acceptor': 1},
        'LYS': {'charge': 1, 'hydrophobicity': -3.9, 'aromatic': 0, 'polar': 1, 'size': 3.0, 'h_bond_donor': 3, 'h_bond_acceptor': 2},
        'MET': {'charge': 0, 'hydrophobicity': 1.9, 'aromatic': 0, 'polar': 0, 'size': 2.5, 'h_bond_donor': 1, 'h_bond_acceptor': 2},
        'PHE': {'charge': 0, 'hydrophobicity': 2.8, 'aromatic': 1, 'polar': 0, 'size': 3.0, 'h_bond_donor': 1, 'h_bond_acceptor': 1},
        'PRO': {'charge': 0, 'hydrophobicity': -1.6, 'aromatic': 0, 'polar': 0, 'size': 1.5, 'h_bond_donor': 1, 'h_bond_acceptor': 2},
        'SER': {'charge': 0, 'hydrophobicity': -0.8, 'aromatic': 0, 'polar': 1, 'size': 1.0, 'h_bond_donor': 1, 'h_bond_acceptor': 2},
        'THR': {'charge': 0, 'hydrophobicity': -0.7, 'aromatic': 0, 'polar': 1, 'size': 1.5, 'h_bond_donor': 1, 'h_bond_acceptor': 2},
        'TRP': {'charge': 0, 'hydrophobicity': -0.9, 'aromatic': 1, 'polar': 0, 'size': 3.5, 'h_bond_donor': 1, 'h_bond_acceptor': 2},
        'TYR': {'charge': 0, 'hydrophobicity': -1.3, 'aromatic': 1, 'polar': 1, 'size': 3.0, 'h_bond_donor': 1, 'h_bond_acceptor': 2},
        'VAL': {'charge': 0, 'hydrophobicity': 4.2, 'aromatic': 0, 'polar': 0, 'size': 2.0, 'h_bond_donor': 1, 'h_bond_acceptor': 1},
    }

    # Normalize values
    for resname, props in residue_properties.items():
        props['hydrophobicity'] = max(-1.0, min(1.0, props['hydrophobicity'] / 4.5))
        props['size'] = props['size'] / 3.5
        props['h_bond_donor'] = props['h_bond_donor'] / 4.0
        props['h_bond_acceptor'] = props['h_bond_acceptor'] / 6.0

    return residue_properties

RESIDUE_PROPERTIES = create_enhanced_residue_features()

def get_perturbation_type(props):
    """Determine perturbation type from residue properties"""
    if props.get('charge', 0) != 0:
        return "ELECTROSTATIC"
    elif props.get('aromatic', 0) == 1:
        return "AROMATIC"
    elif props.get('polar', 0) == 1:
        return "H-BOND"
    elif props.get('hydrophobicity', 0) > 0.3:
        return "HYDROPHOBIC"
    else:
        return "GENERAL"

# ============================================================
#  UTILITY FUNCTIONS
# ============================================================

def detect_all_segids(graphs):
    """Detect ALL segids present in the dataset"""
    print(" Scanning for all available segids...")

    all_segids = set()
    for graph_idx, graph in enumerate(graphs):
        if hasattr(graph, 'segids'):
            if isinstance(graph.segids, (list, tuple)):
                segids_list = graph.segids
            elif hasattr(graph.segids, '__len__') and len(graph.segids) == graph.num_nodes:
                segids_list = graph.segids
            else:
                segids_list = getattr(graph, 'segids', [])

            for segid in segids_list:
                if segid and str(segid).strip() and str(segid) != 'None':
                    all_segids.add(str(segid))

    available_segids = sorted(list(all_segids))
    print(f"✅ Found {len(available_segids)} unique segids: {available_segids}")

    # Show segid distribution
    print("\n Segid distribution in first graph:")
    sample_graph = graphs[0]
    if hasattr(sample_graph, 'segids'):
        segid_counts = defaultdict(int)
        for segid in sample_graph.segids:
            segid_counts[segid] += 1

        for segid, count in sorted(segid_counts.items()):
            print(f"   {segid}: {count} atoms")

    return available_segids

def load_graphs_with_fix():
    """Load graphs using the global file path"""
    global GRAPH_FILE_PATH
    if GRAPH_FILE_PATH is None:
        print("❌ No graph file path set! Call set_file_paths() first.")
        return None

    print(f"📂 Loading graphs from: {GRAPH_FILE_PATH}")
    try:
        graphs = torch.load(GRAPH_FILE_PATH, map_location='cpu', weights_only=True)
    except:
        graphs = torch.load(GRAPH_FILE_PATH, map_location='cpu', weights_only=False)

    print(f"   Loaded {len(graphs)} graphs")
    for graph in graphs:
        if hasattr(graph, 'x') and graph.x is not None:
            graph.x = graph.x.float()
        if hasattr(graph, 'pos') and graph.pos is not None:
            graph.pos = graph.pos.float()
    return graphs

def select_frames(graphs, total_frames, step=1, start_frame=0, max_frames=None):
    """Select frames for analysis"""
    max_available = len(graphs)

    if max_frames is not None:
        max_available = min(max_available, max_frames)

    if start_frame >= max_available:
        print(f"   ⚠️  Start frame {start_frame} exceeds available frames {max_available}")
        start_frame = 0

    if start_frame + total_frames > max_frames if max_frames else start_frame + total_frames > max_available:
        total_frames = (max_frames if max_frames else max_available) - start_frame
        print(f"   ⚠️  Adjusted total_frames to {total_frames} to fit within available frames")

    selected_indices = list(range(start_frame, start_frame + total_frames, step))
    selected_graphs = [graphs[i] for i in selected_indices if i < (max_frames if max_frames else max_available)]

    print(f"    Frame selection: {len(selected_graphs)} frames "
          f"({start_frame} to {start_frame + total_frames - 1}, step={step})")
    return selected_graphs

# ============================================================
#  ENHANCED BIOLOGICAL PERTURBATIONS
# ============================================================

def apply_electrostatic_perturbation(features, target_indices, target_resnames, perturbation_strength=0.8):
    """Perturb electrostatic properties specifically"""
    mutated_features = features.clone()

    for idx, resname in zip(target_indices, target_resnames):
        props = RESIDUE_PROPERTIES.get(resname, {'charge': 0})

        if props['charge'] != 0:
            mutated_features[idx][0] *= (1 - perturbation_strength)
            mutated_features[idx][4] = 0.0

            if perturbation_strength > 0.5:
                mutated_features[idx][0] *= -0.5
        else:
            mutated_features[idx][0] = 0.3 * np.random.choice([-1, 1])
            mutated_features[idx][4] = 0.3 * np.random.choice([-1, 1])

    return mutated_features

def apply_hydrophobic_perturbation(features, target_indices, target_resnames, perturbation_strength=0.8):
    """Perturb hydrophobic properties specifically"""
    mutated_features = features.clone()

    for idx, resname in zip(target_indices, target_resnames):
        props = RESIDUE_PROPERTIES.get(resname, {'hydrophobicity': 0})

        current_hydro = features[idx][3].item()

        if abs(props['hydrophobicity']) > 0.3:
            new_hydro = -current_hydro * perturbation_strength
            mutated_features[idx][3] = new_hydro

            if props['polar'] == 1:
                mutated_features[idx][5] *= (1 - perturbation_strength)
        else:
            mutated_features[idx][3] = 0.5 * perturbation_strength

    return mutated_features

def apply_steric_perturbation(features, target_indices, target_resnames, perturbation_strength=0.8):
    """Perturb steric properties - simulate bulky substitutions"""
    mutated_features = features.clone()

    for idx, resname in zip(target_indices, target_resnames):
        props = RESIDUE_PROPERTIES.get(resname, {'size': 0.5})

        mutated_features[idx] *= (1 - perturbation_strength * 0.3)
        mutated_features[idx][6] *= (1 - perturbation_strength)

        noise = torch.randn_like(mutated_features[idx]) * 0.1 * perturbation_strength
        mutated_features[idx] += noise

    return mutated_features

def apply_aromatic_perturbation(features, target_indices, target_resnames, perturbation_strength=0.8):
    """Specifically target aromatic interactions (π-π stacking, CH-π)"""
    mutated_features = features.clone()

    for idx, resname in zip(target_indices, target_resnames):
        props = RESIDUE_PROPERTIES.get(resname, {'aromatic': 0})

        if props['aromatic'] == 1:
            mutated_features[idx][3] *= (1 - perturbation_strength)

            if mutated_features[idx].shape[0] > 8:
                mutated_features[idx][7] *= (1 - perturbation_strength)
        else:
            mutated_features[idx][3] += 0.3 * perturbation_strength

    return mutated_features

def apply_hydrogen_bond_perturbation(features, target_indices, target_resnames, perturbation_strength=0.8):
    """Simulate disruption of hydrogen bonding networks"""
    mutated_features = features.clone()

    polar_residues = ['SER', 'THR', 'ASN', 'GLN', 'ASP', 'GLU', 'HIS', 'ARG', 'LYS', 'TYR']

    for idx, resname in zip(target_indices, target_resnames):
        if resname in polar_residues:
            mutated_features[idx][4] *= (1 - perturbation_strength)

            scale_factor = 1 - (perturbation_strength * 0.5)
            mutated_features[idx] *= scale_factor

            mutated_features[idx][0] *= (1 - perturbation_strength)

    return mutated_features

def apply_conformational_perturbation(graph, target_indices, perturbation_strength=1.0):
    """Introduce conformational strain by displacing sidechain atoms"""
    modified = graph.clone()

    for idx in target_indices:
        if hasattr(graph, 'resnames') and graph.resnames[idx] in RESIDUE_PROPERTIES:
            displacement = torch.randn(3) * 0.3 * perturbation_strength
            modified.pos[idx] += displacement

    return modified

def apply_residue_specific_masking(features, target_indices, target_resnames):
    """Intelligent masking based on residue chemical properties"""
    mutated_features = features.clone()

    for idx, resname in zip(target_indices, target_resnames):
        props = RESIDUE_PROPERTIES.get(resname, {'charge': 0, 'hydrophobicity': 0, 'aromatic': 0, 'polar': 0})

        if resname in ['LYS', 'ARG']:
            mutated_features[idx][0] *= 0.1
            mutated_features[idx][4] = 0.0
            mutated_features[idx] *= 0.7

        elif resname in ['GLU', 'ASP']:
            mutated_features[idx][0] *= 0.1
            mutated_features[idx][4] = 0.0
            mutated_features[idx] *= 0.7

        elif resname in ['TRP', 'TYR', 'PHE']:
            mutated_features[idx][3] *= 0.3
            mutated_features[idx][1] *= 0.8
            if mutated_features[idx].shape[0] > 8:
                mutated_features[idx][7] *= 0.5

        elif resname in ['SER', 'THR', 'ASN', 'GLN']:
            mutated_features[idx][0] *= 0.5
            mutated_features[idx][3] *= 0.7
            mutated_features[idx][4] *= 0.6

        elif resname in ['ALA', 'VAL', 'LEU', 'ILE', 'MET']:
            mutated_features[idx][3] *= 0.4
            mutated_features[idx][6] *= 0.7

        else:
            mutated_features[idx] *= 0.5

    return mutated_features

# ============================================================
#  COMPREHENSIVE PERTURBATION SCANNING
# ============================================================

def comprehensive_perturbation_scan(model, graph, res1, res2, original_pred, segid1, segid2):
    """Apply all perturbation types and return max effect"""
    target_indices = []
    target_resnames = []

    for i in range(graph.num_nodes):
        if hasattr(graph, 'segids') and hasattr(graph, 'residues'):
            segid = graph.segids[i]
            resid = graph.residues[i]
            resname = graph.resnames[i] if hasattr(graph, 'resnames') else "UNK"

            current_res = f"{segid}-{resname}-{resid}"
            if current_res == res1 or current_res == res2:
                target_indices.append(i)
                target_resnames.append(resname)

    if not target_indices:
        return 0.0

    perturbations = []

    modified = graph.clone()
    modified.x = apply_electrostatic_perturbation(modified.x, target_indices, target_resnames, 0.8)
    with torch.no_grad():
        pred = model(modified).item()
    perturbations.append(abs(pred - original_pred))

    modified = graph.clone()
    modified.x = apply_hydrophobic_perturbation(modified.x, target_indices, target_resnames, 0.8)
    with torch.no_grad():
        pred = model(modified).item()
    perturbations.append(abs(pred - original_pred))

    modified = graph.clone()
    modified.x = apply_steric_perturbation(modified.x, target_indices, target_resnames, 0.8)
    with torch.no_grad():
        pred = model(modified).item()
    perturbations.append(abs(pred - original_pred))

    aromatic_residues = [r for r in target_resnames if RESIDUE_PROPERTIES.get(r, {}).get('aromatic', 0) == 1]
    if aromatic_residues:
        modified = graph.clone()
        modified.x = apply_aromatic_perturbation(modified.x, target_indices, target_resnames, 0.8)
        with torch.no_grad():
            pred = model(modified).item()
        perturbations.append(abs(pred - original_pred))

    polar_residues = [r for r in target_resnames if r in ['SER', 'THR', 'ASN', 'GLN', 'ASP', 'GLU', 'HIS', 'ARG', 'LYS', 'TYR']]
    if polar_residues:
        modified = graph.clone()
        modified.x = apply_hydrogen_bond_perturbation(modified.x, target_indices, target_resnames, 0.8)
        with torch.no_grad():
            pred = model(modified).item()
        perturbations.append(abs(pred - original_pred))

    modified = apply_conformational_perturbation(graph, target_indices, 1.0)
    with torch.no_grad():
        pred = model(modified).item()
    perturbations.append(abs(pred - original_pred))

    return max(perturbations) if perturbations else 0.0

# ============================================================
#  MODEL LOADING FUNCTIONS
# ============================================================

def load_interface_model(segid1, segid2):
    """Load model using the global file path"""
    global MODEL_FILE_PATH
    if MODEL_FILE_PATH is None:
        print("❌ No model file path set! Call set_file_paths() first.")
        return None

    print(f"📂 Loading model for {segid1} ↔ {segid2} interface...")
    try:
        checkpoint = torch.load(MODEL_FILE_PATH, map_location='cpu', weights_only=True)
    except:
        checkpoint = torch.load(MODEL_FILE_PATH, map_location='cpu', weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        if 'interface_predictor.weight' in state_dict:
            interface_weight_shape = state_dict['interface_predictor.weight'].shape
            num_interface_features = interface_weight_shape[1]
        else:
            num_interface_features = 96
    else:
        print("   ❌ No model_state_dict found in checkpoint")
        return None

    graphs = load_graphs_with_fix()
    if graphs is None:
        return None

    node_dim = graphs[0].x.shape[1]

    class UniversalInterfacePredictor(nn.Module):
        def __init__(self, node_dim, num_interface_features=96, segid1="PROA", segid2="PROD"):
            super().__init__()
            self.conv1 = GCNConv(node_dim, 128)
            self.conv2 = GCNConv(128, 64)
            self.conv3 = GCNConv(64, 32)
            self.batch_norm1 = nn.BatchNorm1d(128)
            self.batch_norm2 = nn.BatchNorm1d(64)
            self.batch_norm3 = nn.BatchNorm1d(32)
            self.dropout = nn.Dropout(0.3)
            self.interface_predictor = nn.Linear(num_interface_features, 1)
            self.required_interfaces = num_interface_features // 32
            self.segid1 = segid1
            self.segid2 = segid2

        def forward(self, data):
            x = F.relu(self.batch_norm1(self.conv1(data.x, data.edge_index)))
            x = F.relu(self.batch_norm2(self.conv2(x, data.edge_index)))
            x = F.relu(self.batch_norm3(self.conv3(x, data.edge_index)))
            x = self.dropout(x)
            interface_features = self.generate_interface_features(data, x)
            x_pooled = interface_features.flatten()
            return self.interface_predictor(x_pooled).squeeze()

        def generate_interface_features(self, data, node_features):
            segids = data.segids
            positions = data.pos.cpu().numpy()
            interface_features = []

            segid1_indices = torch.tensor([i for i, s in enumerate(segids) if s == self.segid1])
            segid2_indices = torch.tensor([i for i, s in enumerate(segids) if s == self.segid2])

            if segid1_indices.numel() > 0 and segid2_indices.numel() > 0:
                interface_mask = self.detect_interface(positions, segid1_indices, segid2_indices)
                if interface_mask.sum() > 0:
                    interface_feature = node_features[interface_mask].mean(dim=0)
                else:
                    interface_feature = node_features.mean(dim=0)
            else:
                interface_feature = node_features.mean(dim=0)

            interface_features.append(interface_feature)

            while len(interface_features) < self.required_interfaces:
                if len(interface_features) == 1 and segid1_indices.numel() > 0:
                    variant_feature = node_features[segid1_indices].mean(dim=0)
                elif len(interface_features) == 2 and segid2_indices.numel() > 0:
                    variant_feature = node_features[segid2_indices].mean(dim=0)
                else:
                    variant_feature = interface_feature + 0.02 * torch.randn_like(interface_feature)
                interface_features.append(variant_feature)

            return torch.stack(interface_features[:self.required_interfaces])

        def detect_interface(self, positions, segid1_indices, segid2_indices):
            interface_mask = torch.zeros(len(positions), dtype=torch.bool)
            segid1_positions = positions[segid1_indices.cpu().numpy()]
            segid2_positions = positions[segid2_indices.cpu().numpy()]

            if len(segid1_positions) == 0 or len(segid2_positions) == 0:
                return interface_mask

            tree_segid1 = cKDTree(segid1_positions)
            distances, indices = tree_segid1.query(segid2_positions, k=1)

            for i, segid2_idx in enumerate(segid2_indices):
                if distances[i] < 8.0:
                    interface_mask[segid2_idx] = True
                    if distances[i] < 6.0:
                        segid1_idx = segid1_indices[indices[i]]
                        interface_mask[segid1_idx] = True

            return interface_mask

    model = UniversalInterfacePredictor(
        node_dim=node_dim,
        num_interface_features=num_interface_features,
        segid1=segid1,
        segid2=segid2
    )

    try:
        model.load_state_dict(state_dict)
        print("   ✅ ALL parameters loaded successfully!")
    except RuntimeError as e:
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"   ✅ Partial load: {len(pretrained_dict)}/{len(state_dict)} parameters")

    model.eval()
    return model

# ============================================================
#  UNIVERSAL INTERFACE ANALYSIS FUNCTIONS - ENHANCED
# ============================================================

def find_universal_contacts(graph, segid1, segid2, cutoff=8.0):
    """Find residue pairs between ANY two segids"""
    residue_centers = {}
    residue_atoms = defaultdict(list)

    for i in range(graph.num_nodes):
        if hasattr(graph, 'segids') and hasattr(graph, 'residues') and hasattr(graph, 'resnames'):
            segid = graph.segids[i]
            resid = graph.residues[i]
            resname = graph.resnames[i]

            if segid not in [segid1, segid2]:
                continue

            key = f"{segid}-{resname}-{resid}"
            residue_atoms[key].append(i)

    for residue_key, atom_indices in residue_atoms.items():
        if len(atom_indices) == 0:
            continue
        positions = [graph.pos[atom_idx].cpu().numpy() for atom_idx in atom_indices]
        residue_centers[residue_key] = np.mean(positions, axis=0)

    segid1_residues = {k: v for k, v in residue_centers.items() if k.startswith(segid1)}
    segid2_residues = {k: v for k, v in residue_centers.items() if k.startswith(segid2)}

    if not segid1_residues or not segid2_residues:
        return []

    segid1_keys = list(segid1_residues.keys())
    segid1_centers = np.array(list(segid1_residues.values()))
    segid2_keys = list(segid2_residues.keys())
    segid2_centers = np.array(list(segid2_residues.values()))

    tree = cKDTree(segid2_centers)
    pairs = []

    for i, segid1_key in enumerate(segid1_keys):
        segid1_center = segid1_centers[i]
        distances, indices = tree.query(segid1_center, k=10)

        for j, dist in enumerate(distances):
            if dist < cutoff and j < len(indices):
                segid2_key = segid2_keys[indices[j]]
                pairs.append((segid1_key, segid2_key))

    return pairs

def calculate_universal_delta_masking(model, graph, res1, res2, original_pred, segid1, segid2):
    """Original method 1: COMPLETE feature masking"""
    modified = graph.clone()
    pair_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)

    for i in range(graph.num_nodes):
        if hasattr(graph, 'segids') and hasattr(graph, 'residues'):
            segid = graph.segids[i]
            resid = graph.residues[i]
            resname = graph.resnames[i] if hasattr(graph, 'resnames') else "UNK"

            current_res = f"{segid}-{resname}-{resid}"
            if current_res == res1 or current_res == res2:
                pair_mask[i] = True

    if pair_mask.sum() > 0:
        modified.x[pair_mask] = 0.0
        with torch.no_grad():
            modified_pred = model(modified).item()
        return abs(modified_pred - original_pred)
    return 0.0

def calculate_universal_intelligent_masking(model, graph, res1, res2, original_pred, segid1, segid2):
    """Original method 3: SMART residue-specific masking"""
    modified = graph.clone()
    pair_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)

    target_indices = []
    target_resnames = []

    for i in range(graph.num_nodes):
        if hasattr(graph, 'segids') and hasattr(graph, 'residues'):
            segid = graph.segids[i]
            resid = graph.residues[i]
            resname = graph.resnames[i] if hasattr(graph, 'resnames') else "UNK"

            current_res = f"{segid}-{resname}-{resid}"
            if current_res == res1 or current_res == res2:
                pair_mask[i] = True
                target_indices.append(i)
                target_resnames.append(resname)

    if target_indices:
        modified.x = apply_residue_specific_masking(modified.x, target_indices, target_resnames)
        with torch.no_grad():
            modified_pred = model(modified).item()
        return abs(modified_pred - original_pred)

    return 0.0

def calculate_universal_distance_perturbation(model, graph, res1, res2, original_pred, segid1, segid2):
    """Original method 2: PHYSICAL separation"""
    modified = graph.clone()
    pair_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)

    res1_indices = []
    res2_indices = []

    for i in range(graph.num_nodes):
        if hasattr(graph, 'segids') and hasattr(graph, 'residues'):
            segid = graph.segids[i]
            resid = graph.residues[i]
            resname = graph.resnames[i] if hasattr(graph, 'resnames') else "UNK"

            current_res = f"{segid}-{resname}-{resid}"
            if current_res == res1:
                res1_indices.append(i)
                pair_mask[i] = True
            elif current_res == res2:
                res2_indices.append(i)
                pair_mask[i] = True

    if res1_indices and res2_indices:
        displacement = torch.randn(3) * 3.0
        for idx in res1_indices:
            modified.pos[idx] += displacement
        for idx in res2_indices:
            modified.pos[idx] -= displacement

        with torch.no_grad():
            modified_pred = model(modified).item()
        return abs(modified_pred - original_pred)

    return 0.0

def enhanced_find_universal_interface_pairs(model, selected_graphs, segid1, segid2, 
                                           stage_boundaries=None):
    """Enhanced version with customizable temporal staging"""
    print(f"    Identifying critical {segid1}-{segid2} pairs with COMPREHENSIVE perturbations...")

    model.eval()
    
    # DEFAULT staging: Early (0-33%), Mid (33-66%), Late (66-100%)
    if stage_boundaries is None:
        stage_boundaries = {
            'early': (0.0, 0.33),
            'mid': (0.33, 0.66),
            'late': (0.66, 1.0)
        }
    
    # Initialize accumulators for each stage
    pair_deltas_accumulated = {stage: defaultdict(list) for stage in stage_boundaries.keys()}
    
    total_frames = len(selected_graphs)
    
    print(f"    Custom temporal staging:")
    for stage, (start_frac, end_frac) in stage_boundaries.items():
        start_frame = int(total_frames * start_frac)
        end_frame = int(total_frames * end_frac)
        print(f"      • {stage}: frames {start_frame}-{end_frame} ({start_frac*100:.0f}%-{end_frac*100:.0f}%)")
    
    print(f"    Applying 6 perturbation types: Electrostatic, Hydrophobic, Steric, Aromatic, H-bond, Conformational")

    for frame_idx, graph in enumerate(selected_graphs):
        # Determine which stage this frame belongs to
        frame_frac = frame_idx / total_frames
        current_stage = None
        
        for stage, (start_frac, end_frac) in stage_boundaries.items():
            if start_frac <= frame_frac < end_frac:
                current_stage = stage
                break
        
        if current_stage is None:
            # Default to last stage for edge case
            current_stage = list(stage_boundaries.keys())[-1]

        with torch.no_grad():
            original_pred = model(graph).item()

        interface_pairs = find_universal_contacts(graph, segid1, segid2, cutoff=8.0)

        for res1, res2 in interface_pairs:
            pair_key = f"{res1} ↔ {res2}"

            delta_comprehensive = comprehensive_perturbation_scan(
                model, graph, res1, res2, original_pred, segid1, segid2
            )

            delta_mask = calculate_universal_delta_masking(model, graph, res1, res2, original_pred, segid1, segid2)
            delta_dist = calculate_universal_distance_perturbation(model, graph, res1, res2, original_pred, segid1, segid2)
            delta_intelligent = calculate_universal_intelligent_masking(model, graph, res1, res2, original_pred, segid1, segid2)

            delta = max(delta_comprehensive, delta_mask, delta_dist, delta_intelligent)
            pair_deltas_accumulated[current_stage][pair_key].append(delta)

        if (frame_idx + 1) % 10 == 0:
            print(f"      Frame {frame_idx} ({current_stage}): tested {len(interface_pairs)} pairs with 6+ perturbation types")

    stage_results = {}
    for stage in stage_boundaries.keys():
        pair_deltas_mean = {}
        for pair, deltas in pair_deltas_accumulated[stage].items():
            if deltas:
                pair_deltas_mean[pair] = np.mean(deltas)
        stage_results[stage] = pair_deltas_mean

    print(f"   ✅ Enhanced analysis complete: {', '.join([f'{stage}={len(stage_results[stage])}' for stage in stage_boundaries.keys()])}")
    return stage_results

def run_universal_interface_analysis(segid1, segid2, n_runs=1, total_frames=100, 
                                     step=5, start_frame=0, max_frames=None,
                                     stage_boundaries=None):
    """Universal interface analysis for ANY segid pair - ENHANCED with customizable staging"""
    print(f"\n UNIVERSAL INTERFACE ANALYSIS: {segid1} ↔ {segid2}")
    print(f"   • Runs: {n_runs}")
    print(f"   • Frames: {total_frames} (step={step})")
    print(f"   • Start frame: {start_frame}")
    print(f"   • Max frames: {max_frames}")
    print(f"   • Focus: {segid1} ↔ {segid2} interactions")
    print(f"   • Temporal staging: {stage_boundaries if stage_boundaries else 'Default (Early/Mid/Late)'}")
    print(f"   • Perturbations: Electrostatic, Hydrophobic, Steric, Aromatic, H-bond, Conformational")

    all_pair_effects = []

    for run in range(n_runs):
        print(f"\n   Run {run+1}/{n_runs}...")

        model = load_interface_model(segid1, segid2)
        if model is None:
            print("   ❌ Failed to load model, skipping run")
            continue

        graphs = load_graphs_with_fix()
        if graphs is None:
            continue

        selected_graphs = select_frames(graphs, total_frames, step, start_frame, max_frames)

        print(f"    Analyzing {len(selected_graphs)} frames for {segid1}-{segid2} pairs...")
        pair_effects = enhanced_find_universal_interface_pairs(
            model, selected_graphs, segid1, segid2, stage_boundaries
        )
        all_pair_effects.append(pair_effects)

    return all_pair_effects

# ============================================================
#  ANALYSIS AND OUTPUT FUNCTIONS - SUM Δ VERSION
# ============================================================

def analyze_delta_distribution(all_pair_effects):
    """Analyze the delta distribution to understand the scale"""
    print(" Analyzing delta distribution across all stages...")

    all_deltas = []
    for run in all_pair_effects:
        for stage in ['early', 'mid', 'late']:
            all_deltas.extend(list(run[stage].values()))

    all_deltas = np.array(all_deltas)

    if len(all_deltas) > 0:
        stats = {
            'mean': np.mean(all_deltas),
            'std': np.std(all_deltas),
            'min': np.min(all_deltas),
            'max': np.max(all_deltas),
            'median': np.median(all_deltas),
            'q95': np.percentile(all_deltas, 95),
            'q75': np.percentile(all_deltas, 75),
            'q50': np.percentile(all_deltas, 50)
        }

        print(f"    Delta Statistics (per-measurement):")
        print(f"      • Range: {stats['min']:.6f} - {stats['max']:.6f}")
        print(f"      • Mean per measurement: {stats['mean']:.6f} ± {stats['std']:.6f}")
        print(f"      • Median: {stats['median']:.6f}")
        print(f"      • Percentiles: 50%={stats['q50']:.6f}, 75%={stats['q75']:.6f}, 95%={stats['q95']:.6f}")

        if stats['q95'] > 0.02:
            print(f"      • Top 5% residues (Δ > {stats['q95']:.4f}) have SIGNIFICANT impact")
        if stats['max'] > 0.05:
            print(f"      • Maximum delta {stats['max']:.4f} indicates STRONG binding disruption")

        return stats
    else:
        print("   ⚠️  No delta values found for analysis")
        return None

def aggregate_temporal_stages(all_pair_effects, top_k=15):
    """Aggregate results for each temporal stage using SUM Δ"""
    print(" Aggregating interface pairs by TEMPORAL STAGES (using SUM Δ)...")
    
    # First, detect all stage names from the data
    stage_names = set()
    for run_results in all_pair_effects:
        stage_names.update(run_results.keys())
    
    stage_aggregators = {stage: defaultdict(list) for stage in stage_names}

    for run_results in all_pair_effects:
        for stage in stage_names:
            if stage in run_results:
                for pair, delta in run_results[stage].items():
                    stage_aggregators[stage][pair].append(delta)

    stage_results = {}
    for stage in stage_names:
        pair_total_delta = {}
        for pair, deltas in stage_aggregators[stage].items():
            if deltas:
                pair_total_delta[pair] = np.sum(deltas)  # SUM not MEAN!

        sorted_pairs = sorted(pair_total_delta.items(), key=lambda x: x[1], reverse=True)[:top_k*2]
        stage_results[stage] = sorted_pairs

    return stage_results

def extract_individual_residues_by_stage(stage_results):
    """Extract individual residues from pairs for each stage using SUM Δ"""
    print(" Extracting individual residues by temporal stage (using SUM Δ)...")

    stage_residues = {}

    for stage in stage_results.keys():  # Dynamic stage names
        segid1_residues = defaultdict(list)
        segid2_residues = defaultdict(list)
        segid1_counts = defaultdict(int)
        segid2_counts = defaultdict(int)

        for pair_data in stage_results[stage]:
            pair, total_delta = pair_data
            segid1_part, segid2_part = pair.split(' ↔ ')
            segid1_res = f"{segid1_part.split('-')[1]}-{segid1_part.split('-')[2]}"
            segid2_res = f"{segid2_part.split('-')[1]}-{segid2_part.split('-')[2]}"

            segid1_residues[segid1_res].append(total_delta)
            segid2_residues[segid2_res].append(total_delta)
            segid1_counts[segid1_res] += 1
            segid2_counts[segid2_res] += 1

        segid1_aggregated = []
        for residue, deltas in segid1_residues.items():
            total_delta = np.sum(deltas)
            total_count = segid1_counts[residue]
            segid1_aggregated.append((residue, total_delta, total_count))

        segid2_aggregated = []
        for residue, deltas in segid2_residues.items():
            total_delta = np.sum(deltas)
            total_count = segid2_counts[residue]
            segid2_aggregated.append((residue, total_delta, total_count))

        segid1_sorted = sorted(segid1_aggregated, key=lambda x: x[1], reverse=True)
        segid2_sorted = sorted(segid2_aggregated, key=lambda x: x[1], reverse=True)

        stage_residues[stage] = {
            'segid1': segid1_sorted,
            'segid2': segid2_sorted
        }

    return stage_residues

def calculate_stage_percentiles(stage_residues, all_pair_effects):
    """Calculate percentiles for each temporal stage using SUM Δ"""
    print(" Calculating percentile ranking by temporal stage (using SUM Δ)...")

    # First, we need to calculate the null distribution of SUM Δ values
    # We'll sum all deltas for each residue across all runs to create null distribution
    stage_null_dists = {}

    # Create a temporary structure to collect all deltas by residue
    stage_residue_deltas = {
        'early': defaultdict(list),
        'mid': defaultdict(list),
        'late': defaultdict(list)
    }

    for run_results in all_pair_effects:
        for stage in ['early', 'mid', 'late']:
            for pair, delta in run_results[stage].items():
                # Extract residue names from pair
                segid1_part, segid2_part = pair.split(' ↔ ')
                segid1_res = f"{segid1_part.split('-')[1]}-{segid1_part.split('-')[2]}"
                segid2_res = f"{segid2_part.split('-')[1]}-{segid2_part.split('-')[2]}"

                stage_residue_deltas[stage][segid1_res].append(delta)
                stage_residue_deltas[stage][segid2_res].append(delta)

    # Calculate SUM Δ for each residue to create null distribution
    for stage in ['early', 'mid', 'late']:
        sum_deltas = []
        for residue, deltas in stage_residue_deltas[stage].items():
            if deltas:
                sum_deltas.append(np.sum(deltas))

        if sum_deltas:
            stage_null_dists[stage] = {
                'all_sum_deltas': np.array(sum_deltas)
            }
        else:
            stage_null_dists[stage] = {
                'all_sum_deltas': np.array([])
            }

    stage_percentiles = {}

    for stage in ['early', 'mid', 'late']:
        null_dist = stage_null_dists[stage]
        stage_segid1_percentiles = []
        stage_segid2_percentiles = []

        for residue_data in stage_residues[stage]['segid1']:
            residue, total_delta, count = residue_data
            if len(null_dist['all_sum_deltas']) > 0:
                percentile = (np.sum(total_delta >= null_dist['all_sum_deltas']) / len(null_dist['all_sum_deltas'])) * 100
            else:
                percentile = 0.0
            stage_segid1_percentiles.append((residue, total_delta, count, percentile))

        for residue_data in stage_residues[stage]['segid2']:
            residue, total_delta, count = residue_data
            if len(null_dist['all_sum_deltas']) > 0:
                percentile = (np.sum(total_delta >= null_dist['all_sum_deltas']) / len(null_dist['all_sum_deltas'])) * 100
            else:
                percentile = 0.0
            stage_segid2_percentiles.append((residue, total_delta, count, percentile))

        stage_percentiles[stage] = {
            'segid1': stage_segid1_percentiles,
            'segid2': stage_segid2_percentiles,
            'null_stats': null_dist
        }

    return stage_percentiles

def print_temporal_staging_tables(stage_percentiles, num_residues_display=10):
    """Print clean tables with SUM Δ"""
    first_stage = list(stage_percentiles.keys())[0]
    segid1_name = "Protein1"
    segid2_name = "Protein2"

    if stage_percentiles[first_stage]['segid1']:
        sample_residue = stage_percentiles[first_stage]['segid1'][0][0]
        segid1_name = sample_residue.split('-')[0] if '-' in sample_residue else "Protein1"

    if stage_percentiles[first_stage]['segid2']:
        sample_residue = stage_percentiles[first_stage]['segid2'][0][0]
        segid2_name = sample_residue.split('-')[0] if '-' in sample_residue else "Protein2"

    for stage in ['early', 'mid', 'late']:
        print(f"\n" + "="*85)
        print(f" {stage.upper()} STAGE - RESIDUE RANKING (SUM Δ)")
        print("="*85)
        print(f"{'Protein':<8} {'Residue':<15} {'Sum Δ':<12} {'#Pairs':<8} {'Percentile':<12}")
        print("-" * 85)

        stage_data = stage_percentiles[stage]

        for i, (residue, total_delta, count, percentile) in enumerate(stage_data['segid1'][:num_residues_display]):
            print(f"{segid1_name:<8} {residue:<15} {total_delta:<12.6f} {count:<8} {percentile:>10.1f}%")

        for i, (residue, total_delta, count, percentile) in enumerate(stage_data['segid2'][:num_residues_display]):
            print(f"{segid2_name:<8} {residue:<15} {total_delta:<12.6f} {count:<8} {percentile:>10.1f}%")

        null_stats = stage_data['null_stats']
        print("-" * 85)
        if len(null_stats['all_sum_deltas']) > 0:
            print(f"Stage Statistics: {len(null_stats['all_sum_deltas'])} residues, Sum Δ range: {np.min(null_stats['all_sum_deltas']):.3f}-{np.max(null_stats['all_sum_deltas']):.3f}")

def print_excel_ready_temporal_output(stage_percentiles, segid1, segid2):
    """Clean Excel-ready output with SUM Δ"""
    print("\n" + "="*70)
    print(f" EXCEL-READY OUTPUT: {segid1} ↔ {segid2} (SUM Δ)")
    print("="*70)

    for stage in ['early', 'mid', 'late']:
        print(f"\n{stage.upper()} STAGE:")
        print("Protein\tResidue\tSumDelta\tCount\tPercentile\tStage\tPerturbationType")

        stage_data = stage_percentiles[stage]

        for residue, total_delta, count, percentile in stage_data['segid1']:
            resname = residue.split('-')[0] if '-' in residue else residue
            props = RESIDUE_PROPERTIES.get(resname, {})
            pert_type = get_perturbation_type(props)

            print(f"{segid1}\t{residue}\t{total_delta:.6f}\t{count}\t{percentile:.2f}\t{stage}\t{pert_type}")

        for residue, total_delta, count, percentile in stage_data['segid2']:
            resname = residue.split('-')[0] if '-' in residue else residue
            props = RESIDUE_PROPERTIES.get(resname, {})
            pert_type = get_perturbation_type(props)

            print(f"{segid2}\t{residue}\t{total_delta:.6f}\t{count}\t{percentile:.2f}\t{stage}\t{pert_type}")

def analyze_perturbation_sensitivity(all_pair_effects, stage_residues):
    """Analyze which perturbation types are most sensitive for each residue"""
    print("\n" + "="*85)
    print(" PERTURBATION TYPE SENSITIVITY ANALYSIS")
    print("="*85)

    print("\n BIOLOGICAL INTERPRETATION OF SUM Δ VALUES:")
    print(" ──────────────────────────────────────────────────────")
    print(" HIGH Sum Δ indicates:")
    print("   • Residue is frequently at the interface (high Count)")
    print("   • Residue has strong disruptive effect when perturbed (high Δ per interaction)")
    print("   • Both frequency AND strength contribute to Sum Δ")
    print("\n IMPORTANCE METRICS:")
    print("   Sum Δ = (Mean Δ per interaction) × (Number of interactions)")
    print("   This weights residues by BOTH strength AND frequency of interaction")

    print("\n PERTURBATION TYPE GUIDELINES:")
    print(" ──────────────────────────────────────────────────────")
    print(" 1. ELECTROSTATIC")
    print("    • Sensitive for: Charged residues (ARG, LYS, ASP, GLU, HIS)")
    print("    • High Sum Δ = Frequent AND strong electrostatic interactions")

    print("\n 2. HYDROPHOBIC")
    print("    • Sensitive for: Hydrophobic residues (ALA, VAL, LEU, ILE, PHE, MET)")
    print("    • High Sum Δ = Large hydrophobic interface patch")

    print("\n 3. STERIC")
    print("    • Sensitive for: Bulky residues (TRP, TYR, ARG, LYS)")
    print("    • High Sum Δ = Tight packing with multiple partners")

    print("\n 4. AROMATIC")
    print("    • Sensitive for: Aromatic residues (PHE, TYR, TRP, HIS)")
    print("    • High Sum Δ = Extensive π-π or CH-π networks")

    print("\n 5. H-BOND")
    print("    • Sensitive for: Polar residues (SER, THR, ASN, GLN, backbone)")
    print("    • High Sum Δ = Extensive hydrogen bonding network")

    print("\n 6. CONFORMATIONAL")
    print("    • Sensitive for: Flexible residues, sidechains at interface")
    print("    • High Sum Δ = Conformationally sensitive binding")

    return ["ELECTROSTATIC", "HYDROPHOBIC", "STERIC", "AROMATIC", "H-BOND", "CONFORMATIONAL"]

def calculate_total_interface_strength(stage_residues):
    """Calculate total interface strength (Sum Δ) like paper Table 1"""
    print("\n" + "="*85)
    print(" TOTAL INTERFACE STRENGTH ANALYSIS (Paper Table 1 Methodology)")
    print("="*85)

    total_strengths = {}

    for stage in ['early', 'mid', 'late']:
        segid1_total = sum(total_delta for _, total_delta, _ in stage_residues[stage]['segid1'])
        segid2_total = sum(total_delta for _, total_delta, _ in stage_residues[stage]['segid2'])
        total_interface = segid1_total + segid2_total

        segid1_count = len(stage_residues[stage]['segid1'])
        segid2_count = len(stage_residues[stage]['segid2'])

        total_strengths[stage] = {
            'segid1_total': segid1_total,
            'segid2_total': segid2_total,
            'total_interface': total_interface,
            'segid1_count': segid1_count,
            'segid2_count': segid2_count
        }

    print(f"\n{'Stage':<10} {'#Residues':<12} {'Segid1 Sum Δ':<15} {'Segid2 Sum Δ':<15} {'Total Sum Δ':<15}")
    print("-" * 70)

    for stage in ['early', 'mid', 'late']:
        stats = total_strengths[stage]
        print(f"{stage:<10} {stats['segid1_count']}+{stats['segid2_count']:<11} "
              f"{stats['segid1_total']:<15.3f} {stats['segid2_total']:<15.3f} {stats['total_interface']:<15.3f}")

    # Calculate change over time
    early_total = total_strengths['early']['total_interface']
    late_total = total_strengths['late']['total_interface']
    change_percent = ((late_total - early_total) / early_total * 100) if early_total > 0 else 0

    print("\n TEMPORAL DYNAMICS:")
    print(f"   • Early → Late change: {change_percent:+.1f}%")
    if change_percent > 10:
        print(f"   • Interface STRENGTHENS over time")
    elif change_percent < -10:
        print(f"   • Interface WEAKENS over time")
    else:
        print(f"   • Interface remains STABLE over time")

    return total_strengths

# ============================================================
#  CORE ANALYSIS PIPELINE - REUSABLE FUNCTION
# ============================================================

def analyze_interface_pair(graph_path, model_path, segid1, segid2, 
                          start_frame=0, step=1, n_runs=1, total_frames=100, 
                          max_frames=None, num_residues_display=10,
                          stage_boundaries=None):
    """
    Core analysis function for a single interface pair.
    This can be easily called in a loop for multiple analyses.
    """
    print(f"\n" + "="*85)
    print(f" ANALYZING INTERFACE: {segid1} ↔ {segid2}")
    print("="*85)
    
    # Set file paths for this analysis
    set_file_paths(graph_path, model_path)
    
    ANALYSIS_PARAMS = {
        'n_runs': n_runs,
        'total_frames': total_frames,
        'step': step,
        'start_frame': start_frame,
        'max_frames': max_frames,
        'num_residues_display': num_residues_display,
        'interface': f"{segid1}-{segid2}",
        'stage_boundaries': stage_boundaries
    }

    # Run ENHANCED analysis for this interface
    pair_effects = run_universal_interface_analysis(
        segid1=segid1,
        segid2=segid2,
        n_runs=ANALYSIS_PARAMS['n_runs'],
        total_frames=ANALYSIS_PARAMS['total_frames'],
        step=ANALYSIS_PARAMS['step'],
        start_frame=ANALYSIS_PARAMS['start_frame'],
        max_frames=ANALYSIS_PARAMS['max_frames'],
        stage_boundaries=ANALYSIS_PARAMS['stage_boundaries']
    )

    if not pair_effects:
        print(f"❌ No results obtained for {segid1}-{segid2}")
        return None
    
    # Analysis pipeline for this interface
    delta_stats = analyze_delta_distribution(pair_effects)
    
    # Need to update aggregate_temporal_stages to handle dynamic stage names
    stage_results = aggregate_temporal_stages(pair_effects, top_k=ANALYSIS_PARAMS['num_residues_display']*3)
    stage_residues = extract_individual_residues_by_stage(stage_results)
    stage_percentiles = calculate_stage_percentiles(stage_residues, pair_effects)

    # Print individual interface results
    print_temporal_staging_tables(stage_percentiles, ANALYSIS_PARAMS['num_residues_display'])

    # Calculate total interface strength for this pair
    total_strengths = calculate_total_interface_strength(stage_residues)

    # Print Excel-ready output for this interface
    print_excel_ready_temporal_output(stage_percentiles, segid1, segid2)

    return {
        'pair_effects': pair_effects,
        'stage_percentiles': stage_percentiles,
        'stage_residues': stage_residues,
        'total_strengths': total_strengths,
        'delta_stats': delta_stats
    }

def detect_interface_pairs_from_graph(graph_path):
    """Auto-detect which interfaces are available in the graph"""
    print(f"📂 Loading dataset to detect available interfaces...")
    global GRAPH_FILE_PATH
    GRAPH_FILE_PATH = graph_path
    
    graphs = load_graphs_with_fix()
    if graphs is None:
        return []
    
    # Detect ALL available segids
    available_segids = detect_all_segids(graphs)
    
    if len(available_segids) < 2:
        print("❌ Need at least 2 segids for interface analysis!")
        return []
    
    # Return all possible interface pairs
    interface_pairs = []
    for i in range(len(available_segids)):
        for j in range(i+1, len(available_segids)):
            interface_pairs.append((available_segids[i], available_segids[j]))
    
    return interface_pairs

# ============================================================
#  MAIN EXECUTION - RESEARCHER-FRIENDLY VERSION
# ============================================================

if __name__ == "__main__":
    print(" UNIVERSAL INTERFACE ANALYZER - SUM Δ VERSION (BIOLOGICALLY CORRECT)")
    print(" FEATURES: SUM Δ weighting + 6 perturbation types + Temporal staging")
    print("=" * 85)
    
    # ============================================================
    #  CONFIGURATION SECTION - SET YOUR PARAMETERS HERE
    # ============================================================
    
    # Option 1: Single analysis (one graph, one model, one interface)
    SINGLE_ANALYSIS = True  # Set to False for batch analysis
    
    if SINGLE_ANALYSIS:
        # === SINGLE ANALYSIS CONFIGURATION ===
        GRAPH_PATH = "/path/to/your/graph_file.pt"  # Replace with your graph file
        MODEL_PATH = "/path/to/your/model_file.pth"  # Replace with your model file
        
        # Define which interface to analyze
        SEGID1 = "PROA"  # Main protein
        SEGID2 = "PROD"  # Partner protein
        
        # Or auto-detect interfaces:
        # interface_pairs = detect_interface_pairs_from_graph(GRAPH_PATH)
        # SEGID1, SEGID2 = interface_pairs[0]  # Analyze first detected interface
        
        # Analysis parameters
        START_FRAME = 0
        STEP = 1
        N_RUNS = 1
        TOTAL_FRAMES = 100
        MAX_FRAMES = None  # Set to number if you want to limit
        NUM_RESIDUES_DISPLAY = 10
        
        # Run the analysis
        results = analyze_interface_pair(
            graph_path=GRAPH_PATH,
            model_path=MODEL_PATH,
            segid1=SEGID1,
            segid2=SEGID2,
            start_frame=START_FRAME,
            step=STEP,
            n_runs=N_RUNS,
            total_frames=TOTAL_FRAMES,
            max_frames=MAX_FRAMES,
            num_residues_display=NUM_RESIDUES_DISPLAY
        )
        
    else:
        # === BATCH ANALYSIS CONFIGURATION ===
        # Analyze multiple interfaces from the same graph/model
        
        GRAPH_PATH = "/path/to/your/graph_file.pt"
        MODEL_PATH = "/path/to/your/model_file.pth"
        
        # Detect all possible interfaces
        interface_pairs = detect_interface_pairs_from_graph(GRAPH_PATH)
        
        print(f"\n  Found {len(interface_pairs)} possible interfaces:")
        for i, (s1, s2) in enumerate(interface_pairs):
            print(f"   {i+1:2d}. {s1} ↔ {s2}")
        
        # Common parameters for all analyses
        START_FRAME = 0
        STEP = 5
        N_RUNS = 1
        TOTAL_FRAMES = 50
        MAX_FRAMES = None
        NUM_RESIDUES_DISPLAY = 10
        
        # Option A: Analyze all interfaces
        analyze_all = True  # Set to False to analyze specific ones
        
        if analyze_all:
            # Analyze ALL interfaces
            interfaces_to_analyze = interface_pairs
        else:
            # Analyze specific interfaces (modify this list)
            interfaces_to_analyze = [
                ("PROA", "PROD"),  # Interface 1
                ("PROA", "PROC"),  # Interface 2
                # Add more as needed
            ]
        
        # Store all results
        all_results = {}
        
        # Loop through and analyze each interface
        for segid1, segid2 in interfaces_to_analyze:
            print(f"\n" + "="*85)
            print(f" STARTING ANALYSIS {len(all_results)+1}/{len(interfaces_to_analyze)}: {segid1} ↔ {segid2}")
            print("="*85)
            
            result = analyze_interface_pair(
                graph_path=GRAPH_PATH,
                model_path=MODEL_PATH,
                segid1=segid1,
                segid2=segid2,
                start_frame=START_FRAME,
                step=STEP,
                n_runs=N_RUNS,
                total_frames=TOTAL_FRAMES,
                max_frames=MAX_FRAMES,
                num_residues_display=NUM_RESIDUES_DISPLAY
            )
            
            if result:
                all_results[f"{segid1}-{segid2}"] = result
        
        # Print summary of ALL interfaces analyzed
        print(f"\n" + "="*85)
        print(f" SUMMARY OF ALL INTERFACES ANALYZED")
        print("="*85)

        if all_results:
            print(f"\n{'Interface':<20} {'Early Sum Δ':<15} {'Mid Sum Δ':<15} {'Late Sum Δ':<15} {'Trend':<10}")
            print("-" * 80)

            for interface_name, results in all_results.items():
                if 'total_strengths' in results:
                    strengths = results['total_strengths']
                    early_total = strengths['early']['total_interface'] if 'early' in strengths else 0
                    mid_total = strengths['mid']['total_interface'] if 'mid' in strengths else 0
                    late_total = strengths['late']['total_interface'] if 'late' in strengths else 0

                    # Determine trend
                    if early_total > 0:
                        trend_percent = ((late_total - early_total) / early_total * 100)
                        if trend_percent > 15:
                            trend = "STRONGER"
                        elif trend_percent < -15:
                            trend = "WEAKER"
                        else:
                            trend = "STABLE"
                    else:
                        trend = "N/A"

                    print(f"{interface_name:<20} {early_total:<15.3f} {mid_total:<15.3f} {late_total:<15.3f} {trend:<10}")
            
            # Combined Excel output
            print(f"\n" + "="*70)
            print(f" COMBINED EXCEL-READY OUTPUT FOR ALL INTERFACES")
            print("="*70)
            
            for interface_name, results in all_results.items():
                segid1, segid2 = interface_name.split('-')
                
                if 'stage_percentiles' in results:
                    stage_percentiles = results['stage_percentiles']
                    
                    for stage in ['early', 'mid', 'late']:
                        if stage in stage_percentiles:
                            stage_data = stage_percentiles[stage]
                            
                            for residue, total_delta, count, percentile in stage_data['segid1']:
                                resname = residue.split('-')[0] if '-' in residue else residue
                                props = RESIDUE_PROPERTIES.get(resname, {})
                                pert_type = get_perturbation_type(props)
                                print(f"{segid1}\t{residue}\t{total_delta:.6f}\t{count}\t{percentile:.2f}\t{stage}\t{pert_type}\t{interface_name}")
                            
                            for residue, total_delta, count, percentile in stage_data['segid2']:
                                resname = residue.split('-')[0] if '-' in residue else residue
                                props = RESIDUE_PROPERTIES.get(resname, {})
                                pert_type = get_perturbation_type(props)
                                print(f"{segid2}\t{residue}\t{total_delta:.6f}\t{count}\t{percentile:.2f}\t{stage}\t{pert_type}\t{interface_name}")
    
    # ============================================================
    #  ADVANCED: LOOP THROUGH MULTIPLE GRAPH/MODEL FILES
    # ============================================================
    # Uncomment and modify this section if you have multiple datasets
    
    # BATCH_MULTIPLE_FILES = False
    # if BATCH_MULTIPLE_FILES:
    #     # Define directories containing your files
    #     GRAPH_DIR = "/path/to/graph/files/"
    #     MODEL_DIR = "/path/to/model/files/"
    #     
    #     # Get all graph files (adjust pattern as needed)
    #     graph_files = sorted(glob.glob(os.path.join(GRAPH_DIR, "*.pt")))
    #     model_files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.pth")))
    #     
    #     # Match graph and model files (assuming same names/directories)
    #     for graph_file in graph_files:
    #         # Find corresponding model file
    #         base_name = os.path.splitext(os.path.basename(graph_file))[0]
    #         model_file = os.path.join(MODEL_DIR, f"{base_name}.pth")
    #         
    #         if os.path.exists(model_file):
    #             print(f"\n" + "="*85)
    #             print(f" ANALYZING: {base_name}")
    #             print("="*85)
    #             
    #             # Auto-detect interfaces for this system
    #             interface_pairs = detect_interface_pairs_from_graph(graph_file)
    #             
    #             if interface_pairs:
    #                 # Analyze first interface (or loop through all)
    #                 segid1, segid2 = interface_pairs[0]
    #                 
    #                 results = analyze_interface_pair(
    #                     graph_path=graph_file,
    #                     model_path=model_file,
    #                     segid1=segid1,
    #                     segid2=segid2,
    #                     start_frame=0,
    #                     step=5,
    #                     n_runs=1,
    #                     total_frames=50,
    #                     max_frames=None,
    #                     num_residues_display=10
    #                 )
    #                 
    #                 # Save results to file
    #                 output_file = f"{base_name}_results.txt"
    #                 # ... save logic here ...

    print(f"\n" + "="*85)
    print(f" ANALYSIS COMPLETE!")
    print("="*85)
