
'''
Perturbation Scanning (PS) - v2
By Fodil Azzaz, PhD, All rights reserved
Copyright (c) 2025 Fodil Azzaz

ACADEMIC LICENSE
================
PERMITTED:
- Academic research and teaching
- Non-commercial scientific use
- Integration into research pipelines
- Publication using this software

COMMERCIAL USE:
- Commercial use requires authorization
- Contact: azzaz.fodil@gmail.com

CITATION:
If you use this software in your research, please cite:
coming soon
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy.spatial import cKDTree
from torch_geometric.nn import GATConv
from scipy.stats import norm, pearsonr, spearmanr
import itertools
import umap
from sklearn.cluster import KMeans

print(" UNIVERSAL PROTEIN INTERFACE ANALYZER - IMPROVED")
print("=" * 85)

# ============================================================
#  GLOBAL FILE PATHS
# ============================================================
GRAPH_FILE_PATH = None
MODEL_FILE_PATH = None

def set_file_paths(graph_path, model_path):
    global GRAPH_FILE_PATH, MODEL_FILE_PATH
    GRAPH_FILE_PATH = graph_path
    MODEL_FILE_PATH = model_path
    print(f"File paths set:")
    print(f"   Graph data: {GRAPH_FILE_PATH}")
    print(f"   Model: {MODEL_FILE_PATH}")

# ============================================================
#  RESIDUE PROPERTY DATABASE
# ============================================================
def create_enhanced_residue_features():
    residue_properties = {
        'ALA': {'charge': 0,    'hydrophobicity': 1.8,  'aromatic': 0, 'polar': 0, 'size': 1.0, 'h_bond_donor': 0, 'h_bond_acceptor': 0},
        'ARG': {'charge': 1,    'hydrophobicity': -4.5, 'aromatic': 0, 'polar': 1, 'size': 3.0, 'h_bond_donor': 4, 'h_bond_acceptor': 6},
        'ASN': {'charge': 0,    'hydrophobicity': -3.5, 'aromatic': 0, 'polar': 1, 'size': 1.5, 'h_bond_donor': 2, 'h_bond_acceptor': 4},
        'ASP': {'charge': -1,   'hydrophobicity': -3.5, 'aromatic': 0, 'polar': 1, 'size': 1.5, 'h_bond_donor': 1, 'h_bond_acceptor': 4},
        'CYS': {'charge': 0,    'hydrophobicity': 2.5,  'aromatic': 0, 'polar': 1, 'size': 1.5, 'h_bond_donor': 1, 'h_bond_acceptor': 1},
        'GLN': {'charge': 0,    'hydrophobicity': -3.5, 'aromatic': 0, 'polar': 1, 'size': 2.0, 'h_bond_donor': 2, 'h_bond_acceptor': 4},
        'GLU': {'charge': -1,   'hydrophobicity': -3.5, 'aromatic': 0, 'polar': 1, 'size': 2.0, 'h_bond_donor': 1, 'h_bond_acceptor': 4},
        'GLY': {'charge': 0,    'hydrophobicity': -0.4, 'aromatic': 0, 'polar': 0, 'size': 0.5, 'h_bond_donor': 2, 'h_bond_acceptor': 2},
        'HIS': {'charge': 0.5,  'hydrophobicity': -3.2, 'aromatic': 1, 'polar': 1, 'size': 2.5, 'h_bond_donor': 2, 'h_bond_acceptor': 4},
        'HSD': {'charge': 0.5,  'hydrophobicity': -3.2, 'aromatic': 1, 'polar': 1, 'size': 2.5, 'h_bond_donor': 2, 'h_bond_acceptor': 4},
        'HSE': {'charge': 0.5,  'hydrophobicity': -3.2, 'aromatic': 1, 'polar': 1, 'size': 2.5, 'h_bond_donor': 2, 'h_bond_acceptor': 4},
        'HSP': {'charge': 1,    'hydrophobicity': -3.2, 'aromatic': 1, 'polar': 1, 'size': 2.5, 'h_bond_donor': 3, 'h_bond_acceptor': 4},
        'ILE': {'charge': 0,    'hydrophobicity': 4.5,  'aromatic': 0, 'polar': 0, 'size': 2.5, 'h_bond_donor': 1, 'h_bond_acceptor': 1},
        'LEU': {'charge': 0,    'hydrophobicity': 3.8,  'aromatic': 0, 'polar': 0, 'size': 2.5, 'h_bond_donor': 1, 'h_bond_acceptor': 1},
        'LYS': {'charge': 1,    'hydrophobicity': -3.9, 'aromatic': 0, 'polar': 1, 'size': 3.0, 'h_bond_donor': 3, 'h_bond_acceptor': 2},
        'MET': {'charge': 0,    'hydrophobicity': 1.9,  'aromatic': 0, 'polar': 0, 'size': 2.5, 'h_bond_donor': 1, 'h_bond_acceptor': 2},
        'PHE': {'charge': 0,    'hydrophobicity': 2.8,  'aromatic': 1, 'polar': 0, 'size': 3.0, 'h_bond_donor': 1, 'h_bond_acceptor': 1},
        'PRO': {'charge': 0,    'hydrophobicity': -1.6, 'aromatic': 0, 'polar': 0, 'size': 1.5, 'h_bond_donor': 1, 'h_bond_acceptor': 2},
        'SER': {'charge': 0,    'hydrophobicity': -0.8, 'aromatic': 0, 'polar': 1, 'size': 1.0, 'h_bond_donor': 1, 'h_bond_acceptor': 2},
        'THR': {'charge': 0,    'hydrophobicity': -0.7, 'aromatic': 0, 'polar': 1, 'size': 1.5, 'h_bond_donor': 1, 'h_bond_acceptor': 2},
        'TRP': {'charge': 0,    'hydrophobicity': -0.9, 'aromatic': 1, 'polar': 0, 'size': 3.5, 'h_bond_donor': 1, 'h_bond_acceptor': 2},
        'TYR': {'charge': 0,    'hydrophobicity': -1.3, 'aromatic': 1, 'polar': 1, 'size': 3.0, 'h_bond_donor': 1, 'h_bond_acceptor': 2},
        'VAL': {'charge': 0,    'hydrophobicity': 4.2,  'aromatic': 0, 'polar': 0, 'size': 2.0, 'h_bond_donor': 1, 'h_bond_acceptor': 1},
    }
    for resname, props in residue_properties.items():
        props['hydrophobicity'] = max(-1.0, min(1.0, props['hydrophobicity'] / 4.5))
        props['size'] = props['size'] / 3.5
        props['h_bond_donor'] = props['h_bond_donor'] / 4.0
        props['h_bond_acceptor'] = props['h_bond_acceptor'] / 6.0
    return residue_properties

RESIDUE_PROPERTIES = create_enhanced_residue_features()

def get_perturbation_type(props):
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
    all_segids = set()
    for graph in graphs:
        if hasattr(graph, 'segids'):
            segids_list = graph.segids if isinstance(graph.segids, (list, tuple)) else getattr(graph, 'segids', [])
            for segid in segids_list:
                if segid and str(segid).strip() and str(segid) != 'None':
                    all_segids.add(str(segid))
    available_segids = sorted(list(all_segids))
    print(f"Found {len(available_segids)} unique segids: {available_segids}")
    sample_graph = graphs[0]
    if hasattr(sample_graph, 'segids'):
        segid_counts = defaultdict(int)
        for segid in sample_graph.segids:
            segid_counts[segid] += 1
        print("\n Segid distribution in first graph:")
        for segid, count in sorted(segid_counts.items()):
            print(f"   {segid}: {count} atoms")
    return available_segids

def load_graphs_with_fix():
    global GRAPH_FILE_PATH
    if GRAPH_FILE_PATH is None:
        print("No graph file path set! Call set_file_paths() first.")
        return None
    print(f"Loading graphs from: {GRAPH_FILE_PATH}")
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
    max_available = len(graphs)
    if max_frames is not None:
        max_available = min(max_available, max_frames)
    if start_frame >= max_available:
        print(f"   Start frame {start_frame} exceeds available, resetting to 0")
        start_frame = 0
    if start_frame + total_frames > (max_frames if max_frames else max_available):
        total_frames = (max_frames if max_frames else max_available) - start_frame
        print(f"   Adjusted total_frames to {total_frames}")
    selected_indices = list(range(start_frame, start_frame + total_frames, step))
    selected_graphs = [graphs[i] for i in selected_indices if i < (max_frames if max_frames else max_available)]
    print(f"   Frame selection: {len(selected_graphs)} frames (start={start_frame}, step={step})")
    return selected_graphs

# ============================================================
#  PERTURBATION FUNCTIONS
# ============================================================
def apply_electrostatic_perturbation(features, target_indices, target_resnames, perturbation_strength=0.8):
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
    mutated_features = features.clone()
    for idx, resname in zip(target_indices, target_resnames):
        props = RESIDUE_PROPERTIES.get(resname, {'hydrophobicity': 0})
        current_hydro = features[idx][3].item()
        if abs(props['hydrophobicity']) > 0.3:
            new_hydro = -current_hydro * perturbation_strength
            mutated_features[idx][3] = new_hydro
            if props.get('polar', 0) == 1:
                mutated_features[idx][5] *= (1 - perturbation_strength)
        else:
            mutated_features[idx][3] = 0.5 * perturbation_strength
    return mutated_features

def apply_steric_perturbation(features, target_indices, target_resnames, perturbation_strength=0.8):
    mutated_features = features.clone()
    for idx, resname in zip(target_indices, target_resnames):
        mutated_features[idx] *= (1 - perturbation_strength * 0.3)
        mutated_features[idx][6] *= (1 - perturbation_strength)
        noise = torch.randn_like(mutated_features[idx]) * 0.1 * perturbation_strength
        mutated_features[idx] += noise
    return mutated_features

def apply_aromatic_perturbation(features, target_indices, target_resnames, perturbation_strength=0.8):
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
    mutated_features = features.clone()
    polar_residues = ['SER', 'THR', 'ASN', 'GLN', 'ASP', 'GLU', 'HIS', 'HSD', 'HSE', 'HSP', 'ARG', 'LYS', 'TYR']
    for idx, resname in zip(target_indices, target_resnames):
        if resname in polar_residues:
            mutated_features[idx][4] *= (1 - perturbation_strength)
            scale_factor = 1 - (perturbation_strength * 0.5)
            mutated_features[idx] *= scale_factor
            mutated_features[idx][0] *= (1 - perturbation_strength)
    return mutated_features

# FIX #2: Seeded conformational perturbation
def apply_conformational_perturbation(graph, target_indices, perturbation_strength=1.0, seed=None):
    """
    Introduce conformational strain by displacing sidechain atoms.

    FIX #2: seed parameter ensures reproducibility across n_runs.
    - If seed is provided: use it directly for a deterministic displacement.
    - If seed is None: average over 5 fixed seeds for a stable estimate
      rather than drawing a single random sample (which was the old behavior).

    Do NOT call without a seed when n_runs > 1, as this produces
    irreproducible variance in the conformational perturbation channel.
    """
    modified = graph.clone()

    if seed is not None:
        rng = np.random.RandomState(seed)
        for idx in target_indices:
            if hasattr(graph, 'resnames') and graph.resnames[idx] in RESIDUE_PROPERTIES:
                displacement = torch.tensor(
                    rng.randn(3) * 0.3 * perturbation_strength,
                    dtype=torch.float32
                )
                modified.pos[idx] += displacement
    else:
        # Average over 5 fixed seeds — stable estimate, fully reproducible
        n_seeds = 5
        pos_accumulator = graph.pos.clone().float()
        for s in range(n_seeds):
            rng = np.random.RandomState(s)
            temp_pos = graph.pos.clone().float()
            for idx in target_indices:
                if hasattr(graph, 'resnames') and graph.resnames[idx] in RESIDUE_PROPERTIES:
                    displacement = torch.tensor(
                        rng.randn(3) * 0.3 * perturbation_strength,
                        dtype=torch.float32
                    )
                    temp_pos[idx] += displacement
            pos_accumulator = pos_accumulator + temp_pos
        modified.pos = pos_accumulator / (n_seeds + 1)

    return modified

def apply_residue_specific_masking(features, target_indices, target_resnames):
    mutated_features = features.clone()
    for idx, resname in zip(target_indices, target_resnames):
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
#  COMPREHENSIVE PERTURBATION SCAN — returns (delta, pert_type)
# ============================================================
def comprehensive_perturbation_scan_with_argmax(model, graph, res1, res2,
                                                 original_pred, segid1, segid2,
                                                 run_seed=None):
    """
    Apply all 6 perturbation types and return (max_delta, dominant_perturbation_type).

    FIX #2: run_seed is passed to apply_conformational_perturbation so that
    conformational displacement is reproducible across n_runs.
    """
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
        return 0.0, None

    def apply_elec(g):
        g2 = g.clone()
        g2.x = apply_electrostatic_perturbation(g2.x, target_indices, target_resnames, 0.8)
        return g2

    def apply_hydro(g):
        g2 = g.clone()
        g2.x = apply_hydrophobic_perturbation(g2.x, target_indices, target_resnames, 0.8)
        return g2

    def apply_ster(g):
        g2 = g.clone()
        g2.x = apply_steric_perturbation(g2.x, target_indices, target_resnames, 0.8)
        return g2

    def apply_arom(g):
        g2 = g.clone()
        g2.x = apply_aromatic_perturbation(g2.x, target_indices, target_resnames, 0.8)
        return g2

    def apply_hbond(g):
        g2 = g.clone()
        g2.x = apply_hydrogen_bond_perturbation(g2.x, target_indices, target_resnames, 0.8)
        return g2

    def apply_conf(g):
        # FIX #2: pass run_seed for reproducibility
        return apply_conformational_perturbation(g, target_indices, 1.0, seed=run_seed)

    perturbations = {
        'ELECTROSTATIC': apply_elec,
        'HYDROPHOBIC':   apply_hydro,
        'STERIC':        apply_ster,
        'AROMATIC':      apply_arom,
        'H-BOND':        apply_hbond,
        'CONFORMATIONAL': apply_conf,
    }

    best_delta = 0.0
    best_type = None

    for pname, pfunc in perturbations.items():
        modified = pfunc(graph)
        with torch.no_grad():
            new_pred = model(modified).item()
        delta = abs(new_pred - original_pred)
        if delta > best_delta:
            best_delta = delta
            best_type = pname

    return best_delta, best_type

# ============================================================
#  ORIGINAL DELTA METHODS (unchanged)
# ============================================================
def calculate_universal_delta_masking(model, graph, res1, res2, original_pred, segid1, segid2):
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
    modified = graph.clone()
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
    if target_indices:
        modified.x = apply_residue_specific_masking(modified.x, target_indices, target_resnames)
        with torch.no_grad():
            modified_pred = model(modified).item()
        return abs(modified_pred - original_pred)
    return 0.0

def calculate_universal_distance_perturbation(model, graph, res1, res2, original_pred, segid1, segid2):
    modified = graph.clone()
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
            elif current_res == res2:
                res2_indices.append(i)
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

# ============================================================
#  MODEL LOADING
# ============================================================
def load_interface_model(segid1, segid2):
    global MODEL_FILE_PATH
    if MODEL_FILE_PATH is None:
        print("No model file path set!")
        return None
    print(f"Loading model for {segid1} <-> {segid2}...")
    try:
        checkpoint = torch.load(MODEL_FILE_PATH, map_location='cpu', weights_only=True)
    except:
        checkpoint = torch.load(MODEL_FILE_PATH, map_location='cpu', weights_only=False)

    if 'model_state_dict' not in checkpoint:
        print("No model_state_dict found in checkpoint")
        return None

    state_dict = checkpoint['model_state_dict']
    num_interface_features = state_dict['interface_predictor.weight'].shape[1] if 'interface_predictor.weight' in state_dict else 96

    graphs = load_graphs_with_fix()
    if graphs is None:
        return None
    node_dim = graphs[0].x.shape[1]

    class UniversalInterfacePredictor(nn.Module):
        def __init__(self, node_dim, num_interface_features=96, segid1="PROA", segid2="PROD", gat_heads=4):
            super().__init__()
            self.conv1 = GATConv(node_dim, 128, heads=gat_heads, concat=True)
            self.conv2 = GATConv(128 * gat_heads, 64, heads=1, concat=False)
            self.conv3 = GATConv(64, 32, heads=1, concat=False)
            self.batch_norm1 = nn.BatchNorm1d(128 * gat_heads)
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
            interface_features = []
            segid1_indices = torch.tensor([i for i, s in enumerate(segids) if s == self.segid1])
            segid2_indices = torch.tensor([i for i, s in enumerate(segids) if s == self.segid2])
            if segid1_indices.numel() > 0 and segid2_indices.numel() > 0:
                interface_mask = self.detect_interface(data.pos.cpu().numpy(), segid1_indices, segid2_indices)
                interface_feature = node_features[interface_mask].mean(dim=0) if interface_mask.sum() > 0 else node_features.mean(dim=0)
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
            tree = cKDTree(segid1_positions)
            distances, indices = tree.query(segid2_positions, k=1)
            for i, segid2_idx in enumerate(segid2_indices):
                if distances[i] < 8.0:
                    interface_mask[segid2_idx] = True
                    if distances[i] < 6.0:
                        interface_mask[segid1_indices[indices[i]]] = True
            return interface_mask

    model = UniversalInterfacePredictor(node_dim, num_interface_features, segid1, segid2)
    try:
        model.load_state_dict(state_dict)
        print("   ALL parameters loaded successfully!")
    except RuntimeError:
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"   Partial load: {len(pretrained_dict)}/{len(state_dict)} parameters")
    model.eval()
    return model

# ============================================================
#  INTERFACE CONTACT DETECTION
# ============================================================
def find_universal_contacts(graph, segid1, segid2, cutoff=8.0):
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

    residue_centers = {}
    for residue_key, atom_indices in residue_atoms.items():
        if atom_indices:
            positions = [graph.pos[idx].cpu().numpy() for idx in atom_indices]
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
        distances, indices = tree.query(segid1_centers[i], k=min(10, len(segid2_keys)))
        for j, dist in enumerate(distances):
            if dist < cutoff:
                pairs.append((segid1_key, segid2_keys[indices[j]]))
    return pairs

# ============================================================
#  ENHANCED PERTURBATION SCAN WITH TEMPORAL STAGING
# ============================================================
def enhanced_find_universal_interface_pairs(model, selected_graphs, segid1, segid2, run_seed=None):
    """
    FIX #2: run_seed is threaded through to apply_conformational_perturbation
    via comprehensive_perturbation_scan_with_argmax.
    """
    print(f"   Identifying critical {segid1}-{segid2} pairs with 6 perturbation types...")
    model.eval()

    pair_deltas_accumulated = {
        'early': defaultdict(list),
        'mid':   defaultdict(list),
        'late':  defaultdict(list)
    }

    total_frames = len(selected_graphs)
    early_cutoff = total_frames // 3
    mid_cutoff = 2 * total_frames // 3
    print(f"   Temporal stages: Early (0-{early_cutoff}), Mid ({early_cutoff}-{mid_cutoff}), Late ({mid_cutoff}-{total_frames})")

    for frame_idx, graph in enumerate(selected_graphs):
        stage = 'early' if frame_idx < early_cutoff else ('mid' if frame_idx < mid_cutoff else 'late')

        with torch.no_grad():
            original_pred = model(graph).item()

        interface_pairs = find_universal_contacts(graph, segid1, segid2, cutoff=8.0)

        for res1, res2 in interface_pairs:
            pair_key = f"{res1} <-> {res2}"

            # FIX #2: pass run_seed through
            delta, pert_type = comprehensive_perturbation_scan_with_argmax(
                model, graph, res1, res2, original_pred, segid1, segid2, run_seed=run_seed
            )
            delta_mask = calculate_universal_delta_masking(model, graph, res1, res2, original_pred, segid1, segid2)
            delta_dist = calculate_universal_distance_perturbation(model, graph, res1, res2, original_pred, segid1, segid2)
            delta_intelligent = calculate_universal_intelligent_masking(model, graph, res1, res2, original_pred, segid1, segid2)

            combined_delta = max(delta, delta_mask, delta_dist, delta_intelligent)
            pair_deltas_accumulated[stage][pair_key].append((combined_delta, pert_type))

        if (frame_idx + 1) % 10 == 0:
            print(f"      Frame {frame_idx} ({stage}): tested {len(interface_pairs)} pairs")

    # Aggregate per pair: mean delta, most frequent perturbation type
    stage_results = {}
    for stage in ['early', 'mid', 'late']:
        pair_agg = {}
        for pair, list_of_delta_pert in pair_deltas_accumulated[stage].items():
            deltas = [d for d, _ in list_of_delta_pert]
            pert_types = [p for _, p in list_of_delta_pert if p is not None]
            mean_delta = np.mean(deltas) if deltas else 0.0
            most_common = max(set(pert_types), key=pert_types.count) if pert_types else "UNKNOWN"
            pair_agg[pair] = (mean_delta, most_common)
        stage_results[stage] = pair_agg

    print(f"   Analysis complete: Early={len(stage_results['early'])}, Mid={len(stage_results['mid'])}, Late={len(stage_results['late'])}")
    return stage_results

def run_universal_interface_analysis(segid1, segid2, n_runs=1, total_frames=100,
                                      step=5, start_frame=0, max_frames=None):
    print(f"\n UNIVERSAL INTERFACE ANALYSIS: {segid1} <-> {segid2}")
    all_pair_effects = []
    run_seeds = [42 + i * 100 for i in range(n_runs)]

    for run in range(n_runs):
        print(f"\n   Run {run+1}/{n_runs} (seed={run_seeds[run]})...")
        model = load_interface_model(segid1, segid2)
        if model is None:
            continue
        graphs = load_graphs_with_fix()
        if graphs is None:
            continue
        selected_graphs = select_frames(graphs, total_frames, step, start_frame, max_frames)
        # FIX #2: pass run_seed to ensure conformational perturbation reproducibility
        pair_effects = enhanced_find_universal_interface_pairs(
            model, selected_graphs, segid1, segid2, run_seed=run_seeds[run]
        )
        all_pair_effects.append(pair_effects)

    return all_pair_effects

# ============================================================
#  AGGREGATION
# ============================================================
def aggregate_temporal_stages(all_pair_effects, top_k=15):
    stage_aggregators = {
        'early': defaultdict(list),
        'mid':   defaultdict(list),
        'late':  defaultdict(list)
    }
    for run_results in all_pair_effects:
        for stage in ['early', 'mid', 'late']:
            for pair, (delta, pert_type) in run_results[stage].items():
                stage_aggregators[stage][pair].append((delta, pert_type))

    stage_results = {}
    for stage in ['early', 'mid', 'late']:
        pair_total_delta = {}
        pair_pert_type = {}
        for pair, list_val in stage_aggregators[stage].items():
            deltas = [d for d, _ in list_val]
            pert_types = [p for _, p in list_val if p is not None]
            pair_total_delta[pair] = np.sum(deltas)
            pair_pert_type[pair] = max(set(pert_types), key=pert_types.count) if pert_types else "UNKNOWN"
        sorted_pairs = sorted(pair_total_delta.items(), key=lambda x: x[1], reverse=True)[:top_k * 2]
        stage_results[stage] = [(pair, total_delta, pair_pert_type[pair]) for pair, total_delta in sorted_pairs]

    return stage_results

def extract_individual_residues_by_stage(stage_results):
    stage_residues = {}
    for stage in ['early', 'mid', 'late']:
        segid1_residues = defaultdict(lambda: {'total_delta': 0.0, 'count': 0, 'pert_types': []})
        segid2_residues = defaultdict(lambda: {'total_delta': 0.0, 'count': 0, 'pert_types': []})

        for pair, total_delta, pert_type in stage_results[stage]:
            try:
                segid1_part, segid2_part = pair.split(' <-> ')
                segid1_res = f"{segid1_part.split('-')[1]}-{segid1_part.split('-')[2]}"
                segid2_res = f"{segid2_part.split('-')[1]}-{segid2_part.split('-')[2]}"
            except (IndexError, ValueError):
                continue

            segid1_residues[segid1_res]['total_delta'] += total_delta
            segid1_residues[segid1_res]['count'] += 1
            segid1_residues[segid1_res]['pert_types'].append(pert_type)

            segid2_residues[segid2_res]['total_delta'] += total_delta
            segid2_residues[segid2_res]['count'] += 1
            segid2_residues[segid2_res]['pert_types'].append(pert_type)

        def aggregate_residues(res_dict):
            result = []
            for res, data in res_dict.items():
                dominant = max(set(data['pert_types']), key=data['pert_types'].count) if data['pert_types'] else "UNKNOWN"
                result.append((res, data['total_delta'], data['count'], dominant))
            return sorted(result, key=lambda x: x[1], reverse=True)

        stage_residues[stage] = {
            'segid1': aggregate_residues(segid1_residues),
            'segid2': aggregate_residues(segid2_residues)
        }

    return stage_residues

# ============================================================
#  FIX #1: calculate_stage_percentiles — FULLY PORTED
# ============================================================
def calculate_stage_percentiles(stage_residues, all_pair_effects):
    """
    FIX #1: Full implementation ported from original PS script,
    updated to handle the new (delta, pert_type) tuple format.

    Builds a null distribution of SUM Delta values per residue per stage,
    then ranks each residue by percentile within that distribution.
    """
    print(" Calculating percentile ranking by temporal stage (SUM Delta)...")

    # Collect per-residue deltas across all runs and stages
    stage_residue_deltas = {
        'early': defaultdict(list),
        'mid':   defaultdict(list),
        'late':  defaultdict(list)
    }

    for run_results in all_pair_effects:
        for stage in ['early', 'mid', 'late']:
            for pair, (delta, pert_type) in run_results[stage].items():
                try:
                    segid1_part, segid2_part = pair.split(' <-> ')
                    segid1_res = f"{segid1_part.split('-')[1]}-{segid1_part.split('-')[2]}"
                    segid2_res = f"{segid2_part.split('-')[1]}-{segid2_part.split('-')[2]}"
                    stage_residue_deltas[stage][segid1_res].append(delta)
                    stage_residue_deltas[stage][segid2_res].append(delta)
                except (IndexError, ValueError):
                    continue

    # Build null distribution: SUM Delta per residue
    stage_null_dists = {}
    for stage in ['early', 'mid', 'late']:
        sum_deltas = [np.sum(deltas) for deltas in stage_residue_deltas[stage].values() if deltas]
        stage_null_dists[stage] = np.array(sum_deltas) if sum_deltas else np.array([0.0])

    # Compute percentile for each residue against the null distribution
    stage_percentiles = {}
    for stage in ['early', 'mid', 'late']:
        null_dist = stage_null_dists[stage]
        segid1_percentiles = []
        segid2_percentiles = []

        for residue, total_delta, count, pert_type in stage_residues[stage]['segid1']:
            percentile = (np.sum(total_delta >= null_dist) / len(null_dist)) * 100 if len(null_dist) > 0 else 0.0
            segid1_percentiles.append((residue, total_delta, count, percentile, pert_type))

        for residue, total_delta, count, pert_type in stage_residues[stage]['segid2']:
            percentile = (np.sum(total_delta >= null_dist) / len(null_dist)) * 100 if len(null_dist) > 0 else 0.0
            segid2_percentiles.append((residue, total_delta, count, percentile, pert_type))

        stage_percentiles[stage] = {
            'segid1': segid1_percentiles,
            'segid2': segid2_percentiles,
            'null_stats': {
                'all_sum_deltas': null_dist
            }
        }

        print(f"   {stage.upper()}: null dist from {len(null_dist)} residues, "
              f"range [{null_dist.min():.4f}, {null_dist.max():.4f}]")

    return stage_percentiles

# ============================================================
#  OUTPUT TABLES
# ============================================================
def print_temporal_staging_tables(stage_percentiles, segid1, segid2, num_residues_display=10):
    for stage in ['early', 'mid', 'late']:
        print(f"\n{'='*100}")
        print(f" {stage.upper()} STAGE - RESIDUE RANKING (SUM Delta + Percentile + Dominant Perturbation)")
        print(f"{'Protein':<8} {'Residue':<15} {'Sum Delta':<12} {'#Pairs':<8} {'Percentile':<12} {'Dominant Force':<20}")
        print("-" * 100)

        stage_data = stage_percentiles[stage]

        for residue, total_delta, count, percentile, pert_type in stage_data['segid1'][:num_residues_display]:
            print(f"{segid1:<8} {residue:<15} {total_delta:<12.6f} {count:<8} {percentile:>10.1f}%  {pert_type:<20}")

        for residue, total_delta, count, percentile, pert_type in stage_data['segid2'][:num_residues_display]:
            print(f"{segid2:<8} {residue:<15} {total_delta:<12.6f} {count:<8} {percentile:>10.1f}%  {pert_type:<20}")

        null_stats = stage_data['null_stats']
        print("-" * 100)
        nd = null_stats['all_sum_deltas']
        print(f"Null dist: {len(nd)} residues, range [{nd.min():.4f}, {nd.max():.4f}], median {np.median(nd):.4f}")

def print_excel_ready_output(stage_percentiles, segid1, segid2):
    print(f"\n{'='*70}")
    print(f" EXCEL-READY OUTPUT: {segid1} <-> {segid2} (SUM Delta)")
    print("="*70)
    print("Protein\tResidue\tSumDelta\tCount\tPercentile\tStage\tDominantForce")
    for stage in ['early', 'mid', 'late']:
        stage_data = stage_percentiles[stage]
        for residue, total_delta, count, percentile, pert_type in stage_data['segid1']:
            print(f"{segid1}\t{residue}\t{total_delta:.6f}\t{count}\t{percentile:.2f}\t{stage}\t{pert_type}")
        for residue, total_delta, count, percentile, pert_type in stage_data['segid2']:
            print(f"{segid2}\t{residue}\t{total_delta:.6f}\t{count}\t{percentile:.2f}\t{stage}\t{pert_type}")

def calculate_total_interface_strength(stage_residues, segid1, segid2):
    print(f"\n{'='*70}")
    print(f" TOTAL INTERFACE STRENGTH: {segid1} <-> {segid2}")
    print("="*70)
    print(f"{'Stage':<10} {'#Residues':<12} {'Segid1 Sum':<15} {'Segid2 Sum':<15} {'Total':<15}")
    print("-" * 70)

    total_strengths = {}
    for stage in ['early', 'mid', 'late']:
        s1_total = sum(td for _, td, _, _ in stage_residues[stage]['segid1'])
        s2_total = sum(td for _, td, _, _ in stage_residues[stage]['segid2'])
        total = s1_total + s2_total
        n1 = len(stage_residues[stage]['segid1'])
        n2 = len(stage_residues[stage]['segid2'])
        total_strengths[stage] = {'segid1_total': s1_total, 'segid2_total': s2_total,
                                   'total_interface': total}
        print(f"{stage:<10} {n1}+{n2:<11} {s1_total:<15.3f} {s2_total:<15.3f} {total:<15.3f}")

    early = total_strengths['early']['total_interface']
    late = total_strengths['late']['total_interface']
    change = ((late - early) / early * 100) if early > 0 else 0.0
    trend = "STRONGER" if change > 10 else ("WEAKER" if change < -10 else "STABLE")
    print(f"\n Early -> Late change: {change:+.1f}% ({trend})")
    return total_strengths

# ============================================================
#  FIX #3 + FIX #4: UMAP WITH MEAN DELTA ACROSS ALL PAIRS
# ============================================================
def get_graph_embedding(model, graph):
    """
    Extract the latent embedding (32 * n_interfaces dimensions) before the
    final linear prediction layer.

    FIX #3 — Safety note:
    model.eval() is called explicitly here. In eval mode, nn.Dropout is a
    no-op, so the model.dropout(x) call below is safe. Do NOT call this
    function while the model is in train mode (model.train()), as dropout
    would then randomly zero activations and corrupt the embedding.
    """
    model.eval()  # Explicit safety — ensures dropout disabled
    with torch.no_grad():
        x = F.relu(model.batch_norm1(model.conv1(graph.x, graph.edge_index)))
        x = F.relu(model.batch_norm2(model.conv2(x, graph.edge_index)))
        x = F.relu(model.batch_norm3(model.conv3(x, graph.edge_index)))
        # model.dropout is nn.Dropout — no-op in eval mode, safe to call
        x = model.dropout(x)
        interface_features = model.generate_interface_features(graph, x)
        embedding = interface_features.flatten().cpu().numpy()
    return embedding

def run_umap_with_delta(model, graphs, segid1, segid2, run_seed=None):
    """
    FIX #4: frame_deltas computed as MEAN across ALL interface pairs,
    not max over first 5. This gives an unbiased per-frame sensitivity
    estimate for large heterogeneous interfaces.

    FIX #3: get_graph_embedding called with model in eval mode.
    """
    print("\n Running UMAP on latent space embeddings...")

    # Extract embeddings
    embeddings = []
    for g in graphs:
        embeddings.append(get_graph_embedding(model, g))
    embeddings = np.array(embeddings)

    # FIX #4: compute per-frame delta as MEAN across ALL pairs
    print(" Computing per-frame perturbation sensitivity (mean across all pairs)...")
    frame_deltas = []
    for g in graphs:
        with torch.no_grad():
            orig = model(g).item()
        pairs = find_universal_contacts(g, segid1, segid2, cutoff=8.0)
        if pairs:
            pair_deltas = []
            for r1, r2 in pairs:  # ALL pairs
                d, _ = comprehensive_perturbation_scan_with_argmax(
                    model, g, r1, r2, orig, segid1, segid2, run_seed=run_seed
                )
                pair_deltas.append(d)
            frame_deltas.append(np.mean(pair_deltas))  # MEAN not max
        else:
            frame_deltas.append(0.0)

    frame_deltas = np.array(frame_deltas)

    # UMAP reduction
    reducer = umap.UMAP(n_components=2, random_state=42)
    coords = reducer.fit_transform(embeddings)

    # Plot colored by mean Delta
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sc1 = axes[0].scatter(coords[:, 0], coords[:, 1],
                          c=frame_deltas, cmap='coolwarm', s=15, alpha=0.8)
    plt.colorbar(sc1, ax=axes[0], label='Mean Perturbation Delta')
    axes[0].set_xlabel('UMAP 1')
    axes[0].set_ylabel('UMAP 2')
    axes[0].set_title(f'Conformational landscape colored by PS Delta\n{segid1} <-> {segid2}')

    # Also plot colored by frame index for temporal reference
    frame_indices = np.arange(len(graphs))
    sc2 = axes[1].scatter(coords[:, 0], coords[:, 1],
                          c=frame_indices, cmap='viridis', s=15, alpha=0.8)
    plt.colorbar(sc2, ax=axes[1], label='Frame index')
    axes[1].set_xlabel('UMAP 1')
    axes[1].set_ylabel('UMAP 2')
    axes[1].set_title('Conformational landscape colored by frame index')

    plt.tight_layout()
    plt.savefig(f'umap_{segid1}_{segid2}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"   UMAP saved as umap_{segid1}_{segid2}.png")

    # Detect transition frames: top 10% farthest from KMeans cluster centers
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(coords)
    dists = np.linalg.norm(coords - kmeans.cluster_centers_[labels], axis=1)
    thresh = np.percentile(dists, 90)
    trans_frames = np.where(dists > thresh)[0]
    print(f"   Detected {len(trans_frames)} transition frames (top 10% by distance from cluster centers)")
    print(f"   Transition frame indices: {trans_frames.tolist()}")
    print(f"   Mean Delta in transition frames: {frame_deltas[trans_frames].mean():.4f}")
    print(f"   Mean Delta in stable frames: {frame_deltas[dists <= thresh].mean():.4f}")

    return coords, trans_frames, frame_deltas

# ============================================================
#  MODULE A: GAT ATTENTION WEIGHT EXTRACTION
# ============================================================

def extract_attention_weights(model, graph):
    """
    Extract GAT attention coefficients from all three layers.

    Returns a dict with keys 'layer1', 'layer2', 'layer3'.
    Each value is a tensor of shape [E, heads] (layer1) or [E, 1] (layers 2-3).
    edge_index is the shared graph connectivity [2, E].

    Biological meaning: high attention weight on edge (i->j) means node j
    heavily weights information from node i during message passing.
    Inter-segid edges with high attention are the contacts the model
    relies on most for interface representation.

    Note: model must be in eval mode. Called internally with model.eval().
    """
    model.eval()
    with torch.no_grad():
        x1, (ei1, alpha1) = model.conv1(
            graph.x, graph.edge_index, return_attention_weights=True
        )
        x1 = F.relu(model.batch_norm1(x1))

        x2, (ei2, alpha2) = model.conv2(
            x1, graph.edge_index, return_attention_weights=True
        )
        x2 = F.relu(model.batch_norm2(x2))

        _, (ei3, alpha3) = model.conv3(
            x2, graph.edge_index, return_attention_weights=True
        )

    return {
        'layer1': alpha1.cpu(),   # [E, heads]
        'layer2': alpha2.cpu(),   # [E, 1]
        'layer3': alpha3.cpu(),   # [E, 1]
        'edge_index': graph.edge_index.cpu()
    }


def aggregate_attention_over_trajectory(model, graphs, segid1, segid2):
    """
    Aggregate mean attention weights across all trajectory frames
    for inter-segid edges only (segid1 <-> segid2).

    Returns a dict mapping residue_pair -> mean_attention (layer 3,
    which is closest to the prediction output and most informative).

    Layer 3 is used because it is the final representation layer
    before interface pooling — its attention weights directly reflect
    what the model considers important for stability prediction.
    """
    print(f"\n Aggregating attention weights: {segid1} <-> {segid2}...")

    pair_attention = defaultdict(list)

    for frame_idx, graph in enumerate(graphs):
        attn = extract_attention_weights(model, graph)
        edge_index = attn['edge_index']
        # Layer 3 attention: shape [E, 1] — squeeze to [E]
        alpha3 = attn['layer3'].squeeze(-1)

        for e in range(edge_index.shape[1]):
            src = edge_index[0, e].item()
            tgt = edge_index[1, e].item()

            seg_src = graph.segids[src]
            seg_tgt = graph.segids[tgt]

            # Only inter-segid edges between the two molecules of interest
            if set([seg_src, seg_tgt]) != set([segid1, segid2]):
                continue

            if not (hasattr(graph, 'resnames') and hasattr(graph, 'residues')):
                continue

            res_src = f"{seg_src}-{graph.resnames[src]}-{graph.residues[src]}"
            res_tgt = f"{seg_tgt}-{graph.resnames[tgt]}-{graph.residues[tgt]}"
            pair_key = f"{res_src} <-> {res_tgt}"

            pair_attention[pair_key].append(alpha3[e].item())

        if (frame_idx + 1) % 20 == 0:
            print(f"   Frame {frame_idx+1}/{len(graphs)} processed")

    # Compute mean attention per pair
    pair_mean_attention = {
        pair: np.mean(vals)
        for pair, vals in pair_attention.items()
        if vals
    }

    sorted_pairs = sorted(pair_mean_attention.items(), key=lambda x: x[1], reverse=True)
    print(f"   Attention aggregated over {len(graphs)} frames, {len(sorted_pairs)} inter-segid pairs")

    return pair_mean_attention, sorted_pairs


def plot_top_attention_edges(sorted_pairs, segid1, segid2, top_k=15):
    """Bar plot of top-k inter-segid edges by mean attention weight."""
    top = sorted_pairs[:top_k]
    labels = [p[0].replace(' <-> ', '\n') for p in top]
    values = [p[1] for p in top]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(labels[::-1], values[::-1], color='steelblue', alpha=0.8)
    ax.set_xlabel('Mean GAT Attention Weight (Layer 3)', fontsize=12)
    ax.set_title(f'Top {top_k} interface contacts by learned attention\n{segid1} <-> {segid2}', fontsize=13)
    ax.grid(axis='x', alpha=0.4)
    plt.tight_layout()
    fname = f'attention_top{top_k}_{segid1}_{segid2}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"   Saved: {fname}")


# ============================================================
#  MODULE B: GRADCAM NODE ATTRIBUTION
# ============================================================

def gradcam_node_attribution(model, graph):
    """
    Compute gradient-based node attribution (GradCAM for graphs).

    For each node, computes the L2 norm of d(predicted_stability)/d(node_features).
    High attribution = small changes to this node's features strongly affect
    the predicted stability score.

    This is the continuous, differentiable complement to PS perturbation scanning:
    - PS perturbation: finite difference, biologically interpretable force decomposition
    - GradCAM: exact gradient, fast (one backward pass), architecture-level attribution

    Both methods ranking the same residues = high-confidence hotspot.
    Disagreement = mechanistically interesting residue worth investigating.

    Returns: attribution scores per node [N], numpy array.
    """
    model.eval()

    # Need gradients — clone and enable grad on node features
    x_input = graph.x.clone().detach().requires_grad_(True)

    # Rebuild a minimal data object with grad-enabled features
    class _GraphProxy:
        pass

    proxy = _GraphProxy()
    proxy.x = x_input
    proxy.edge_index = graph.edge_index
    proxy.pos = graph.pos
    proxy.segids = graph.segids
    proxy.resnames = getattr(graph, 'resnames', None)
    proxy.residues = getattr(graph, 'residues', None)
    proxy.num_nodes = graph.num_nodes

    pred = model(proxy)
    pred.backward()

    # Attribution = L2 norm of gradient per node
    attribution = torch.norm(x_input.grad, dim=1).detach().cpu().numpy()
    return attribution


def aggregate_gradcam_over_trajectory(model, graphs, segid1, segid2):
    """
    Compute mean GradCAM attribution per residue across all trajectory frames,
    restricted to interface residues of segid1 and segid2.

    Returns dict mapping residue_key -> mean_attribution.
    """
    print(f"\n Computing GradCAM attribution: {segid1} <-> {segid2}...")

    residue_attributions = defaultdict(list)

    for frame_idx, graph in enumerate(graphs):
        try:
            attribution = gradcam_node_attribution(model, graph)
        except Exception as e:
            print(f"   Frame {frame_idx}: GradCAM failed ({e}), skipping")
            continue

        for i in range(graph.num_nodes):
            seg = graph.segids[i]
            if seg not in [segid1, segid2]:
                continue
            if not (hasattr(graph, 'resnames') and hasattr(graph, 'residues')):
                continue
            res_key = f"{seg}-{graph.resnames[i]}-{graph.residues[i]}"
            residue_attributions[res_key].append(attribution[i])

        if (frame_idx + 1) % 20 == 0:
            print(f"   Frame {frame_idx+1}/{len(graphs)} processed")

    residue_mean_attribution = {
        res: np.mean(vals)
        for res, vals in residue_attributions.items()
        if vals
    }

    sorted_residues = sorted(residue_mean_attribution.items(), key=lambda x: x[1], reverse=True)
    print(f"   GradCAM computed for {len(sorted_residues)} residues over {len(graphs)} frames")

    return residue_mean_attribution, sorted_residues


def plot_gradcam_attribution(sorted_residues, segid1, segid2, top_k=15):
    """Bar plot of top-k residues by mean GradCAM attribution."""
    top = sorted_residues[:top_k]
    labels = [r[0] for r in top]
    values = [r[1] for r in top]

    # Color by segid
    colors = ['#E74C3C' if segid1 in l else '#3498DB' for l in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels[::-1], values[::-1], color=colors[::-1], alpha=0.85)
    ax.set_xlabel('Mean GradCAM Attribution Score', fontsize=12)
    ax.set_title(f'Top {top_k} residues by gradient attribution\n{segid1} (red) vs {segid2} (blue)', fontsize=13)
    ax.grid(axis='x', alpha=0.4)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#E74C3C', label=segid1),
                       Patch(facecolor='#3498DB', label=segid2)]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    fname = f'gradcam_top{top_k}_{segid1}_{segid2}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"   Saved: {fname}")


# ============================================================
#  MODULE C: ATTENTION vs SUM DELTA CROSS-CORRELATION
# ============================================================

def build_residue_sum_delta(stage_percentiles):
    """
    Aggregate Sum Delta per residue across all three temporal stages
    from the stage_percentiles dict. Returns dict: residue_key -> total_sum_delta.
    """
    residue_sum_delta = defaultdict(float)
    for stage in ['early', 'mid', 'late']:
        for residue, total_delta, count, percentile, pert_type in stage_percentiles[stage]['segid1']:
            residue_sum_delta[residue] += total_delta
        for residue, total_delta, count, percentile, pert_type in stage_percentiles[stage]['segid2']:
            residue_sum_delta[residue] += total_delta
    return dict(residue_sum_delta)


def build_residue_attention(pair_mean_attention, segid1, segid2):
    """
    Aggregate mean attention per residue from pair-level attention dict.
    Each residue's attention score = mean of all pair-level attentions it participates in.
    Returns dict: residue_key (short: RESNAME-RESID) -> mean_attention.
    """
    residue_attn = defaultdict(list)
    for pair_key, attn in pair_mean_attention.items():
        try:
            part1, part2 = pair_key.split(' <-> ')
            # Extract short key: RESNAME-RESID (drop segid prefix)
            res1_short = f"{part1.split('-')[1]}-{part1.split('-')[2]}"
            res2_short = f"{part2.split('-')[1]}-{part2.split('-')[2]}"
            residue_attn[res1_short].append(attn)
            residue_attn[res2_short].append(attn)
        except (IndexError, ValueError):
            continue
    return {res: np.mean(vals) for res, vals in residue_attn.items() if vals}


def cross_correlate_attention_vs_delta(residue_sum_delta, residue_attention,
                                        segid1, segid2, top_k=20):
    """
    Cross-correlate Sum Delta (PS perturbation) vs mean attention weight (GAT)
    per residue. Produces:
    1. Pearson and Spearman correlations between the two rankings
    2. Scatter plot with quadrant annotation
    3. Identification of high-confidence hotspots (high in both)
       and mechanistically interesting residues (high in one only)

    This is the dual-validation figure: residues that rank high in BOTH
    methods are the most robust hotspot predictions.
    """
    print(f"\n Cross-correlating Sum Delta vs Attention: {segid1} <-> {segid2}...")

    # Find common residues
    common_residues = set(residue_sum_delta.keys()) & set(residue_attention.keys())
    if len(common_residues) < 5:
        print(f"   Only {len(common_residues)} common residues — not enough for cross-correlation.")
        return None

    residues = sorted(common_residues)
    delta_vals = np.array([residue_sum_delta[r] for r in residues])
    attn_vals  = np.array([residue_attention[r] for r in residues])

    # Normalize both to [0, 1] for comparability
    def norm01(x):
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 1e-8 else np.zeros_like(x)

    delta_norm = norm01(delta_vals)
    attn_norm  = norm01(attn_vals)

    # Correlations
    pearson_r,  pearson_p  = pearsonr(delta_norm,  attn_norm)
    spearman_r, spearman_p = spearmanr(delta_norm, attn_norm)

    print(f"   Pearson  r = {pearson_r:.3f}  (p={pearson_p:.4f})")
    print(f"   Spearman r = {spearman_r:.3f}  (p={spearman_p:.4f})")

    # Quadrant thresholds (median split)
    delta_thresh = np.median(delta_norm)
    attn_thresh  = np.median(attn_norm)

    high_both  = [r for r, d, a in zip(residues, delta_norm, attn_norm) if d >= delta_thresh and a >= attn_thresh]
    high_delta = [r for r, d, a in zip(residues, delta_norm, attn_norm) if d >= delta_thresh and a <  attn_thresh]
    high_attn  = [r for r, d, a in zip(residues, delta_norm, attn_norm) if d <  delta_thresh and a >= attn_thresh]
    low_both   = [r for r, d, a in zip(residues, delta_norm, attn_norm) if d <  delta_thresh and a <  attn_thresh]

    print(f"\n   QUADRANT ANALYSIS:")
    print(f"   High Delta + High Attention (robust hotspots): {len(high_both)}")
    for r in sorted(high_both, key=lambda x: residue_sum_delta[x], reverse=True)[:5]:
        print(f"      {r}  SumDelta={residue_sum_delta[r]:.4f}  Attention={residue_attention[r]:.4f}")
    print(f"   High Delta only (perturbation-sensitive, attention-invisible): {len(high_delta)}")
    for r in sorted(high_delta, key=lambda x: residue_sum_delta[x], reverse=True)[:3]:
        print(f"      {r}  SumDelta={residue_sum_delta[r]:.4f}  Attention={residue_attention[r]:.4f}")
    print(f"   High Attention only (model-attended, perturbation-robust): {len(high_attn)}")
    for r in sorted(high_attn, key=lambda x: residue_attention[x], reverse=True)[:3]:
        print(f"      {r}  SumDelta={residue_sum_delta[r]:.4f}  Attention={residue_attention[r]:.4f}")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(8, 7))

    quadrant_colors = []
    for d, a in zip(delta_norm, attn_norm):
        if d >= delta_thresh and a >= attn_thresh:
            quadrant_colors.append('#E74C3C')   # red: high both
        elif d >= delta_thresh:
            quadrant_colors.append('#E67E22')   # orange: high delta only
        elif a >= attn_thresh:
            quadrant_colors.append('#3498DB')   # blue: high attention only
        else:
            quadrant_colors.append('#95A5A6')   # grey: low both

    ax.scatter(delta_norm, attn_norm, c=quadrant_colors, s=60, alpha=0.85, edgecolors='white', linewidth=0.5)

    # Label top_k by combined score
    combined_score = delta_norm + attn_norm
    top_indices = np.argsort(combined_score)[-top_k:]
    for idx in top_indices:
        ax.annotate(residues[idx], (delta_norm[idx], attn_norm[idx]),
                    fontsize=7, ha='left', va='bottom',
                    xytext=(3, 3), textcoords='offset points')

    # Quadrant lines
    ax.axvline(delta_thresh, color='grey', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(attn_thresh,  color='grey', linestyle='--', alpha=0.5, linewidth=1)

    # Quadrant labels
    ax.text(0.02, 0.98, 'High Attention\nLow Delta', transform=ax.transAxes,
            fontsize=8, color='#3498DB', va='top')
    ax.text(0.98, 0.98, 'HIGH BOTH\n(Robust hotspots)', transform=ax.transAxes,
            fontsize=8, color='#E74C3C', va='top', ha='right', fontweight='bold')
    ax.text(0.02, 0.02, 'Low Both', transform=ax.transAxes,
            fontsize=8, color='grey')
    ax.text(0.98, 0.02, 'High Delta\nLow Attention', transform=ax.transAxes,
            fontsize=8, color='#E67E22', ha='right')

    ax.set_xlabel('Normalized Sum Delta (PS perturbation)', fontsize=12)
    ax.set_ylabel('Normalized Mean Attention Weight (GAT)', fontsize=12)
    ax.set_title(
        f'PS Delta vs GAT Attention — dual hotspot validation\n'
        f'{segid1} <-> {segid2}  |  '
        f'Pearson r={pearson_r:.2f}  Spearman r={spearman_r:.2f}',
        fontsize=12
    )
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fname = f'crosscorr_delta_attention_{segid1}_{segid2}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"   Saved: {fname}")

    return {
        'pearson_r':   pearson_r,
        'spearman_r':  spearman_r,
        'high_both':   high_both,
        'high_delta':  high_delta,
        'high_attn':   high_attn,
        'residues':    residues,
        'delta_norm':  delta_norm,
        'attn_norm':   attn_norm,
    }


# ============================================================
#  MODULE D: NODE EMBEDDING TRAJECTORY
# ============================================================

def get_node_embeddings(model, graph):
    """
    Extract per-node 32-dim embeddings from the final GCN layer
    (before interface pooling and linear prediction).

    Shape: [N, 32] — one 32-dim vector per atom.

    These embeddings encode what the model has learned about each
    atom's role in the interface, integrating information from its
    3-hop neighborhood via message passing.
    """
    model.eval()
    with torch.no_grad():
        x = F.relu(model.batch_norm1(model.conv1(graph.x, graph.edge_index)))
        x = F.relu(model.batch_norm2(model.conv2(x, graph.edge_index)))
        x = F.relu(model.batch_norm3(model.conv3(x, graph.edge_index)))
        # model.dropout is no-op in eval mode
        x = model.dropout(x)
    return x.cpu()  # [N, 32]


def aggregate_residue_embeddings(model, graphs, segid1, segid2):
    """
    For each interface residue, collect its 32-dim embedding across
    all trajectory frames. Returns:
      residue_embedding_traj: dict residue_key -> np.array [n_frames, 32]
      frame_indices: list of frame indices processed
    """
    print(f"\n Extracting node embedding trajectories: {segid1} <-> {segid2}...")

    # First pass: identify which residues are interface residues
    # (present in at least one frame's contact list)
    interface_residues = set()
    sample_graph = graphs[0]
    pairs = find_universal_contacts(sample_graph, segid1, segid2, cutoff=8.0)
    for r1, r2 in pairs:
        interface_residues.add(r1)
        interface_residues.add(r2)

    print(f"   Tracking {len(interface_residues)} interface residues across {len(graphs)} frames")

    # Build residue -> node index mapping per frame
    residue_embedding_traj = defaultdict(list)
    frame_indices = []

    for frame_idx, graph in enumerate(graphs):
        node_emb = get_node_embeddings(model, graph)  # [N, 32]

        # Average embedding per residue (over all atoms of that residue)
        residue_atoms = defaultdict(list)
        for i in range(graph.num_nodes):
            if not (hasattr(graph, 'segids') and hasattr(graph, 'resnames') and hasattr(graph, 'residues')):
                continue
            seg = graph.segids[i]
            if seg not in [segid1, segid2]:
                continue
            res_key = f"{seg}-{graph.resnames[i]}-{graph.residues[i]}"
            if res_key in interface_residues:
                residue_atoms[res_key].append(node_emb[i].numpy())

        for res_key, atom_embs in residue_atoms.items():
            # Mean over atoms of the same residue
            residue_embedding_traj[res_key].append(np.mean(atom_embs, axis=0))

        frame_indices.append(frame_idx)

        if (frame_idx + 1) % 20 == 0:
            print(f"   Frame {frame_idx+1}/{len(graphs)} processed")

    # Convert lists to arrays
    residue_embedding_traj = {
        res: np.array(embs)  # [n_frames, 32]
        for res, embs in residue_embedding_traj.items()
        if len(embs) == len(graphs)  # only residues present in all frames
    }

    print(f"   {len(residue_embedding_traj)} residues tracked consistently across all frames")
    return residue_embedding_traj, frame_indices


def compute_embedding_drift(residue_embedding_traj):
    """
    For each residue, compute its embedding drift = mean L2 distance
    between consecutive frames. High drift = the model's representation
    of this residue's interface role changes significantly over time.

    This identifies residues that undergo conformational role transitions,
    not just positional changes.
    """
    residue_drift = {}
    for res, traj in residue_embedding_traj.items():
        # L2 distance between consecutive frame embeddings
        diffs = np.linalg.norm(np.diff(traj, axis=0), axis=1)  # [n_frames-1]
        residue_drift[res] = np.mean(diffs)
    return residue_drift


def plot_node_embedding_trajectories(residue_embedding_traj, residue_drift,
                                      stage_percentiles, segid1, segid2,
                                      top_k=6):
    """
    Four-panel figure:
    1. Top-k residues by embedding drift — UMAP of their embedding trajectories
    2. Drift score bar chart colored by segid
    3. Drift vs Sum Delta scatter — do high-perturbation residues also drift?
    4. Heatmap of embedding drift over time for top residues
    """
    print(f"\n Plotting node embedding trajectories...")

    # Get Sum Delta per residue for cross-reference
    residue_sum_delta = build_residue_sum_delta(stage_percentiles)

    # Top drifting residues
    sorted_drift = sorted(residue_drift.items(), key=lambda x: x[1], reverse=True)
    top_residues = [r for r, _ in sorted_drift[:top_k]]

    fig = plt.figure(figsize=(18, 14))

    # ---- Panel 1: UMAP of embedding trajectories for top residues ----
    ax1 = fig.add_subplot(2, 2, 1)

    all_embeddings = []
    all_labels = []
    all_frame_idx = []

    for res in top_residues:
        traj = residue_embedding_traj[res]
        all_embeddings.append(traj)
        all_labels.extend([res] * len(traj))
        all_frame_idx.extend(list(range(len(traj))))

    all_embeddings = np.vstack(all_embeddings)

    reducer = umap.UMAP(n_components=2, random_state=42)
    coords_2d = reducer.fit_transform(all_embeddings)

    colors = plt.cm.tab10(np.linspace(0, 1, len(top_residues)))
    offset = 0
    for idx, res in enumerate(top_residues):
        n = len(residue_embedding_traj[res])
        c = coords_2d[offset:offset+n]
        ax1.scatter(c[:, 0], c[:, 1], c=[colors[idx]], s=20, alpha=0.7, label=res.split('-')[1]+'-'+res.split('-')[2])
        # Draw trajectory arrow from first to last point
        ax1.annotate('', xy=c[-1], xytext=c[0],
                     arrowprops=dict(arrowstyle='->', color=colors[idx], lw=1.5))
        offset += n

    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    ax1.set_title(f'Embedding trajectory (top {top_k} drifting residues)')
    ax1.legend(fontsize=7, loc='best')
    ax1.grid(alpha=0.3)

    # ---- Panel 2: Drift score bar chart ----
    ax2 = fig.add_subplot(2, 2, 2)

    top20 = sorted_drift[:20]
    labels20 = [r[0].split('-')[1]+'-'+r[0].split('-')[2] for r in top20]
    drift20  = [r[1] for r in top20]
    bar_colors = ['#E74C3C' if segid1 in r[0] else '#3498DB' for r in top20]

    ax2.barh(labels20[::-1], drift20[::-1], color=bar_colors[::-1], alpha=0.85)
    ax2.set_xlabel('Mean Embedding Drift (L2 distance / frame)', fontsize=11)
    ax2.set_title(f'Residue embedding drift\n{segid1} (red) vs {segid2} (blue)')
    ax2.grid(axis='x', alpha=0.4)

    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(facecolor='#E74C3C', label=segid1),
                         Patch(facecolor='#3498DB', label=segid2)], loc='lower right')

    # ---- Panel 3: Drift vs Sum Delta scatter ----
    ax3 = fig.add_subplot(2, 2, 3)

    common = set(residue_drift.keys()) & set(residue_sum_delta.keys())
    if common:
        common_res = sorted(common)
        x_drift = np.array([residue_drift[r] for r in common_res])
        y_delta = np.array([residue_sum_delta[r] for r in common_res])

        seg_colors = ['#E74C3C' if segid1 in r else '#3498DB' for r in common_res]
        ax3.scatter(x_drift, y_delta, c=seg_colors, s=50, alpha=0.8, edgecolors='white', linewidth=0.5)

        # Label top combined score residues
        combined = x_drift / (x_drift.max() + 1e-8) + y_delta / (y_delta.max() + 1e-8)
        top_idx = np.argsort(combined)[-8:]
        for idx in top_idx:
            ax3.annotate(common_res[idx].split('-')[1]+'-'+common_res[idx].split('-')[2],
                         (x_drift[idx], y_delta[idx]),
                         fontsize=7, xytext=(3, 3), textcoords='offset points')

        if len(x_drift) > 3:
            r_val, p_val = pearsonr(x_drift, y_delta)
            ax3.set_title(f'Embedding drift vs Sum Delta\nPearson r={r_val:.2f} (p={p_val:.3f})')
        else:
            ax3.set_title('Embedding drift vs Sum Delta')

        ax3.set_xlabel('Mean Embedding Drift', fontsize=11)
        ax3.set_ylabel('Sum Delta (PS)', fontsize=11)
        ax3.grid(alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No common residues', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Embedding drift vs Sum Delta')

    # ---- Panel 4: Temporal drift heatmap for top residues ----
    ax4 = fig.add_subplot(2, 2, 4)

    heatmap_data = []
    heatmap_labels = []
    for res in top_residues:
        traj = residue_embedding_traj[res]
        # Frame-by-frame drift
        frame_drift = np.concatenate([[0], np.linalg.norm(np.diff(traj, axis=0), axis=1)])
        heatmap_data.append(frame_drift)
        heatmap_labels.append(res.split('-')[1]+'-'+res.split('-')[2])

    heatmap_array = np.array(heatmap_data)  # [top_k, n_frames]

    im = ax4.imshow(heatmap_array, aspect='auto', cmap='hot', interpolation='nearest')
    ax4.set_yticks(range(len(heatmap_labels)))
    ax4.set_yticklabels(heatmap_labels, fontsize=9)
    ax4.set_xlabel('Frame index')
    ax4.set_title(f'Per-frame embedding drift heatmap\n(top {top_k} residues)')
    plt.colorbar(im, ax=ax4, label='Embedding drift (L2)')

    plt.suptitle(f'Node Embedding Trajectory Analysis: {segid1} <-> {segid2}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = f'node_embedding_trajectory_{segid1}_{segid2}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"   Saved: {fname}")

    return sorted_drift


# ============================================================
#  COMBINED NODE/EDGE ANALYSIS RUNNER
# ============================================================

def run_node_edge_analysis(model, graphs, segid1, segid2,
                            stage_percentiles, top_k=15):
    """
    Master function running all four node/edge-level analyses
    for a given interface pair.

    Outputs:
    - attention_top_pairs: top interface contacts by GAT attention
    - gradcam_top_residues: top residues by gradient attribution
    - crosscorr_results: dual-validation scatter + correlations
    - drift_results: per-residue embedding drift ranking

    All figures saved as PNG files.
    """
    print(f"\n{'='*85}")
    print(f" NODE/EDGE-LEVEL ANALYSIS: {segid1} <-> {segid2}")
    print("="*85)

    # Module A: GAT attention weights
    pair_mean_attention, attention_sorted = aggregate_attention_over_trajectory(
        model, graphs, segid1, segid2
    )
    plot_top_attention_edges(attention_sorted, segid1, segid2, top_k=top_k)

    # Module B: GradCAM
    residue_mean_attribution, gradcam_sorted = aggregate_gradcam_over_trajectory(
        model, graphs, segid1, segid2
    )
    plot_gradcam_attribution(gradcam_sorted, segid1, segid2, top_k=top_k)

    # Module C: Cross-correlation
    residue_sum_delta = build_residue_sum_delta(stage_percentiles)
    residue_attention  = build_residue_attention(pair_mean_attention, segid1, segid2)
    crosscorr_results  = cross_correlate_attention_vs_delta(
        residue_sum_delta, residue_attention, segid1, segid2, top_k=top_k
    )

    # Module D: Node embedding trajectory
    residue_embedding_traj, frame_indices = aggregate_residue_embeddings(
        model, graphs, segid1, segid2
    )
    residue_drift = compute_embedding_drift(residue_embedding_traj)
    drift_results = plot_node_embedding_trajectories(
        residue_embedding_traj, residue_drift,
        stage_percentiles, segid1, segid2, top_k=min(6, top_k)
    )

    # Summary table
    print(f"\n{'='*85}")
    print(f" NODE/EDGE ANALYSIS SUMMARY: {segid1} <-> {segid2}")
    print("="*85)
    print(f"{'Residue':<20} {'Sum Delta':<12} {'Attention':<12} {'GradCAM':<12} {'Drift':<12} {'Consensus'}")
    print("-" * 85)

    # Build consensus ranking
    all_residues = set(residue_sum_delta.keys()) & \
                   set(residue_attention.keys()) & \
                   set(residue_mean_attribution.keys()) & \
                   set(residue_drift.keys())

    if all_residues:
        def norm01_dict(d):
            vals = np.array(list(d.values()))
            rng = vals.max() - vals.min()
            if rng < 1e-8:
                return {k: 0.5 for k in d}
            return {k: (v - vals.min()) / rng for k, v in d.items()}

        nd   = norm01_dict(residue_sum_delta)
        na   = norm01_dict(residue_attention)
        ng   = norm01_dict(residue_mean_attribution)
        ndr  = norm01_dict(residue_drift)

        consensus = {
            r: nd.get(r, 0) + na.get(r, 0) + ng.get(r, 0) + ndr.get(r, 0)
            for r in all_residues
        }

        top_consensus = sorted(consensus.items(), key=lambda x: x[1], reverse=True)[:top_k]
        for res, score in top_consensus:
            short = res.split('-')[1] + '-' + res.split('-')[2] if '-' in res else res
            print(f"{short:<20} "
                  f"{residue_sum_delta.get(res, 0):<12.4f} "
                  f"{residue_attention.get(res, 0):<12.4f} "
                  f"{residue_mean_attribution.get(res, 0):<12.4f} "
                  f"{residue_drift.get(res, 0):<12.4f} "
                  f"{score:.3f}")
    else:
        print("   Not enough overlapping residues for consensus table.")

    return {
        'pair_mean_attention':      pair_mean_attention,
        'gradcam_sorted':           gradcam_sorted,
        'crosscorr_results':        crosscorr_results,
        'residue_drift':            residue_drift,
        'residue_embedding_traj':   residue_embedding_traj,
    }


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    print(" UNIVERSAL INTERFACE ANALYZER - FIXED & IMPROVED")
    print("=" * 85)

    graph_path = input("   Enter path to .pt graph file: ").strip()
    model_path = input("   Enter path to .pth model file: ").strip()
    set_file_paths(graph_path, model_path)

    graphs = load_graphs_with_fix()
    if graphs is None:
        exit()

    available_segids = detect_all_segids(graphs)
    if len(available_segids) < 2:
        print("Need at least 2 segids for interface analysis!")
        exit()

    segid1 = "PROA"
    if segid1 not in available_segids:
        segid1 = input(f"   PROA not found. Enter main protein segid from {available_segids}: ").strip().upper()

    other_segids = [s for s in available_segids if s != segid1]
    print(f"\n Found {len(other_segids)} partners for {segid1}: {other_segids}")

    print("\n INTERFACE SELECTION:")
    print("   1. Analyze ALL interfaces")
    print("   2. Analyze specific interfaces")
    choice = input("   Enter choice (1 or 2): ").strip()

    if choice == "1":
        interface_pairs = [(segid1, s2) for s2 in other_segids]
    else:
        for i, s in enumerate(other_segids):
            print(f"   {i+1}. {segid1} <-> {s}")
        selection = input("   Enter choices (e.g. 1,2): ").strip()
        selected_indices = [int(x.strip()) - 1 for x in selection.split(',') if x.strip().isdigit()]
        interface_pairs = [(segid1, other_segids[i]) for i in selected_indices if 0 <= i < len(other_segids)]
        if not interface_pairs:
            interface_pairs = [(segid1, other_segids[0])]

    start_frame = int(input("   Start frame [0]: ") or 0)
    step = int(input("   Frame step [1]: ") or 1)
    n_runs = int(input("   Number of trials [1]: ") or 1)
    total_frames = int(input("   Total frames [100]: ") or 100)
    max_frames_input = input("   Max frames (0=all): ").strip()
    max_frames = int(max_frames_input) if max_frames_input and max_frames_input.isdigit() and int(max_frames_input) > 0 else None
    num_residues_display = int(input("   Residues to display per stage [15]: ") or 15)
    run_umap_flag      = input("\n Run UMAP + Delta overlay? (y/n): ").strip().lower() == 'y'
    run_node_edge_flag = input(" Run node/edge-level analysis (attention, GradCAM, cross-correlation, embedding trajectory)? (y/n): ").strip().lower() == 'y'

    all_results = {}

    for seg1, seg2 in interface_pairs:
        print(f"\n{'='*85}")
        print(f" ANALYZING INTERFACE: {seg1} <-> {seg2}")
        print("="*85)

        pair_effects = run_universal_interface_analysis(
            seg1, seg2,
            n_runs=n_runs,
            total_frames=total_frames,
            step=step,
            start_frame=start_frame,
            max_frames=max_frames
        )

        if not pair_effects:
            print(f"   No results for {seg1} <-> {seg2}, skipping.")
            continue

        stage_results   = aggregate_temporal_stages(pair_effects, top_k=num_residues_display * 3)
        stage_residues  = extract_individual_residues_by_stage(stage_results)
        # FIX #1: fully functional percentile calculation
        stage_percentiles = calculate_stage_percentiles(stage_residues, pair_effects)

        all_results[f"{seg1}-{seg2}"] = {
            'pair_effects':     pair_effects,
            'stage_percentiles': stage_percentiles,
            'stage_residues':   stage_residues
        }

        print_temporal_staging_tables(stage_percentiles, seg1, seg2, num_residues_display)
        total_strengths = calculate_total_interface_strength(stage_residues, seg1, seg2)
        all_results[f"{seg1}-{seg2}"]['total_strengths'] = total_strengths
        print_excel_ready_output(stage_percentiles, seg1, seg2)

        # FIX #3 + FIX #4: UMAP with mean delta across all pairs
        if run_umap_flag:
            model = load_interface_model(seg1, seg2)
            if model is not None:
                graphs_local = load_graphs_with_fix()
                if graphs_local is not None:
                    selected = select_frames(graphs_local, total_frames, step, start_frame, max_frames)
                    run_umap_with_delta(model, selected, seg1, seg2, run_seed=42)

        # NODE/EDGE-LEVEL ANALYSIS (Modules A, B, C, D)
        if run_node_edge_flag:
            model = load_interface_model(seg1, seg2)
            if model is not None:
                graphs_local = load_graphs_with_fix()
                if graphs_local is not None:
                    selected = select_frames(graphs_local, total_frames, step, start_frame, max_frames)
                    node_edge_results = run_node_edge_analysis(
                        model, selected, seg1, seg2,
                        stage_percentiles=stage_percentiles,
                        top_k=num_residues_display
                    )
                    all_results[f"{seg1}-{seg2}"]['node_edge'] = node_edge_results

    # Summary across all interfaces
    print(f"\n{'='*85}")
    print(f" SUMMARY OF ALL INTERFACES")
    print("="*85)
    print(f"{'Interface':<20} {'Early Sum':<15} {'Mid Sum':<15} {'Late Sum':<15} {'Trend':<10}")
    print("-" * 80)

    for interface_name, results in all_results.items():
        if 'total_strengths' in results:
            s = results['total_strengths']
            early = s.get('early', {}).get('total_interface', 0)
            mid   = s.get('mid',   {}).get('total_interface', 0)
            late  = s.get('late',  {}).get('total_interface', 0)
            change = ((late - early) / early * 100) if early > 0 else 0.0
            trend = "STRONGER" if change > 15 else ("WEAKER" if change < -15 else "STABLE")
            print(f"{interface_name:<20} {early:<15.3f} {mid:<15.3f} {late:<15.3f} {trend:<10}")

    print(f"\n MULTI-INTERFACE ANALYSIS COMPLETE!")
    print(f" Interfaces analyzed: {len(all_results)}")
    print(f" Fixes applied: percentiles, seeded conformational perturbation, UMAP mean Delta")
    print(f" Node/edge additions: GAT attention, GradCAM, cross-correlation, embedding trajectory")