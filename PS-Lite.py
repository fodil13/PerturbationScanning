# @title PS-Lite - Dr Fodil Azzaz (Non AI-driven Method)
"""
PS-Lite - Fixed Version with Chain Separation
Copyright (c) 2025 Fodil Azzaz - All Rights Reserved
Non-commercial use only

"""

print(" Installing in FRESH runtime...")
!pip install -q torch torchvision torchaudio
!pip install -q torch-geometric
!pip install -q MDAnalysis
!pip install -q scipy numpy matplotlib

print("✅ All installed! Now running TPS-Lite...")

# ====== NOW THE ACTUAL CODE ======
import torch
import numpy as np
from scipy.spatial import cKDTree
import MDAnalysis as mda
import os

print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")

# === FIXED TPS-LITE CLASS WITH CHAIN SEPARATION ===
class TPS_Lite_Fixed:
    """TPS-Lite with proper chain separation and improved scoring"""

    def __init__(self, psf_path, pdb_path):
        self.psf_path = psf_path
        self.pdb_path = pdb_path
        self.graph = self._create_graph()

    def _create_graph(self):
        """Create simple graph from PSF+PDB"""
        print(f"📁 Loading {os.path.basename(self.psf_path)} + {os.path.basename(self.pdb_path)}")

        # Load with MDAnalysis
        u = mda.Universe(self.psf_path, self.pdb_path)

        # Collect data
        positions = []
        segids = []
        resnames = []
        residues = []
        atom_names = []

        for atom in u.atoms:
            positions.append(atom.position)
            segids.append(getattr(atom, 'segid', 'UNK'))
            resnames.append(atom.resname)
            residues.append(atom.resid)
            atom_names.append(atom.name)

        positions = np.array(positions)

        # Create simple features (7D)
        features = []
        for resname, atom_name in zip(resnames, atom_names):
            features.append(self._get_features(resname, atom_name))

        # Create edges (distance-based)
        edges = self._create_edges(positions, segids)

        # Convert to tensors
        pos_tensor = torch.tensor(positions, dtype=torch.float32)
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        edge_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Store graph
        graph = {
            'x': feature_tensor,
            'pos': pos_tensor,
            'edge_index': edge_tensor,
            'segids': segids,
            'resnames': resnames,
            'residues': residues,
            'atom_names': atom_names,
            'num_nodes': len(positions),
            'num_edges': len(edges) // 2
        }

        print(f"✅ Graph created: {graph['num_nodes']} atoms, {graph['num_edges']} edges")
        print(f"   Segids/Chains available: {set(segids)}")

        return graph

    def _get_features(self, resname, atom_name):
        """7 simple features"""
        # Charge
        if resname in ['LYS', 'ARG']:
            charge = 1.0
        elif resname in ['ASP', 'GLU']:
            charge = -1.0
        else:
            charge = 0.0

        # Mass proxy
        if atom_name.startswith('H'):
            mass = 0.1
        elif atom_name.startswith('C'):
            mass = 1.2
        elif atom_name.startswith('N'):
            mass = 1.4
        elif atom_name.startswith('O'):
            mass = 1.6
        elif atom_name.startswith('S'):
            mass = 3.2
        else:
            mass = 1.0

        # Hydrophobicity
        hydrophobic = ['ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'MET', 'PRO', 'CYS']
        hydro = 1.0 if resname in hydrophobic else -1.0

        # Aromatic
        aromatic = ['PHE', 'TYR', 'TRP', 'HIS']
        is_arom = 1.0 if resname in aromatic else 0.0

        # Polar
        polar = ['SER', 'THR', 'ASN', 'GLN', 'ASP', 'GLU', 'LYS', 'ARG', 'HIS']
        is_polar = 1.0 if resname in polar else 0.0

        # Backbone
        is_backbone = 1.0 if atom_name in ['N', 'CA', 'C', 'O'] else 0.0

        # Sidechain
        is_sidechain = 1.0 if (is_backbone == 0.0 and resname not in ['HOH', 'WAT']) else 0.0

        return [charge, mass, hydro, is_arom, is_polar, is_backbone, is_sidechain]

    def _create_edges(self, positions, segids, cutoff=6.0):
        """Create edges between different segids"""
        edges = []

        tree = cKDTree(positions)
        pairs = tree.query_pairs(cutoff)

        for i, j in pairs:
            if segids[i] != segids[j]:  # Only different molecules
                edges.append([i, j])
                edges.append([j, i])  # Undirected

        return edges

    def _calculate_residue_score(self, resname, contact_count):
        """Improved scoring without saturation"""
        # Normalize contacts (0-1 scale)
        contact_norm = 1 - np.exp(-contact_count / 3.0)

        # Residue type weights
        type_weights = {
            'ARG': 0.35, 'LYS': 0.30, 'ASP': 0.30, 'GLU': 0.30,
            'TRP': 0.40, 'TYR': 0.35, 'PHE': 0.30,
            'ILE': 0.25, 'LEU': 0.25, 'VAL': 0.20, 'MET': 0.25,
            'HIS': 0.30, 'ASN': 0.20, 'GLN': 0.20,
            'SER': 0.15, 'THR': 0.15,
            'ALA': 0.10, 'GLY': 0.10, 'PRO': 0.15, 'CYS': 0.20
        }

        base_weight = type_weights.get(resname, 0.10)

        # Combine with non-linear mixing
        total_score = 0.7 * contact_norm + 0.3 * base_weight

        return min(total_score, 1.0)

    def analyze_interface(self, segid1, segid2, cutoff=6.0):
        """Main analysis function - separates results by chain"""
        print(f"\n🔍 Analyzing {segid1} ↔ {segid2} (cutoff={cutoff}Å)")

        positions = self.graph['pos'].numpy()
        segids = self.graph['segids']
        resnames = self.graph['resnames']
        residues = self.graph['residues']
        atom_names = self.graph['atom_names']

        # Get indices for each segid
        idx1 = [i for i, s in enumerate(segids) if s == segid1]
        idx2 = [i for i, s in enumerate(segids) if s == segid2]

        if not idx1:
            print(f"❌ {segid1} not found in segids: {set(segids)}")
            return {'chain1': [], 'chain2': [], 'all': []}
        if not idx2:
            print(f"❌ {segid2} not found in segids: {set(segids)}")
            return {'chain1': [], 'chain2': [], 'all': []}

        print(f"   • {segid1}: {len(idx1)} atoms")
        print(f"   • {segid2}: {len(idx2)} atoms")

        # Find contacts
        tree = cKDTree(positions[idx2])
        distances, neighbors = tree.query(positions[idx1], k=1)

        # Collect interface residues SEPARATELY
        interface_residues_chain1 = {}
        interface_residues_chain2 = {}

        for i, dist in enumerate(distances):
            if dist < cutoff:
                atom1 = idx1[i]
                atom2 = idx2[neighbors[i]]

                # Determine if backbone or sidechain
                bb1 = "(bb)" if atom_names[atom1] in ['N', 'CA', 'C', 'O'] else "(sc)"
                bb2 = "(bb)" if atom_names[atom2] in ['N', 'CA', 'C', 'O'] else "(sc)"

                # Chain 1 residue
                res1_key = f"{segid1}:{resnames[atom1]}-{residues[atom1]}{bb1}"

                # Chain 2 residue
                res2_key = f"{segid2}:{resnames[atom2]}-{residues[atom2]}{bb2}"

                # Count contacts
                interface_residues_chain1[res1_key] = interface_residues_chain1.get(res1_key, 0) + 1
                interface_residues_chain2[res2_key] = interface_residues_chain2.get(res2_key, 0) + 1

        # Convert to scores for chain 1
        results_chain1 = []
        for residue_str, count in interface_residues_chain1.items():
            # Extract resname from string like "A:ASP-39(sc)"
            parts = residue_str.split(':')
            if len(parts) < 2:
                continue

            residue_part = parts[1].split('(')[0]  # Get "ASP-39"
            resname = residue_part.split('-')[0]

            # Calculate score
            score = self._calculate_residue_score(resname, count)

            # Determine residue type
            if resname in ['ARG', 'LYS', 'ASP', 'GLU']:
                res_type = "ELECTROSTATIC"
            elif resname in ['TRP', 'TYR', 'PHE']:
                res_type = "AROMATIC"
            elif resname in ['ALA', 'VAL', 'LEU', 'ILE', 'MET']:
                res_type = "HYDROPHOBIC"
            elif resname in ['SER', 'THR', 'ASN', 'GLN']:
                res_type = "H-BOND"
            else:
                res_type = "GENERAL"

            results_chain1.append({
                'residue': residue_str,
                'chain': segid1,
                'score': score,
                'type': res_type,
                'contacts': count
            })

        # Convert to scores for chain 2
        results_chain2 = []
        for residue_str, count in interface_residues_chain2.items():
            parts = residue_str.split(':')
            if len(parts) < 2:
                continue

            residue_part = parts[1].split('(')[0]
            resname = residue_part.split('-')[0]

            score = self._calculate_residue_score(resname, count)

            if resname in ['ARG', 'LYS', 'ASP', 'GLU']:
                res_type = "ELECTROSTATIC"
            elif resname in ['TRP', 'TYR', 'PHE']:
                res_type = "AROMATIC"
            elif resname in ['ALA', 'VAL', 'LEU', 'ILE', 'MET']:
                res_type = "HYDROPHOBIC"
            elif resname in ['SER', 'THR', 'ASN', 'GLN']:
                res_type = "H-BOND"
            else:
                res_type = "GENERAL"

            results_chain2.append({
                'residue': residue_str,
                'chain': segid2,
                'score': score,
                'type': res_type,
                'contacts': count
            })

        # Sort both lists
        results_chain1.sort(key=lambda x: x['score'], reverse=True)
        results_chain2.sort(key=lambda x: x['score'], reverse=True)

        print(f"✅ Found {len(results_chain1)} interface residues in {segid1}")
        print(f"✅ Found {len(results_chain2)} interface residues in {segid2}")

        return {
            'chain1': results_chain1,
            'chain2': results_chain2,
            'all': results_chain1 + results_chain2
        }

    def print_results(self, results, top_n=10):
        """Pretty print results separated by chain"""
        print(f"\n{'='*80}")
        print(f" TPS-Lite Results - SEPARATED BY CHAIN")
        print(f"{'='*80}")

        # Print chain 1 results
        if results.get('chain1'):
            chain_id = results['chain1'][0]['chain'] if results['chain1'] else 'A'
            print(f"\n CHAIN {chain_id}:")
            print(f"{'-'*80}")
            print(f"{'Residue':<25} {'Score':<8} {'Type':<15} {'Contacts':<10}")
            print(f"{'-'*80}")

            for res in results['chain1'][:top_n]:
                print(f"{res['residue']:<25} {res['score']:<8.3f} {res['type']:<15} {res['contacts']:<10}")

        # Print chain 2 results
        if results.get('chain2'):
            chain_id = results['chain2'][0]['chain'] if results['chain2'] else 'B'
            print(f"\n CHAIN {chain_id}:")
            print(f"{'-'*80}")
            print(f"{'Residue':<25} {'Score':<8} {'Type':<15} {'Contacts':<10}")
            print(f"{'-'*80}")

            for res in results['chain2'][:top_n]:
                print(f"{res['residue']:<25} {res['score']:<8.3f} {res['type']:<15} {res['contacts']:<10}")

        # Combined top residues
        if results.get('all'):
            print(f"\n Top 5 Critical Interface Residues Overall:")
            all_results = sorted(results['all'], key=lambda x: x['score'], reverse=True)
            for i, res in enumerate(all_results[:5]):
                print(f"   {i+1}. {res['residue']} ({res['type']}, score={res['score']:.3f})")

        print(f"\n Legend: (bb) = backbone, (sc) = sidechain")
        print(f"• Score 0.8-1.0 = Critical interface residue")
        print(f"• Score 0.6-0.8 = Important")
        print(f"• Score <0.6 = Peripheral")

    def validate_with_known_residues(self, results, known_dict):
        """Validate predictions against known interface residues"""
        print(f"\n{'='*80}")
        print(f" VALIDATION vs. KNOWN RESIDUES")
        print(f"{'='*80}")

        for chain, known_residues in known_dict.items():
            print(f"\n🔍 Chain {chain}:")
            print(f"   Known critical residues: {', '.join(known_residues)}")

            # Get results for this chain
            chain_results = []
            if f'chain{chain}' in results:
                chain_results = results[f'chain{chain}']
            else:
                # Try to find by matching chain ID in residue string
                chain_results = [r for r in results.get('all', []) if f"{chain}:" in r.get('residue', '')]

            detected = []
            for known in known_residues:
                found = False
                for result in chain_results:
                    # Extract residue number from result string
                    res_str = result['residue']
                    # Format is like "A:ASP-39(sc)" or "B:LYS-27(bb)"
                    parts = res_str.split(':')
                    if len(parts) > 1:
                        residue_part = parts[1].split('(')[0]  # Get "ASP-39"
                        # Check if known residue matches
                        if known.upper() in residue_part.upper():
                            detected.append((known, result['score']))
                            found = True
                            break

                if not found:
                    print(f"   ✗ {known}: NOT detected")

            # Print detected residues
            if detected:
                print(f"   ✓ Detected: ", end="")
                for res, score in detected:
                    print(f"{res}(score={score:.2f}) ", end="")
                print()

                accuracy = len(detected) / len(known_residues) if known_residues else 0
                print(f"    Detection rate: {accuracy:.1%}")

    def analyze_multiple_cutoffs(self, segid1, segid2):
        """Test different distance cutoffs"""
        print(f"\n{'='*80}")
        print(f" Testing Multiple Cutoffs: {segid1} ↔ {segid2}")
        print(f"{'='*80}")

        for cutoff in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
            print(f"\n📏 Cutoff = {cutoff}Å")
            results = self.analyze_interface(segid1, segid2, cutoff)

            if results['chain1'] and results['chain2']:
                print(f"   • Chain {segid1}: {len(results['chain1'])} residues")
                print(f"   • Chain {segid2}: {len(results['chain2'])} residues")

                # Show top residue for each chain
                if results['chain1']:
                    top1 = results['chain1'][0]
                    print(f"   • Top in {segid1}: {top1['residue']} (score={top1['score']:.3f})")
                if results['chain2']:
                    top2 = results['chain2'][0]
                    print(f"   • Top in {segid2}: {top2['residue']} (score={top2['score']:.3f})")

# === BENCHMARK FUNCTION ===
def run_benchmark(pdb_id, segid1='A', segid2='B', cutoff=4.0):
    """Run benchmark on a PDB structure"""
    print(f"\n{'='*80}")
    print(f" BENCHMARK: {pdb_id}")
    print(f"{'='*80}")

    # Download PDB
    import urllib.request
    pdb_file = f"/content/{pdb_id}.pdb"

    if not os.path.exists(pdb_file):
        print(f" Downloading {pdb_id} from RCSB...")
        try:
            urllib.request.urlretrieve(
                f"https://files.rcsb.org/download/{pdb_id}.pdb",
                pdb_file
            )
            print(f" Downloaded {pdb_id}.pdb")
        except:
            print(f" Failed to download {pdb_id}")
            return None

    # Create a minimal PSF from PDB (simplified for benchmarking)
    # For real use, you'd need a proper PSF file
    print(" Note: Using simplified PSF generation for benchmarking")

    # For now, create a dummy PSF file
    dummy_psf = f"/content/{pdb_id}_dummy.psf"
    with open(dummy_psf, 'w') as f:
        f.write("PSF\n")
        f.write("Dummy PSF for benchmarking\n")

    try:
        # Run TPS-Lite
        tps = TPS_Lite_Fixed(dummy_psf, pdb_file)
        results = tps.analyze_interface(segid1, segid2, cutoff)

        if results:
            tps.print_results(results, top_n=8)

            # Known residues for validation
            known_residues = {}
            if pdb_id == '1BRS':  # Barnase-Barstar
                known_residues = {'A': ['D39', 'R59', 'E73', 'R87'], 'B': ['D35', 'D39', 'R59']}
            elif pdb_id == '1F8S':  # Trypsin-BPTI
                known_residues = {'E': ['D189', 'S190', 'G216'], 'I': ['K15', 'R17']}
            elif pdb_id == '1JRH':  # Antibody-Lysozyme
                known_residues = {'H': ['Y33', 'Y50'], 'L': ['D101', 'G102']}

            if known_residues:
                tps.validate_with_known_residues(results, known_residues)

        return results
    except Exception as e:
        print(f" Error analyzing {pdb_id}: {e}")
        return None

# === MAIN EXECUTION ===
def main():
    print("\n" + "="*80)
    print(" TPS-Lite FIXED - Protein Interface Analysis")
    print("="*80)
    print("\nChoose an option:")
    print("1. Analyze your own PSF/PDB files")
    print("2. Run benchmark on known complexes")
    print("3. Both")

    choice = input("\nEnter choice (1, 2, or 3): ").strip()

    if choice in ['1', '3']:
        # ============ USER INPUTS ============
        PSF_FILE = "/content/ionized.psf"  # Your PSF file
        PDB_FILE = "/content/ionized.pdb"  # Your PDB file
        SEGID1 = "PROA"  # First chain/segid
        SEGID2 = "HETA"  # Second chain/segid
        CUTOFF = 3.0     # Interface cutoff
        # =====================================

        # Check files
        if not os.path.exists(PSF_FILE):
            print(f" PSF not found: {PSF_FILE}")
            print("   Please upload your PSF file to Colab")
            return

        if not os.path.exists(PDB_FILE):
            print(f" PDB not found: {PDB_FILE}")
            print("   Please upload your PDB file to Colab")
            return

        # Run analysis
        print(f"\n{'='*80}")
        print(f" Analyzing: {os.path.basename(PDB_FILE)}")
        print(f"{'='*80}")

        tps = TPS_Lite_Fixed(PSF_FILE, PDB_FILE)

        # Test multiple cutoffs
        tps.analyze_multiple_cutoffs(SEGID1, SEGID2)

        # Main analysis with chosen cutoff
        print(f"\n{'='*80}")
        print(f" MAIN ANALYSIS (cutoff={CUTOFF}Å)")
        print(f"{'='*80}")

        results = tps.analyze_interface(SEGID1, SEGID2, CUTOFF)

        if results:
            tps.print_results(results)

            # Save results
            output_file = f"tps_lite_{SEGID1}_{SEGID2}.txt"
            with open(output_file, "w") as f:
                f.write(f"TPS-Lite Results: {SEGID1} ↔ {SEGID2}\n")
                f.write(f"PDB: {os.path.basename(PDB_FILE)}\n")
                f.write(f"Cutoff: {CUTOFF}Å\n")
                f.write("="*80 + "\n\n")

                if results.get('chain1'):
                    f.write(f"CHAIN {SEGID1} RESULTS:\n")
                    f.write("-"*40 + "\n")
                    for res in results['chain1'][:20]:
                        f.write(f"{res['residue']}\t{res['score']:.3f}\t{res['type']}\t{res['contacts']}\n")

                if results.get('chain2'):
                    f.write(f"\nCHAIN {SEGID2} RESULTS:\n")
                    f.write("-"*40 + "\n")
                    for res in results['chain2'][:20]:
                        f.write(f"{res['residue']}\t{res['score']:.3f}\t{res['type']}\t{res['contacts']}\n")

            print(f"\n Results saved to: {output_file}")

        print(f"\n Analysis complete!")

    if choice in ['2', '3']:
        # Run benchmarks
        print(f"\n{'='*80}")
        print(f" RUNNING BENCHMARKS")
        print(f"{'='*80}")

        benchmark_pdbs = ['1BRS', '1F8S', '1JRH']  # Barnase, Trypsin, Antibody

        for pdb_id in benchmark_pdbs:
            run_benchmark(pdb_id, cutoff=4.0)
            print("\n")

# Run the main function
if __name__ == "__main__":
    main()
