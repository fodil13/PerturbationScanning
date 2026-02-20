# Perturbation Scanning (PS) 

Before using, PerturbationScanning needs:
-  .pt file from md-graph-converter: https://github.com/fodil13/md-graph-converter
-  .pth trained model from Stability Interface trainer: https://github.com/fodil13/stability-interface-TRAINER
-  any question? contact me: azzaz.fodil@gmail.com

# USAGE
On your own machin or using Google Colab. If you use Google Colab, copy paste each code in your notebook and then upload your files in your Google Drive. Then, set the correct path inside the code (everything is explained, but you are welcome to send a message if you request help). 
You will also need to install the following dependencies: 
- !pip install numpy
- !pip install MDAnalysis
- !pip install torch
- !pip install torch_geometric

# Other tools

You will find Perturbation Scanning-Lite (PS-Lite) and (IIOS) on the following links: 
- PS-Lite:
- IIOS:

##  AI-Driven Biomolecular Interface Analysis & Design

Perturbation Scanning (PS) is a biologically grounded interpretability framework for Graph Neural Networks (GNNs) applied to protein–protein interfaces.

Rather than relying on abstract gradients or attention weights, PS identifies functionally critical residues by measuring how biophysically meaningful perturbations alter the model’s prediction of interface strength.

Key idea:
If perturbing a residue in a biologically realistic way strongly changes the model output, that residue is functionally important.

> **Paper:** "An AI-Driven Platform for Deconstructing and Engineering Biomolecular Recognition"  
> **Preprint:** [DOI 10.64898/2025.12.09.692808](https://doi.org/10.64898/2025.12.09.692808)  
> 
---
## Key Features

- Six biologically motivated perturbation types
- Residue-level interpretability (not atom-level noise)
- Temporal staging (early / mid / late binding dynamics) ps: user can define more or less stages
- SUM Δ metric combining interaction strength and frequency
- Model-agnostic (works with any trained GNN)
- Fully compatible with molecular dynamics trajectories

---
## Biological Perturbation Types

Perturbation Scanning applies chemically meaningful modifications instead of arbitrary masking:

- Perturbation Type |	Biological Meaning
- Electrostatic	| Charge neutralization / inversion
- Hydrophobic | Polarity reversal of hydrophobic patches
- Steric |	Bulky side-chain substitutions
- Aromatic |	Disruption of π–π and CH–π interactions
- Hydrogen bond	| Collapse of H-bond networks
- Conformational | 	Local structural displacement

Each residue pair is scored using the maximum disruptive effect across perturbations.


---
##  Quick Start

### Prerequisites
- Python 3.8+
- PyTorch ≥1.12
- torch_geometric
- numpy, scipy, matplotlib


---

##  What PS Delivers

### 1. **Mechanistic Deconstruction**
- **Stage-resolved analysis** (early/mid/late binding)
- **Force-specific contributions** (electrostatic, hydrophobic, steric, aromatic, H-bond)
- **Sum Δ metric** = (Perturbation effect) × (Interaction frequency)

### 2. **Key Outputs**
```
PROTEIN    RESIDUE    Sum Δ      #Pairs    Percentile    Stage      PerturbationType
PROA       ARG-59     4.821      8        99.8%         early      ELECTROSTATIC
PROA       TYR-112    3.456      5        97.2%         mid        AROMATIC
PROD       ASP-35     2.891      6        95.1%         late       H-BOND
```

### 3. **Total Interface Strength**
PS calculates comprehensive interface metrics comparable to experimental measurements:
- **Early stage:** 15.23 Sum Δ
- **Mid stage:** 18.45 Sum Δ (+21.1%)
- **Late stage:** 16.78 Sum Δ (-9.0%)

---

### **Perturbation Scanning (PS)**
- **Core Framework:** Graph neural network with systematic perturbations
- **Perturbation Types:** Electrostatic, hydrophobic, steric, aromatic, H-bond, conformational
- **Input:** Molecular dynamics trajectories or static PDB structures
- **Output:** Residue-level, stage-resolved, force-specific contributions


---

##  Configuration Options

### Basic Settings (Single Analysis)
```python
# In main.py, set these parameters:
GRAPH_PATH = "path/to/your/graph_file.pt"
MODEL_PATH = "path/to/your/model_file.pth"
SEGID1 = "PROA"
SEGID2 = "PROD"
START_FRAME = 0
STEP = 1
TOTAL_FRAMES = 100
N_RUNS = 1
NUM_RESIDUES_DISPLAY = 10
```

##  Output Interpretation

### **Sum Δ Metric**
- **High Sum Δ:** Residue participates in **many interactions** with **strong effects** when perturbed
- **Low Sum Δ:** Residue either rarely interacts or has minimal effect when perturbed

### **Stage Dynamics**
- **Strengthening interface:** Sum Δ increases from early → late stages
- **Weakening interface:** Sum Δ decreases from early → late stages
- **Transient interactions:** High early Sum Δ, low late Sum Δ

### **Percentile Ranking**
- **>95%:** Critical residue—experimental mutation likely disruptive
- **80-95%:** Important contributor—consider for engineering
- **<50%:** Peripheral—unlikely to affect function if mutated

---


##  Citation

If you use PS or IIOS in your research, please cite:

```
Azzaz, F., & Fantini, J. (2025). An AI-Driven Platform for Deconstructing and Engineering 
Biomolecular Recognition. Preprint. https://doi.org/10.64898/2025.12.09.692808
```

##  License

**ACADEMIC USE:** ✅ Permitted - Research, teaching, publication  
**COMMERCIAL USE:** 🚫 Requires authorization - Contact azzaz.fodil@gmail.com  

See full license terms in the code header.

---

##  Support & Issues

- **Documentation:** This README + comments in code
- **Issues:** GitHub Issues tab
- **Contact:** azzaz.fodil@gmail.com
- **Related Tools:** [MD Graph Converter](https://github.com/fodil13/md-graph-converter)

---
Fodil Azzaz, PhD
