# Perturbation Scanning (PS) 

#Before using
PerturbationScanning needs:
-  .pt file from md-graph-converter: https://github.com/fodil13/md-graph-converter
-  .pth trained model from Stability Interface trainer: https://github.com/fodil13/stability-interface-TRAINER
-  any question? contact me: azzaz.fodil@gmail.com

##  AI-Driven Biomolecular Interface Analysis & Design

Perturbation Scanning (PS) is a biologically grounded interpretability framework for Graph Neural Networks (GNNs) applied to protein–protein interfaces.

Rather than relying on abstract gradients or attention weights, PS identifies functionally critical residues by measuring how biophysically meaningful perturbations alter the model’s prediction of interface strength.

Key idea:
If perturbing a residue in a biologically realistic way strongly changes the model output, that residue is functionally important.

> **Paper:** "An AI-Driven Platform for Deconstructing and Engineering Biomolecular Recognition"  
> **Preprint:** [DOI 10.64898/2025.12.09.692808](https://doi.org/10.64898/2025.12.09.692808)  
> 

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

### Advanced: Custom Temporal Staging
```python
stage_boundaries = {
    'initial_encounter': (0.0, 0.15),
    'docking': (0.15, 0.4),
    'optimization': (0.4, 0.8),
    'final_complex': (0.8, 1.0)
}
```

### Batch Processing
```python
# Analyze multiple interfaces automatically
interface_pairs = detect_interface_pairs_from_graph("graphs.pt")
# Returns: [("PROA", "PROB"), ("PROA", "PROC"), ("PROB", "PROC")]
```

---

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

BibTeX:
```bibtex
@article{azzaz2025perturbation,
  title={An AI-Driven Platform for Deconstructing and Engineering Biomolecular Recognition},
  author={Azzaz, Fodil and Fantini, Jacques},
  journal={Preprint},
  year={2025},
  doi={10.64898/2025.12.09.692808}
}
```

---

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

##  Related Resources

1. **Paper:** [DOI 10.64898/2025.12.09.692808](https://doi.org/10.64898/2025.12.09.692808)
2. **Graph Converter:** [github.com/fodil13/md-graph-converter](https://github.com/fodil13/md-graph-converter)
3. **Example Datasets:** Available upon request

---

**Perturbation Scanning transforms molecular recognition from observational science to engineering discipline.**

---
*Fodil Azzaz, PhD
