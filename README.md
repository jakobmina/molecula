# Molecula - Molecule ID From Mass Spectra (Enveda CASMI 2026) with H7 Framework
**Current Project Status and Delivery Summary**

**Project:** Molecula - Competition Launch: Enveda CASMI 2026 - Molecule ID From Mass Spectra  
**Snapshot date:** September 20, 2026  
**Status:** Integration and validation in progress  
**Repository branch:** `main` tracking `origin/main`

---

## 1. Overview & Project Scope
## Author's Declaration & Methodology Constraint

**I am competing in this challenge as a solo developer.** 

As a self-imposed constraint rooted in my prior thesis work on the H7 Framework and QuoreMind architecture, **I am strictly limiting my solutions to quantum computing implementations and metriplectic dynamical systems.** 

This project intentionally bypasses standard classical machine learning models (such as classical random forests, standard deep learning classifiers, or baseline GNNs). Every molecular candidate generation, feature extraction, and candidate ranking step must pass through the quantum neural network (QNN) embeddings, Z7 topological routing, and $SU(2)$ state representations defined by the H7 layer. This constraint is designed to test the absolute real-world viability of these quantum-architectural concepts against noisy, empirical mass spectrometry data.
Molecula is an experimental software project that combines a modular H7 representation layer with a molecular-data and mass-spectrometry workflow. The main goal is to map experimental tandem mass spectrometry ($MS/MS$) precursor $m/z$ values, collision energies, and peak intensity spectra to candidate chemical structures represented as **SMILES** (Simplified Molecular Input Line Entry System).

The H7 layer maps deterministic inputs into Z7 sectors, complementary values, continuous phase-derived quantities, SU(2)-related state representations, audit metrics, and compact JSON/binary records. This framework integrates:
1. **Data Exploration & Ingestion Pipelines:** Streaming and chunking large Parquet files.
2. **H7 Unified Framework & Algebra:** Mathematical constructs for algebraic state evolution and topological priors.
3. **Quantum Circuit & Hash Embeddings:** Converting spectrographic and text seeds into quantum statevectors and Metriplectic state representations.
4. **C & Python Interoperability:** High-performance C loaders bridging Metriplectic states and external reasoning models (e.g., Ollama / Llama3.2).

**Interpretation Boundary:** The project does **not** present H7 labels as new physical theory or as direct chemical evidence. Terms such as *vacuum*, *node*, *pair*, and *virtual isotope* are computational conventions. Molecular identification must remain grounded in validated precursor, fragmentation, library, and structural data. H7 values are deterministic features, routing context, and audit annotations.

---

## 2. Problem Statement

The challenge is to create an understandable, high-level interface for molecular records and mass spectra while preserving a traceable H7 computational representation. The system must not only generate a response but ensure it is tied to the evidence supplied.

For a real MS/MS identification request, the input contract must include:
1. Numeric precursor m/z;
2. Ionization mode, adduct, and charge;
3. One or more arrays of paired fragment m/z and normalized intensity values;
4. Collision method and energy for each MS2 spectrum;
5. Mass tolerances and instrument metadata; and
6. Validated library hits when a structural candidate is reported.

The H7 layer can annotate this record, but it must not fabricate missing precursor values, formulas, library accessions, or SMILES strings.

---

## 3. Repository Structure & Key Modules

### Directory Tree
```text
.
├── README.md                     # Project documentation
├── requirements.txt              # Python package dependencies
├── LICENSE                       # Project license
├── train.csv / test.parquet      # Dataset placeholders / references
├── submissions.csv               # Output submission file
│
├── explorar_datos.py             # Data exploration & parquet batch processing script
│
├── h7_unified_v2.py              # Core H7 mathematical framework (Node, Braid, Jacobi, Engine)
├── h7_bridge.py                  # Bidirectional bridge between H7 Python objects and C Metriplectic CTypes
├── h7_qnn_hash.py                # Quantum circuit generation & Metriplectic hash synthesis
├── h7_qnn_ollama.py              # QNN pipeline integrated with LLM reasoning via Ollama
├── h7_qnn_batch_nodesort.py      # Batch processing and node sorting algorithms
├── utf8_qnn_poc.py               # Proof-of-Concept for mapping UTF-8 character bytes to quantum state amplitudes
│
├── metriplectic.h                # C header defining MetriplecticState, Cuaternion, & TorsionObservables
├── qnn_loader.c                  # C binary loader and raw-socket Ollama HTTP client
└── his-torial/                   # Historical binary stream exports and state checkpoints (.bin, .json)

```

### Core Components

| File | Role | Current evidence |
| --- | --- | --- |
| `h7_unified_v2.py` | Core H7 framework: `H7Node`, braid and Jacobi components, subgroup covariance, Mahalanobis-based controller. | 978 lines; primary computational module. |
| `h7_qnn_ollama.py` | Interactive multiline interface, input-to-seed conversion, H7 audit logging, dynamic Ollama parameters. | 315 lines. Runs via `python3 -m h7_qnn_ollama`. |
| `h7_bridge.py` | `ctypes` data structures plus JSON and binary export of H7 state, torsion, and QNN-grid information. | 324 lines; emits per-query artifacts. |
| `h7_qnn_hash.py` | Metriplectic hash generation and QNN-step utilities. | 156 lines. |
| `h7_qnn_batch_nodesort.py` | Batch tokenization, covariance-based node ordering, prompt construction, and batch inference. | 279 lines. |
| `utf8_qnn_poc.py` | UTF-8 character amplitudes, QNN cipher circuit, and text-to-seed proof of concept. | 114 lines. |
| `explorar_datos.py` | Dataset exploration utility (parquet datasets, MS/MS peaks, adducts). | Present and locally modified. |
| `qnn_loader.c` & `metriplectic.h` | Native loader and C declarations for the metriplectic layer. | 421 C lines and 85 header lines. |

---

## 4. Installation & Usage Guide

### Requirements

* Python 3.10+
* `gcc` or standard C compiler
* Python dependencies: `numpy`, `scipy`, `sympy`, `qiskit`, `requests`, `pyarrow`
* (Optional) Local [Ollama](https://ollama.ai/?utm_source=gemini) instance running `llama3.2:latest` on port `11434`.

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd molecula

# Install Python dependencies
pip install -r requirements.txt

# Compile C components (Optional)
gcc -O2 -o qnn_loader qnn_loader.c -lm

```

### Quick Start Commands

* **Minimal H7 Inspection:**
`python3 -c "from h7_unified_v2 import H7Node; print(H7Node(15).summary(verbose=True))"`
* **Syntax Check:**
`python3 -m py_compile h7_bridge.py h7_qnn_hash.py h7_unified_v2.py utf8_qnn_poc.py h7_qnn_batch_nodesort.py explorar_datos.py`
* **Dataset Inspection:**
`python3 explorar_datos.py`
* **Export H7 States to C Binary:**
`python3 h7_bridge.py`
* **Load Binary States in C:**
`./qnn_loader h7_state.bin`
* **Test UTF-8 QNN Mapping:**
`python3 utf8_qnn_poc.py`

### Interactive H7 & Ollama Path

```bash
python3 -m h7_qnn_ollama

```

*Input Example for MS/MS Request:*

```text
molecule_id: m_005e53
Spectrum 1: ionization=positive, adduct=[M+H]+, precursor_mz=xxx.xxxx, collision_energy=20 eV
peaks (m/z:intensity): 85.0284:100 ; 113.0233:45 ; 141.0182:12 ; ...
Spectrum 2: ionization=positive, adduct=[M+H]+, precursor_mz=xxx.xxxx, collision_energy=40 eV
peaks (m/z:intensity): 57.0335:100 ; 85.0284:60 ; ...

Reference library hits (precursor within 10 ppm, ranked by cosine similarity):
1. SMILES=..., formula=..., cosine=0.91
...

```

---

## 5. System Architecture: Working H7 Flow

```text
text or structured input
  -> UTF-8 / QNN-derived integer seed
  -> H7Node and Z7 sector assignment
  -> SVD-free audit distance plus Mahalanobis trace
  -> dynamic generation parameters for Ollama
  -> JSON and binary bridge exports
  -> response text and per-character stream artifacts

```

The interactive path records critical audit values including:

* Seed and Z7 node.
* `m_star`, symplectic, and metriplectic quantities.
* Covariance proxy, asymmetry, and Mahalanobis distance.
* Dynamic LLM parameters (temperature, top-p, repetition penalty, top-k).
* JSON, binary, and stream artifacts stored under `his-torial/`.

---

## 6. Implementation Metrics & Data Artifacts

**Source Size:** Total tracked source is approximately 2,675 lines across 8 primary files.

**Data Artifacts & Observations:**

| Resource | Current local state | Interpretation |
| --- | --- | --- |
| `his-torial/` | ~5,436 files (`hash_n*.json`, `hash_n*.bin`, stream files). | Runtime trace artifacts; generated outputs, not training labels. |
| `h7_state.bin` | Present. | Binary state artifact exported by the bridge. |
| `train.csv` | ~6.2 MB; wavelength-like fields. | **Not** the expected MS/MS training schema. Must be separated. |
| `submissions.csv` | ~44 KB; `molecule_id` and `smiles`. | Candidate output file requiring chemical validation. |
| `train.parquet` | *Reported previously as 2,539,608 MS/MS rows.* | Currently absent. Must be restored before training. |

---

## 7. Project Status: Progress & Risks

### Achievements & Verified Progress

* **Consolidated Core:** A modular H7 implementation consolidated into a substantial core module.
* **Interactive App:** Connects text-derived seeds, H7 audits, exports, and a local LLM endpoint.
* **Modularity:** Distinct batch and bridge pathways rather than a monolithic script.
* **Traceability:** Runtime exports demonstrate that per-query traceability artifacts are reliably created.
* **Compilation:** Principal Python modules compile successfully in the local environment.

### Known Gaps & Risks

* **Data Alignment:** The local `train.csv` is incorrect. The reported `train.parquet` corpus (2.5M rows) is absent and must be restored to validate training claims.
* **LLM Control:** General chemistry questions route incorrectly into MS/MS flows. Placeholders in prompts can lead to malformed chemical text and unsupported assertions.
* **Validation:** `submissions.csv` strings lack established chemical validity/provenance.
* **Repository Hygiene:** Generated `his-torial/` artifacts are mixed into the working tree. No dedicated automated test suite or dependency manifest exists yet.

---

## 8. Roadmap & Next Steps

### Immediate Actions

1. Restore/mount the `train.parquet` MS/MS dataset.
2. Separate the wavelength-style `train.csv` from molecular training inputs.
3. Add a dependency manifest (`requirements.txt`) and `.gitignore` policy for `his-torial/`.
4. Implement an intent router for educational vs. MS/MS-identification requests.

### Short & Medium Term

* **Data Contract:** Implement the canonical MS/MS JSON evidence contract.
* **Validation:** Validate SMILES with a chemistry parser before reporting.
* **Retrieval:** Add deterministic library retrieval and candidate de-duplication.
* **Testing:** Train on grouped molecular splits to prevent leakage. Add regression tests for baseline chemical logic.
* **Evaluation:** Compare spectral similarity baselines with/without H7-derived features. Calibrate abstention thresholds.

### Quality Checklist

* [x] Core H7, bridge, batch, seed, interactive, and native loader sources present.
* [x] Python modules pass local syntax checks.
* [ ] Dependency versions declared in reproducible manifest.
* [ ] Dedicated automated test suite implemented.
* [ ] Generated history artifacts separated from source control.
* [ ] Precursor fields, adducts, and collision metadata validated against dataset schema.
* [ ] SMILES canonicalized; outputs cite real library records and similarity values.

---

## 9. Recommended Learning Path

* **Level 1 - Orientation:** Read this document. Run `H7Node(15).summary(verbose=True)` and test the interactive module with a conceptual question to inspect the H7 trace.
* **Level 2 - Core Implementation:** Study `h7_unified_v2.py` (`H7Node`), `h7_bridge.py` (exports), and `utf8_qnn_poc.py` (text-to-seed).
* **Level 3 - Molecular Integration:** Restore the MS/MS parquet corpus, validate schema, write an evidence serializer, and build deterministic candidate retrieval.
* **Level 4 - Validation & Extension:** Add unit tests, grouped data splits, evaluate retrieval quality, and document performance claims.

---

## 10. Conclusion

Molecula is an active integration-stage project with a functioning H7 computational path and a clear route toward molecular-spectrum applications. The current priority is disciplined data alignment and validation: enforcing an evidence-first schema, routing conversational intent correctly, and ensuring all chemical claims derive from experimentally validated records.

## License

This project is licensed under the terms included in the `LICENSE` file.

```
