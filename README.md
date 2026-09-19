# molecula Molecule ID From Mass Spectra (Enveda CASMI 2026)


An advanced framework combining Quantum Neural Network (QNN) representations, H7 mathematical dynamics, and Metriplectic systems to identify small chemical structures (SMILES) from tandem mass spectrometry (MS/MS) data.

---

## Overview

This repository provides tools and data processing pipelines developed for the **Competition Launch: Enveda CASMI 2026 - Molecule ID From Mass Spectra**.

The main goal is to map experimental tandem mass spectrometry ($MS/MS$) precursor $m/z$ values, collision energies, and peak intensity spectra to candidate chemical structures represented as **SMILES** (Simplified Molecular Input Line Entry System).

To address this challenge, the repository integrates:
1. **Data Exploration & Ingestion Pipelines:** Streaming and chunking large Parquet files (`train.parquet`, `test.parquet`).
2. **H7 Unified Framework & Algebra:** Mathematical constructs (`H7Node`, `BraidH7`, `JacobiH7`, `QuoreMindH7`) for algebraic state evolution and topological priors.
3. **Quantum Circuit & Hash Embeddings:** Converting spectrographic and text seeds into quantum statevectors and Metriplectic state representations (`qiskit`, `metriplectic.h`).
4. **C & Python Interoperability:** High-performance C loaders (`qnn_loader.c`) bridging Metriplectic states and external reasoning models (e.g., Ollama / Llama3.2).

---

## Repository Structure

```
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

---

## Key Modules & Components

### 1. Data Ingestion & Exploration (`explorar_datos.py`)
- Reads parquet datasets efficiently using row group iterations via `pyarrow`.
- Extracts spectral features such as precursor $m/z$, base peak intensities, adducts, and $MS/MS$ peaks ($m/z$ and normalized intensity arrays).

### 2. H7 Unified Framework (`h7_unified_v2.py`)
- **`H7Node`**: Models extended golden-ratio operators, graph Laplacians, $SU(2)$ representations, and eigenvalues.
- **`BraidH7`**: Implements $B_3$ braid matrix representations ($8\times 8$), Yang-Baxter relations, and topological priors.
- **`JacobiH7`**: Computes modular theta functions $\vartheta(z|i\varphi)$ and functional equations along critical lines.
- **`H7SubgroupEngine`**: SVD-free covariance estimation across $G_0/G_1$ subgroups.
- **`QuoreMindH7`**: Adaptive decision-making via Mahalanobis distance and H7-guided Thompson Sampling.

### 3. C & Python Bridge (`h7_bridge.py`, `metriplectic.h`, `qnn_loader.c`)
- **`h7_bridge.py`**: Serializes Python H7 states into 152-byte C-compatible binary payloads (`h7_state.bin`).
- **`metriplectic.h`**: Header file defining algebraic structures (`Cuaternion`, `MetriplecticState`, `TorsionObservables`).
- **`qnn_loader.c`**: Low-level C binary loader that consumes exported state binaries and communicates with local LLM endpoints via raw C sockets.

### 4. Quantum Neural Network & Hashes (`utf8_qnn_poc.py`, `h7_qnn_hash.py`, `h7_qnn_ollama.py`)
- Maps text representations or spectral seeds to 8-bit quantum statevectors in `qiskit`.
- Generates Metriplectic hashes combining statevector amplitudes, operators, and Aer simulators.

---

## Requirements & Installation

### Prerequisites
- Python 3.10+
- `gcc` or any standard C compiler (for building `qnn_loader.c`)
- (Optional) Local [Ollama](https://ollama.ai/) instance running `llama3.2:latest` on port `11434` for LLM candidate generation.

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd molecula
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Compile C components (Optional):**
   ```bash
   gcc -O2 -o qnn_loader qnn_loader.c -lm
   ```

---

## Usage Guide

### 1. Dataset Inspection
To inspect the dataset structure, metadata, and row groups from parquet/csv files:
```bash
python3 explorar_datos.py
```

### 2. Exporting H7 States to C Metriplectic Binary Format
Run the bridge script to generate `h7_state.bin`:
```bash
python3 h7_bridge.py
```

### 3. Loading Binary States in C
Execute the compiled C loader to inspect serialized states:
```bash
./qnn_loader h7_state.bin
```

### 4. Running QNN & Hash Workflows
To test UTF-8 to quantum amplitude mappings:
```bash
python3 utf8_qnn_poc.py
```

To run the QNN Hash pipeline:
```bash
python3 h7_qnn_hash.py
```

To run the full QNN pipeline integrated with Ollama (ensure Ollama service is active):
```bash
python3 h7_qnn_ollama.py
```

---

## License

This project is licensed under the terms included in the `LICENSE` file.
modelo para Competition Launch: Enveda CASMI 2026 - Molecule ID From Mass Spectra
