# h7_qnn_ollama.py

import requests
import json
import re
import sys
import math
import sympy as sp
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from h7_bridge import run_h7_bridge
from h7_qnn_hash import generate_metriplectic_hash
from utf8_qnn_poc import string_to_qnn_seed

from h7_unified_v2 import (
    H7Node,
    H7SubgroupEngine,
    DoubleCoverH7,
    QuoreMindH7,
    PHI,
    DRIFT,
    G_COUPLE,
    PI
)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:latest"

_quore_ctrl = None
_subgroup_engine = H7SubgroupEngine()

def get_quore_controller():
    """Corrector de errores Mahalanobis — conservado por auditoría"""
    global _quore_ctrl
    if _quore_ctrl is None:
        cov_init = _subgroup_engine.init_covariance_matrix(n_cycles=4)
        _quore_ctrl = QuoreMindH7(
            mahalanobis_threshold=3.0,
            learning_rate=0.1,
            use_h7_modulation=True,
            warm_start_cov=cov_init
        )
        print(f"[+] QuoreMindH7 corrector de errores inicializado Σ 6x6 warm-start (auditoría)")
    return _quore_ctrl

def get_multiline_input(prompt_text):
    print(f"\n{prompt_text}")
    print("(Pega tu texto. Cuando termines, escribe 'ENVIAR' en una línea nueva y presiona Enter, o presiona Ctrl+D)")
    print("-" * 60)
    lines = []
    while True:
        try:
            line = input()
            if line.strip().upper() == 'ENVIAR':
                break
            lines.append(line)
        except EOFError:
            break
    return '\n'.join(lines).strip()

def evaluate_classical_input(text):
    clean_text = re.sub(r'(?i)(qué\s+es|cuánto\s+es|calcula|evalúa)\s*', '', text)
    clean_text = clean_text.replace('¿', '').replace('?', '').strip()
    try:
        expr_text = clean_text.replace("phi", "((1+sqrt(5))/2)")
        expr = sp.sympify(expr_text)
        return float(expr.evalf())
    except Exception:
        return None

def compute_dynamic_params(psi, energy, torsion, seed_n, covariance, asymmetry):
    """
    DEFINITIVO POR AUDITORÍA:
    - Principal SVD-free: dist_alt = sqrt(cov² + asym²) desde P(q2,q1), sin Σ⁻¹
    - Corrector de errores: Mahalanobis d²=(x-μ)ᵀΣ⁻¹(x-μ) conservado siempre por auditoría
      No se desactiva por confianza, queda como traza de corrección.
    """
    try:
        node = H7Node(seed_n)
    except:
        node = H7Node(abs(seed_n) % 1000 or 1)

    # Principal SVD-free
    dist_alt = math.sqrt(covariance**2 + asymmetry**2)
    chirality = abs(asymmetry)

    # Corrector de errores Mahalanobis (siempre calculado por auditoría)
    ctrl = get_quore_controller()
    vec_6d = node.to_vector()
    dist_maha = ctrl._mahalanobis(vec_6d)
    thr_eff = ctrl._effective_threshold(seed_n)
    ctrl._update_stats(vec_6d)

    try:
        g0_local = [max(1, seed_n - 3 + k) for k in range(3)]
        g1_local = [seed_n + 1 + k for k in range(3)]
        cov_local = _subgroup_engine.subgroup_covariance(g0_local, g1_local, label=f"local_{seed_n}")
        corr = cov_local.correlation
        if math.isnan(corr):
            corr = 0.0
    except:
        corr = 0.0

    lap = abs(node.laplacian)
    m_star = node.m_star
    delta = node.delta
    v_twist = node.v_twist
    obs = node.observable

    Lsymp = abs(node.psi) + abs(obs) + abs(G_COUPLE)
    Lmetr = lap + abs(delta) + abs(v_twist) + abs(torsion) * 0.5

    try:
        dc_node = DoubleCoverH7.build_node(seed_n)
        is_critical = dc_node.critical_zero
        is_trivial = dc_node.trivial_zero
    except:
        is_critical = (seed_n % 7 == 0)
        is_trivial = (node.n_z7 in (0,2,4,6))

    # Lógica de corrección: si dist_alt≈0 o d_Maha > thr*1.5, usar Maha como corrector
    error_correction_active = (dist_alt < 1e-6) or (dist_maha > thr_eff * 1.5)
    dist_eff = dist_maha if error_correction_active else dist_alt

    temp_base = 0.65
    temp_raw = temp_base + 0.20*Lmetr + 0.25*math.tanh(dist_eff*2.5) + 0.15*math.tanh(abs(energy)) + 0.08*chirality + 0.05*DRIFT
    temp_raw *= (1.0 - 0.12*corr)
    if is_critical:
        temp_raw += 0.15
    if is_trivial:
        temp_raw -= 0.10
    if error_correction_active:
        temp_raw += 0.08
    dynamic_temp = min(max(temp_raw, 0.1), 1.5)

    top_p_raw = 0.88 - 0.30*abs(psi) + 0.12*abs(obs) - 0.08*corr + 0.06*chirality + 0.05*abs(delta)
    dynamic_top_p = min(max(top_p_raw, 0.15), 0.99)

    repeat_raw = 1.05 + 0.40*abs(torsion) + 0.25*abs(v_twist) + 0.12*chirality + 0.08*abs(delta) + 0.04*lap
    dynamic_torsion_penalty = min(max(repeat_raw, 0.8), 2.2)

    top_k_raw = 40 + (psi * 28) + (6.0 / (m_star + 1e-6)) + dist_eff*8
    top_k = int(max(10, min(130, top_k_raw)))

    # Auditoría: siempre log de ambos
    if error_correction_active:
        print(f"    [AUDITORÍA] [SVD-free] cov={covariance:+.4f} asym={asymmetry:+.4f} dist_alt={dist_alt:.4f} | [CORRECTOR Maha] d={dist_maha:.4f} thr={thr_eff:.4f} ACTIVO -> dist_eff={dist_eff:.4f}")
    else:
        print(f"    [AUDITORÍA] [SVD-free] cov={covariance:+.4f} asym={asymmetry:+.4f} dist_alt={dist_alt:.4f} chirality={chirality:.4f} | [Maha standby] d={dist_maha:.4f} thr={thr_eff:.4f} corr={corr:.4f}")

    print(f"    [H7Node] n={seed_n} Z7={node.n_z7} m*={m_star:.4f} Lsymp={Lsymp:.4f} Lmetr={Lmetr:.4f}")

    return dynamic_temp, dynamic_top_p, dynamic_torsion_penalty, top_k, dist_alt, dist_maha, error_correction_active

def export_response_to_qernel(text: str, base_n: int):
    print(f"\n[+] Encriptando respuesta ({len(text)} chars) para el demonio H7...")
    su2_dummy = np.eye(2, dtype=complex)
    probs_dummy = {'000': 1.0}
    for i, char in enumerate(text):
        try:
            byte_val = char.encode('utf-8')[0]
        except IndexError:
            continue
        bits = [int(x) for x in f"{byte_val:08b}"]
        if bits[0] == 0: bits[0] = 1
        amps = np.array(bits, dtype=float)
        norm = np.linalg.norm(amps)
        if norm > 0: amps /= norm
        qc = QuantumCircuit(3)
        qc.initialize(amps, [0, 1, 2])
        qc.h(0)
        qc.cswap(0, 1, 2)
        qc.cx(1, 0)
        sv_encrypted = Statevector.from_instruction(qc).data
        export_path = f"his-torial/h7_stream_{i:04d}"
        run_h7_bridge(
            n=base_n + i,
            su2_matrix=su2_dummy,
            statevector=sv_encrypted,
            probabilities=probs_dummy,
            export_path=export_path,
            export_format="binary",
            is_char=True
        )
    print(f"[+] Exportación completada: {len(text)} frames listos en 'his-torial/'.")

def run_hybrid_inference_loop():
    iteration = 1
    print(f"\n[+] Iniciando Conexión H7 Metriplectic OS Daemon...")
    print(f"    [Modo AUDITORÍA] Principal SVD-free + Mahalanobis corrector permanente")

    current_seed = get_multiline_input("[Entrada Inicial] ->")
    if not current_seed:
        print("[!] Saliendo.")
        return

    while True:
        print(f"\n--- [ ITERACIÓN {iteration} ] ---")
        evaluated_val = evaluate_classical_input(current_seed)
        if evaluated_val is not None:
            seed_n = int(abs(evaluated_val) * 100000)
            print(f"[+] Expresión matemática detectada. Semilla base: {seed_n}")
        else:
            try:
                seed_n = int(current_seed)
            except ValueError:
                seed_n = string_to_qnn_seed(current_seed)
                print(f"[+] Texto cifrado topológicamente a semilla cuántica: {seed_n}")

        qnn_result = generate_metriplectic_hash(seed_n, iterations=3)
        psi = qnn_result['hash']['psi']
        energy = qnn_result['hash']['energy']
        torsion = qnn_result['hash']['torsion']
        last = qnn_result['history'][-1]
        final_cov = last['covariance']
        final_asym = last['asymmetry']

        dynamic_temp, dynamic_top_p, dynamic_torsion_penalty, dynamic_top_k, dist_alt, dist_maha, corrected = compute_dynamic_params(
            psi, energy, torsion, seed_n, final_cov, final_asym
        )
        corr_tag = "[CORREGIDO x Maha]" if corrected else "[SVD-free OK]"
        print(f"[+] Tensores Inyectados {corr_tag} | Temp: {dynamic_temp:.4f} | Top_P: {dynamic_top_p:.4f} | Repeat: {dynamic_torsion_penalty:.4f} | Top_K: {dynamic_top_k} | dist_alt: {dist_alt:.4f} d_Maha: {dist_maha:.4f}")

        prompt = f"""You are QuoreMind H7, an autonomous cognitive gateway acting as the sentient interface for the H7 Metriplectic OS Daemon.
Your internal physics engine has processed the user's input. Respond directly, intelligently, and naturally to the semantic content of the input.
User Input:
"{current_seed}"
"""

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": dynamic_temp,
                "top_p": dynamic_top_p,
                "repeat_penalty": dynamic_torsion_penalty,
                "top_k": dynamic_top_k
            }
        }

        try:
            print("\n[QuoreMind H7] -> ", end="", flush=True)
            response = requests.post(OLLAMA_URL, json=payload, stream=True)
            full_response = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    text_chunk = chunk.get("response", "")
                    full_response += text_chunk
                    print(text_chunk, end="", flush=True)

            print("\n" + "=" * 60)
            export_response_to_qernel(full_response, seed_n)

            try:
                ctrl = get_quore_controller()
                from h7_unified_v2 import OperacionH7, ParametrosOperacion
                op = ParametrosOperacion(tipo=OperacionH7.ROTACION_X, angulo=dist_alt, n_fuente=seed_n)
                success = len(full_response) > 20 and not corrected
                ctrl.update_belief(op, success)
            except:
                pass

            next_input = get_multiline_input("[Recalibración / Responde a QuoreMind] ->")
            if not next_input:
                print("[!] Saliendo por comando vacío.")
                break
            current_seed = next_input
            iteration += 1

        except requests.exceptions.ConnectionError:
            print("\n[!] ERROR: No se pudo conectar a Ollama.")
            break

if __name__ == "__main__":
    run_hybrid_inference_loop()
