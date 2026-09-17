/**
 * h7_loader.c
 * Consumidor C del binary export de h7_bridge.py
 * Carga h7_state_nN.bin → MetriplecticState + TorsionObservables + EstadoCuantico
 * Compilar: gcc -o h7_loader h7_loader.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>   /* strncasecmp */
#include <ctype.h>     /* isxdigit */
#include <math.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "metriplectic.h"

/* ────────────────────────────────────────────────────────────
 * Cliente Ollama minimalista (sockets crudos, sin libcurl ni deps)
 * Cierra el loop del "pipeline C": h7_main.py exporta -> qnn_loader
 * decodifica -> qnn_loader le pregunta a Ollama sobre el estado.
 * ──────────────────────────────────────────────────────────── */
#define OLLAMA_HOST  "127.0.0.1"
#define OLLAMA_PORT  11434
#define OLLAMA_MODEL "llama3.2:latest"
#define OLLAMA_BUF_SIZE 65536

/* Layout binario: 152 bytes little-endian
   [0..6]  : MetriplecticState  (psi, v, energy, q.w, q.x, q.y, q.z)
   [7..10] : TorsionObservables (energy_density, entropy_gradient, spatial_torsion, chirality)
   [11..18]: EstadoCuantico.psi[8]
*/
#define BINARY_DOUBLES 19

typedef struct {
    MetriplecticState state;
    TorsionObservables torsion;
    EstadoCuantico quantum;
} H7FullExport;

/* ---- JSON: escapar un string para meterlo como valor entre comillas ---- */
static void json_escape(const char *src, char *dst, size_t dst_size) {
    size_t j = 0;
    for (size_t i = 0; src[i] != '\0' && j + 6 < dst_size; i++) {
        unsigned char c = (unsigned char)src[i];
        switch (c) {
            case '\"': dst[j++] = '\\'; dst[j++] = '\"'; break;
            case '\\': dst[j++] = '\\'; dst[j++] = '\\'; break;
            case '\n': dst[j++] = '\\'; dst[j++] = 'n';  break;
            case '\r': dst[j++] = '\\'; dst[j++] = 'r';  break;
            case '\t': dst[j++] = '\\'; dst[j++] = 't';  break;
            default:
                if (c < 0x20) j += (size_t)snprintf(dst + j, dst_size - j, "\\u%04x", c);
                else          dst[j++] = (char)c;
        }
    }
    dst[j] = '\0';
}

/* Codifica un code point BMP (sin pares surrogate / emoji) como UTF-8. */
static int utf8_encode(long code, char *dst) {
    if (code < 0x80) {
        dst[0] = (char)code; return 1;
    } else if (code < 0x800) {
        dst[0] = (char)(0xC0 | (code >> 6));
        dst[1] = (char)(0x80 | (code & 0x3F));
        return 2;
    } else {
        dst[0] = (char)(0xE0 | (code >> 12));
        dst[1] = (char)(0x80 | ((code >> 6) & 0x3F));
        dst[2] = (char)(0x80 | (code & 0x3F));
        return 3;
    }
}

/* ---- JSON: extraer y des-escapar un campo string simple "field":"valor" ---- */
static int extract_json_string_field(const char *json, const char *field,
                                      char *out, size_t out_size) {
    char key[64];
    snprintf(key, sizeof(key), "\"%s\"", field);
    const char *p = strstr(json, key);
    if (!p) return -1;
    p += strlen(key);
    while (*p == ' ' || *p == '\t') p++;   /* tolera "field" : "valor" */
    if (*p != ':') return -1;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    if (*p != '\"') return -1;
    p++;

    size_t j = 0;
    while (*p && *p != '\"' && j + 1 < out_size) {
        if (*p == '\\' && *(p + 1)) {
            p++;
            switch (*p) {
                case 'n': out[j++] = '\n'; break;
                case 't': out[j++] = '\t'; break;
                case 'r': out[j++] = '\r'; break;
                case '\"': out[j++] = '\"'; break;
                case '\\': out[j++] = '\\'; break;
                case '/':  out[j++] = '/';  break;
                case 'u':
                    if (isxdigit((unsigned char)p[1]) && isxdigit((unsigned char)p[2]) &&
                        isxdigit((unsigned char)p[3]) && isxdigit((unsigned char)p[4])) {
                        char hex[5] = { p[1], p[2], p[3], p[4], 0 };
                        long code = strtol(hex, NULL, 16);
                        /* Nota: no maneja pares surrogate (emoji fuera del BMP);
                           suficiente para texto en español devuelto por Ollama. */
                        int n = utf8_encode(code, out + j);
                        if (j + (size_t)n + 1 < out_size) j += (size_t)n;
                        p += 4;
                    } else {
                        out[j++] = '?';
                    }
                    break;
                default: out[j++] = *p; break;
            }
            p++;
        } else {
            out[j++] = *p++;
        }
    }
    out[j] = '\0';
    return 0;
}

/* ---- HTTP: ubica el body tras las cabeceras; desempaqueta chunked si aplica ---- */
static const char *extract_http_body(char *raw_response, char *scratch, size_t scratch_size) {
    char *sep = strstr(raw_response, "\r\n\r\n");
    if (!sep) return raw_response;
    char *body = sep + 4;

    int chunked = 0;
    size_t header_len = (size_t)(sep - raw_response);
    for (size_t i = 0; i + 26 <= header_len; i++) {
        if (strncasecmp(raw_response + i, "Transfer-Encoding: chunked", 27) == 0) {
            chunked = 1;
            break;
        }
    }
    if (!chunked) return body;

    size_t out_pos = 0;
    char *p = body;
    while (*p && out_pos < scratch_size - 1) {
        char *endptr;
        long chunk_size = strtol(p, &endptr, 16);
        if (endptr == p || chunk_size <= 0) break;
        p = endptr;
        while (*p == '\r' || *p == '\n') p++;
        for (long k = 0; k < chunk_size && out_pos < scratch_size - 1; k++)
            scratch[out_pos++] = p[k];
        p += chunk_size;
        while (*p == '\r' || *p == '\n') p++;
    }
    scratch[out_pos] = '\0';
    return scratch;
}

/* ---- HTTP: POST JSON crudo por socket TCP. 0 = ok (out_buf = respuesta
   completa con cabeceras), -1 = no se pudo conectar/enviar/recibir. ---- */
static int http_post_json(const char *host, int port, const char *path,
                           const char *json_body, char *out_buf, size_t out_buf_size) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return -1;

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) { close(sock); return -1; }

    struct timeval tv = { .tv_sec = 30, .tv_usec = 0 };
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        close(sock);
        return -1;
    }

    char request[OLLAMA_BUF_SIZE];
    int body_len = (int)strlen(json_body);
    int req_len = snprintf(request, sizeof(request),
        "POST %s HTTP/1.1\r\n"
        "Host: %s:%d\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n"
        "\r\n"
        "%s",
        path, host, port, body_len, json_body);

    if (req_len < 0 || send(sock, request, (size_t)req_len, 0) != req_len) {
        close(sock);
        return -1;
    }

    size_t total = 0;
    ssize_t n;
    while (total < out_buf_size - 1 &&
           (n = recv(sock, out_buf + total, out_buf_size - 1 - total, 0)) > 0) {
        total += (size_t)n;
    }
    out_buf[total] = '\0';
    close(sock);
    return (total > 0) ? 0 : -1;
}

/* ---- Construye el prompt desde el estado físico decodificado, se lo manda
   a Ollama (POST /api/generate, stream:false) e imprime la respuesta.
   Si Ollama no está corriendo, avisa claro y NUNCA truena el programa. ---- */
void ask_ollama(const H7FullExport *e, char decoded_char) {
    char char_desc[64];
    if (decoded_char >= 32 && decoded_char < 127) {
        snprintf(char_desc, sizeof(char_desc), "'%c' (ASCII %d)", decoded_char, (int)decoded_char);
    } else {
        snprintf(char_desc, sizeof(char_desc), "no imprimible (byte %d)", (int)(unsigned char)decoded_char);
    }

    char prompt[512];
    snprintf(prompt, sizeof(prompt),
        "Eres QuoreMind H7, la interfaz cognitiva del H7 Metriplectic OS Daemon. "
        "El motor de fisica acaba de procesar un estado con estos observables: "
        "psi=%.6f, energy=%.6f, torsion=%.6f, chirality=%.6f. "
        "El caracter decodificado del statevector fue %s. "
        "Responde en una o dos frases interpretando este estado.",
        e->state.psi, e->state.energy, e->torsion.spatial_torsion, e->torsion.chirality,
        char_desc);

    char prompt_escaped[768];
    json_escape(prompt, prompt_escaped, sizeof(prompt_escaped));

    char body[1024];
    snprintf(body, sizeof(body), "{\"model\":\"%s\",\"prompt\":\"%s\",\"stream\":false}",
             OLLAMA_MODEL, prompt_escaped);

    printf("\n[QuoreMind H7 -> Ollama] Enviando estado a %s (%s:%d)...\n",
           OLLAMA_MODEL, OLLAMA_HOST, OLLAMA_PORT);

    char raw_response[OLLAMA_BUF_SIZE];
    if (http_post_json(OLLAMA_HOST, OLLAMA_PORT, "/api/generate", body,
                        raw_response, sizeof(raw_response)) != 0) {
        printf("  [!] No se pudo conectar a Ollama en %s:%d.\n", OLLAMA_HOST, OLLAMA_PORT);
        printf("      Verifica que 'ollama serve' esté corriendo y que '%s' esté descargado.\n",
               OLLAMA_MODEL);
        return;
    }

    char body_scratch[OLLAMA_BUF_SIZE];
    const char *json_body = extract_http_body(raw_response, body_scratch, sizeof(body_scratch));

    char answer[4096];
    if (extract_json_string_field(json_body, "response", answer, sizeof(answer)) != 0) {
        printf("  [!] Ollama respondió pero no se encontró el campo \"response\".\n");
        printf("      Respuesta cruda (primeros 300 bytes):\n  %.300s\n", json_body);
        return;
    }

    printf("  %s\n", answer);
}

int h7_load_binary(const char *path, H7FullExport *out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[h7_loader] No se pudo abrir: %s\n", path); return -1; }

    double buf[BINARY_DOUBLES];
    size_t read = fread(buf, sizeof(double), BINARY_DOUBLES, f);
    fclose(f);

    if (read != BINARY_DOUBLES) {
        fprintf(stderr, "[h7_loader] Archivo incompleto: leídos %zu de %d doubles\n",
                read, BINARY_DOUBLES);
        return -2;
    }

    /* MetriplecticState */
    out->state.psi    = buf[0];
    out->state.v      = buf[1];
    out->state.energy = buf[2];
    out->state.q.w    = buf[3];
    out->state.q.x    = buf[4];
    out->state.q.y    = buf[5];
    out->state.q.z    = buf[6];

    /* TorsionObservables */
    out->torsion.energy_density   = buf[7];
    out->torsion.entropy_gradient = buf[8];
    out->torsion.spatial_torsion  = buf[9];   /* DRIFT_072 = 2π-7 */
    out->torsion.chirality        = buf[10];

    /* EstadoCuantico */
    for (int i = 0; i < HILBERT_DIM; i++)
        out->quantum.psi[i] = buf[11 + i];

    return 0;
}

void h7_print_export(const H7FullExport *e) {
    printf("\n=== H7 Export (Python → C) ===\n");
    printf("MetriplecticState:\n");
    printf("  psi    = %.8f  (cos(πφn) quasiperiod)\n", e->state.psi);
    printf("  v      = %.8f  (cos(πn)  parity)\n",      e->state.v);
    printf("  energy = %.8f  (Ψn classifier)\n",         e->state.energy);
    printf("  q      = (w=%.6f, x=%.6f, y=%.6f, z=%.6f)\n",
           e->state.q.w, e->state.q.x, e->state.q.y, e->state.q.z);

    printf("\nTorsionObservables:\n");
    printf("  energy_density   = %.8f\n", e->torsion.energy_density);
    printf("  entropy_gradient = %.8f  (Shannon normalizado)\n", e->torsion.entropy_gradient);
    printf("  spatial_torsion  = %.10f  (DRIFT_072 = 2π-7)\n", e->torsion.spatial_torsion);
    printf("  chirality        = %.8f  (paridad Z7)\n", e->torsion.chirality);

    printf("\nEstadoCuantico psi[8]:\n  ");
    for (int i = 0; i < HILBERT_DIM; i++)
        printf("%.4f ", e->quantum.psi[i]);
    printf("\n");

    /* golden_operator sobre psi como verificación cruzada */
    printf("\ngolden_operator check:\n");
    printf("  golden_operator(1) = %.8f\n", golden_operator(1));
    printf("  psi × v            = %.8f  (debe aproximar energy - v)\n",
           e->state.psi * e->state.v);
}

/* forward_pass stub: demonstra el flujo QNN con datos H7 */
void demo_forward_pass(const H7FullExport *e) {
    QNNGrid grid = initialize_qnn_grid();
    EstadoCuantico output;

    double On = golden_operator(1);  /* n=1 demo; en prod. pasar n real */
    forward_pass(&grid, (EstadoCuantico *)&e->quantum, &output, On);

    printf("\nQNN forward_pass output psi[8]:\n  ");
    for (int i = 0; i < HILBERT_DIM; i++)
        printf("%.4f ", output.psi[i]);
    printf("\n");
}

/* Codec UTF-8 a Cuántico */
char decode_char(EstadoCuantico estado) {
    double temp[8];
    for(int i=0; i<8; i++) temp[i] = estado.psi[i];
    
    // 1. Inversa de CX(1, 0)
    // Qubit 1 es control, Qubit 0 es target
    // Intercambia estados 2<->3 y 6<->7
    double swap_val;
    swap_val = temp[2]; temp[2] = temp[3]; temp[3] = swap_val;
    swap_val = temp[6]; temp[6] = temp[7]; temp[7] = swap_val;
    
    // 2. Inversa de CSWAP(0, 1, 2)
    // Qubit 0 es control, Qubit 1 y 2 targets
    // Intercambia estados 3<->5
    swap_val = temp[3]; temp[3] = temp[5]; temp[5] = swap_val;
    
    // 3. Inversa de H(0)
    // Mezcla pares (0,1), (2,3), (4,5), (6,7)
    for(int i=0; i<7; i+=2) {
        double a = temp[i];
        double b = temp[i+1];
        temp[i]   = (a + b) / sqrt(2.0);
        temp[i+1] = (a - b) / sqrt(2.0);
    }
    
    char out_char = 0;
    for(int i=0; i<8; i++) {
        if(temp[i] > 0.2) {
            // Reconstruimos el bit (ignorando el bit 0 que es el Flag de Seguridad).
            // Peso 2^(7-i): bits[i] en la codificación original (f"{byte:08b}")
            // pesa 2^(7-i) para i=1..7. El +1 que había aquí duplicaba cada
            // peso y corrompía el byte reconstruido.
            if(i != 0) {
                out_char |= (1 << (7 - i));
            }
        }
    }
    return out_char;
}

EstadoCuantico encode_char(char c) {
    // Stub for C-side encoding (feed-forward)
    EstadoCuantico ec = {0};
    // TODO: implement C-side initialization if needed
    return ec;
}

QNNGrid initialize_qnn_grid() {
    QNNGrid grid = {0};
    return grid;
}

void forward_pass(QNNGrid *grid, EstadoCuantico *input, EstadoCuantico *output, double On) {
    // Stub
    for(int i=0; i<8; i++) {
        output->psi[i] = input->psi[i] * On;
    }
}

int main(int argc, char *argv[]) {
    const char *path = (argc > 1) ? argv[1] : "his-torial/h7_state_char_A.bin";

    H7FullExport export_data;
    if (h7_load_binary(path, &export_data) != 0)
        return 1;

    h7_print_export(&export_data);
    demo_forward_pass(&export_data);
    
    // Intentar descifrar como carácter UTF-8
    char recuperado = decode_char(export_data.quantum);
    printf("\n[Decodificador UTF-8 Cuántico]\n");
    printf("  Carácter recuperado del Statevector: '%c' (ASCII: %d)\n", recuperado, (int)recuperado);

    // Cierra el loop del pipeline C: el estado ya decodificado se le manda
    // a Ollama para que lo interprete. Si no hay Ollama corriendo, avisa
    // y el programa sigue (no depende de la red para terminar bien).
    ask_ollama(&export_data, recuperado);

    return 0;
}
