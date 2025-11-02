# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np

# =========================
# 0) ENTRADA / CONFIG
# =========================
FOLDER = "fotos"
EXT = ".jpeg"
SELECCION = "imagen_09"

IMAGENES = {
    "imagen_01": "b_01",
    "imagen_02": "b_02",
    "imagen_03": "b_03",
    "imagen_04": "b_04",
    "imagen_05": "b_05",
    "imagen_06": "b_06",
    "imagen_07": "b_07",

    "imagen_08": "sp_01",
    "imagen_09": "sp_02",
    "imagen_10": "sp_03",
    "imagen_11": "sp_04",
    "imagen_12": "sp_05",
    "imagen_13": "sp_06",
    "imagen_14": "sp_07",

    "imagen_15": "af_01",
    "imagen_16": "af_02",
    "imagen_17": "af_03",
    "imagen_18": "af_04",
    "imagen_19": "af_05",
    "imagen_20": "af_06",
    "imagen_21": "af_07",
    "imagen_22": "af_08",
    "imagen_23": "af_09",
    "imagen_24": "af_10",
    "imagen_25": "af_11",

    "imagen_26": "fon_AF1",
    "imagen_27": "fon_AF2",
    "imagen_28": "fon_AF3",
    "imagen_29": "fon_bla",
    "imagen_30": "fon_salpi",
}

TARGET_W, TARGET_H = 800, 600
DEBUG = True

# --- Detección ADAPTATIVA de Sal-y-Pimienta (ajústalos si hace falta) ---
SP_P_LOW = 1       # percentil bajo de brillo para "extremos"
SP_P_HIGH = 99     # percentil alto
SP_K_MAD = 4.0       # k * MAD sobre el residuo
SP_MAX_CC_AREA = 10  # área máx. (px) de un impulso
SP_FRAC_MIN = 0.01  # fracción absoluta mínima (0.5%)
SP_MULT_BASE = 3.0   # veces por encima de la cola base del residuo para activar
PRINT_SP_DEBUG = False  # True para ver métricas y calibrar

# --- FFT (ruido frecuencial) ---
FFT_CENTER_R = 20
FFT_K_STD = 6.0
FFT_MIN_PIXELS = 6

# --- Segmentación robusta ---
MIN_AREA_REL   = 0.001
MAX_AREA_REL   = 0.15
MIN_SOLIDITY   = 0.60
AREA_RATIO_MIN = 0.35
BG_SIGMA       = 21

# =========================
# Utils
# =========================
def build_path(alias: str) -> str:
    assert alias in IMAGENES, f'Alias "{alias}" no existe en IMAGENES.'
    return os.path.join(FOLDER, IMAGENES[alias] + EXT)

def resize_to_target(img, w, h):
    interp = cv2.INTER_AREA if (img.shape[1] > w or img.shape[0] > h) else cv2.INTER_CUBIC
    return cv2.resize(img, (w, h), interpolation=interp)

def ensure_gray(img):
    if img.ndim == 3 and img.shape[2] == 3:
        print("INFO: imagen en COLOR -> convirtiendo a GRIS.")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# =========================
# 1) Clasificación de ruido (sin modificar imagen)
# =========================
def _salt_pepper_adapt(gray):
    """
    Devuelve (is_sp, stats) usando un criterio ADAPTATIVO:
    - Residuo por mediana 3x3 (impulsos)
    - Umbral robusto con MAD
    - Cuenta solo extremos (percentiles) y componentes diminutas
    - Compara contra la 'cola' base del residuo para decidir
    """
    g = gray.astype(np.uint8)

    # Residuo impulsivo con mediana 3x3
    m3 = cv2.medianBlur(g, 3)
    r = cv2.absdiff(g, m3).astype(np.float32)

    # Umbral robusto del residuo: mediana + k * MAD
    med_r = np.median(r)
    mad_r = np.median(np.abs(r - med_r)) + 1e-6
    thr_r = med_r + SP_K_MAD * 1.4826 * mad_r

    # Extremos por percentiles (adaptativos a iluminación)
    low = np.percentile(g, SP_P_LOW)
    high = np.percentile(g, SP_P_HIGH)
    extremes = (g <= low + 2) | (g >= high - 2)

    # Impulsos = residuo alto Y extremo
    impulses = (r > thr_r) & extremes
    imp_u8 = (impulses.astype(np.uint8) * 255)

    # Mantener solo componentes diminutas
    n, lab, stats, _ = cv2.connectedComponentsWithStats(imp_u8, connectivity=8)
    small_mask = np.zeros_like(imp_u8)
    for i in range(1, n):
        if stats[i, -1] <= SP_MAX_CC_AREA:
            small_mask[lab == i] = 255

    frac_imp = small_mask.mean() / 255.0   # fracción de impulsos

    # Cola base del residuo (sin exigir extremos)
    tail = float((r > thr_r).mean())

    # Decisión final
    thr_frac = max(SP_FRAC_MIN, SP_MULT_BASE * tail)
    is_sp = frac_imp >= thr_frac

    stats_out = {"frac_imp": frac_imp, "tail": tail, "thr_frac": thr_frac,
                 "thr_r": float(thr_r), "low": float(low), "high": float(high)}
    return is_sp, stats_out

def detect_ruido(gray):
    # 1) Sal y pimienta ADAPTATIVO
    is_sp, sp_stats = _salt_pepper_adapt(gray)
    if PRINT_SP_DEBUG:
        print(f"[SP] frac_imp={sp_stats['frac_imp']:.4f}  tail={sp_stats['tail']:.4f}  thr_frac={sp_stats['thr_frac']:.4f}")
    if is_sp:
        return "sal_pimienta"

    # 2) Frecuencial (picos en FFT)
    g = gray.astype(np.uint8)
    f = np.fft.fft2(g)
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift)).astype(np.float32)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    mask_centro = (yy - cy)**2 + (xx - cx)**2 <= FFT_CENTER_R**2
    mag2 = mag.copy(); mag2[mask_centro] = mag2.mean()
    thr = mag2.mean() + FFT_K_STD * mag2.std()
    if int(np.count_nonzero(mag2 > thr)) >= FFT_MIN_PIXELS:
        return "frecuencial"

    return "limpio"

# =========================
# 2) Segmentación robusta (usa copias internas, no cambia tu imagen)
# =========================
def _flatten_background(gray, sigma=BG_SIGMA):
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    flat = cv2.addWeighted(gray, 1.0, bg, -1.0, 128)
    return np.clip(flat, 0, 255).astype(np.uint8)

def _clear_border(mask):
    h, w = mask.shape
    n, labels = cv2.connectedComponents(mask)
    border_labels = np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
    out = mask.copy()
    for bl in border_labels:
        out[labels == bl] = 0
    return out

def _filtrar_componentes(labels, stats, cents, shape):
    h, w = shape
    img_area = h * w
    min_area = max(300, int(MIN_AREA_REL * img_area))
    max_area = int(MAX_AREA_REL * img_area)

    candidatos = []
    for lab in range(1, stats.shape[0]):
        x, y, ww, hh, area = stats[lab, 0], stats[lab, 1], stats[lab, 2], stats[lab, 3], stats[lab, -1]
        if area < min_area or area > max_area:  continue
        if x == 0 or y == 0 or (x + ww) >= w or (y + hh) >= h:  continue

        mask_i = (labels == lab).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:  continue
        cnt = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        hull_area = max(1.0, cv2.contourArea(hull))
        solidity = area / hull_area
        if solidity < MIN_SOLIDITY:  continue

        cx, cy = map(int, cents[lab])
        candidatos.append({"lab": lab, "area": int(area), "centro": (cx, cy), "bbox": (int(x), int(y), int(ww), int(hh))})

    candidatos.sort(key=lambda p: p["area"], reverse=True)
    if len(candidatos) >= 2 and candidatos[1]["area"] < AREA_RATIO_MIN * candidatos[0]["area"]:
        candidatos = candidatos[:1]
    return candidatos[:2]

def segmentar_y_encontrar(gray):
    flat = _flatten_background(gray, sigma=BG_SIGMA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(flat)

    invert = eq.mean() < 127
    _, th = cv2.threshold(eq, 0, 255,
                          (cv2.THRESH_BINARY_INV if not invert else cv2.THRESH_BINARY) + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    th = _clear_border(th)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(th, connectivity=8)
    piezas = _filtrar_componentes(labels, stats, cents, th.shape)

    chosen_labels = [p["lab"] for p in piezas]
    mask_two = np.isin(labels, chosen_labels).astype(np.uint8) * 255

    piezas_fmt = []
    for i, p in enumerate(piezas, 1):
        piezas_fmt.append({"id": f"pieza_{i}", "centro": p["centro"], "bbox": p["bbox"], "area": p["area"]})
    return piezas_fmt, mask_two

def dibujar(gray, piezas):
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for p in piezas:
        (x, y, w, h) = p["bbox"]; (cx, cy) = p["centro"]
        cv2.rectangle(out, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(out, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(out, f'{p["id"]} ({cx},{cy})', (x, max(15, y-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
    return out

# =========================
# 3) MAIN
# =========================
def main():
    ruta = build_path(SELECCION)
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró la imagen: {ruta}")
    img = cv2.imread(ruta, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"No se pudo leer '{ruta}'.")
    gray = ensure_gray(img)
    gray = resize_to_target(gray, TARGET_W, TARGET_H)

    # 1) Solo CLASIFICAMOS el ruido (NO se filtra)
    tipo = detect_ruido(gray)
    print(f"Ruido clasificado: {tipo}")

    # 2) Segmentación (se calcula en copias internas); se dibuja sobre la imagen original
    piezas, mask = segmentar_y_encontrar(gray)

    if len(piezas) >= 2:
        print(f'{piezas[0]["id"]}: centro = {piezas[0]["centro"]}')
        print(f'{piezas[1]["id"]}: centro = {piezas[1]["centro"]}')
    elif len(piezas) == 1:
        print(f'Solo una pieza: {piezas[0]["centro"]}')
    else:
        print("No se detectaron piezas.")

    if DEBUG:
        cv2.imshow("Entrada (gris 800x600) — SIN modificar", gray)
        cv2.imshow("Máscara (piezas)", mask)
        cv2.imshow("Resultado", dibujar(gray, piezas))
        cv2.waitKey(0); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
