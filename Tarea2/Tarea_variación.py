import os
import cv2
import numpy as np

# =========================
# 0) ENTRADA / CONFIG
# =========================
FOLDER = "fotos"
EXT = ".jpeg"
SELECCION = "imagen_00"

IMAGENES = {
    # piezas con fondos blancos
    "imagen_01": "b_01",
    "imagen_02": "b_02",
    "imagen_03": "b_03",
    "imagen_04": "b_04",
    "imagen_05": "b_05",
    "imagen_06": "b_06",
    "imagen_07": "b_07",

    # piezas con fondos con sal y pimienta
    "imagen_08": "sp_01",
    "imagen_09": "sp_02",
    "imagen_10": "sp_03",
    "imagen_11": "sp_04",
    "imagen_12": "sp_05",
    "imagen_13": "sp_06",
    "imagen_14": "sp_07",

    # piezas con fondos de alta frecuencia
    "imagen_00": "af_00",
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

    # solo fondos (sin piezas)
    "imagen_26": "fon_AF1",
    "imagen_27": "fon_AF2",
    "imagen_28": "fon_AF3",
    "imagen_29": "fon_bla",
    "imagen_30": "fon_salpi",
}

TARGET_W, TARGET_H = 800, 600
DEBUG = True

# --- Parámetros de detección de ruido ---
SP_LOW, SP_HIGH = 5, 200
SP_MIN_FRAC = 0.1983

FFT_CENTER_R = 20
FFT_K_STD = 3
FFT_MIN_PIXELS = 1

# --- Parámetros de segmentación / piezas ---
# AHORA: umbrales DIRECTOS de área en píxeles
# Si quieres detectar piezas más pequeñas -> baja MIN_AREA_PX
# Si quieres que ignore cosas grandes -> baja MAX_AREA_PX
MIN_AREA_PX = 5000     # área mínima de una pieza
MAX_AREA_PX = 23000  # área máxima de una pieza

MIN_SOLIDITY   = 0.4
AREA_RATIO_MIN = 0.35
BG_SIGMA       = 21

# filtros geométricos para descartar barras finas/alargadas
AR_MAX   = 20   # máximo aspect ratio permitido (lado_largo / lado_corto)
MIN_SIDE = 25   # lado mínimo en píxeles para considerar una pieza

DEBUG_AREAS = False    # pon True si quieres ver áreas de todos los blobs


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
def detect_ruido(gray):
    # 0: limpio, 1: sal y pimienta, 2: frecuencial
    g = gray.astype(np.uint8)

    # --- Métricas para sal y pimienta ---
    N = g.size
    n_low  = int((g <= SP_LOW).sum())
    n_high = int((g >= SP_HIGH).sum())
    frac_ext = (n_low + n_high) / float(N)

    corr = cv2.Laplacian(g, cv2.CV_64F).var()

    print(f"[S&P] frac_ext={frac_ext:.4f}  var(Lap)={corr:.2f}")

    LAP_MAX_VAR_SP = 150  # ajustable

    if frac_ext >= SP_MIN_FRAC and corr < LAP_MAX_VAR_SP:
        print("Ruido tipo: Sal y pimienta")
        return 1

    # --- Métricas para ruido frecuencial (picos en FFT) ---
    f = np.fft.fft2(g)
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift)).astype(np.float32)

    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    mask_centro = (yy - cy)**2 + (xx - cx)**2 <= FFT_CENTER_R**2

    mag2 = mag.copy()
    mag2[mask_centro] = mag2.mean()

    thr = mag2.mean() + FFT_K_STD * mag2.std()
    n_peaks = int(np.count_nonzero(mag2 > thr))

    print(f"[FFT] n_peaks={n_peaks}")

    if n_peaks >= FFT_MIN_PIXELS:
        print("Ruido tipo: Frecuencial")
        return 2

    print("Limpio")
    return 0


# =========================
# 2) Eliminación de ruido
# =========================
def eliminar_ruido(gray, tipo):
    denoised = gray.copy()

    if tipo == 1:   # Sal y pimienta (lo dejas como ya lo tenías)
        denoised = cv2.medianBlur(denoised, 3)
        denoised = cv2.medianBlur(denoised, 5)

    elif tipo == 2:  # Alta frecuencia: low-pass gaussiano FUERTE
        g = gray.astype(np.float32)

        # FFT
        f = np.fft.fft2(g)
        fshift = np.fft.fftshift(f)

        h, w = g.shape
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        r2 = (yy - cy)**2 + (xx - cx)**2

         # sigma pequeño => filtrado más fuerte (más borroso)
        SIGMA_LP = 0.006 * min(h, w)   # prueba 0.05, 0.07, 0.1, etc.
        H = np.exp(-r2 / (2.0 * SIGMA_LP * SIGMA_LP))
        # ------------------------------------------------------

        fshift_filt = fshift * H

        # Volver al dominio espacial
        f_ishift = np.fft.ifftshift(fshift_filt)
        img_back = np.fft.ifft2(f_ishift).real
        denoised = np.clip(img_back, 0, 255).astype(np.uint8)

        # Opcional: un poco más de suavizado si aún ves patrón
        # denoised = cv2.GaussianBlur(denoised, (5, 5), 0)

    return denoised



# =========================
# 3) Segmentación robusta
# =========================
def _flatten_background(gray, sigma=BG_SIGMA):
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    flat = cv2.addWeighted(gray, 1.0, bg, -1.0, 128)
    return np.clip(flat, 0, 255).astype(np.uint8)

def _clear_border(mask):
    h, w = mask.shape
    n, labels = cv2.connectedComponents(mask)
    border_labels = np.unique(
        np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    )
    out = mask.copy()
    for bl in border_labels:
        out[labels == bl] = 0
    return out

def _filtrar_componentes(labels, stats, cents, shape):
    h, w = shape

    candidatos = []
    for lab in range(1, stats.shape[0]):
        x, y, ww, hh, area = stats[lab, 0], stats[lab, 1], stats[lab, 2], stats[lab, 3], stats[lab, -1]

        # 1) Filtro por área absoluta
        if area < MIN_AREA_PX or area > MAX_AREA_PX:
            continue

        # 2) No tocar bordes
        if x == 0 or y == 0 or (x + ww) >= w or (y + hh) >= h:
            continue

        # 3) Filtro geométrico básico
        side_min = min(ww, hh)
        side_max = max(ww, hh)
        aspect   = side_max / (side_min + 1e-6)

        if side_min < MIN_SIDE:
            continue
        if aspect > AR_MAX:
            continue

        # 4) Solidity (más relajado)
        mask_i = (labels == lab).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        hull_area = max(1.0, cv2.contourArea(hull))
        solidity = area / hull_area
        if solidity < MIN_SOLIDITY:
            continue

        cx, cy = map(int, cents[lab])
        candidatos.append({
            "lab": lab,
            "area": int(area),
            "centro": (cx, cy),
            "bbox": (int(x), int(y), int(ww), int(hh))
        })

    candidatos.sort(key=lambda p: p["area"], reverse=True)

    if len(candidatos) >= 2 and candidatos[1]["area"] < AREA_RATIO_MIN * candidatos[0]["area"]:
        candidatos = candidatos[:1]

    return candidatos[:2]


def segmentar_y_encontrar(gray):
    flat = _flatten_background(gray, sigma=BG_SIGMA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(flat)

    invert = eq.mean() < 127
    _, th = cv2.threshold(
        eq,
        0,
        255,
        (cv2.THRESH_BINARY_INV if not invert else cv2.THRESH_BINARY) + cv2.THRESH_OTSU
    )

    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    th = _clear_border(th)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(th, connectivity=8)
    piezas = _filtrar_componentes(labels, stats, cents, th.shape)

    chosen_labels = [p["lab"] for p in piezas]
    mask_two = np.isin(labels, chosen_labels).astype(np.uint8) * 255

    piezas_fmt = []
    for i, p in enumerate(piezas, 1):
        piezas_fmt.append({
            "id": f"pieza_{i}",
            "centro": p["centro"],
            "bbox": p["bbox"],
            "area": p["area"]
        })
    return piezas_fmt, mask_two

def dibujar(gray, piezas):
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for p in piezas:
        (x, y, w, h) = p["bbox"]
        (cx, cy) = p["centro"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(out, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(
            out,
            f'{p["id"]} ({cx},{cy})',
            (x, max(15, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return out


# =========================
# 4) MAIN
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

    gray_ruidosa = gray.copy()

    tipo = detect_ruido(gray)

    if tipo == 2:  # alta frecuencia
        piezas, mask = segmentar_y_encontrar(gray)
        fondo_filtrado = eliminar_ruido(gray, 2)
        gray_limpia = gray.copy()
        gray_limpia[mask == 0] = fondo_filtrado[mask == 0]

    elif tipo == 1:  # sal y pimienta
        gray_limpia = eliminar_ruido(gray, 1)
        piezas, mask = segmentar_y_encontrar(gray_limpia)

    else:  # limpio
        gray_limpia = gray
        piezas, mask = segmentar_y_encontrar(gray_limpia)

    if len(piezas) >= 2:
        print(f'{piezas[0]["id"]}: centro = {piezas[0]["centro"]}')
        print(f'{piezas[1]["id"]}: centro = {piezas[1]["centro"]}')
    elif len(piezas) == 1:
        print(f'Solo una pieza: {piezas[0]["centro"]}')
    else:
        print("No se detectaron piezas.")

    if DEBUG:
        cv2.imshow("Entrada ruidosa (gris 800x600)", gray_ruidosa)
        cv2.imshow("Entrada filtrada", gray_limpia)
        cv2.imshow("Mascara (piezas)", mask)
        cv2.imshow("Resultado", dibujar(gray_limpia, piezas))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
