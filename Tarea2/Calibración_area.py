# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np

# =========================
# 0) ENTRADA / CONFIG
# =========================
FOLDER = "fotos"
EXT = ".jpeg"
SELECCION = "imagen_00"   # <--- cambia aquí el alias de la imagen que quieras

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

TARGET_W, TARGET_H = 800, 600    # mismo tamaño que usas en tu proyecto

# =========================
# Utils
# =========================
def build_path(alias: str) -> str:
    assert alias in IMAGENES, f'Alias "{alias}" no existe en IMAGENES.'
    return os.path.join(FOLDER, IMAGENES[alias] + EXT)

def resize_to_target(img, w, h):
    interp = cv2.INTER_AREA if (img.shape[1] > w or img.shape[0] > h) else cv2.INTER_CUBIC
    return cv2.resize(img, (w, h), interpolation=interp)

# =========================
# Herramienta de calibración de área
# =========================
drawing = False
ix, iy = -1, -1
img = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
img_show = img.copy()

def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, img, img_show

    if event == cv2.EVENT_LBUTTONDOWN:
        # inicio del rectángulo
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        # dibujar rectángulo temporal mientras se arrastra
        img_show = img.copy()
        cv2.rectangle(img_show, (ix, iy), (x, y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        # rectángulo final
        drawing = False
        img_show = img.copy()
        cv2.rectangle(img_show, (ix, iy), (x, y), (0, 255, 0), 2)

        # normalizar coordenadas (por si arrastras "al revés")
        x0, y0 = min(ix, x), min(iy, y)
        x1, y1 = max(ix, x), max(iy, y)

        w = x1 - x0
        h = y1 - y0
        area = w * h

        if w > 0 and h > 0:
            img_h, img_w = img.shape[:2]
            total_area = img_w * img_h
            pct = 100.0 * area / float(total_area)

            # mostrar texto en la imagen
            txt = f"{w}x{h}  area={area} px ({pct:.2f}%)"
            cv2.putText(img_show, txt, (x0, max(15, y0-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

            # imprimir en consola
            print("====================================")
            print(f"Rectángulo seleccionado:")
            print(f"  x={x0}, y={y0}, w={w}, h={h}")
            print(f"  Área = {area} píxeles")
            print(f"  Porcentaje de la imagen = {pct:.4f} %")
            print("====================================")
        else:
            print("Rectángulo con ancho/alto cero, ignorado.")


def main():
    global img, img_show

    ruta = build_path(SELECCION)
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró la imagen: {ruta}")

    img0 = cv2.imread(ruta, cv2.IMREAD_COLOR)
    if img0 is None:
        raise ValueError(f"No se pudo leer '{ruta}'.")

    img = resize_to_target(img0, TARGET_W, TARGET_H)
    img_show = img.copy()

    cv2.namedWindow("Calibrar área")
    cv2.setMouseCallback("Calibrar área", mouse_callback)

    print("Instrucciones:")
    print("  - Haz clic izquierdo y arrastra para dibujar un rectángulo.")
    print("  - Al soltar el botón se mostrará el área en consola.")
    print("  - Pulsa 'c' para limpiar la imagen.")
    print("  - Pulsa 'q' o ESC para salir.\n")

    while True:
        cv2.imshow("Calibrar área", img_show)
        key = cv2.waitKey(20) & 0xFF

        if key == 27 or key == ord('q'):  # ESC o 'q'
            break
        elif key == ord('c'):
            img_show = img.copy()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
