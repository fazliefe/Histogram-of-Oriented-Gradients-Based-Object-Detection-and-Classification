import numpy as np
import cv2
import matplotlib.pyplot as plt


# ============================================================
# 3.1 TEMEL İMPLEMENTASYON
# ============================================================

def compute_gradients(image: np.ndarray,
                      method: str = "kernel"):
    """
    Görüntünün x ve y yönündeki gradyanlarını, büyüklük ve yönelimini hesaplar.

    Girdi
    -----
    image  : Gri tonlamalı (H, W) veya BGR (H, W, 3) numpy array
    method : "kernel" -> [-1, 0, 1] filtresi ile
             "np"     -> np.gradient ile

    Çıktı
    -----
    gx, gy             : (H, W) float32, yatay ve dikey gradyan
    gradient_magnitude : (H, W) float32
    gradient_angle     : (H, W) float32, derece cinsinden [0, 180)
    """
    # Renkli ise griye çevir
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img = image.astype(np.float32)

    if method == "np":
        # np.gradient: gy (dikey), gx (yatay)
        gy, gx = np.gradient(img)
    elif method == "kernel":
        # [-1, 0, 1] filtresi ile konvolüsyon
        kernel = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        gx = cv2.filter2D(img, -1, kernel[None, :],
                          borderType=cv2.BORDER_REPLICATE)
        gy = cv2.filter2D(img, -1, kernel[:, None],
                          borderType=cv2.BORDER_REPLICATE)
    else:
        raise ValueError("method 'kernel' veya 'np' olmalı.")

    # Gradyan büyüklüğü
    gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2)

    # Gradyan yönü (derece)
    gradient_angle = np.rad2deg(np.arctan2(gy, gx))  # [-180, 180]

    # Negatif açıları 0–180 aralığına kaydır (unsigned gradient)
    gradient_angle[gradient_angle < 0] += 180.0
    gradient_angle[gradient_angle >= 180.0] -= 180.0

    return (gx.astype(np.float32),
            gy.astype(np.float32),
            gradient_magnitude.astype(np.float32),
            gradient_angle.astype(np.float32))


def create_cell_histogram(cell_magnitude: np.ndarray,
                          cell_angle: np.ndarray,
                          num_bins: int = 9):
    """
    Bir hücre için yönelim histogramı oluşturur.

    Girdi
    -----
    cell_magnitude : (h, w) hücre içindeki gradyan büyüklükleri
    cell_angle     : (h, w) hücre içindeki gradyan açıları [0, 180)
    num_bins       : histogram bin sayısı (varsayılan 9)

    Çıktı
    -----
    hist : (num_bins,) 1D numpy array

    Yaptığımız
    ----------
    0–180° aralığını num_bins eşit parçaya böleriz.
    Her piksel yönünü iki komşu bine lineer interpolasyonla dağıtırız.
    """
    bin_width = 180.0 / num_bins
    hist = np.zeros(num_bins, dtype=np.float32)

    mag = cell_magnitude.flatten().astype(np.float32)
    ang = cell_angle.flatten().astype(np.float32)

    for m, a in zip(mag, ang):
        # Örnek: 37° / 20° = 1.85 -> bin 1 ile 2 arasında
        bin_float = a / bin_width
        bin_low = int(np.floor(bin_float)) % num_bins
        bin_high = (bin_low + 1) % num_bins

        high_weight = bin_float - bin_low   # 0.85
        low_weight = 1.0 - high_weight      # 0.15

        hist[bin_low] += m * low_weight
        hist[bin_high] += m * high_weight

    return hist


def normalize_block(block_histogram: np.ndarray, method: str = "L2"):
    """
    Blok histogramını normalize eder.

    Girdi
    -----
    block_histogram : 1D numpy array
    method          : "L1", "L2", "L2-Hys"

    Çıktı
    -----
    normalized_hist : 1D numpy array
    """
    eps = 1e-5
    h = block_histogram.astype(np.float32)

    if method == "L2":
        norm = np.sqrt(np.sum(h ** 2) + eps ** 2)
        return h / norm

    if method == "L1":
        norm = np.sum(np.abs(h)) + eps
        return h / norm

    if method == "L2-Hys":
        # 1) L2 normalize
        norm = np.sqrt(np.sum(h ** 2) + eps ** 2)
        h = h / norm
        # 2) Clipping
        h = np.clip(h, 0, 0.2)
        # 3) Tekrar L2 normalize
        norm = np.sqrt(np.sum(h ** 2) + eps ** 2)
        return h / norm

    raise ValueError(f"Bilinmeyen normalize metodu: {method}")


def compute_hog_descriptor(image: np.ndarray,
                           cell_size=(8, 8),
                           block_size=(2, 2),
                           num_bins: int = 9,
                           grad_method: str = "kernel",
                           norm_method: str = "L2"):
    """
    Tam HOG descriptor hesaplar.

    Girdi
    -----
    image      : Gri tonlamalı veya BGR görüntü
    cell_size  : (h, w) piksel cinsinden hücre boyutu
    block_size : (bh, bw) hücre cinsinden blok boyutu
    num_bins   : histogram bin sayısı
    grad_method: "kernel" veya "np" (compute_gradients için)
    norm_method: "L1", "L2", "L2-Hys"

    Çıktı
    -----
    hog_vector     : 1D numpy array (HOG özellik vektörü)
    cell_histograms: (n_cells_y, n_cells_x, num_bins) hücre histogramları
                     (visualize_hog için geri döndürüyoruz)
    """
    gx, gy, magnitude, angle = compute_gradients(image, method=grad_method)

    H, W = magnitude.shape
    cell_h, cell_w = cell_size

    # Hücre sayıları
    n_cells_y = H // cell_h
    n_cells_x = W // cell_w

    # Tam oturan alanı kullan
    Hc = n_cells_y * cell_h
    Wc = n_cells_x * cell_w
    magnitude = magnitude[:Hc, :Wc]
    angle = angle[:Hc, :Wc]

    # Her hücre için histogram
    cell_histograms = np.zeros((n_cells_y, n_cells_x, num_bins),
                               dtype=np.float32)

    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            y0 = cy * cell_h
            y1 = y0 + cell_h
            x0 = cx * cell_w
            x1 = x0 + cell_w

            cell_mag = magnitude[y0:y1, x0:x1]
            cell_ang = angle[y0:y1, x0:x1]

            hist = create_cell_histogram(cell_mag, cell_ang, num_bins=num_bins)
            cell_histograms[cy, cx, :] = hist

    # Blok tarama (sliding window)
    block_h, block_w = block_size  # hücre cinsinden
    n_blocks_y = n_cells_y - block_h + 1
    n_blocks_x = n_cells_x - block_w + 1

    hog_features = []

    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            # (block_h x block_w x num_bins) blok
            block = cell_histograms[
                by:by + block_h,
                bx:bx + block_w,
                :
            ]
            block_vec = block.ravel()
            block_vec = normalize_block(block_vec, method=norm_method)
            hog_features.append(block_vec)

    if len(hog_features) == 0:
        return np.array([], dtype=np.float32), cell_histograms

    hog_vector = np.concatenate(hog_features, axis=0)
    return hog_vector.astype(np.float32), cell_histograms


# ============================================================
# 3.2 GÖRSELLEŞTİRME
# ============================================================

def visualize_gradients(image: np.ndarray,
                        grad_method: str = "sobel",
                        figsize=(12, 6)):
    """
    Problem 1'deki görsel çıktılara benzeyen gradyan görselleştirmesi.
    (Original, Gx, Gy, Magnitude, Orientation, Empty)

    grad_method:
        "sobel"  -> OpenCV Sobel (PDF’teki kare örneğine daha yakın)
        "kernel" -> compute_gradients(..., method="kernel")
        "np"     -> compute_gradients(..., method="np")
    """

    # --- Görüntü griye çevir ---
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray_f = gray.astype(np.float32)

    method = grad_method.lower()

    # --- Gradyanları hesapla ---
    if method == "sobel":
        gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        angle = cv2.phase(gx, gy, angleInDegrees=True)  # [0,360)
        angle[angle >= 180.0] -= 180.0                   # unsigned gradient [0,180)
    elif method in ("kernel", "np"):
        gx, gy, magnitude, angle = compute_gradients(gray, method=method)
    else:
        raise ValueError("grad_method 'sobel', 'kernel' veya 'np' olmalı.")

    # --- Görselleri normalize et (0–1 aralığı) ---
    def norm(im):
        im = im - im.min()
        im = im / (im.max() + 1e-8)
        return im

    gx_vis = norm(gx)
    gy_vis = norm(gy)
    mag_vis = norm(magnitude)
    ori_vis = norm(angle)        # açı haritası mozaik gibi görünür
    empty = np.zeros_like(ori_vis)

    # --- Panel çizimi ---
    titles = ["Original", "Gx (Horizontal)", "Gy (Vertical)",
              "Magnitude", "Orientation", "Empty"]
    ims = [gray, gx_vis, gy_vis, mag_vis, ori_vis, empty]

    plt.figure(figsize=figsize)
    for i, (t, im) in enumerate(zip(titles, ims)):
        plt.subplot(2, 3, i + 1)
        plt.title(t)
        plt.axis("off")
        plt.imshow(im, cmap="gray")
    plt.tight_layout()
    plt.show()

    return gx, gy, magnitude, angle


def visualize_hog(image: np.ndarray,
                  cell_histograms: np.ndarray,
                  cell_size=(8, 8),
                  num_bins: int = 9,
                  scale_factor: float = 1.0,
                  figsize=(10, 5)):
    """
    HOG özelliklerini görselleştirir.

    Gereksinimler:
    • Her hücre için, histogram bin değerlerine karşılık gelen yönlerde çizgiler çizin
    • Çizgi uzunluğu histogram değeri ile orantılı olsun
    • Orijinal görüntü ile HOG görselleştirmesini yan yana gösterin
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_for_show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        gray = image.copy()
        img_for_show = gray

    H, W = gray.shape
    cell_h, cell_w = cell_size

    n_cells_y, n_cells_x, _ = cell_histograms.shape

    hog_image = np.zeros_like(gray, dtype=np.float32)

    # Histogramları normalize et (görsel dengeli olsun)
    max_val = cell_histograms.max()
    if max_val > 0:
        cell_histograms = cell_histograms / max_val

    bin_width = 180.0 / num_bins
    bin_angles = (np.arange(num_bins) + 0.5) * bin_width  # bin merkez açıları

    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            hist = cell_histograms[cy, cx]
            y_center = int(cy * cell_h + cell_h / 2)
            x_center = int(cx * cell_w + cell_w / 2)

            for b in range(num_bins):
                v = hist[b]
                if v <= 0:
                    continue

                angle_deg = bin_angles[b]
                angle_rad = np.deg2rad(angle_deg)

                length = v * (min(cell_h, cell_w) / 2) * scale_factor

                dx = length * np.cos(angle_rad)
                dy = length * np.sin(angle_rad)

                x1 = int(round(x_center - dx))
                y1 = int(round(y_center - dy))
                x2 = int(round(x_center + dx))
                y2 = int(round(y_center + dy))

                h, w = hog_image.shape
                x1 = np.clip(x1, 0, w - 1)
                x2 = np.clip(x2, 0, w - 1)
                y1 = np.clip(y1, 0, h - 1)
                y2 = np.clip(y2, 0, h - 1)

                cv2.line(hog_image, (x1, y1), (x2, y2), 255, 1)

    # Orijinal + HOG yan yana
    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    plt.title("Orijinal Görüntü")
    plt.axis("off")
    plt.imshow(img_for_show, cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title("HOG Görselleştirme")
    plt.axis("off")
    plt.imshow(hog_image, cmap="gray")

    plt.tight_layout()
    plt.show()

    return hog_image


# ============================================================
# TEST FONKSİYONLARI (SENTETİK + GERÇEK)
# ============================================================

def create_synthetic_shapes(size=128):
    """
    Basit geometrik şekiller:
      - kare
      - daire
      - üçgen
    üretir.
    """
    imgs = {}
    base = np.zeros((size, size), dtype=np.uint8)

    # Kare
    img_square = base.copy()
    cv2.rectangle(img_square, (32, 32), (96, 96), 255, -1)
    imgs["square"] = img_square

    # Daire
    img_circle = base.copy()
    cv2.circle(img_circle, (size // 2, size // 2), 32, 255, -1)
    imgs["circle"] = img_circle

    # Üçgen
    img_triangle = base.copy()
    pts = np.array([
        [size // 2, 24],
        [24, size - 24],
        [size - 24, size - 24]
    ], np.int32)
    cv2.fillConvexPoly(img_triangle, pts, 255)
    imgs["triangle"] = img_triangle

    return imgs


def run_hog_test_on_image(image,
                          name: str,
                          cell_size=(8, 8),
                          block_size=(2, 2),
                          num_bins=9,
                          grad_method="kernel",
                          norm_method="L2"):
    """
    Verilen görüntü üzerinde:
      - HOG descriptor hesapla
      - Vektör boyutunu yazdır
      - HOG görselleştirmesini göster
    """
    print("\n======================================")
    print(f"Görüntü adı: {name}")
    print(f"Boyut: {image.shape}")
    print(f"Parametreler: cell_size={cell_size}, block_size={block_size}, "
          f"num_bins={num_bins}, grad_method={grad_method}")

    hog_vec, cell_hists = compute_hog_descriptor(
        image,
        cell_size=cell_size,
        block_size=block_size,
        num_bins=num_bins,
        grad_method=grad_method,
        norm_method=norm_method
    )

    print("HOG özellik vektör boyutu:", hog_vec.shape)

    visualize_hog(image, cell_hists,
                  cell_size=cell_size,
                  num_bins=num_bins,
                  scale_factor=1.0)
    print("======================================")


# ============================================================
# MAIN – ÖRNEK ÇALIŞMA
# ============================================================

if __name__ == "__main__":
    # 1) Sentetik geometrik şekiller
    synthetic_imgs = create_synthetic_shapes(size=128)

    # Kare görüntü için Gx, Gy, Magnitude, Orientation, Empty panosu
    square = synthetic_imgs["square"]
    # PDF’teki örneğe daha benzeyen sonuç için "sobel" kullanıyoruz.
    visualize_gradients(square, grad_method="sobel", figsize=(10, 6))

    # Sentetik şekiller için HOG testleri
    for name, img in synthetic_imgs.items():
        run_hog_test_on_image(
            img,
            name + " (8x8 hücre, 9 bin)",
            cell_size=(8, 8),
            block_size=(2, 2),
            num_bins=9,
            grad_method="kernel"
        )

        run_hog_test_on_image(
            img,
            name + " (16x16 hücre, 9 bin)",
            cell_size=(16, 16),
            block_size=(2, 2),
            num_bins=9,
            grad_method="kernel"
        )

    # 2) Kendi gerçek görüntülerin (dosyaları klasöre sen koyacaksın)
    real_image_paths = [
        "object1.png",
        "object2.png",
        "person1.png",
        "person2.png",
    ]

    for path in real_image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"{path} bulunamadı, klasöre ekleyip ismini bu listeye yazabilirsin.")
            continue

        run_hog_test_on_image(
            img,
            path + " (8x8 hücre, 9 bin)",
            cell_size=(8, 8),
            block_size=(2, 2),
            num_bins=9,
            grad_method="kernel"
        )

        run_hog_test_on_image(
            img,
            path + " (8x8 hücre, 18 bin)",
            cell_size=(8, 8),
            block_size=(2, 2),
            num_bins=18,
            grad_method="kernel"
        )
