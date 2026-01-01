import os
import cv2
import numpy as np
import joblib
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import csv

# ================== AYARLAR ==================

# Proje kök dizini
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Özel nesne verisi klasörleri
CUSTOM_DIR      = os.path.join(BASE_DIR, "data", "custom")
TRAIN_POS_DIR   = os.path.join(CUSTOM_DIR, "train_pos")
TRAIN_NEG_DIR   = os.path.join(CUSTOM_DIR, "train_neg")
TEST_IMAGES_DIR = os.path.join(CUSTOM_DIR, "test_images")

MODEL_PATH = os.path.join(CUSTOM_DIR, "hog_svm_model.joblib")

# HOG penceresi (pozitif örneklerin boyutu)
WINDOW_SIZE = (64, 128)  # (width, height)

# SVM çıktısı için karar eşiği
# 0.0: çok hassas, çok false positive
# 0.5–1.5: daha seçici, daha az kutu
DECISION_THRESHOLD = 1.0

# NMS çakışma eşiği (daha küçük -> daha agresif birleştirme)
NMS_IOU_THRESH = 0.2

# ======================================================


def get_hog():
    """Sabit HOG parametrizasyonu."""
    hog = cv2.HOGDescriptor(
        _winSize=WINDOW_SIZE,
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    return hog


def load_images_and_extract_hog(folder, label):
    """Verilen klasördeki tüm görüntüler için HOG çıkarır."""
    hog = get_hog()
    features = []
    labels = []

    if not os.path.exists(folder):
        print(f"HATA: Klasör bulunamadı: {folder}")
        return np.empty((0,)), np.empty((0,))

    for img_name in os.listdir(folder):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Görüntü okunamadı: {img_path}")
            continue

        img_resized = cv2.resize(img, WINDOW_SIZE)
        hog_desc = hog.compute(img_resized)
        features.append(hog_desc.flatten())
        labels.append(label)

    if len(features) == 0:
        print(f"UYARI: {folder} içinde uygun görüntü bulunamadı.")
        return np.empty((0,)), np.empty((0,))

    return np.array(features), np.array(labels)


def train_svm():
    """Pozitif ve negatif örneklerle Linear SVM eğitir."""
    print("Pozitif örneklerden HOG çıkarılıyor...")
    X_pos, y_pos = load_images_and_extract_hog(TRAIN_POS_DIR, 1)

    print("Negatif örneklerden HOG çıkarılıyor...")
    X_neg, y_neg = load_images_and_extract_hog(TRAIN_NEG_DIR, 0)

    if X_pos.size == 0 or X_neg.size == 0:
        print("HATA: Pozitif veya negatif eğitim verisi yok. train_pos/train_neg kontrol et.")
        return False

    X = np.vstack((X_pos, X_neg))
    y = np.hstack((y_pos, y_neg))

    print(f"Toplam eğitim örneği: {len(y)} "
          f"({len(y_pos)} pozitif, {len(y_neg)} negatif)")

    svm = LinearSVC()
    svm.fit(X, y)

    joblib.dump(svm, MODEL_PATH)
    print(f"Model kaydedildi: {MODEL_PATH}")
    return True


def sliding_window(image, step_size, window_size):
    """
    image: gri tonlamalı görüntü
    step_size: (sx, sy)
    window_size: (w, h)
    """
    for y in range(0, image.shape[0] - window_size[1], step_size[1]):
        for x in range(0, image.shape[1] - window_size[0], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


def non_max_suppression_fast(boxes, scores, overlapThresh=0.3):
    """NMS (IoU threshold overlapThresh)."""
    if len(boxes) == 0:
        return [], []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    pick = []

    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)

        suppress = [len(idxs) - 1]

        for pos in range(0, len(idxs) - 1):
            i = idxs[pos]

            xx1 = max(x1[last], x1[i])
            yy1 = max(y1[last], y1[i])
            xx2 = min(x2[last], x2[i])
            yy2 = min(y2[last], y2[i])

            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            overlap = float(w * h) / areas[i]

            if overlap > overlapThresh:
                suppress.append(pos)

        idxs = np.delete(idxs, suppress)

    picked_boxes = boxes[pick].tolist()
    picked_scores = scores[pick].tolist()
    return picked_boxes, picked_scores


def detect_on_image(img_bgr, svm, decision_threshold=0.0):
    """Verilen BGR görüntü üzerinde sliding window + HOG + SVM ile tespit yapar."""
    hog = get_hog()
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    boxes = []
    scores = []

    scale = 1.0
    scaled_img = img_gray.copy()

    # Multi-scale: görüntüyü sürekli küçülterek tara
    while scaled_img.shape[0] >= WINDOW_SIZE[1] and scaled_img.shape[1] >= WINDOW_SIZE[0]:
        for (x, y, window) in sliding_window(
            scaled_img,
            step_size=(8, 8),
            window_size=WINDOW_SIZE
        ):
            if window.shape[0] != WINDOW_SIZE[1] or window.shape[1] != WINDOW_SIZE[0]:
                continue

            hog_desc = hog.compute(window).flatten().reshape(1, -1)
            score = svm.decision_function(hog_desc)[0]

            if score > decision_threshold:
                x_orig = int(x / scale)
                y_orig = int(y / scale)
                w_orig = int(WINDOW_SIZE[0] / scale)
                h_orig = int(WINDOW_SIZE[1] / scale)

                boxes.append([x_orig, y_orig, w_orig, h_orig])
                scores.append(score)

        # ölçeği küçült
        scale *= 1.2
        new_w = int(img_gray.shape[1] / scale)
        new_h = int(img_gray.shape[0] / scale)
        if new_w < WINDOW_SIZE[0] or new_h < WINDOW_SIZE[1]:
            break
        scaled_img = cv2.resize(img_gray, (new_w, new_h))

    nms_boxes, nms_scores = non_max_suppression_fast(
        boxes, scores, overlapThresh=NMS_IOU_THRESH
    )
    return nms_boxes, nms_scores


def test_on_folder(decision_threshold=None, save_images=True, results_subdir="results_default"):
    """
    TEST_IMAGES_DIR içindeki görüntülerde tespit yapar, sonuçları kaydeder
    ve temel istatistikleri döndürür.

    return:
        img_names          : kullanılan test görüntü isimleri (sıralı liste)
        per_image_counts   : her görüntü için tespit edilen kişi sayısı listesi
    """
    if decision_threshold is None:
        decision_threshold = DECISION_THRESHOLD

    if not os.path.exists(MODEL_PATH):
        print("Model bulunamadı, önce eğitiliyor...")
        ok = train_svm()
        if not ok:
            return [], []

    svm = joblib.load(MODEL_PATH)

    results_root = os.path.join(CUSTOM_DIR, "results")
    os.makedirs(results_root, exist_ok=True)

    results_dir = os.path.join(results_root, results_subdir)
    if save_images:
        os.makedirs(results_dir, exist_ok=True)

    total_images = 0
    total_detections = 0

    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"HATA: Test görüntü klasörü yok: {TEST_IMAGES_DIR}")
        return [], []

    img_names = []
    per_image_counts = []

    for img_name in sorted(os.listdir(TEST_IMAGES_DIR)):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        img_path = os.path.join(TEST_IMAGES_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Görüntü okunamadı: {img_path}")
            continue

        total_images += 1
        img_names.append(img_name)

        boxes, scores = detect_on_image(
            img, svm, decision_threshold=decision_threshold
        )

        num_det = len(boxes)
        per_image_counts.append(num_det)
        total_detections += num_det

        if save_images:
            # tespit kutularını çiz
            for (x, y, w, h), s in zip(boxes, scores):
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(
                    img, f"{s:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
                )

            out_path = os.path.join(results_dir, f"det_{img_name}")
            cv2.imwrite(out_path, img)

        print(f"{img_name}: {num_det} tespit (eşik={decision_threshold})")

    print("\n===== ÖZEL NESNE TESPİT ÖZETİ =====")
    print(f"Test görüntü sayısı   : {total_images}")
    print(f"Toplam tespit sayısı   : {total_detections}")
    if total_images > 0:
        print(f"Görüntü başına ortalama tespit: {total_detections / total_images:.2f}")
    print(f"Decision threshold     : {decision_threshold}")
    print(f"NMS IoU threshold      : {NMS_IOU_THRESH}")
    print("===================================")

    return img_names, per_image_counts


# ---------- Değerlendirme + Grafik Üretimi ----------

def load_gt_counts(gt_counts_path):
    """
    Opsiyonel: Her test görüntüsünde GERÇEK kişi sayısını içeren CSV okur.
    Format örneği (header dahil):
        image,count
        img_0001.jpg,3
        img_0002.jpg,1
    """
    gt_counts = {}
    if gt_counts_path is None:
        return gt_counts

    if not os.path.exists(gt_counts_path):
        print(f"UYARI: gt_counts CSV bulunamadı: {gt_counts_path}")
        return gt_counts

    with open(gt_counts_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("image") or row.get("img") or row.get("filename")
            cnt = row.get("count") or row.get("persons") or row.get("people")
            if not name or cnt is None:
                continue
            try:
                gt_counts[name] = int(cnt)
            except ValueError:
                continue

    print(f"Yüklenen ground truth kayıt sayısı: {len(gt_counts)}")
    return gt_counts


def evaluate_thresholds(threshold_list, gt_counts_path=None):
    """
    Farklı decision threshold değerleri için:
    - Her görüntüde kaç kişi tespit edilmiş (istatistik)
    - (Varsa) ground truth üzerinden yaklaşık TP/FP/FN sayıları
    - Grafikler:
        * detection_counts_threshold_X.png  (her test görüntüsü için tespit sayısı)
        * thresholds_vs_avg_detections.png (eşik vs. ortalama tespit sayısı)
        * thresholds_vs_tp_fp_fn.png       (varsa GT ile eşik vs TP/FP/FN)
    Ayrıca false_positive_negative_examples.txt içine örnek görsellerin isimlerini yazar.
    """
    results_root = os.path.join(CUSTOM_DIR, "results")
    os.makedirs(results_root, exist_ok=True)

    gt_counts = load_gt_counts(gt_counts_path)

    # Her threshold için istatistikler
    stats_per_threshold = {}
    img_names_global = None  # tüm thresholdlarda aynı sıralama

    for th in threshold_list:
        print(f"\n=== Threshold = {th} için değerlendirme başlıyor ===")
        img_names, counts = test_on_folder(
            decision_threshold=th,
            save_images=False,                    # Görselleri sadece ana testte kaydettik
            results_subdir=f"th_{th:.2f}"
        )

        if img_names_global is None:
            img_names_global = img_names

        avg_det = float(np.mean(counts)) if len(counts) > 0 else 0.0

        tp_total = fp_total = fn_total = 0
        fp_imgs = []
        fn_imgs = []

        if gt_counts:
            for name, det_count in zip(img_names, counts):
                if name not in gt_counts:
                    continue
                gt = gt_counts[name]

                # Çok kaba bir metrik: kutu sayısı üzerinden
                tp = min(det_count, gt)
                fp = max(0, det_count - gt)
                fn = max(0, gt - det_count)

                tp_total += tp
                fp_total += fp
                fn_total += fn

                if fp > 0:
                    fp_imgs.append(name)
                if fn > 0:
                    fn_imgs.append(name)

        stats_per_threshold[th] = {
            "counts": counts,
            "avg_detections": avg_det,
            "tp": tp_total,
            "fp": fp_total,
            "fn": fn_total,
            "fp_imgs": fp_imgs,
            "fn_imgs": fn_imgs
        }

        print(f"Eşik={th} -> Ortalama tespit: {avg_det:.2f}, "
              f"TP={tp_total}, FP={fp_total}, FN={fn_total}")

    # 1) Her test görüntüsü için tespit sayısı (seçilen bir threshold için)
    # Burada DECISION_THRESHOLD'ü referans alıyoruz, yoksa ilk threshold
    ref_th = DECISION_THRESHOLD
    if ref_th not in stats_per_threshold and len(threshold_list) > 0:
        ref_th = threshold_list[0]

    if img_names_global is not None and ref_th in stats_per_threshold:
        counts_ref = stats_per_threshold[ref_th]["counts"]
        indices = list(range(len(img_names_global)))

        plt.figure(figsize=(max(8, len(indices) * 0.4), 4))
        plt.bar(indices, counts_ref)
        plt.xticks(indices, img_names_global, rotation=90)
        plt.xlabel("Test görüntüleri")
        plt.ylabel("Tespit edilen kişi sayısı")
        plt.title(f"Her test görüntüsü için tespit sayısı (threshold={ref_th})")
        plt.tight_layout()
        out_path = os.path.join(results_root, f"detection_counts_threshold_{ref_th}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"-> Grafik kaydedildi: {out_path}")

    # 2) Threshold vs ortalama tespit sayısı grafiği
    ths_sorted = sorted(threshold_list)
    avg_list = [stats_per_threshold[th]["avg_detections"] for th in ths_sorted]

    plt.figure(figsize=(6, 4))
    plt.plot(ths_sorted, avg_list, marker="o")
    plt.xlabel("Decision threshold")
    plt.ylabel("Görüntü başına ortalama tespit")
    plt.title("Farklı threshold değerlerinin tespit sayısına etkisi")
    plt.grid(True)
    plt.tight_layout()
    out_path = os.path.join(results_root, "thresholds_vs_avg_detections.png")
    plt.savefig(out_path)
    plt.close()
    print(f"-> Grafik kaydedildi: {out_path}")

    # 3) Eğer ground truth varsa: threshold vs TP/FP/FN grafiği
    if gt_counts:
        tp_list = [stats_per_threshold[th]["tp"] for th in ths_sorted]
        fp_list = [stats_per_threshold[th]["fp"] for th in ths_sorted]
        fn_list = [stats_per_threshold[th]["fn"] for th in ths_sorted]

        plt.figure(figsize=(6, 4))
        plt.plot(ths_sorted, tp_list, marker="o", label="TP")
        plt.plot(ths_sorted, fp_list, marker="o", label="FP")
        plt.plot(ths_sorted, fn_list, marker="o", label="FN")
        plt.xlabel("Decision threshold")
        plt.ylabel("Kişi sayısı (toplam)")
        plt.title("Threshold değişimine göre TP / FP / FN")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        out_path = os.path.join(results_root, "thresholds_vs_tp_fp_fn.png")
        plt.savefig(out_path)
        plt.close()
        print(f"-> Grafik kaydedildi: {out_path}")

        # False positive / false negative örnekleri için bir txt dosyası
        fp_fn_path = os.path.join(results_root, "false_positive_negative_examples.txt")
        with open(fp_fn_path, "w", encoding="utf-8") as f:
            f.write("False positive / false negative örnek görüntü isimleri\n")
            f.write("Not: Buradaki FP/FN, sadece kişi SAYISI üzerinden yaklaşık hesaplanmıştır.\n\n")
            for th in ths_sorted:
                f.write(f"=== Threshold = {th} ===\n")
                f.write("False Positive (fazla tespit edilenler):\n")
                for name in stats_per_threshold[th]["fp_imgs"]:
                    f.write(f"  - {name}\n")
                f.write("False Negative (eksik tespit edilenler):\n")
                for name in stats_per_threshold[th]["fn_imgs"]:
                    f.write(f"  - {name}\n")
                f.write("\n")
        print(f"-> FP/FN örnek listesi kaydedildi: {fp_fn_path}")


if __name__ == "__main__":
    # 1) Model eğitimi + temel test (her test görüntüsü için bounding box'lı sonuçlar)
    train_svm()
    # Bu çağrıda bounding box çizili sonuç görselleri data/custom/results/results_default altına kaydedilir.
    test_on_folder(decision_threshold=DECISION_THRESHOLD,
                   save_images=True,
                   results_subdir="results_default")

    # 2) Farklı threshold değerleri için analiz + grafikler
    # Burayı istersen kendi threshold listenle değiştirebilirsin.
    threshold_list = [0.0, 0.5, 1.0, 1.5]

    # Opsiyonel: data/custom/gt_counts.csv hazırlarsan (image,count formatında),
    # aşağıdaki satır otomatik olarak o dosyayı kullanır.
    gt_csv_default = os.path.join(CUSTOM_DIR, "gt_counts.csv")
    if os.path.exists(gt_csv_default):
        gt_path = gt_csv_default
    else:
        gt_path = None

    evaluate_thresholds(threshold_list, gt_counts_path=gt_path)
