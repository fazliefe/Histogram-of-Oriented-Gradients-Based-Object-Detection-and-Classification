import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import csv

# =============== AYARLAR ===============

# Proje kök dizini = bu dosyanın bulunduğu yerden yukarı çık
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Test görüntüleri klasörü (insan içeren fotoğraflar)
# Gerekirse burayı kendi klasör yapına göre değiştir
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "data", "custom", "test_images")

# Sonuçları kaydedeceğimiz klasör (görseller + grafikler)
RESULTS_DIR = os.path.join(BASE_DIR, "veri", "pedestrians_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# =======================================


def non_max_suppression(boxes, scores, overlapThresh=0.5):
    """
    Basit NMS (Non-Maximum Suppression) implementasyonu.
    boxes: [x, y, w, h]
    scores: confidence list
    """
    if len(boxes) == 0:
        return [], []

    # boxes: x, y, w, h -> x1, y1, x2, y2'ye çevir
    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)  # puana göre sırala (düşükten yükseğe)

    pick = []

    while len(idxs) > 0:
        last = idxs[-1]  # en yüksek skor
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

    # Seçilen kutu ve skorları geri döndür
    picked_boxes = boxes[pick].tolist()
    picked_scores = scores[pick].tolist()
    return picked_boxes, picked_scores


def run_detector_on_folder(hit_threshold=0.0,
                           save_images=True,
                           subdir_name="default"):
    """
    Verilen hit_threshold ile tüm TEST_IMAGES_DIR üzerinde insan tespiti yapar.

    return:
        img_names        : görüntü isimleri listesi
        per_image_counts : her görüntüde tespit edilen kişi sayısı listesi
    """
    print(f"\n=== İnsan tespiti çalışıyor (hit_threshold={hit_threshold}) ===")
    print("Python şu klasöre bakıyor:", TEST_IMAGES_DIR)
    print("Bu klasör gerçekten var mı?:", os.path.exists(TEST_IMAGES_DIR))

    if not os.path.exists(TEST_IMAGES_DIR):
        print("HATA: Test görüntü klasörü bulunamadı!")
        return [], []

    # OpenCV HOG + SVM insan dedektörü
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    total_images = 0
    total_detections = 0

    img_names = []
    per_image_counts = []

    if save_images:
        out_dir = os.path.join(RESULTS_DIR, subdir_name)
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = None

    for img_name in sorted(os.listdir(TEST_IMAGES_DIR)):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".png")):
            continue

        img_path = os.path.join(TEST_IMAGES_DIR, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Görüntü okunamadı: {img_path}")
            continue

        total_images += 1
        img_names.append(img_name)

        # 2. Multi-scale detection
        rects, weights = hog.detectMultiScale(
            image,
            hitThreshold=hit_threshold,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05
        )

        # Bazı OpenCV sürümlerinde rects/weights tuple oluyor, listeye çeviriyoruz
        rects = list(rects)
        weights = [float(w) for w in list(weights)]

        # 3. NMS uygula
        nms_boxes, nms_scores = non_max_suppression(
            rects, weights, overlapThresh=0.5
        )

        num_det = len(nms_boxes)
        per_image_counts.append(num_det)
        total_detections += num_det

        # 4. Bounding box + confidence score çiz
        if save_images and out_dir is not None:
            for (x, y, w, h), score in zip(nms_boxes, nms_scores):
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = f"{score:.2f}"
                cv2.putText(
                    image,
                    text,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA
                )

            # Sonucu kaydet
            out_path = os.path.join(out_dir, f"det_{img_name}")
            cv2.imwrite(out_path, image)

        print(f"{img_name}: {num_det} kişi tespit edildi. (hit_threshold={hit_threshold})")

    print("--------- ÖZET ---------")
    print(f"Toplam görüntü sayısı: {total_images}")
    print(f"Toplam tespit sayısı: {total_detections}")
    if total_images > 0:
        print(f"Görüntü başına ortalama tespit: {total_detections / total_images:.2f}")
    print("------------------------")

    return img_names, per_image_counts


# ---- Opsiyonel: ground-truth kişi sayısı (sadece görüntü başına sayı) ----

def load_gt_counts(gt_counts_path):
    """
    Her test görüntüsünde GERÇEK kişi sayısını içeren CSV okur.
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


def evaluate_thresholds_pedestrian(threshold_list, gt_counts_path=None):
    """
    Farklı hit_threshold değerleri için:
    - Her görüntüde kaç kişi tespit edilmiş (istatistik)
    - (Varsa) ground truth üzerinden yaklaşık TP/FP/FN sayıları
    - Grafikler:
        * ped_detection_counts_threshold_X.png  (her test görüntüsü için tespit sayısı)
        * ped_thresholds_vs_avg_detections.png (eşik vs ortalama tespit sayısı)
        * ped_thresholds_vs_tp_fp_fn.png       (varsa GT ile eşik vs TP/FP/FN)
    Ayrıca false_positive_negative_examples_pedestrian.txt içine
    örnek görsellerin isimlerini yazar.
    """
    gt_counts = load_gt_counts(gt_counts_path)

    stats_per_threshold = {}
    img_names_global = None

    for th in threshold_list:
        # Bu çağrıda görselleri KAYDETME, sadece sayıları topla
        img_names, counts = run_detector_on_folder(
            hit_threshold=th,
            save_images=False,
            subdir_name=f"th_{th:.2f}"
        )

        if img_names_global is None:
            img_names_global = img_names

        if len(counts) > 0:
            avg_det = float(np.mean(counts))
        else:
            avg_det = 0.0

        tp_total = fp_total = fn_total = 0
        fp_imgs = []
        fn_imgs = []

        if gt_counts:
            for name, det_count in zip(img_names, counts):
                if name not in gt_counts:
                    continue
                gt = gt_counts[name]

                # Sadece kişi SAYISI üzerinden kaba TP/FP/FN hesabı
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
    if len(threshold_list) > 0:
        ref_th = threshold_list[0]
    else:
        ref_th = 0.0

    if img_names_global is not None and ref_th in stats_per_threshold:
        counts_ref = stats_per_threshold[ref_th]["counts"]
        indices = list(range(len(img_names_global)))

        plt.figure(figsize=(max(8, len(indices) * 0.4), 4))
        plt.bar(indices, counts_ref)
        plt.xticks(indices, img_names_global, rotation=90)
        plt.xlabel("Test görüntüleri")
        plt.ylabel("Tespit edilen kişi sayısı")
        plt.title(f"Her test görüntüsü için insan tespiti (hit_threshold={ref_th})")
        plt.tight_layout()
        out_path = os.path.join(
            RESULTS_DIR, f"ped_detection_counts_threshold_{ref_th}.png"
        )
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"-> Grafik kaydedildi: {out_path}")

    # 2) Threshold vs ortalama tespit sayısı grafiği
    if len(threshold_list) > 0:
        ths_sorted = sorted(threshold_list)
        avg_list = [stats_per_threshold[th]["avg_detections"] for th in ths_sorted]

        plt.figure(figsize=(6, 4))
        plt.plot(ths_sorted, avg_list, marker="o")
        plt.xlabel("hitThreshold")
        plt.ylabel("Görüntü başına ortalama tespit")
        plt.title("Farklı hitThreshold değerlerinin insan tespitine etkisi")
        plt.grid(True)
        plt.tight_layout()
        out_path = os.path.join(
            RESULTS_DIR, "ped_thresholds_vs_avg_detections.png"
        )
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"-> Grafik kaydedildi: {out_path}")

    # 3) Eğer ground truth varsa: threshold vs TP/FP/FN grafiği + FP/FN örnek listesi
    if gt_counts and len(threshold_list) > 0:
        ths_sorted = sorted(threshold_list)
        tp_list = [stats_per_threshold[th]["tp"] for th in ths_sorted]
        fp_list = [stats_per_threshold[th]["fp"] for th in ths_sorted]
        fn_list = [stats_per_threshold[th]["fn"] for th in ths_sorted]

        plt.figure(figsize=(6, 4))
        plt.plot(ths_sorted, tp_list, marker="o", label="TP")
        plt.plot(ths_sorted, fp_list, marker="o", label="FP")
        plt.plot(ths_sorted, fn_list, marker="o", label="FN")
        plt.xlabel("hitThreshold")
        plt.ylabel("Kişi sayısı (toplam)")
        plt.title("hitThreshold değişimine göre TP / FP / FN")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        out_path = os.path.join(
            RESULTS_DIR, "ped_thresholds_vs_tp_fp_fn.png"
        )
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"-> Grafik kaydedildi: {out_path}")

        fp_fn_path = os.path.join(
            RESULTS_DIR, "false_positive_negative_examples_pedestrian.txt"
        )
        with open(fp_fn_path, "w", encoding="utf-8") as f:
            f.write("False positive / false negative örnek görüntü isimleri\n")
            f.write("Not: Buradaki FP/FN, sadece kişi SAYISI üzerinden yaklaşık hesaplanmıştır.\n\n")
            for th in ths_sorted:
                f.write(f"=== hitThreshold = {th} ===\n")
                f.write("False Positive (fazla tespit edilenler):\n")
                for name in stats_per_threshold[th]["fp_imgs"]:
                    f.write(f"  - {name}\n")
                f.write("False Negative (eksik tespit edilenler):\n")
                for name in stats_per_threshold[th]["fn_imgs"]:
                    f.write(f"  - {name}\n")
                f.write("\n")
        print(f"-> FP/FN örnek listesi kaydedildi: {fp_fn_path}")


if __name__ == "__main__":
    # 1) Önce klasik insan tespiti koşusu (görselleri kaydeder)
    # Buradaki hit_threshold'u default deneyin (0.0) – rapora görsel koymak için yeterli
    run_detector_on_folder(
        hit_threshold=0.0,
        save_images=True,
        subdir_name="default"
    )

    # 2) Farklı threshold değerlerinin analizi + grafikler
    threshold_list = [0.0, 0.2, 0.4, 0.6]

    # Opsiyonel: Eğer TEST görüntüleri için kişi sayısını el ile sayıp
    # CSV dosyası hazırlarsan (image,count), buraya yolunu ver:
    gt_csv_path = os.path.join(RESULTS_DIR, "gt_counts_pedestrian.csv")
    if not os.path.exists(gt_csv_path):
        gt_csv_path = None  # ground truth yoksa basit istatistik + grafik

    evaluate_thresholds_pedestrian(threshold_list, gt_counts_path=gt_csv_path)
