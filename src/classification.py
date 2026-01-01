import os
import cv2
import numpy as np
import joblib

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt  # GRAFİK İÇİN

# ================== PATH ve PARAMETRELER ==================

# Proje kök dizini
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Aynı custom klasör yapısını kullanıyoruz
CUSTOM_DIR = os.path.join(BASE_DIR, "data", "custom")
TRAIN_POS_DIR = os.path.join(CUSTOM_DIR, "train_pos")  # sınıf 1
TRAIN_NEG_DIR = os.path.join(CUSTOM_DIR, "train_neg")  # sınıf 0

# Eğitilmiş en iyi sınıflandırıcıyı buraya kaydedeceğiz (ÖDEVDE İSTENEN İSİM)
CLASSIFIER_PATH = os.path.join(BASE_DIR, "trained_classifier.pkl")

# Grafiklerin kaydedileceği klasör
FIG_DIR = os.path.join(BASE_DIR, "report", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# HOG penceresi
WINDOW_SIZE = (64, 128)  # (width, height)

# ==========================================================


def get_hog():
    """
    Sabit HOG parametrizasyonu döndürür.
    Problem 2'de kullandığımız HOG ile aynı olsun diye ayarladık.
    """
    hog = cv2.HOGDescriptor(
        _winSize=WINDOW_SIZE,
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    return hog


def load_images_and_extract_hog(folder, label):
    """
    Verilen klasördeki tüm görüntüler için HOG feature çıkarır.
    """
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

        # Tüm patch'leri aynı boyuta getir
        img_resized = cv2.resize(img, WINDOW_SIZE)

        # HOG hesapla
        hog_desc = hog.compute(img_resized)  # (N,1) vektör
        features.append(hog_desc.flatten())
        labels.append(label)

    if len(features) == 0:
        print(f"UYARI: {folder} içinde uygun görüntü bulunamadı.")
        return np.empty((0,)), np.empty((0,))

    X = np.array(features)
    y = np.array(labels)

    print(f"{folder} -> {len(y)} örnek, feature boyutu: {X.shape[1]}")
    return X, y


def build_dataset():
    """
    train_pos ve train_neg klasörlerinden dataset'i oluşturur.
    """
    print("Pozitif (sınıf 1) örnekler yükleniyor...")
    X_pos, y_pos = load_images_and_extract_hog(TRAIN_POS_DIR, 1)

    print("Negatif (sınıf 0) örnekler yükleniyor...")
    X_neg, y_neg = load_images_and_extract_hog(TRAIN_NEG_DIR, 0)

    if X_pos.size == 0 or X_neg.size == 0:
        print("HATA: Pozitif veya negatif veri yok. Klasörleri kontrol et.")
        return None, None

    X = np.vstack((X_pos, X_neg))
    y = np.hstack((y_pos, y_neg))

    print(f"Toplam dataset boyutu: {X.shape[0]} örnek, {X.shape[1]} feature\n")
    return X, y


def train_and_evaluate_classifiers(X, y):
    """
    Farklı sınıflandırıcıları eğitir ve test setinde karşılaştırır.
    En iyi modeli (accuracy'ye göre) kaydeder.
    Konsola tablo basar ve accuracy bar grafiğini PNG olarak kaydeder.
    """
    # Eğitim / test bölünmesi
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    print(f"Eğitim seti   : {X_train.shape[0]} örnek")
    print(f"Test seti     : {X_test.shape[0]} örnek\n")

    classifiers = {
        "Linear SVM": LinearSVC(),
        "RBF SVM": SVC(kernel="rbf", gamma="scale"),
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
    }

    best_name = None
    best_clf = None
    best_acc = -1.0

    # Tablo + grafik için sonuçları bir listede tutalım
    results = []

    for name, clf in classifiers.items():
        print(f"--- {name} eğitiliyor ---")
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"{name} doğruluk (accuracy): {acc:.4f}")
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification report:")
        print(classification_report(y_test, y_pred, digits=3))

        results.append((name, acc))

        if acc > best_acc:
            best_acc = acc
            best_clf = clf
            best_name = name

        print("-" * 60)

    # En iyi modeli kaydet
    if best_clf is not None:
        joblib.dump(best_clf, CLASSIFIER_PATH)
        print(
            f"\nEn iyi sınıflandırıcı: {best_name} "
            f"(accuracy = {best_acc:.4f})"
        )
        print(f"Kaydedilen model yolu: {CLASSIFIER_PATH}")
    else:
        print("HATA: Hiçbir sınıflandırıcı eğitilemedi.")

    # === Konsola tablo olarak özet bas ===
    print("\n=== MODELLERİN DOĞRULUK KARŞILAŞTIRMASI ===")
    print("{:<15} {:>10}".format("Model", "Accuracy"))
    print("-" * 27)
    for name, acc in results:
        print("{:<15} {:>10.4f}".format(name, acc))
    print("===========================================")

    # === BAR GRAFİK OLUŞTUR VE KAYDET ===
    model_names = [r[0] for r in results]
    accuracies = [r[1] for r in results]

    plt.figure(figsize=(6, 4))
    plt.bar(model_names, accuracies)
    plt.ylim(0, 1.0)
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.title("HOG Tabanlı Sınıflandırıcıların Doğruluk Karşılaştırması")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    fig_path = os.path.join(FIG_DIR, "classification_accuracy.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"\nAccuracy bar grafiği kaydedildi: {fig_path}")


def main():
    X, y = build_dataset()
    if X is None:
        return

    train_and_evaluate_classifiers(X, y)


if __name__ == "__main__":
    main()
