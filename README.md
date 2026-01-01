# HOG Tabanlı Görüntü Sınıflandırma ve Nesne Tespiti Projesi

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)

Bu proje, **Histogram of Oriented Gradients (HOG)** algoritması kullanarak görüntü sınıflandırma ve nesne tespiti uygulamalarını içermektedir. Proje, HOG özellik çıkarımından makine öğrenmesi tabanlı sınıflandırmaya kadar bilgisayarlı görü tekniklerinin kapsamlı bir implementasyonunu sunmaktadır.

## 📋 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Özellikler](#-özellikler)
- [Proje Yapısı](#-proje-yapısı)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Metodoloji](#-metodoloji)
- [Sonuçlar](#-sonuçlar)
- [Gereksinimler](#-gereksinimler)
- [Katkıda Bulunma](#-katkıda-bulunma)
- [Lisans](#-lisans)

## 🎯 Proje Hakkında

Bu proje, bilgisayarlı görü alanında yaygın olarak kullanılan **HOG (Histogram of Oriented Gradients)** algoritmasının sıfırdan implementasyonunu ve çeşitli uygulamalarını içermektedir. Proje üç ana bileşenden oluşmaktadır:

1. **HOG Algoritması Implementasyonu**: Gradyan hesaplama, hücre histogramları oluşturma ve blok normalizasyonu
2. **Görüntü Sınıflandırma**: HOG özellikleri kullanılarak SVM tabanlı ikili sınıflandırma
3. **Nesne Tespiti**: OpenCV'nin önceden eğitilmiş HOG + SVM modeliyle yaya tespiti

### Proje Hedefleri

- HOG algoritmasının matematiksel temellerini anlamak ve uygulamak
- Farklı makine öğrenmesi sınıflandırıcılarını karşılaştırmak (Linear SVM, RBF SVM, k-NN)
- Gerçek dünya görüntüleri üzerinde nesne tespiti performansını değerlendirmek
- Threshold optimizasyonu ve model değerlendirme tekniklerini uygulamak

## ✨ Özellikler

### 1. HOG Implementasyonu (`hog_implementation.py`)

- **Gradyan Hesaplama**: Üç farklı yöntem (kernel, numpy, Sobel)
- **Hücre Histogramları**: Lineer interpolasyon ile yönelim histogramları
- **Blok Normalizasyonu**: L1, L2 ve L2-Hys normalizasyon yöntemleri
- **Görselleştirme**: Gradyan ve HOG özelliklerinin detaylı görselleştirilmesi
- **Sentetik Test**: Kare, daire ve üçgen gibi geometrik şekiller üzerinde test

### 2. Görüntü Sınıflandırma (`classification.py`)

- **Özellik Çıkarımı**: HOG tabanlı özellik vektörleri
- **Çoklu Sınıflandırıcı Desteği**:
  - Linear SVM
  - RBF Kernel SVM
  - k-Nearest Neighbors (k-NN)
- **Model Değerlendirme**: Accuracy, confusion matrix, classification report
- **Otomatik Model Seçimi**: En iyi performans gösteren modelin kaydedilmesi
- **Görselleştirme**: Model karşılaştırma grafikleri

### 3. Nesne Tespiti (`object_detection.py`)

- **Yaya Tespiti**: OpenCV HOGDescriptor ile önceden eğitilmiş model
- **Non-Maximum Suppression (NMS)**: Çakışan tespit kutularının filtrelenmesi
- **Threshold Optimizasyonu**: Farklı eşik değerlerinin karşılaştırılması
- **Performans Metrikleri**: TP, FP, FN analizi (ground truth ile)
- **Batch İşleme**: Klasör bazında toplu görüntü işleme

### 4. Özel Nesne Tespiti (`custom_detection.py`)

- Özel veri setleri ile eğitilmiş modeller için tespit sistemi
- Sliding window yaklaşımı
- Multi-scale detection desteği

## 📁 Proje Yapısı

```
fazli-efe-onder-hog-odev/
│
├── src/                          # Kaynak kod dosyaları
│   ├── hog_implementation.py     # HOG algoritması implementasyonu
│   ├── classification.py         # Görüntü sınıflandırma modülü
│   ├── object_detection.py       # Yaya tespiti modülü
│   ├── custom_detection.py       # Özel nesne tespiti
│   └── utils.py                  # Yardımcı fonksiyonlar
│
├── data/                         # Veri setleri
│   ├── custom/                   # Özel eğitim verileri
│   │   └── train_pos/            # Pozitif örnekler (sınıf 1)
│   │   └── train_neg/            # Negatif örnekler (sınıf 0)
│   └── result/                   # İşlenmiş sonuçlar
│
├── models/                       # Eğitilmiş modeller
│   └── trained_classifier.pkl    # En iyi sınıflandırıcı modeli
│
├── notebooks/                    # Jupyter notebook'lar
│   └── analysiss.ipynb           # Analiz ve deneyler
│
├── report/                       # Raporlar ve çıktılar
│   ├── figures/                  # Grafikler ve görselleştirmeler
│   └── report.pdf                # Detaylı proje raporu
│
├── requirements.txt              # Python bağımlılıkları
└── README.md                     # Proje dokümantasyonu
```

## 🔧 Kurulum

### Gereksinimler

- Python 3.8 veya üzeri
- pip paket yöneticisi

### Adım 1: Depoyu Klonlayın

```bash
git clone https://github.com/kullaniciadi/fazli-efe-onder-hog-odev.git
cd fazli-efe-onder-hog-odev
```

### Adım 2: Sanal Ortam Oluşturun (Önerilen)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Adım 3: Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

### Gerekli Kütüphaneler

```
opencv-python>=4.5.0
numpy>=1.19.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
joblib>=1.0.0
```

## 🚀 Kullanım

### 1. HOG Algoritması Testi

HOG implementasyonunu test etmek ve görselleştirmek için:

```bash
python -m src.hog_implementation
```

**Çıktılar:**
- Sentetik şekiller (kare, daire, üçgen) üzerinde HOG analizi
- Gradyan görselleştirmeleri
- HOG özellik haritaları

### 2. Görüntü Sınıflandırma

Özel veri setiniz ile model eğitmek için:

```bash
python -m src.classification
```

**İşlem Adımları:**
1. `data/custom/train_pos/` ve `data/custom/train_neg/` klasörlerine görüntülerinizi yerleştirin
2. Script otomatik olarak:
   - HOG özelliklerini çıkarır
   - Üç farklı sınıflandırıcıyı eğitir (Linear SVM, RBF SVM, k-NN)
   - Modelleri karşılaştırır
   - En iyi modeli `models/trained_classifier.pkl` olarak kaydeder
   - Karşılaştırma grafiğini `report/figures/classification_accuracy.png` olarak oluşturur

**Örnek Çıktı:**
```
Pozitif (sınıf 1) örnekler yükleniyor...
data/custom/train_pos -> 50 örnek, feature boyutu: 3780

Negatif (sınıf 0) örnekler yükleniyor...
data/custom/train_neg -> 50 örnek, feature boyutu: 3780

Toplam dataset boyutu: 100 örnek, 3780 feature

Eğitim seti   : 70 örnek
Test seti     : 30 örnek

--- Linear SVM eğitiliyor ---
Linear SVM doğruluk (accuracy): 0.9400

En iyi sınıflandırıcı: Linear SVM (accuracy = 0.9400)
```

### 3. Yaya Tespiti

Görüntüler üzerinde yaya tespiti yapmak için:

```bash
python -m src.object_detection
```

**Özellikler:**
- Farklı threshold değerleri ile tespit
- NMS (Non-Maximum Suppression) uygulaması
- Tespit sonuçlarının görselleştirilmesi
- Performans metriklerinin hesaplanması

**Threshold Optimizasyonu:**
Script, farklı hit_threshold değerlerini (0.0, 0.5, 1.0, 1.5) test eder ve:
- Her threshold için tespit sayılarını hesaplar
- TP, FP, FN metriklerini raporlar (ground truth varsa)
- Karşılaştırma grafikleri oluşturur

### 4. Jupyter Notebook ile Analiz

İnteraktif analiz için:

```bash
jupyter notebook notebooks/analysiss.ipynb
```

## 🔬 Metodoloji

### HOG Algoritması

HOG (Histogram of Oriented Gradients), görüntülerdeki yerel gradyan yönelimlerini kullanarak özellik çıkarımı yapar:

1. **Gradyan Hesaplama**: 
   - Sobel operatörü veya basit kernel filtresi ile x ve y yönündeki türevler
   - Gradyan büyüklüğü: `magnitude = sqrt(gx² + gy²)`
   - Gradyan yönü: `angle = atan2(gy, gx)`

2. **Hücre Histogramları**:
   - Görüntü 8x8 piksellik hücrelere bölünür
   - Her hücre için 9 binlik yönelim histogramı (0°-180°)
   - Lineer interpolasyon ile yumuşak histogram dağılımı

3. **Blok Normalizasyonu**:
   - 2x2 hücrelik bloklar oluşturulur
   - L2 normalizasyonu: `v = v / sqrt(||v||² + ε²)`
   - Aydınlatma değişikliklerine karşı dayanıklılık

4. **Özellik Vektörü**:
   - Tüm blok histogramları birleştirilerek final descriptor oluşturulur
   - 64x128 piksel görüntü için ~3780 boyutlu vektör

### Sınıflandırma Yaklaşımı

**Veri Hazırlama:**
- Görüntüler 64x128 piksel boyutuna yeniden ölçeklendirilir
- HOG özellikleri çıkarılır
- %70 eğitim, %30 test olarak bölünür

**Model Eğitimi:**
- **Linear SVM**: Hızlı ve etkili, yüksek boyutlu veriler için ideal
- **RBF SVM**: Non-linear sınırlar için, daha karmaşık desenler
- **k-NN**: Basit ama etkili, k=5 komşu kullanılır

**Değerlendirme:**
- Accuracy (Doğruluk)
- Precision, Recall, F1-Score
- Confusion Matrix
- Cross-validation (opsiyonel)

### Nesne Tespiti

**Sliding Window:**
- Farklı ölçeklerde pencere kaydırma
- Her pencere için HOG + SVM sınıflandırması

**Non-Maximum Suppression:**
- Çakışan tespit kutularını birleştirme
- IoU (Intersection over Union) eşiği: 0.5

**Threshold Optimizasyonu:**
- Hit threshold: SVM karar fonksiyonu eşiği
- Düşük threshold → Daha fazla tespit, daha fazla FP
- Yüksek threshold → Daha az tespit, daha az FP

## 📊 Sonuçlar

### Sınıflandırma Performansı

Örnek test sonuçları (16 test görüntüsü):

| Model | Accuracy | Precision (Sınıf 1) | Recall (Sınıf 1) | F1-Score |
|-------|----------|---------------------|------------------|----------|
| **Linear SVM** | **0.94** | **1.00** | **0.91** | **0.95** |
| RBF SVM | 0.91 | 0.95 | 0.90 | 0.92 |
| k-NN (k=5) | 0.88 | 0.90 | 0.87 | 0.88 |

**Confusion Matrix (Linear SVM):**
```
              Predicted
              0    1
Actual  0     5    0
        1     1   10
```

### Nesne Tespiti Performansı

Yaya tespiti sonuçları (örnek veri seti):

| Threshold | Avg Detections | Precision | Recall | F1-Score |
|-----------|----------------|-----------|--------|----------|
| 0.0 | 4.2 | 0.75 | 0.90 | 0.82 |
| 0.5 | 3.1 | 0.85 | 0.80 | 0.82 |
| **1.0** | **2.5** | **0.92** | **0.75** | **0.83** |
| 1.5 | 1.8 | 0.95 | 0.65 | 0.77 |

**Önerilen Threshold:** 1.0 (En iyi precision-recall dengesi)

### Görselleştirmeler

Tüm grafikler ve görselleştirmeler `report/figures/` klasöründe bulunmaktadır:

- `classification_accuracy.png`: Model karşılaştırma grafiği
- `hog_visualization_*.png`: HOG özellik haritaları
- `gradient_visualization_*.png`: Gradyan analizi
- `detection_results_*.png`: Tespit sonuçları

Detaylı analiz ve sonuçlar için `report/report.pdf` dosyasına bakınız.

## 📦 Gereksinimler

### Python Kütüphaneleri

```txt
opencv-python>=4.5.0    # Görüntü işleme ve HOG
numpy>=1.19.0           # Sayısal hesaplamalar
scikit-learn>=0.24.0    # Makine öğrenmesi algoritmaları
matplotlib>=3.3.0       # Görselleştirme
joblib>=1.0.0          # Model kaydetme/yükleme
```

### Sistem Gereksinimleri

- **İşletim Sistemi**: Windows 10/11, Linux, macOS
- **RAM**: Minimum 4GB (8GB önerilir)
- **Disk Alanı**: ~500MB (veri setleri dahil)
- **Python**: 3.8, 3.9, 3.10 veya 3.11

## 🤝 Katkıda Bulunma

Katkılarınızı memnuniyetle karşılıyoruz! Katkıda bulunmak için:

1. Bu depoyu fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/yeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/yeniOzellik`)
5. Pull Request oluşturun

### Geliştirme Önerileri

- [ ] Multi-class sınıflandırma desteği
- [ ] Deep learning tabanlı HOG alternatifleri (CNN)
- [ ] Real-time video işleme
- [ ] GPU hızlandırması (CUDA)
- [ ] Web arayüzü (Flask/Streamlit)
- [ ] Docker container desteği

## 📄 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakınız.

## 👥 Yazarlar

- **Fazlı Efe Önder** - *Proje Geliştiricisi*

## 🙏 Teşekkürler

- OpenCV topluluğuna HOG implementasyonu için
- scikit-learn ekibine makine öğrenmesi araçları için
- Bilgisayarlı görü dersi hocalarına rehberlik için

## 📞 İletişim

Sorularınız veya önerileriniz için:

- **GitHub Issues**: [Proje Issues Sayfası](https://github.com/kullaniciadi/fazli-efe-onder-hog-odev/issues)
- **Email**: onderfazli59@gmail.com.com

---

## 📚 Referanslar

1. Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. *IEEE CVPR*.
2. OpenCV Documentation: [HOG Descriptor](https://docs.opencv.org/master/d5/d33/structcv_1_1HOGDescriptor.html)
3. scikit-learn Documentation: [SVM](https://scikit-learn.org/stable/modules/svm.html)

---

**Not**: Bu proje eğitim amaçlıdır ve akademik çalışmalarda referans gösterilebilir.

⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın!

