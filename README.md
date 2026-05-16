# Tugas Besar 2 IF3270 Pembelajaran Mesin
## Convolutional Neural Network dan Recurrent Neural Network

Implementasi *forward propagation from scratch* untuk CNN, Simple RNN, dan LSTM, serta eksplorasi pipeline *image captioning* menggunakan arsitektur encoder-decoder CNN LSTM. Tugas ini menggunakan dua buah dataset, yakni Intel Image Classification (klasifikasi gambar, 6 kelas) dan Flickr8k (*image captioning*).

## Struktur Repository

```
Tubes2-ML/
├── src/
│   ├── notebook/
│   │   ├── notebook_cnn.ipynb      # Eksperimen CNN
│   │   └── notebook_rnn_lstm.ipynb # Eksperimen RNN & LSTM
│   ├── scratchlayers/
│   │   ├── cnn/                    # Implementasi Conv2D, LocallyConnected2D, Pooling, dll.
│   │   ├── rnn/                    # Implementasi SimpleRNN cell
│   │   ├── lstm/                   # Implementasi LSTM cell
│   │   ├── dense/                  # Implementasi Dense layer
│   │   ├── embedding/              # Implementasi Embedding layer
│   │   └── core/                   # Layer base, Sequential model, aktivasi
│   └── test/                       # Script pengujian
├── dataset/                        # Dataset
├── weights/                        # Bobot model hasil pelatihan
└── doc/                            # Laporan PDF
```

---

## Setup

### 1. Clone Repository

```bash
git clone https://github.com/karolyangqian/Tubes2-ML.git
cd Tubes2-ML
```

### 2. Install Dependencies

```bash
pip install tensorflow==2.21.0 tf-keras==2.21.0 numpy matplotlib seaborn scikit-learn pillow h5py jupyter
```

### 3. Persiapkan Dataset

- Intel Image Classification: Unduh dari [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification), ekstrak ke `dataset/` sehingga strukturnya:
  ```
  dataset/
  ├── seg_train/seg_train/
  │   ├── buildings/
  │   ├── forest/
  │   └── ...
  └── seg_test/seg_test/
  ```
- Flickr8k: Unduh dari [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k), ekstrak ke `dataset/flickr8k/`.

## Menjalankan Notebook

```bash
cd src/notebook
jupyter notebook
```

- `notebook_cnn.ipynb` (Eksperimen CNN)
- `notebook_rnn_lstm.ipynb` (Eksperimen RNN & LSTM)

Jalankan seluruh sel secara berurutan. Bobot model akan disimpan otomatis ke folder `weights/` sehingga tidak perlu melatih ulang pada run berikutnya.

## Pembagian Tugas

| No. | NIM | Nama | Tugas |
|---|---|---|---|
| 1 | 13523077 | Albertus Christian Poandy | Implementasi *Recurrent Neural Network* (RNN), Notebook RNN dan LSTM |
| 2 | 13523087 | Grace Evelyn Simon | Implementasi *Convolutional Neural Network* (CNN), Notebook CNN, Bonus Visualisasi Fitur CNN |
| 3 | 13523093 | Karol Yangqian Poetracahya | Implementasi *Long Short-Term Memory Network* (LSTM), Analisis Hasil Pengujian CNN, RNN, LSTM |
