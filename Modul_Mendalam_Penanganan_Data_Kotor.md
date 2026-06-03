# Modul Pendalaman — Penanganan Data Kotor & Berantakan
### Pelengkap Minggu 2 (Preprocessing & Cleaning) — GEMASTIK Divisi III

> Modul ini mengasumsikan data Anda **berantakan** seperti data hasil *scraping* nyata: encoding rusak, tipe tercampur, angka berbentuk teks, tanggal beragam format, kategori tidak konsisten, duplikat samar, dan teks penuh sampah Unicode. Setiap bagian menjelaskan **mekanisme/teori → mengapa bermasalah → kode yang dijelaskan baris demi baris → jebakan**.

---

## Filosofi: Diagnosa Dulu, Jangan "Tembak Langsung"

Kesalahan pemula: langsung `dropna()` atau `fillna(0)` tanpa tahu *kenapa* data hilang. Pembersihan yang benar berurutan:

```
1. DIAGNOSA  → potret kondisi data (profiling)
2. PERBAIKI STRUKTUR → encoding, tipe data, bentuk tabel
3. PERBAIKI ISI → missing, outlier, duplikat, kategori, teks
4. VALIDASI  → buktikan data sudah bersih (assertion/uji)
5. DOKUMENTASI → catat setiap keputusan + alasannya
```

Urutan ini penting. Membersihkan isi sebelum tipe data benar = sia-sia (mis. menghitung mean kolom yang masih berisi "1.000" sebagai string).

---

## BAGIAN 1 — Diagnosa & *Data Profiling*

### Teori
Sebelum menyentuh data, jawab dulu: kolom apa saja, tipe apa, berapa persen kosong, berapa banyak nilai unik (kardinalitas), dan **pola** kekosongannya. *Profiling* mengubah "data kotor" yang abstrak menjadi daftar masalah konkret yang bisa ditangani.

Konsep penting: **kekosongan yang sistematis**. Kalau kolom B selalu kosong ketika kolom A kosong, itu bukan kebetulan — ada penyebab struktural (mis. baris kosong hasil *scraping* gagal). Ini menentukan strategi imputasi nanti.

### Kode (dijelaskan)
```python
import pandas as pd, numpy as np

def potret_data(df: pd.DataFrame) -> pd.DataFrame:
    """Ringkasan kesehatan data dalam satu tabel — panggil ini PERTAMA."""
    out = pd.DataFrame({
        "tipe": df.dtypes,                                   # tipe data terdeteksi pandas
        "n_unik": df.nunique(dropna=False),                  # kardinalitas (termasuk NaN)
        "persen_kosong": (df.isna().mean() * 100).round(2),  # % missing per kolom
        "contoh": [df[c].dropna().unique()[:3] for c in df.columns],  # 3 contoh nilai
    })
    # Deteksi "kolom object yang sebenarnya angka" → kandidat perlu konversi tipe
    out["dugaan_numerik"] = [
        df[c].dropna().astype(str).str.replace(r"[^\d]", "", regex=True).str.len().gt(0).mean() > 0.8
        if df[c].dtype == "object" else False
        for c in df.columns
    ]
    return out

potret = potret_data(df)
print(potret)
```

**Penjelasan kunci:**
- `df.nunique(dropna=False)` memasukkan NaN ke hitungan unik — penting untuk tahu apakah kolom sebenarnya konstan/biner.
- Kolom bertipe `object` dengan kardinalitas sangat tinggi biasanya: ID, teks bebas, atau angka yang salah baca sebagai string.
- Kolom `dugaan_numerik` menandai kolom teks yang isinya didominasi digit → target konversi di Bagian 3.

### Visualisasi pola kekosongan
```python
import missingno as msno          # pip install missingno
msno.matrix(df)                   # baris putih = kosong; pola visual langsung terlihat
msno.heatmap(df)                  # korelasi kemunculan NaN antar kolom (-1..1)
```
**Cara baca heatmap:** nilai mendekati **+1** = dua kolom cenderung kosong bersamaan (kekosongan terkait/MAR). Nilai mendekati **0** = kosong secara acak (MCAR). Ini bukti untuk memilih strategi imputasi.

> **Trik lomba:** jalankan `from ydata_profiling import ProfileReport; ProfileReport(df).to_file("profil.html")` untuk laporan otomatis lengkap. Sangat membantu untuk bagian "Data Understanding" di makalah.

---

## BAGIAN 2 — Memperbaiki Encoding & Teks Rusak (*Mojibake*)

### Teori
Teks komputer adalah byte yang ditafsirkan dengan suatu *encoding* (UTF-8, Latin-1, CP1252, dll). **Mojibake** terjadi saat byte yang ditulis dengan satu encoding dibaca dengan encoding lain — hasilnya teks rusak seperti `Ã©` (seharusnya `é`), `â€™` (seharusnya tanda kutip `'`), atau `Indonesiaâ€¦`. Data hasil *scraping* sangat rentan ini karena sumbernya bercampur.

Ada juga masalah **karakter tak terlihat**: zero-width space (`\u200b`), non-breaking space (`\xa0`), BOM (`\ufeff`), dan karakter kontrol — semuanya merusak pencocokan string dan model teks.

### Kode (dijelaskan)
```python
# Langkah 0: deteksi encoding file sebelum membaca (jangan asal UTF-8)
import charset_normalizer
with open("data.csv", "rb") as f:
    hasil = charset_normalizer.from_bytes(f.read(100_000)).best()
print(hasil.encoding)          # mis. 'utf_8', 'cp1252', 'latin_1'

df = pd.read_csv("data.csv", encoding=hasil.encoding,
                 encoding_errors="replace")  # 'replace' agar tidak crash di byte rusak

# Langkah 1: perbaiki mojibake otomatis dengan ftfy (fix text for you)
import ftfy                      # pip install ftfy
df["teks"] = df["teks"].astype(str).map(ftfy.fix_text)
# ftfy mengenali pola "Ã©" -> "é", "â€™" -> "'", dll. secara cerdas

# Langkah 2: normalisasi Unicode (samakan bentuk karakter yang "kelihatan sama")
import unicodedata
def normalisasi_unicode(s: str) -> str:
    # NFKC menggabungkan bentuk-kompatibel: "ﬁ"->"fi", "①"->"1", lebar-penuh->normal
    return unicodedata.normalize("NFKC", s)
df["teks"] = df["teks"].map(normalisasi_unicode)

# Langkah 3: buang karakter tak terlihat & kontrol
import re
SAMPAH = re.compile(r"[\u200b-\u200f\ufeff\xa0\u00ad]")  # zero-width, BOM, nbsp, soft-hyphen
def bersihkan_tak_terlihat(s: str) -> str:
    s = SAMPAH.sub(" ", s)
    # buang karakter kontrol (kategori Unicode 'C') kecuali newline/tab biasa
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C" or ch in "\n\t")
    return re.sub(r"\s+", " ", s).strip()   # rapikan spasi berlebih
df["teks"] = df["teks"].map(bersihkan_tak_terlihat)
```

**Jebakan:** `\xa0` (non-breaking space) **terlihat** seperti spasi biasa tapi bukan — `"Jakarta\xa0Pusat" != "Jakarta Pusat"`. Banyak duplikat kategori "misterius" sebenarnya disebabkan karakter ini.

---

## BAGIAN 3 — Memperbaiki Tipe Data & Angka Berbentuk Teks

### Teori
Data *scraping*/ekspor sering menyimpan angka sebagai teks bercampur simbol: `"Rp 1.250.000,50"`, `"12,5%"`, `"1,5 jt"`, `"N/A"`. Format Indonesia memakai **titik sebagai pemisah ribuan** dan **koma sebagai desimal** — kebalikan dari format Inggris. Salah konversi = angka kacau (1.250.000 jadi 1.25).

Kolom bertipe `object` juga bisa menyembunyikan **tipe tercampur** (sebagian int, sebagian string), yang akan menggagalkan operasi numerik.

### Kode (dijelaskan)
```python
def ke_angka_id(seri: pd.Series) -> pd.Series:
    """Konversi teks bergaya Indonesia -> float. 'Rp 1.250.000,50' -> 1250000.50"""
    s = (seri.astype(str)
              .str.replace(r"[^\d,.-]", "", regex=True)  # buang 'Rp', spasi, '%', huruf
              .str.replace(".", "", regex=False)          # hapus titik (pemisah ribuan)
              .str.replace(",", ".", regex=False))        # koma desimal -> titik
    return pd.to_numeric(s, errors="coerce")              # gagal konversi -> NaN, bukan error

df["harga"] = ke_angka_id(df["harga"])

# Untuk persentase: "12,5%" -> 0.125
df["diskon"] = ke_angka_id(df["diskon"]) / 100

# Deteksi kolom bertipe TERCAMPUR (bahaya tersembunyi)
def cek_tipe_campur(seri):
    tipe = seri.dropna().map(type).value_counts()
    return tipe if len(tipe) > 1 else None
for c in df.select_dtypes("object"):
    t = cek_tipe_campur(df[c])
    if t is not None: print(f"⚠️ {c} bertipe campur:\n{t}\n")
```

**Penjelasan `errors="coerce"`:** ini fondasi pembersihan tahan-banting. Alih-alih *crash* saat ketemu `"N/A"`, nilai gagal-konversi diubah jadi `NaN` sehingga bisa ditangani seragam di tahap missing values. **Selalu** cek berapa banyak yang jadi NaN setelah `coerce` — kalau banyak, mungkin pola pembersihan Anda salah:
```python
print("Gagal dikonversi:", df["harga"].isna().sum())
```

### Optimasi tipe (untuk data besar hasil scraping)
```python
# Hemat memori & percepat: kategori untuk kolom kardinalitas rendah
for c in df.select_dtypes("object"):
    if df[c].nunique() / len(df) < 0.5:       # < 50% nilai unik = layak jadi category
        df[c] = df[c].astype("category")
# Turunkan presisi numerik bila aman
df = df.apply(lambda s: pd.to_numeric(s, downcast="integer") if s.dtype=="int64" else s)
```

---

## BAGIAN 4 — Missing Values (Pendalaman)

### Teori: tiga mekanisme & dampaknya
1. **MCAR** (*Missing Completely At Random*) — hilang tanpa pola. Imputasi sederhana (median/modus) relatif aman; menghapus baris tidak membuat bias (tapi membuang data).
2. **MAR** (*Missing At Random*) — kehilangan bergantung pada kolom **lain** yang teramati (mis. pendapatan kosong lebih sering pada usia muda). Imputasi harus **berbasis fitur lain** (KNN/iterative), bukan rata-rata global.
3. **MNAR** (*Missing Not At Random*) — kehilangan bergantung pada **nilai itu sendiri** (mis. orang berpenghasilan tinggi enggan mengisi). Paling berbahaya; sering perlu dimodelkan eksplisit / dijadikan fitur penanda.

**Konsep tersembunyi:** *missing yang menyamar*. Nilai seperti `-999`, `0`, `"-"`, `"unknown"`, `"null"`, `9999` sering sebenarnya "kosong" yang disamarkan. Wajib dicari dan disatukan jadi NaN dulu.

### Kode (dijelaskan)
```python
# Langkah 1: satukan "missing yang menyamar" menjadi NaN sungguhan
TANDA_KOSONG = ["", "-", "n/a", "na", "null", "none", "unknown", "tidak ada", "?", "9999", "-999"]
df = df.replace(r"(?i)^\s*(%s)\s*$" % "|".join(TANDA_KOSONG), np.nan, regex=True)
# (?i) = case-insensitive; ^\s* dan \s*$ menoleransi spasi di tepi

# Langkah 2: tambahkan FITUR PENANDA sebelum imputasi (krусial untuk MNAR/MAR)
for c in ["pendapatan", "usia"]:
    df[f"{c}_was_missing"] = df[c].isna().astype(int)
# Mengapa? Fakta "data ini kosong" sendiri bisa prediktif. Model boleh memanfaatkannya.

# Langkah 3a: imputasi sederhana (MCAR) — median tahan terhadap skew/outlier
from sklearn.impute import SimpleImputer
df["usia"] = SimpleImputer(strategy="median").fit_transform(df[["usia"]])

# Langkah 3b: imputasi berbasis tetangga (MAR) — gunakan korelasi antar fitur
from sklearn.impute import KNNImputer
num = ["usia", "pendapatan", "lama_langganan"]
df[num] = KNNImputer(n_neighbors=5, weights="distance").fit_transform(df[num])
# Tiap NaN diisi rata-rata tertimbang 5 baris paling mirip (perlu data sudah di-scale)

# Langkah 3c: imputasi iteratif/MICE (paling kuat untuk MAR kompleks)
from sklearn.experimental import enable_iterative_imputer   # WAJIB diimpor dulu
from sklearn.impute import IterativeImputer
df[num] = IterativeImputer(random_state=42, max_iter=10).fit_transform(df[num])
# Memodelkan tiap kolom sebagai fungsi kolom lain, berulang sampai konvergen
```

**Jebakan terbesar (lagi): leakage.** Semua imputasi di atas harus di-`fit` **hanya pada data training** lalu di-`transform` ke test. Saat lomba, bungkus dalam `Pipeline` (lihat Bagian 9) supaya otomatis benar. Median/parameter KNN yang dihitung dari seluruh data = kebocoran informasi test.

**Kapan menghapus, bukan mengimputasi?** Jika kolom >60–70% kosong dan tidak prediktif → buang kolomnya. Jika baris kunci (mis. target) kosong → buang barisnya. Imputasi besar-besaran pada kolom yang hampir kosong hanya menambah noise.

---

## BAGIAN 5 — Outliers (Pendalaman)

### Teori
Outlier bisa berarti tiga hal berbeda: (a) **error** (salah input, mis. usia 999), (b) **anomali sah** yang justru penting (transaksi fraud!), atau (c) **ekor distribusi wajar**. Keputusan menghapus/menahan **bergantung konteks**, bukan rumus. Membuang outlier secara membabi buta bisa membuang sinyal terpenting.

Metode univariat klasik (z-score) **tidak tahan** terhadap outlier itu sendiri karena mean & std-nya ikut tergeser. Solusi lebih kuat: **Modified Z-score berbasis MAD** (Median Absolute Deviation) yang memakai median.

Untuk data berdimensi banyak, outlier bisa "tersembunyi": tiap kolom normal, tapi **kombinasinya** janggal (mis. tinggi 150 cm berat 150 kg). Ini butuh metode multivariat.

### Kode (dijelaskan)
```python
# --- Univariat tahan-banting: Modified Z-score (MAD) ---
def outlier_mad(seri, ambang=3.5):
    med = seri.median()
    mad = (seri - med).abs().median()           # sebaran berbasis median
    if mad == 0: return pd.Series(False, index=seri.index)
    z_mod = 0.6745 * (seri - med) / mad         # 0.6745 = faktor konsistensi normal
    return z_mod.abs() > ambang                  # True = outlier
mask = outlier_mad(df["harga"])
print("Outlier harga:", mask.sum())

# Strategi penanganan: WINSORIZE (capping), bukan buang, agar tak kehilangan baris
lo, hi = df["harga"].quantile([0.01, 0.99])
df["harga"] = df["harga"].clip(lo, hi)

# --- Multivariat: Isolation Forest (deteksi anomali kombinasi fitur) ---
from sklearn.ensemble import IsolationForest
fitur = df[["tinggi", "berat", "usia"]].dropna()
iso = IsolationForest(contamination=0.02, random_state=42)  # asumsi ~2% anomali
df.loc[fitur.index, "skor_anomali"] = iso.fit_predict(fitur)  # -1 = anomali, 1 = normal
print(df["skor_anomali"].value_counts())
```

**Penjelasan `contamination`:** estimasi proporsi anomali. Jangan terlalu tinggi — kalau set 0.1 padahal anomali nyata 1%, Anda membuang 9% data sehat. Untuk kasus *fraud/anomaly detection* (sering jadi tema GEMASTIK), justru baris ber-skor -1 inilah **targetnya**, bukan sampah yang dibuang.

> **Aturan emas:** sebelum menghapus outlier, tanya "apakah ini error atau fenomena nyata?". Catat alasannya di makalah. Juri akan menguji ini.

---

## BAGIAN 6 — Duplikat: Persis & Samar (*Fuzzy*)

### Teori
Duplikat ada dua jenis. **Persis** (semua kolom identik) mudah. Yang sulit adalah **duplikat samar**: baris yang merujuk entitas sama tapi ditulis berbeda — "PT. Maju Jaya", "PT Maju Jaya", "pt maju jaya  " — akibat spasi, kapitalisasi, tanda baca, atau salah ketik. Data *scraping* penuh ini (komentar yang di-*repost*, entri ganda).

Membersihkannya butuh dua langkah: **normalisasi** (samakan bentuk) lalu **pencocokan kemiripan** (*string similarity*).

### Kode (dijelaskan)
```python
# Duplikat persis
print("Duplikat persis:", df.duplicated().sum())
df = df.drop_duplicates()

# Duplikat dalam grup: simpan entri TERBARU per ID
df = (df.sort_values("tanggal")
        .drop_duplicates(subset="user_id", keep="last"))

# --- Duplikat SAMAR ---
# Langkah 1: kunci normalisasi (lowercase, buang tanda baca & spasi berlebih)
import re
def kunci_norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"[^\w\s]", "", s)     # buang tanda baca
    s = re.sub(r"\s+", " ", s)        # spasi tunggal
    return s
df["nama_kunci"] = df["nama_perusahaan"].map(kunci_norm)
df = df.drop_duplicates(subset="nama_kunci")   # tangkap dupe yang beda hanya di format

# Langkah 2: pencocokan fuzzy untuk salah ketik (rapidfuzz — cepat)
from rapidfuzz import fuzz, process     # pip install rapidfuzz
kandidat = df["nama_kunci"].unique().tolist()
# contoh: cari yang mirip "maju jaya" di atas ambang 90/100
mirip = process.extract("maju jaya", kandidat, scorer=fuzz.token_sort_ratio, limit=5)
print(mirip)   # [('maju jaya', 100, ..), ('maju jayaa', 95, ..), ...]
```

**Penjelasan `token_sort_ratio`:** mengurutkan kata sebelum membandingkan, sehingga "jaya maju" cocok dengan "maju jaya". Untuk membangun daftar kanonik, kelompokkan semua string ber-skor di atas ambang ke satu nama baku.

**Jebakan:** pencocokan fuzzy itu O(n²) — untuk data besar, lakukan *blocking* dulu (bandingkan hanya dalam grup yang berbagi huruf awal/kota sama) agar tidak meledak.

---

## BAGIAN 7 — Standardisasi Kategori yang Tidak Konsisten

### Teori
Kolom kategorik sering punya banyak ejaan untuk hal sama: "Jakarta", "JKT", "Jakarta Pusat", "DKI Jakarta". Bagi model, ini lima kategori berbeda → informasi terpecah, fitur membengkak. Tujuan: **memetakan semua varian ke satu nilai baku**.

Juga ada masalah **kategori langka** (*rare categories*): nilai yang muncul hanya 1–2 kali. Setelah one-hot, ini menciptakan kolom hampir-nol yang menambah noise & risiko overfitting → biasanya digabung jadi "Lainnya".

### Kode (dijelaskan)
```python
# Langkah 1: normalisasi dasar
df["kota"] = (df["kota"].astype(str).str.strip().str.lower()
                .str.replace(r"\s+", " ", regex=True))

# Langkah 2: peta baku manual (untuk kasus yang Anda tahu)
peta_kota = {
    "jkt": "jakarta", "dki jakarta": "jakarta", "jakarta pusat": "jakarta",
    "bdg": "bandung", "kota bandung": "bandung",
}
df["kota"] = df["kota"].replace(peta_kota)

# Langkah 3: gabung kategori langka -> "lainnya"
ambang = 10                                   # minimal kemunculan
freq = df["kota"].value_counts()
langka = freq[freq < ambang].index
df["kota"] = df["kota"].where(~df["kota"].isin(langka), "lainnya")
# .where: pertahankan nilai jika kondisi True, selain itu ganti "lainnya"

print(df["kota"].value_counts())
```

**Penjelasan langkah 3:** `value_counts()` menghitung frekuensi, lalu `where(~isin(langka), ...)` menyisakan kategori umum dan melebur sisanya. Ini menstabilkan distribusi & mengurangi dimensi setelah encoding. Catat ambang yang Anda pilih di makalah (keputusan = bisa dipertanggungjawabkan).

---

## BAGIAN 8 — Tanggal & Pembersihan Teks Mendalam

### Tanggal: beragam format
```python
# Format campur ("12/03/2024", "2024-03-12", "12 Mar 2024") -> errors='coerce'
df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce", dayfirst=True)
# dayfirst=True penting untuk format Indonesia/Eropa (hari/bulan/tahun)

# Nama bulan Bahasa Indonesia -> standar sebelum parsing
bulan_id = {"januari":"January","februari":"February","maret":"March","mei":"May",
            "agustus":"August","oktober":"October","desember":"December"}  # dst.
df["tgl_str"] = df["tgl_str"].str.lower().replace(bulan_id, regex=True)
df["tanggal"] = pd.to_datetime(df["tgl_str"], errors="coerce")

print("Tanggal gagal di-parse:", df["tanggal"].isna().sum())   # selalu cek!
```

### Teori pembersihan teks untuk data *scraping* (sangat relevan NLP)
Teks media sosial penuh: tag HTML (`&amp;`, `<br>`), URL, *mention* (`@user`), tagar, emoji, huruf berulang ekspresif ("baguuuss", "mantull"), dan slang. Tiap elemen perlu keputusan: **buang** (URL, HTML) atau **manfaatkan sebagai fitur** (jumlah emoji bisa menandakan emosi pada analisis sentimen). Jangan buang membabi buta.

```python
import re, html

def bersihkan_teks_sosmed(s: str, simpan_emoji=False) -> str:
    s = html.unescape(s)                                  # &amp; -> &, &lt; -> <
    s = re.sub(r"<[^>]+>", " ", s)                        # buang tag HTML
    s = re.sub(r"http\S+|www\.\S+", " ", s)               # buang URL
    s = re.sub(r"@\w+", " ", s)                           # buang mention
    s = re.sub(r"#(\w+)", r"\1", s)                       # tagar -> kata isinya
    if not simpan_emoji:
        s = re.sub(r"[^\w\s.,!?]", " ", s, flags=re.UNICODE)  # buang simbol/emoji
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)                  # "baguuuss" -> "baguss"
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

df["bersih"] = df["teks"].astype(str).map(bersihkan_teks_sosmed)
```

**Penjelasan `(.)\1{2,}` → `\1\1`:** menormalkan huruf berulang ≥3 menjadi 2 huruf ("kerennnnn" → "kerenn"), mengurangi varian token tanpa menghilangkan penekanan total. Untuk Bahasa Indonesia, lanjutkan dengan **stopword removal (Sastrawi)** dan opsional **stemming** seperti di Minggu 6.

> **Trik fitur:** sebelum membuang emoji, hitung dulu — `df["n_emoji"] = ...`. Jumlah emoji/tanda seru sering jadi fitur kuat untuk sentimen.

---

## BAGIAN 9 — Menjadikan Pembersihan Reproducible & Anti-Leakage

### Teori
Semua langkah di atas harus bisa **dijalankan ulang persis sama** pada data baru (saat demo final juri memberi data uji!). Caranya: bungkus pembersihan yang **belajar dari data** (imputasi, scaling, daftar kategori baku) ke dalam objek yang bisa `fit`/`transform`. Pembersihan **deterministik** (regex, normalisasi teks) boleh pakai `FunctionTransformer`; yang **belajar parameter** sebaiknya jadi *custom transformer*.

### Kode: custom transformer yang aman dari leakage
```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

class PembersihKategori(BaseEstimator, TransformerMixin):
    """Belajar daftar kategori 'umum' DARI TRAIN, terapkan ke data apa pun."""
    def __init__(self, kolom, ambang=10):
        self.kolom, self.ambang = kolom, ambang
    def fit(self, X, y=None):
        self.kategori_umum_ = {}
        for c in self.kolom:                          # pelajari HANYA dari X (train)
            freq = X[c].astype(str).str.lower().str.strip().value_counts()
            self.kategori_umum_[c] = set(freq[freq >= self.ambang].index)
        return self
    def transform(self, X):
        X = X.copy()
        for c in self.kolom:
            s = X[c].astype(str).str.lower().str.strip()
            X[c] = s.where(s.isin(self.kategori_umum_[c]), "lainnya")
        return X

# Rangkai jadi pipeline; .fit(train) lalu .transform(test) -> otomatis tanpa leakage
pipa_bersih = Pipeline([
    ("kategori", PembersihKategori(kolom=["kota", "segmen"], ambang=10)),
    # ... transformer numerik (RobustScaler tahan outlier), imputasi, dst.
])
pipa_bersih.fit(X_train)
X_train_bersih = pipa_bersih.transform(X_train)
X_test_bersih  = pipa_bersih.transform(X_test)   # memakai kategori dari TRAIN saja
```

**Inti pelajaran:** parameter apa pun yang "dihitung dari data" (kategori umum, median imputasi, batas winsorize, mean/std scaler) **wajib dipelajari di `fit` (train)** dan dipakai ulang di `transform` (test/data baru). Inilah yang membedakan pipeline juara dari skrip yang skornya bagus di latihan tapi jatuh di penilaian.

---

## BAGIAN 10 — Validasi: Buktikan Data Sudah Bersih

### Teori
Setelah membersihkan, **jangan berasumsi** sudah benar — uji dengan *assertion*. Validasi otomatis menangkap regresi (mis. setelah ubah kode, tiba-tiba muncul NaN lagi). Ini juga bukti kualitas yang bisa Anda tunjukkan ke juri.

### Kode (dijelaskan)
```python
def validasi(df):
    assert df["harga"].between(0, 1e9).all(), "Ada harga di luar rentang wajar"
    assert df["usia"].between(0, 120).all(),  "Ada usia tidak masuk akal"
    assert df.duplicated().sum() == 0,        "Masih ada duplikat"
    assert df["target"].notna().all(),        "Target tidak boleh kosong"
    assert df["kota"].nunique() < 50,         "Kategori kota tidak terstandardisasi"
    print("✅ Semua validasi lolos")
validasi(df_bersih)
```

Untuk proyek serius, gunakan pustaka **`pandera`** (skema data deklaratif) atau **Great Expectations** untuk validasi yang lebih rapi dan terdokumentasi.

```python
import pandera as pa
skema = pa.DataFrameSchema({
    "harga": pa.Column(float, pa.Check.in_range(0, 1e9)),
    "usia":  pa.Column(float, pa.Check.in_range(0, 120), nullable=False),
    "kota":  pa.Column(str,   pa.Check.isin(["jakarta","bandung","lainnya"])),
})
skema.validate(df_bersih)   # melempar error jelas bila ada yang melanggar
```

---

## Daftar Periksa Pembersihan (Cetak & Tempel)

```
[ ] Profiling dijalankan; daftar masalah konkret dibuat
[ ] Encoding terdeteksi benar; mojibake & karakter tak terlihat dibersihkan (ftfy + NFKC)
[ ] Semua kolom bertipe benar; angka-teks dikonversi (errors='coerce' dicek)
[ ] 'Missing yang menyamar' (-999, 'N/A', dll.) disatukan jadi NaN
[ ] Fitur penanda *_was_missing dibuat sebelum imputasi
[ ] Strategi imputasi sesuai mekanisme (MCAR/MAR/MNAR), di-fit di TRAIN saja
[ ] Outlier didiagnosa (error vs sinyal); ditangani dengan justifikasi
[ ] Duplikat persis & samar (fuzzy) ditangani
[ ] Kategori distandardisasi; kategori langka digabung
[ ] Tanggal di-parse (cek jumlah gagal); teks scraping dibersihkan
[ ] Seluruh proses dibungkus Pipeline (reproducible, anti-leakage)
[ ] Validasi/assertion lolos; keputusan didokumentasikan untuk makalah
```

## Prinsip yang Wajib Diingat
1. **Diagnosa sebelum aksi** — kenali pola masalah dulu.
2. **`errors='coerce'` adalah teman** — gagal jadi NaN, bukan crash; lalu cek jumlahnya.
3. **Outlier & missing bukan selalu sampah** — kadang itu justru sinyal/target.
4. **Apa pun yang dipelajari dari data → fit di train, transform di test** (anti-leakage).
5. **Dokumentasikan setiap keputusan + alasannya** — inilah yang dinilai juri, bukan sekadar kode yang jalan.
