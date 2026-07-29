# DEA + Gecikmeli Malmquist + Panel Analizi - Web Sitesi

## Dosyalar
- `dea_module.py` — CCR/BCC iki asamali DEA (GAMS'la dogrulandi, bkz. asagida)
- `malmquist_module.py` — Gecikmeli Malmquist mesafe fonksiyonu (GAMS'la dogrulandi)
- `panel_module.py` — Pooled OLS -> FE -> RE -> Hausman -> robust/clustered panel analizi
- `excel_okuma.py` — Excel okuma + veri dogrulama
- `pipeline.py` — Tum adimlari birbirine baglayan orkestrasyon katmani
- `app.py` — Streamlit web arayuzu

## Kurulum (kendi bilgisayarinizda)
```
pip install -r requirements.txt
streamlit run app.py
```
Tarayicida otomatik acilir (genelde http://localhost:8501).

## Ucretsiz internete acma (Streamlit Community Cloud)
1. Bu dosyalari bir GitHub reposuna yukleyin (app.py, tum .py modulleri, requirements.txt)
2. https://share.streamlit.io adresinden GitHub hesabinizla giris yapin
3. "New app" -> reponuzu secin -> main file olarak `app.py` gosterin -> Deploy
4. Birkaç dakika icinde herkese acik bir URL alirsiniz (ucretsiz katmanda)

## Excel sablonu
Uzun format, sutunlar:
`Donem | DMU | Girdi_<isim> | Girdi_<isim2> | ... | Cikti_<isim> | ...`

Donemler Excel'de kronolojik sirada olmali. Girdi_/Cikti_ sutun sayisi
sinirsiz — kod otomatik algilar.

## Dogrulama notu
DEA ve Malmquist modulleri, orijinal GAMS/CPLEX kodlariniza karsi test
edildi: CCR/BCC stage-1 amac fonksiyonu degerleri (11.4133 / 11.4623) ve
Malmquist mesafe fonksiyonu LP cozumleri (orn. D(t1,t1)=0.357143) GAMS
log dosyalarinizdaki degerlerle ondalik hanesine kadar eslesti.
