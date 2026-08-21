# Ar-Ge Simulasyon Etkinligi - VZA + Malmquist + Panel Analizi Web Uygulamasi

## Dosyalar
- `dea_module.py` — CCR/BCC iki asamali DEA (GAMS'la dogrulandi, bkz. asagida)
- `malmquist_module.py` — Gecikmeli Malmquist mesafe fonksiyonu (GAMS'la dogrulandi)
- `panel_module.py` — Pooled OLS -> FE -> RE -> Hausman/Mundlak -> robust/clustered
  panel analizi; coklu baglanti (rank) kontrolu; VIF/korelasyon teshisi
- `excel_okuma.py` — Excel okuma + veri dogrulama
- `pipeline.py` — DEA + Malmquist + panel adimlarini birbirine baglayan
  orkestrasyon katmani (DEA girdi/ciktilariyla panel girdileri AYRI secilir)
- `senaryo_module.py` — "Gelecek Donem Tahmini" sayfasindaki iki deterministik
  senaryo araci: A) DEA girdi senaryosu (LP yeniden cozumu), B) Panel girdi
  senaryosu (Bolum 6 panel modelinin kendi katsayilariyla)
- `backtest_module.py` — Tek katli + rolling/walk-forward geriye donuk
  dogrulama (MAE/RMSE, esik yon dogrulugu, siralama korelasyonu)
- `yorumlama.py` — DEA/Malmquist/panel sayisal ciktilarini duz Turkce
  metne/gorsele ceviren yardimci fonksiyonlar ("Sonuc Yorumlama" sayfasi)
- `app.py` — Streamlit web arayuzu (7 sekme: DEA, Malmquist, Panel Analizi,
  Sonuc Yorumlama, Aciklayicilik, Backtest, Gelecek Donem Tahmini)

## Kurulum (kendi bilgisayarinizda)
```
pip install -r requirements.txt
streamlit run app.py
```
Tarayicida otomatik acilir (genelde http://localhost:8501).

## Ucretsiz internete acma (Streamlit Community Cloud)
1. Bu dosyalari bir GitHub reposuna yukleyin (app.py, tum .py modulleri, requirements.txt)
2. https://share.streamlit.io adresinden GitHub hesabinizla giris yapin
3. "New app" -> reponuzu secin -> main file olarak `app.py` gosterin -> Advanced
   settings'ten Python surumunu 3.12 secin -> Deploy
4. Birkaç dakika icinde herkese acik bir URL alirsiniz (ucretsiz katmanda)

## Kutuphane surumleri neden sabitlendi (requirements.txt)
Rastgele Orman gibi agac-tabanli modeller, sabit random_state'e ragmen farkli
Python/kutuphane surumlerinde farkli sonuc uretebiliyor (ampirik olarak
gozlemlendi -- bkz. metodoloji raporu Bolum 3.3). Bu nedenle TUM kutuphaneler
ve Python surumu (3.12) sabitlenmistir; deploy sirasinda "Advanced settings"
adiminda Python surumunu mutlaka 3.12 secin.

## Excel sablonu
Uzun format, sutunlar:
`Donem | DMU | Girdi_<isim> | Girdi_<isim2> | ... | Cikti_<isim> | ...`

Donemler Excel'de kronolojik sirada olmali. Girdi_/Cikti_ sutun sayisi
sinirsiz — kod otomatik algilar. DEA'da kullanilacak girdi/ciktilar ile
panel analizinde kullanilacak girdiler, uygulama arayuzunde AYRI AYRI
secilir (ayni girdi ikisinde birden secilebilir ama uygulama bu durumda
ayrilabilirlik uyarisi gosterir).

## Dogrulama notu
DEA ve Malmquist modulleri, orijinal GAMS/CPLEX kodlariniza karsi test
edildi: CCR/BCC stage-1 amac fonksiyonu degerleri (11.4133 / 11.4623) ve
Malmquist mesafe fonksiyonu LP cozumleri (orn. D(t1,t1)=0.357143) GAMS
log dosyalarinizdaki degerlerle ondalik hanesine kadar eslesti.

