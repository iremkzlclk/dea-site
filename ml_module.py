# -*- coding: utf-8 -*-
"""
MAKINE OGRENMESI TAHMIN MODULU (kucuk-N panel icin kalibre edilmis)
========================================================================
Panel analizindeki anlamli girdi degiskenlerinin MI uzerindeki etkisini,
DUZENLILESTIRILMIS (regularized) regresyon modelleriyle -- Ridge, Lasso,
ElasticNet -- somutlastirir. Bu, "makine ogrenmesi" kapsaminda ama sizin
veri buyuklugunuze (N~12-15 DMU, T~6-8 donem, ~60-100 gozlem) UYGUN olan
yontem ailesidir.

NEDEN Random Forest / Gradient Boosting / Neural Network DEGIL:
Bu yontemler genelde YUZLERCE-BINLERCE gozlem gerektirir. Sizin
buyuklugunuzde, bu modeller egitim verisini EZBERLER (overfit) ve
gercekte panel regresyonundan DAHA KOTU tahmin eder -- bu bir varsayim
degil, ML literaturunde iyi bilinen "kucuk-N buyuk-varyans" sorunudur.

Yine de bir KARSILASTIRMA NOKTASI olarak, agir sinirlandirilmis (sig
derinlikli, yuksek min_samples_leaf) bir Random Forest da sunulur --
"daha karmasik modelin bu veri buyuklugunde islev gormedigini gostermek"
bile basli basina degerli, dogru bir bulgudur.

Tum modeller, panel_module.py/backtest_module.py ile AYNI degerlendirme
cercevesini (leave-last-period-out + rolling backtest, MAE/RMSE/MAPE/
yon dogrulugu, naif M=1 karsilastirmasi) kullanir -- boylece klasik panel
regresyonuyla DOGRUDAN, adil bir kiyaslama yapilabilir.
"""
import numpy as np
import pandas as pd
from scipy.stats import gmean

from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

MODEL_ACIKLAMALARI = {
    "ridge": "**Ridge:** Tüm girdi/çıktıları hesaba katar, hiçbirini tamamen göz ardı etmez -- "
             "aralarında güçlü bir ilişki (çoklu doğrusal bağlantı) varsa bile kararlı sonuç verir. "
             "Az veriyle güvenle kullanılabilir.",
    "lasso": "**Lasso:** Zayıf/önemsiz görünen girdi-çıktıların etkisini otomatik olarak SIFIRA çeker "
             "-- yani size 'bunlar önemli değil' diye otomatik bir eleme de yapar.",
    "elasticnet": "**ElasticNet:** Ridge ile Lasso'nun ortası -- hem kararlı sonuç verir hem de "
                  "önemsiz değişkenleri kısmen eler.",
    "random_forest": "**Random Forest:** Karar ağaçlarından oluşan, doğrusal olmayan ilişkileri de "
                      "yakalayabilen bir model. ⚠️ Sadece KARŞILAŞTIRMA amaçlı sunuluyor -- bu kadar "
                      "az veriyle (60-100 gözlem) ezberleme riski yüksek, sonuçlarına diğerleri kadar "
                      "güvenmeyin.",
}


def _model_kur(model_tipi: str):
    """Model tipine gore bir (StandardScaler + regresor) pipeline'i kurar.
    Kucuk-N icin uygun capraz dogrulama (LOOCV -- leave-one-out) kullanilir."""
    if model_tipi == "ridge":
        alfa_araligi = np.logspace(-3, 3, 50)
        regresor = RidgeCV(alphas=alfa_araligi, cv=None)  # cv=None -> LOOCV (kucuk N icin en uygun)
    elif model_tipi == "lasso":
        # NOT: scikit-learn 1.9+ 'n_alphas' parametresini kaldirdi -- artik 'alphas'
        # parametresi hem tam sayi (eskiden n_alphas'in yaptigi gibi otomatik uretim
        # sayisi) hem de liste (belirli degerler) kabul ediyor. Bu yuzden 'alphas=50'
        # kullaniyoruz, 'n_alphas=50' DEGIL (eski sklearn surumlerinde calisirdi,
        # 1.9+'da TypeError verir).
        regresor = LassoCV(cv=5, max_iter=20000, alphas=50)
    elif model_tipi == "elasticnet":
        regresor = ElasticNetCV(cv=5, max_iter=20000, alphas=50, l1_ratio=[.1, .3, .5, .7, .9, .95, .99])
    elif model_tipi == "random_forest":
        # Agir sinirlandirma: sig agaclar, yuksek min_samples_leaf -- overfit riskini azaltmak icin
        regresor = RandomForestRegressor(
            n_estimators=200, max_depth=3, min_samples_leaf=5, random_state=42,
        )
    else:
        raise ValueError(f"Bilinmeyen model_tipi: {model_tipi}")

    return Pipeline([("olcekleme", StandardScaler()), ("model", regresor)])


def model_egit(panel_df: pd.DataFrame, bagimsizlar: list, model_tipi: str, bagimli: str = "MI"):
    """
    Verilen panel verisiyle (TUMU, egitim/test ayirimi YOK -- bu fonksiyon
    "nihai model"i kurmak icindir, dogrulama icin ml_backtest_calistir
    kullanilir) bir ML modeli egitir.

    Returns: dict -- pipeline (egitilmis), katsayilar (Ridge/Lasso/ElasticNet
             icin DataFrame; Random Forest icin ozellik onemleri), model_tipi
    """
    X = panel_df[bagimsizlar].to_numpy(dtype=float)
    y = panel_df[bagimli].to_numpy(dtype=float)

    pipeline = _model_kur(model_tipi)
    pipeline.fit(X, y)

    regresor = pipeline.named_steps["model"]
    if model_tipi in ("ridge", "lasso", "elasticnet"):
        std_katsayilar = regresor.coef_
        # OLCEKLENMIS (standardize edilmis) katsayilar dogrudan "1 birim degisince
        # ne olur" diye yorumlanamaz -- StandardScaler'in olcek faktoruyle (std)
        # bolerek GERCEK OLCEGE (orijinal birimlere) ceviriyoruz.
        olcekleyici = pipeline.named_steps["olcekleme"]
        ham_katsayilar = std_katsayilar / olcekleyici.scale_

        katsayilar = pd.DataFrame({
            "degisken": bagimsizlar, "katsayi_standart": std_katsayilar, "katsayi_gercek_olcek": ham_katsayilar,
        }).set_index("degisken")
        katsayilar["katsayi_standart"] = katsayilar["katsayi_standart"].round(5)
        katsayilar["katsayi_gercek_olcek"] = katsayilar["katsayi_gercek_olcek"].round(6)
        secilen_alpha = getattr(regresor, "alpha_", None)
    else:  # random_forest
        katsayilar = pd.DataFrame({
            "degisken": bagimsizlar, "ozellik_onemi": regresor.feature_importances_,
        }).set_index("degisken")
        katsayilar["ozellik_onemi"] = katsayilar["ozellik_onemi"].round(4)
        secilen_alpha = None

    return {
        "pipeline": pipeline, "katsayilar": katsayilar, "model_tipi": model_tipi,
        "secilen_alpha": secilen_alpha, "aciklama": MODEL_ACIKLAMALARI[model_tipi],
    }


def ml_yorum_metni(model_paketi: dict, girdi_cols: list, cikti_cols: list) -> str:
    """
    Katsayilari/ozellik onemlerini, konuyu hic bilmeyen bir kullanicinin
    anlayacagi, DOGRUDAN AKSIYONA donusturulebilir bir yorum metnine cevirir.
    Buyukluge (mutlak deger) gore siralanmis, en etkili degiskenden baslar.
    """
    model_tipi = model_paketi["model_tipi"]
    katsayilar = model_paketi["katsayilar"]

    satirlar = [f"**Model: {model_tipi.upper()}** -- MI (verimlilik endeksi) üzerindeki etkiler, "
                f"en güçlüden en zayıfa doğru sıralandı:\n"]

    if model_tipi in ("ridge", "lasso", "elasticnet"):
        siralama = katsayilar.reindex(
            katsayilar["katsayi_standart"].abs().sort_values(ascending=False).index
        )
        for degisken, satir in siralama.iterrows():
            ham = satir["katsayi_gercek_olcek"]
            tip = "Girdi" if degisken in girdi_cols else "Çıktı"
            if abs(satir["katsayi_standart"]) < 1e-6:
                satirlar.append(
                    f"- **{degisken}** ({tip}): Model bu değişkenin etkisini **sıfıra çekti** -- "
                    f"yani MI'yi açıklamada işe yaramadığını tespit etti. Bu değişkene dayanarak "
                    f"aksiyon almanızı önermiyoruz."
                )
                continue
            yon = "artırır" if ham > 0 else "azaltır"
            satirlar.append(
                f"- **{degisken}** ({tip}): Bu değişken **1 birim arttığında**, MI'nin ortalama "
                f"**{ham:+.5f} birim** değişmesi bekleniyor -- yani bu değişkeni artırmak MI'yi "
                f"**{yon}**."
            )
        if girdi_cols:
            en_guclu_girdi = None
            for degisken, satir in siralama.iterrows():
                if degisken in girdi_cols and abs(satir["katsayi_standart"]) > 1e-6:
                    en_guclu_girdi = (degisken, satir["katsayi_gercek_olcek"])
                    break
            if en_guclu_girdi:
                deg, ham = en_guclu_girdi
                yon_metni = "artırmayı" if ham > 0 else "azaltmayı"
                satirlar.append(
                    f"\n**Özet:** Modele göre, verimliliği artırmak için en güçlü kaldıraç "
                    f"**{deg}**'i **{yon_metni}** denemek olabilir. ⚠️ Bu, panel analizindeki DEA-"
                    f"tutarlılık kontrolüyle birlikte değerlendirilmeli -- tek başına kesin bir "
                    f"yatırım kararı için yeterli değildir."
                )
    else:  # random_forest
        siralama = katsayilar.sort_values("ozellik_onemi", ascending=False)
        for degisken, satir in siralama.iterrows():
            tip = "Girdi" if degisken in girdi_cols else "Çıktı"
            satirlar.append(
                f"- **{degisken}** ({tip}): Modelin tahmininde **%{satir['ozellik_onemi']*100:.1f}** "
                f"oranında rol oynuyor. ⚠️ Random Forest **yön** (artış/azalış) bilgisi vermez, "
                f"sadece 'ne kadar önemli' bilgisi verir -- ve bu veri büyüklüğünde bu önem "
                f"sıralaması da güvenilmez olabilir."
            )

    return "\n".join(satirlar)


def cikti_tahmin_modeli_egit(sonuc: dict, girdi_cols: list, cikti_cols: list) -> dict:
    """
    Her CIKTI icin, o ciktiyi TUM girdilerden (coklu degiskenli, Ridge
    regresyon) tahmin eden ayri bir model egitir -- eskiden kullanilan
    basit ikili (bivariate, beta=Cov/Var) yaklasimin YERINE.

    NEDEN GEREKLI (kullanicinin kendi ampirik bulgusuyla dogrulandi):
    Eski yontem, ciktinin girdiye tepkisini TEK BIR SABIT sayiya (beta)
    indirgiyordu. Ama bu iliski GERCEKTE BELIRSIZ -- ayni girdi degisimi
    icin farkli makul carpanlar (orn. 0.98 vs 1.2) denendiginde, nihai
    Malmquist sonucu TAMAMEN ZIT cikabiliyordu (M=0.96 vs M=1.09). Tek bir
    "sabit sayi" bu belirsizligi GIZLIYORDU. Bu fonksiyon, hem coklu-
    degiskenli bir tahmin (TUM girdileri ayni anda hesaba katar, sadece
    tek bir girdiyle ikili iliskiye bakmaz) hem de bir BELIRSIZLIK OLCUSU
    (kalintilarin standart sapmasi -- tahminin ne kadar "titrek" oldugunun
    gostergesi) saglayarak, senaryo_tahmin_et'in ALT/NOKTA/UST olmak uzere
    UC senaryo uretmesine olanak tanir.

    Returns: dict -- {cikti_adi: {"pipeline": egitilmis Ridge pipeline'i,
             "kalinti_std": float (tahmin belirsizligi -- ne kadar buyukse,
             o kadar az guvenilir bir tahmin)}}
    """
    panel_df = sonuc["panel_df"]
    modeller = {}
    gecerli_girdiler = [g for g in girdi_cols if g in panel_df.columns]
    if not gecerli_girdiler:
        return modeller

    for c in cikti_cols:
        if c not in panel_df.columns:
            continue
        X = panel_df[gecerli_girdiler].to_numpy(dtype=float)
        y = panel_df[c].to_numpy(dtype=float)
        try:
            pipeline = Pipeline([
                ("olcekleme", StandardScaler()),
                ("model", RidgeCV(alphas=np.logspace(-3, 3, 50), cv=None)),
            ])
            pipeline.fit(X, y)
            tahminler = pipeline.predict(X)
            kalintilar = y - tahminler
            serbestlik = max(len(y) - len(gecerli_girdiler) - 1, 1)
            kalinti_std = float(np.sqrt(np.sum(kalintilar ** 2) / serbestlik))
        except Exception:
            kalinti_std = float(np.std(y)) if len(y) > 1 else 0.0
            pipeline = None
        modeller[c] = {"pipeline": pipeline, "kalinti_std": kalinti_std, "girdiler": gecerli_girdiler}
    return modeller


def senaryo_tahmin_et(model_paketi: dict, sonuc: dict, girdi_cols: list, cikti_cols: list,
                       girdi_yuzdeleri: dict) -> dict:
    """
    Egitilmis ML modelini kullanarak, kullanicinin sectigi girdi yuzde
    degisiklikleriyle BIR SONRAKI DONEM icin MI tahmini uretir -- DEA'yi
    yeniden cozmeden, dogrudan modelin ogrendigi iliskiyi kullanarak
    (bu yuzden ANINDA sonuc verir).

    CIKTI DEGISKENLERI ARTIK COKLU-DEGISKENLI, BELIRSIZLIK-FARKINDALIKLI BIR
    MODELLE TAHMIN EDILIYOR (onemli guncelleme -- kullanicinin kendi ampirik
    bulgusuyla motive edildi): Eskiden, bir girdi degisince korele ciktinin
    ne kadar hareket edecegi TEK BIR SABIT sayiyla (basit ikili beta=Cov/Var)
    hesaplaniyordu. Ama bu iliski GERCEKTE BELIRSIZ -- ayni girdi degisimi
    icin farkli makul carpanlar (kullanicinin kendi testinde: 0.98 vs 1.2)
    TAMAMEN ZIT MI sonuclari verebiliyordu (M=0.96 vs M=1.09). Artik:
    1) cikti_tahmin_modeli_egit() ile TUM girdilerden (coklu-degiskenli
       Ridge) her cikti icin bir tahmin modeli VE bir belirsizlik olcusu
       (kalinti std'si) kuruluyor.
    2) Senaryo, bu tek bir NOKTA tahmini yerine UC PARALEL versiyon olarak
       hesaplaniyor: ALT (belirsizlik araliginin kotumser ucu), NOKTA (en
       olasi tahmin), UST (iyimser ucu) -- boylece "tek bir sabit sayiya"
       guvenmek yerine, kullaniciya GERCEK BELIRSIZLIGI gosteriyoruz.

    Cikti degiskenleri, HALA sizin dogrudan Artir/Azalt secebileceginiz bir
    sey degil -- donem basinda karar verilebilecek degil, donem sonunda
    gozlemlenen bir sonuc oldugu icin (bkz. modul felsefesi). Ama artik
    "girdi degisince cikti nasil tepki verir" sorusuna TEK bir cevap yerine
    bir AKARALIK veriliyor.

    IKI AYRI KARSILASTIRMA dondurulur (HER BIRI ARTIK ALT/NOKTA/UST olarak
    UCLU) -- bunlar FARKLI SORULARA cevap verir, birbirine KARISTIRILMAMALIDIR:

    1) senaryo_etkisi_yuzde_{alt,nokta,ust}: SADECE sizin girdi degisikliginizin
       (+ korele ciktinin BELIRSIZ tepkisinin) MARJINAL etkisi.
    2) degisim_yuzde_{alt,nokta,ust}: senaryonuzun, GERCEK son donem MI
       degerine gore TOPLAM farki.

    girdi_yuzdeleri: dict -- {girdi_adi: yuzde (orn. 0.10 = %10 artis, -0.10 = %10 azalis)}

    Returns: dict -- son_gercek_ortalama_MI, taban_sifir_degisim_MI,
             senaryo_etkisi_yuzde (NOKTA tahmini, geriye-uyumluluk icin),
             senaryo_etkisi_yuzde_alt/nokta/ust (uc senaryo),
             tahmini_degisim_yuzde_alt/nokta/ust, belirsizlik_genis_mi
             (bool -- alt ve ust senaryolar ZIT YONDEYSE True, yani "artis
             mi azalis mi" sorusuna bile guvenilir cevap veremiyoruz demektir),
             detay_df (DMU bazinda, NOKTA senaryosu icin)
    """
    pipeline = model_paketi["pipeline"]
    son_donem = sonuc["veri"]["donem_sirali"][-1]
    X_son = sonuc["X"][son_donem]
    Y_son = sonuc["Y"][son_donem]
    dmu_sirali = sonuc["veri"]["dmu_sirali"]
    cikti_modelleri = cikti_tahmin_modeli_egit(sonuc, girdi_cols, cikti_cols)

    # Son gecisin GERCEK (olculmus, tahmin edilmemis) MI degerleri
    panel_df = sonuc["panel_df"]
    son_gecis_zamani = sorted(panel_df.index.get_level_values("time").unique())[-1]
    son_gercek_MI = panel_df.xs(son_gecis_zamani, level="time")["MI"]
    son_gercek_ort = float(gmean(son_gercek_MI.to_numpy()))

    taban_satirlari = []
    # Uc paralel senaryo: ALT (kotumser cikti tepkisi), NOKTA (en olasi), UST (iyimser)
    senaryo_satirlari_versiyon = {"alt": [], "nokta": [], "ust": []}
    detay_satirlar_nokta = []

    for dmu in dmu_sirali:
        satir_taban = []
        girdi_yeni_degerler = {}
        for g in girdi_cols:
            deger = float(X_son.loc[dmu, g])
            satir_taban.append(deger)
            yuzde = girdi_yuzdeleri.get(g, 0.0)
            girdi_yeni_degerler[g] = deger * (1 + yuzde)
        for c in cikti_cols:
            satir_taban.append(float(Y_son.loc[dmu, c]))  # taban: cikti GERCEK, degismemis deger
        taban_satirlari.append(satir_taban)

        satir_senaryo_v = {"alt": [], "nokta": [], "ust": []}
        for c in cikti_cols:
            son_deger_c = float(Y_son.loc[dmu, c])
            model_bilgi = cikti_modelleri.get(c)
            if model_bilgi is None or model_bilgi["pipeline"] is None:
                # Model kurulamadiysa, guvenli bir sekilde SABIT birak (eski davranis)
                for v in ("alt", "nokta", "ust"):
                    satir_senaryo_v[v].append(son_deger_c)
                continue

            model_girdiler = model_bilgi["girdiler"]
            kalinti_std = model_bilgi["kalinti_std"]

            taban_girdi_vektoru = np.array([[float(X_son.loc[dmu, g]) for g in model_girdiler]])
            senaryo_girdi_vektoru = np.array([[girdi_yeni_degerler[g] for g in model_girdiler]])

            taban_cikti_tahmin = float(model_bilgi["pipeline"].predict(taban_girdi_vektoru)[0])
            senaryo_cikti_tahmin_nokta = float(model_bilgi["pipeline"].predict(senaryo_girdi_vektoru)[0])
            # Modelin tahmini ile ciktinin GERCEK son deger arasindaki fark,
            # modelin "kalinti"si -- bu farki taban VE senaryo tahminine
            # ekleyerek, modelin sistematik sapmasini nötrlüyoruz (kalibrasyon).
            kalibrasyon = son_deger_c - taban_cikti_tahmin
            cikti_delta_nokta = (senaryo_cikti_tahmin_nokta + kalibrasyon) - son_deger_c

            for v, kat in (("alt", -1.0), ("nokta", 0.0), ("ust", 1.0)):
                delta = cikti_delta_nokta + kat * 1.96 * kalinti_std
                yeni_deger = max(son_deger_c + delta, 0.01)
                satir_senaryo_v[v].append(yeni_deger)

        for v in ("alt", "nokta", "ust"):
            senaryo_satirlari_versiyon[v].append(satir_senaryo_v[v])

    # Girdi kismini (senaryo degerleriyle) her versiyona ekle
    tam_senaryo_satirlari = {"alt": [], "nokta": [], "ust": []}
    for i, dmu in enumerate(dmu_sirali):
        girdi_kismi = [taban_satirlari[i][j] * (1 + girdi_yuzdeleri.get(g, 0.0)) for j, g in enumerate(girdi_cols)]
        for v in ("alt", "nokta", "ust"):
            tam_senaryo_satirlari[v].append(girdi_kismi + senaryo_satirlari_versiyon[v][i])

    taban_tahminleri = pipeline.predict(np.array(taban_satirlari))
    taban_tahminleri_guvenli = np.clip(taban_tahminleri, 0.01, None)
    taban_ort = float(gmean(taban_tahminleri_guvenli))

    sonuclar_versiyon = {}
    for v in ("alt", "nokta", "ust"):
        senaryo_tahminleri = pipeline.predict(np.array(tam_senaryo_satirlari[v]))
        senaryo_tahminleri_guvenli = np.clip(senaryo_tahminleri, 0.01, None)
        senaryo_ort = float(gmean(senaryo_tahminleri_guvenli))
        sonuclar_versiyon[v] = {
            "senaryo_ort": senaryo_ort,
            "senaryo_tahminleri": senaryo_tahminleri,
            "senaryo_etkisi_yuzde": round((senaryo_ort - taban_ort) / taban_ort * 100, 2) if taban_ort else None,
            "degisim_yuzde": round((senaryo_ort - son_gercek_ort) / son_gercek_ort * 100, 2) if son_gercek_ort else None,
        }

    # Detay tablosu: NOKTA senaryosu uzerinden (eskisiyle uyumlu format)
    nokta_tahminleri = sonuclar_versiyon["nokta"]["senaryo_tahminleri"]
    detay = pd.DataFrame({
        "DMU": dmu_sirali,
        "son_gercek_MI": [round(float(son_gercek_MI.get(dmu, np.nan)), 4) for dmu in dmu_sirali],
        "taban_sifir_degisim_MI": np.round(taban_tahminleri, 4),
        "senaryo_tahmin_MI": np.round(nokta_tahminleri, 4),
    })
    detay["senaryo_etkisi"] = (detay["senaryo_tahmin_MI"] - detay["taban_sifir_degisim_MI"]).round(4)
    detay["tahmini_gelecek_MI"] = (detay["son_gercek_MI"] + detay["senaryo_etkisi"]).round(4)
    detay["gercege_gore_fark"] = (detay["senaryo_tahmin_MI"] - detay["son_gercek_MI"]).round(4)
    detay = detay.set_index("DMU")

    tahmini_gelecek_ort_nokta = float(gmean(np.clip(detay["tahmini_gelecek_MI"].to_numpy(), 0.01, None)))

    etki_alt = sonuclar_versiyon["alt"]["senaryo_etkisi_yuzde"]
    etki_ust = sonuclar_versiyon["ust"]["senaryo_etkisi_yuzde"]
    belirsizlik_genis_mi = (
        etki_alt is not None and etki_ust is not None and
        ((etki_alt > 0.5 and etki_ust < -0.5) or (etki_alt < -0.5 and etki_ust > 0.5))
    )

    return {
        "son_gercek_ortalama_MI": round(son_gercek_ort, 4),
        "taban_sifir_degisim_MI": round(taban_ort, 4),
        "senaryo_ortalama_MI": round(sonuclar_versiyon["nokta"]["senaryo_ort"], 4),
        "tahmini_gelecek_ortalama_MI": round(tahmini_gelecek_ort_nokta, 4),
        "senaryo_etkisi_yuzde": sonuclar_versiyon["nokta"]["senaryo_etkisi_yuzde"],
        "degisim_yuzde": sonuclar_versiyon["nokta"]["degisim_yuzde"],
        "tahmini_degisim_yuzde": sonuclar_versiyon["nokta"]["degisim_yuzde"],
        "senaryo_etkisi_yuzde_alt": sonuclar_versiyon["alt"]["senaryo_etkisi_yuzde"],
        "senaryo_etkisi_yuzde_nokta": sonuclar_versiyon["nokta"]["senaryo_etkisi_yuzde"],
        "senaryo_etkisi_yuzde_ust": sonuclar_versiyon["ust"]["senaryo_etkisi_yuzde"],
        "belirsizlik_genis_mi": belirsizlik_genis_mi,
        "detay_df": detay,
    }


def _tek_kat_ml(panel_df: pd.DataFrame, egitim_zamanlari: list, test_zamani,
                 bagimsizlar: list, model_tipi: str, bagimli: str):
    """TEK BIR kat icin ML modeli egitir ve test doneminde tahmin uretir."""
    egitim_df = panel_df[panel_df.index.get_level_values("time").isin(egitim_zamanlari)]
    test_df = panel_df[panel_df.index.get_level_values("time") == test_zamani]

    if egitim_df.shape[0] < 5:  # cok az gozlemle model kurmaya calismayalim
        return None

    try:
        model_paketi = model_egit(egitim_df, bagimsizlar, model_tipi, bagimli)
    except Exception:
        return None

    X_test = test_df[bagimsizlar].to_numpy(dtype=float)
    try:
        tahminler = model_paketi["pipeline"].predict(X_test)
    except Exception:
        return None

    tahmin_satirlar = []
    for entity, tahmin in zip(test_df.index.get_level_values("entity"), tahminler):
        gercek = float(test_df.loc[(entity, test_zamani), bagimli])
        hata = float(tahmin) - gercek
        yon_gercek = "Artis" if gercek > 1.0 else "Azalis"
        yon_tahmin = "Artis" if tahmin > 1.0 else "Azalis"
        tahmin_satirlar.append({
            "DMU": entity, "gercek_MI": round(gercek, 4), "tahmin_MI": round(float(tahmin), 4),
            "hata": round(hata, 4),
            "mutlak_yuzde_hata": round(abs(hata / gercek) * 100, 2) if gercek else np.nan,
            "yon_gercek": yon_gercek, "yon_tahmin": yon_tahmin, "yon_dogru_mu": yon_gercek == yon_tahmin,
        })
    tahmin_df = pd.DataFrame(tahmin_satirlar).set_index("DMU")

    gercek_arr = tahmin_df["gercek_MI"].to_numpy()
    tahmin_arr = tahmin_df["tahmin_MI"].to_numpy()
    hata_arr = tahmin_arr - gercek_arr

    mae = float(np.mean(np.abs(hata_arr)))
    rmse = float(np.sqrt(np.mean(hata_arr ** 2)))
    mape = float(tahmin_df["mutlak_yuzde_hata"].mean())
    yon_dogruluk = float(tahmin_df["yon_dogru_mu"].mean() * 100)
    naif_mae = float(np.mean(np.abs(np.ones_like(gercek_arr) - gercek_arr)))

    return {
        "egitim_zamanlari": list(egitim_zamanlari), "test_zamani": test_zamani,
        "model_paketi": model_paketi, "tahmin_df": tahmin_df,
        "metrikler": {
            "MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE_%": round(mape, 2),
            "yon_dogruluk_%": round(yon_dogruluk, 1), "naif_baseline_MAE": round(naif_mae, 4),
            "modelin_naiften_iyi_mi": mae < naif_mae,
        },
    }


def ml_backtest_calistir(panel_df: pd.DataFrame, bagimsizlar: list, model_tipi: str,
                          bagimli: str = "MI") -> dict:
    """
    TEK KATLI (leave-last-period-out) backtest -- panel_module tabanli
    backtest_module.backtest_calistir ile AYNI mantik, ama ML modeliyle.
    """
    zaman_seviyeleri = sorted(panel_df.index.get_level_values("time").unique())
    if len(zaman_seviyeleri) < 2:
        return {"yeterli_veri": False, "mesaj": "Backtest icin en az 2 gecis donemi gerekli."}

    holdout_zaman = zaman_seviyeleri[-1]
    egitim_zamanlari = zaman_seviyeleri[:-1]

    sonuc_kati = _tek_kat_ml(panel_df, egitim_zamanlari, holdout_zaman, bagimsizlar, model_tipi, bagimli)
    if sonuc_kati is None:
        return {"yeterli_veri": False, "mesaj": "Model kurulamadi (muhtemelen egitim seti cok kucuk)."}

    return {
        "yeterli_veri": True, "model_tipi": model_tipi, "aciklama": MODEL_ACIKLAMALARI[model_tipi],
        "egitim_zamanlari": sonuc_kati["egitim_zamanlari"], "holdout_zaman": sonuc_kati["test_zamani"],
        "katsayilar": sonuc_kati["model_paketi"]["katsayilar"],
        "tahmin_df": sonuc_kati["tahmin_df"], "metrikler": sonuc_kati["metrikler"],
    }


def ml_rolling_backtest_calistir(panel_df: pd.DataFrame, bagimsizlar: list, model_tipi: str,
                                  bagimli: str = "MI", min_egitim_donemi: int = 2) -> dict:
    """
    COK KATLI (walk-forward) backtest -- backtest_module.rolling_backtest_calistir
    ile AYNI mantik, ama ML modeliyle. Klasik panel regresyonuyla DOGRUDAN
    karsilastirilabilir sonuc verir (ayni metrik seti).
    """
    zaman_seviyeleri = sorted(panel_df.index.get_level_values("time").unique())
    toplam_kat_adayi = len(zaman_seviyeleri) - min_egitim_donemi
    if toplam_kat_adayi < 1:
        return {
            "yeterli_veri": False,
            "mesaj": f"Rolling backtest icin en az {min_egitim_donemi + 1} gecis donemi gerekli.",
        }

    kat_detaylari = []
    for i in range(min_egitim_donemi, len(zaman_seviyeleri)):
        egitim_zamanlari = zaman_seviyeleri[:i]
        test_zamani = zaman_seviyeleri[i]
        sonuc_kati = _tek_kat_ml(panel_df, egitim_zamanlari, test_zamani, bagimsizlar, model_tipi, bagimli)
        if sonuc_kati is not None:
            kat_detaylari.append(sonuc_kati)

    if not kat_detaylari:
        return {"yeterli_veri": False, "mesaj": "Hicbir kat basariyla tamamlanamadi."}

    mae_listesi = [k["metrikler"]["MAE"] for k in kat_detaylari]
    yon_listesi = [k["metrikler"]["yon_dogruluk_%"] for k in kat_detaylari]
    naif_listesi = [k["metrikler"]["naif_baseline_MAE"] for k in kat_detaylari]
    ortalama_mae = float(np.mean(mae_listesi))
    ortalama_naif = float(np.mean(naif_listesi))

    return {
        "yeterli_veri": True, "model_tipi": model_tipi, "aciklama": MODEL_ACIKLAMALARI[model_tipi],
        "kat_sayisi": len(kat_detaylari), "denenen_kat_sayisi": toplam_kat_adayi,
        "kat_detaylari": kat_detaylari,
        "ortalama_metrikler": {
            "MAE_ortalama": round(ortalama_mae, 4), "MAE_std": round(float(np.std(mae_listesi)), 4),
            "yon_dogruluk_ortalama_%": round(float(np.mean(yon_listesi)), 1),
            "yon_dogruluk_std": round(float(np.std(yon_listesi)), 1),
            "naif_baseline_MAE_ortalama": round(ortalama_naif, 4),
            "modelin_naiften_iyi_mi": ortalama_mae < ortalama_naif,
            "kat_basina_naiften_iyi_sayisi": sum(1 for k in kat_detaylari if k["metrikler"]["modelin_naiften_iyi_mi"]),
        },
    }


def en_iyi_modeli_sec(panel_df: pd.DataFrame, bagimsizlar: list, bagimli: str = "MI") -> dict:
    """
    Panel Analizi sekmesindeki model-secim mantigina (Poolability->Hausman->SE
    zinciri) benzer sekilde: 4 ML modelinin (ridge/lasso/elasticnet/random_forest)
    HEPSINI backtest'ten gecirir ve GECMISI EN IYI TAHMIN EDEN modeli otomatik
    secer -- boylece kullanicinin 4 model arasinda KENDI SECIM YAPMASI gerekmez.

    ONEMLI: Random Forest, KARSILASTIRMA TABLOSUNDA gosterilir (seffaflik icin)
    ama HICBIR ZAMAN otomatik SECILEN model olamaz. Nedeni: Random Forest
    dogrusal degildir (karar agaclari kullanir) -- bu yuzden bir girdiyi HEM
    artirmak HEM azaltmak, ayni yonde (orn. ikisi de artis) bir tahmin
    degisikligi VEREBILIR ("U seklinde" yerel davranis). Dogrusal modellerde
    (Ridge/Lasso/ElasticNet) bu MATEMATIKSEL OLARAK IMKANSIZDIR -- katsayinin
    isareti sabittir, artirmak ve azaltmak HER ZAMAN zit yonde sonuc verir.
    Bir yatirim-yonu araci icin bu tutarlilik kritik oldugundan, sadece
    dogrusal modeller otomatik secim havuzuna girer.

    Secim kriteri: once Rolling Backtest (varsa) ortalama YON DOGRULUGU'nu
    (yuksek=iyi) esas alir -- bu sekmenin amaci (girdiyi artir/azalt
    kararı) icin en pratik olcut budur; esitlik durumunda ortalama MAE
    (dusuk=iyi) ile tiebreak yapilir. Rolling yeterli veri sunmuyorsa
    tek-katli backtest metriklerine dusulur.

    Returns: dict -- secilen_model (str), gerekce (str, kullaniciya gosterilecek
             aciklama), karsilastirma_df (DataFrame, 4 modelin yan yana
             metrikleri -- seffaflik icin, Random Forest dahil)
    """
    adaylar = []
    for model_tipi in ["ridge", "lasso", "elasticnet", "random_forest"]:
        rbt = ml_rolling_backtest_calistir(panel_df, bagimsizlar, model_tipi, bagimli)
        if rbt["yeterli_veri"]:
            mae = rbt["ortalama_metrikler"]["MAE_ortalama"]
            yon = rbt["ortalama_metrikler"]["yon_dogruluk_ortalama_%"]
            kaynak = "rolling"
        else:
            bt = ml_backtest_calistir(panel_df, bagimsizlar, model_tipi, bagimli)
            if not bt["yeterli_veri"]:
                continue
            mae = bt["metrikler"]["MAE"]
            yon = bt["metrikler"]["yon_dogruluk_%"]
            kaynak = "tek-katli"
        adaylar.append({"model": model_tipi, "MAE": mae, "yon_dogruluk_%": yon, "kaynak": kaynak})

    if not adaylar:
        return {
            "basarili": False,
            "mesaj": "Hicbir model icin yeterli veri bulunamadi (cok kisa panel).",
        }

    karsilastirma_df = pd.DataFrame(adaylar).sort_values(
        ["yon_dogruluk_%", "MAE"], ascending=[False, True]
    ).reset_index(drop=True)

    # Secim havuzu SADECE dogrusal modeller (Ridge/Lasso/ElasticNet) --
    # Random Forest karsilastirma tablosunda goruntulenir ama secilemez
    # (bkz. fonksiyon docstring'i -- dogrusal olmayan yon tutarsizligi riski).
    dogrusal_adaylar = karsilastirma_df[karsilastirma_df["model"] != "random_forest"]
    if dogrusal_adaylar.empty:
        return {
            "basarili": False,
            "mesaj": "Hicbir dogrusal model (Ridge/Lasso/ElasticNet) icin yeterli veri bulunamadi.",
        }

    secilen = dogrusal_adaylar.iloc[0]
    rf_notu = ""
    if "random_forest" in karsilastirma_df["model"].values:
        rf_satiri = karsilastirma_df[karsilastirma_df["model"] == "random_forest"].iloc[0]
        if (rf_satiri["yon_dogruluk_%"] > secilen["yon_dogruluk_%"]) or \
           (rf_satiri["yon_dogruluk_%"] == secilen["yon_dogruluk_%"] and rf_satiri["MAE"] < secilen["MAE"]):
            rf_notu = (
                f" (Random Forest'in backtest metrikleri aslında daha iyi çıktı, ama doğrusal "
                f"olmadığı için -- bir girdiyi artırmak VE azaltmak aynı yönde sonuç verebildiği "
                f"için -- otomatik seçim dışında tutuldu.)"
            )
    gerekce = (
        f"**{secilen['model'].upper()}** seçildi -- geçmiş dönemleri tahmin etmede "
        f"({secilen['kaynak']} backtest) diğer doğrusal modellerden daha isabetliydi "
        f"(MAE={secilen['MAE']}, yön doğruluğu=%{secilen['yon_dogruluk_%']})."
        f"{rf_notu}"
    )

    return {
        "basarili": True, "secilen_model": secilen["model"],
        "gerekce": gerekce, "karsilastirma_df": karsilastirma_df,
    }


def gelecek_donem_dea_senaryo(sonuc: dict, girdi_cols: list, cikti_cols: list,
                               girdi_yuzdeleri: dict, cikti_modelleri: dict = None) -> dict:
    """
    ML TAHMIN SEKMESININ YENI TEMEL MEKANIZMASI -- eski (regresyon-tahminli)
    senaryo_tahmin_et'in YERINE gecer. Farki: MI'yi bir ML modelinden DOGRUDAN
    TAHMIN ETMEK yerine, GERCEK bir "sonraki donem" verisi insa edip, bu yeni
    donem ile SON GERCEK donem arasinda GERCEKTEN DEA + Malmquist COZER --
    yani sonuc, istatistiksel bir yaklastirma degil, DEA'nin kendi
    optimizasyonunun DOGRUDAN cikisi.

    ADIMLAR:
    1) Kullanicinin sectigi girdi yuzde degisiklikleri, SON DONEMIN gercek
       girdi degerlerine uygulanarak yeni bir "senaryo" donemi kurulur.
    2) Ciktilar, cikti_tahmin_modeli_egit() ile egitilen coklu-degiskenli
       (TUM girdilerden, Ridge) modelle, yeni girdi degerlerine gore
       yeniden tahmin edilir -- boylece ciktilar da girdilerle TUTARLI
       sekilde hareket eder (saf sabit tutulmaz).
    3) Bu YENI donem + SON GERCEK donem birlikte, malmquist_module.solve_malmquist
       fonksiyonuna verilir -- GERCEK bir DEA/Malmquist LP cozumu yapilir.
    4) Sonucta EC (etkinlik degisimi), TC (sinir/teknoloji degisimi) ve
       M=EC*TC (toplam verimlilik degisimi), DMU bazinda ve ortalama olarak
       donuyor.

    girdi_yuzdeleri: dict -- {girdi_adi: yuzde (orn. 0.10 = %10 artis)}
    cikti_modelleri: onceden egitilmisse tekrar egitmemek icin verilebilir
                     (cikti_tahmin_modeli_egit() ciktisi)

    Returns: dict -- detay_df (DMU bazinda EC/TC/M), ortalama_EC, ortalama_TC,
             ortalama_M, degisim_yuzde (ortalama_M'in %1'den sapmasi),
             X_senaryo, Y_senaryo (olusturulan yeni donem verisi, seffaflik icin)
    """
    from dea_module import solve_dea_period  # noqa: F401 (dogrudan kullanilmiyor, solve_malmquist icinde)
    from malmquist_module import solve_malmquist

    son_donem = sonuc["veri"]["donem_sirali"][-1]
    X_son = sonuc["X"][son_donem]
    Y_son = sonuc["Y"][son_donem]

    if cikti_modelleri is None:
        cikti_modelleri = cikti_tahmin_modeli_egit(sonuc, girdi_cols, cikti_cols)

    # 1) Yeni "senaryo" girdi donemi
    X_senaryo = X_son.copy()
    for g in girdi_cols:
        yuzde = girdi_yuzdeleri.get(g, 0.0)
        X_senaryo[g] = X_son[g] * (1 + yuzde)

    # 2) Yeni "senaryo" cikti donemi -- coklu-degiskenli model + kalibrasyon
    Y_senaryo = Y_son.copy()
    for c in cikti_cols:
        model_bilgi = cikti_modelleri.get(c)
        if model_bilgi is None or model_bilgi["pipeline"] is None:
            continue  # model kurulamadiysa, cikti sabit kalir (guvenli varsayilan)
        model_girdiler = model_bilgi["girdiler"]
        taban_tahmin = model_bilgi["pipeline"].predict(X_son[model_girdiler].to_numpy(dtype=float))
        senaryo_tahmin = model_bilgi["pipeline"].predict(X_senaryo[model_girdiler].to_numpy(dtype=float))
        # Kalibrasyon: modelin GERCEK son deger ile kendi tahmini arasindaki
        # sapmasini nötrler -- boylece sadece GIRDI DEGISIMININ etkisi tasinir.
        kalibrasyon = Y_son[c].to_numpy(dtype=float) - taban_tahmin
        yeni_deger = senaryo_tahmin + kalibrasyon
        Y_senaryo[c] = np.clip(yeni_deger, 0.01, None)

    # 3) GERCEK DEA + Malmquist cozumu -- son gercek donem ile yeni senaryo donemi arasinda
    X = {son_donem: X_son, "senaryo": X_senaryo}
    Y = {son_donem: Y_son, "senaryo": Y_senaryo}
    malmquist_sonuc = solve_malmquist(X, Y, periods=[son_donem, "senaryo"])

    detay = malmquist_sonuc.xs(son_donem, level="donem")[["EC", "TC", "M"]].round(4)

    ort_EC = float(gmean(detay["EC"].to_numpy()))
    ort_TC = float(gmean(detay["TC"].to_numpy()))
    ort_M = float(gmean(detay["M"].to_numpy()))

    return {
        "detay_df": detay,
        "ortalama_EC": round(ort_EC, 4),
        "ortalama_TC": round(ort_TC, 4),
        "ortalama_M": round(ort_M, 4),
        "degisim_yuzde": round((ort_M - 1.0) * 100, 2),
        "X_senaryo": X_senaryo, "Y_senaryo": Y_senaryo,
    }


def panel_regresyon_senaryo(sonuc: dict, panel_girdi_yuzdeleri: dict) -> dict:
    """
    DEA'yi YENIDEN COZMEZ -- sadece panel regresyonunun (Pooled/FE/RE, hangisi
    nihai olarak secildiyse) KATSAYILARINI kullanarak, kullanicinin panel
    girdilerinde belirttigi yuzdesel degisikliklere gore GELECEK DONEM MI
    tahmini uretir. Bu, gelecek_donem_dea_senaryo'dan (DEA'yi mekanik olarak
    yeniden cozen fonksiyon) YONTEM OLARAK FARKLIDIR: burada sonuc SAF
    ISTATISTIKSEL bir tahmindir (nokta tahmin + kaba belirsizlik araligi),
    EC/TC ayristirmasi YOKTUR (bu ayristirma sadece DEA'nin kendi
    optimizasyonundan gelir, regresyondan degil).

    ONEMLI: modelin R^2'si dusukse (panel analizi sekmesinde gordugunuz gibi),
    nokta tahmin etrafindaki belirsizlik ARALIGI genis olacaktir -- bu, modelin
    hatasi degil, dusuk R^2'nin dogal/beklenen bir sonucudur.

    panel_girdi_yuzdeleri: {degisken_adi: yuzdesel_degisim} (orn. {"Girdi_X": 0.10})
                            Sadece nihai modelde GERCEKTEN kullanilan
                            (zaman-sabitlik/dejenerelik nedeniyle elenmemis)
                            degiskenler icin anlamlidir.

    Returns: dict -- nihai_baslik, panel_bagimsizlar, r_kare, resid_std,
             detay_df (index=DMU: Son_Gercek_MI, Tahmin_MI, Tahmin_Alt,
             Tahmin_Ust, Degisim_Yuzde), ortalama_tahmin_MI, ortalama_son_MI,
             ortalama_degisim_yuzde, X_senaryo
    """
    from backtest_module import _tahmin_et

    p = sonuc["panel_sonuc"]
    oneri = p["oneri"]
    tablo_map = {
        "pooled_robust": p["pooled_robust"], "pooled_clustered": p["pooled_clustered"],
        "fe_robust": p["fe_robust"], "fe_clustered": p["fe_clustered"],
        "re_robust": p["re_robust"], "re_clustered": p["re_clustered"],
    }
    nihai_res = tablo_map[oneri["sonuc_tablo"]]
    panel_bagimsizlar = [v for v in nihai_res.params.index if v != "const"]

    panel_df = sonuc["panel_df"]
    son_zaman = panel_df.index.get_level_values("time").max()
    son_df = panel_df.xs(son_zaman, level="time")  # index=entity
    yeni_zaman = son_zaman + 1

    satirlar = []
    for entity in son_df.index:
        satir = {"entity": entity, "time": yeni_zaman}
        for col in panel_bagimsizlar:
            taban = float(son_df.loc[entity, col])
            yuzde = panel_girdi_yuzdeleri.get(col, 0.0)
            satir[col] = taban * (1 + yuzde)
        satirlar.append(satir)
    X_senaryo = pd.DataFrame(satirlar).set_index(["entity", "time"])

    tahmin_MI = _tahmin_et(nihai_res, X_senaryo)

    try:
        resid_std = float(nihai_res.resids.std())
    except Exception:
        resid_std = float("nan")

    detay_satirlari = []
    for entity in son_df.index:
        tahmin = float(tahmin_MI.loc[(entity, yeni_zaman)])
        gercek_son = float(son_df.loc[entity, "MI"])
        alt = tahmin - 1.96 * resid_std if not np.isnan(resid_std) else None
        ust = tahmin + 1.96 * resid_std if not np.isnan(resid_std) else None
        detay_satirlari.append({
            "DMU": entity,
            "Son_Gercek_MI": round(gercek_son, 4),
            "Tahmin_MI": round(tahmin, 4),
            "Tahmin_Alt_%95": round(alt, 4) if alt is not None else None,
            "Tahmin_Ust_%95": round(ust, 4) if ust is not None else None,
            "Degisim_Yuzde": round((tahmin - gercek_son) / gercek_son * 100, 2) if gercek_son else None,
        })
    detay_df = pd.DataFrame(detay_satirlari).set_index("DMU")

    return {
        "nihai_baslik": oneri["sonuc_basligi"],
        "panel_bagimsizlar": panel_bagimsizlar,
        "r_kare": float(nihai_res.rsquared),
        "resid_std": resid_std,
        "detay_df": detay_df,
        "ortalama_tahmin_MI": round(float(detay_df["Tahmin_MI"].mean()), 4),
        "ortalama_son_MI": round(float(detay_df["Son_Gercek_MI"].mean()), 4),
        "ortalama_degisim_yuzde": round(float(detay_df["Degisim_Yuzde"].mean()), 2),
        "X_senaryo": X_senaryo,
    }
