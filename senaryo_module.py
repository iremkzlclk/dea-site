# -*- coding: utf-8 -*-
"""
SENARYO MODULU (gelecek donem verimlilik senaryolari)
==========================================================
Uygulamanin "Gelecek Dönem Tahmini" sayfasindaki İKİ AYRI, tamamen
deterministik senaryo aracini icerir -- ayri bir makine ogrenmesi
model karsilastirma katmani (Ridge/Lasso/ElasticNet/Random Forest
arasinda secim) KASITLI OLARAK YOKTUR (bkz. metodoloji raporu Bolum
3.3 -- kucuk orneklemde Rastgele Orman'in platform-bagimli
tekrarlanabilirlik sorunu ampirik olarak gozlemlenmis, bu nedenle
tahmin katmani cikarim modeliyle (Bolum 6) ayni dogrusal cerceveye
indirgenmistir).

A) gelecek_donem_dea_senaryo(): DEA girdilerini degistirip GERCEK bir
   DEA/Malmquist LP'si yeniden cozer -- deterministik, belirsizlik
   araligi yok. Ciktilari girdilerle TUTARLI tahmin etmek icin TEK
   bir Ridge regresyonu kullanir (cikti_tahmin_modeli_egit) -- bu,
   "hangi ML modeli daha iyi" sorusuna cevap ARAMAZ, sadece senaryo
   verisini insa etmek icin gerekli bir ara adimdir.

B) panel_regresyon_senaryo(): DEA'yi yeniden COZMEZ -- Bolum 6'daki
   NIHAI panel regresyon modelinin (Pooled/FE/RE) kendi katsayilarini
   kullanarak MI tahmini uretir; %95 belirsizlik araligi, kalinti
   standart sapmasindan hesaplanir.
"""
import numpy as np
import pandas as pd
from scipy.stats import gmean

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

