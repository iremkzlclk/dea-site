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

from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

MODEL_ACIKLAMALARI = {
    "ridge": "Ridge (L2 duzenlilestirme) -- kucuk-N icin guvenli, coklu dogrusal baglantiya dayanikli.",
    "lasso": "Lasso (L1 duzenlilestirme) -- zayif degiskenlerin katsayisini SIFIRA cekerek otomatik degisken secimi yapar.",
    "elasticnet": "ElasticNet (L1+L2 karisimi) -- Ridge ve Lasso'nun dengeli bir birlesimi.",
    "random_forest": "Random Forest (agir sinirlandirilmis) -- KARSILASTIRMA AMACLI; bu veri buyuklugunde "
                      "asiri ogrenme (overfit) riski yuksektir, sonuclarini temkinli yorumlayin.",
}


def _model_kur(model_tipi: str):
    """Model tipine gore bir (StandardScaler + regresor) pipeline'i kurar.
    Kucuk-N icin uygun capraz dogrulama (LOOCV -- leave-one-out) kullanilir."""
    if model_tipi == "ridge":
        alfa_araligi = np.logspace(-3, 3, 50)
        regresor = RidgeCV(alphas=alfa_araligi, cv=None)  # cv=None -> LOOCV (kucuk N icin en uygun)
    elif model_tipi == "lasso":
        regresor = LassoCV(cv=5, max_iter=20000, n_alphas=50)
    elif model_tipi == "elasticnet":
        regresor = ElasticNetCV(cv=5, max_iter=20000, n_alphas=50, l1_ratio=[.1, .3, .5, .7, .9, .95, .99])
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
        katsayilar = pd.DataFrame({
            "degisken": bagimsizlar, "katsayi": regresor.coef_,
        }).set_index("degisken")
        katsayilar["katsayi"] = katsayilar["katsayi"].round(5)
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
