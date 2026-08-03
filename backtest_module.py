# -*- coding: utf-8 -*-
"""
BACKTEST MODULU (leave-last-period-out)
==========================================
Panel modelinin GERCEK tahmin gucunu, DEA'yi tekrar cozmeden, TEK BIR
regresyon dogrulamasiyla olcer:

  1) Panel modelini SON GECIS DONEMI HARIC butun gecis donemleriyle egitir
     (ayni Pooled/FE/RE + Hausman + robust/clustered karar zincirini,
     panel_module.run_panel_analysis uzerinden, egitim setine uygular).
  2) Bu egitilen modelin katsayilarini kullanarak, SON GECISIN GERCEK
     (senaryo/varsayimsal DEGIL) girdi/cikti degerleriyle MI'yi tahmin eder.
  3) Tahmini, o gecisin GERCEKTEN GERCEKLESEN MI degeriyle karsilastirir.

Bu, "Gelecek Verimlilik Tahmini" sekmesindeki senaryo mekanizmasindan
BAGIMSIZ bir dogrulamadir -- DEA'yi yeniden cozmez, sadece panel modelinin
kendi basina ne kadar isabetli oldugunu olcer. Naif bir baseline (MI'de
degisim olmayacagini varsayan, yani "M=1" tahmini) ile de karsilastirir.
"""
import numpy as np
import pandas as pd

from panel_module import run_panel_analysis


def _tahmin_et(nihai_res, X_holdout: pd.DataFrame) -> pd.Series:
    """
    Egitilen nihai modelin katsayilariyla holdout donemi icin MI tahmini uretir.

    Pooled OLS / RE (params'ta 'const' olan modeller):
        tahmin = const + X @ katsayilar
    FE (entity_effects=True, 'const' icermez):
        tahmin = X @ katsayilar + entity'nin egitim setinde tahmin edilen sabit etkisi
    """
    params = nihai_res.params
    has_const = "const" in params.index

    if has_const:
        katsayilar = params.drop("const")
        sabit = params["const"]
        tahmin = X_holdout[katsayilar.index] @ katsayilar + sabit
    else:
        katsayilar = params
        taban_tahmin = X_holdout[katsayilar.index] @ katsayilar
        entity_etkileri = nihai_res.estimated_effects
        entity_etki_map = entity_etkileri.groupby(level="entity").first().iloc[:, 0]
        ekleme = X_holdout.index.get_level_values("entity").map(entity_etki_map).to_numpy(dtype=float)
        tahmin = pd.Series(taban_tahmin.to_numpy() + ekleme, index=X_holdout.index)

    return tahmin


def backtest_calistir(panel_df: pd.DataFrame, bagimsizlar: list, bagimli: str = "MI") -> dict:
    """
    Leave-last-period-out backtest calistirir.

    Returns: dict --
      yeterli_veri (bool), mesaj (yetersizse aciklama),
      egitim_zamanlari, holdout_zaman, nihai_baslik,
      tahmin_df (index=DMU: gercek_MI, tahmin_MI, hata, mutlak_yuzde_hata, yon_gercek, yon_tahmin, yon_dogru_mu),
      metrikler (MAE, RMSE, MAPE_%, Pearson_r, yon_dogruluk_%, naif_baseline_MAE, modelin_naiften_iyi_mi)
    """
    zaman_seviyeleri = sorted(panel_df.index.get_level_values("time").unique())
    if len(zaman_seviyeleri) < 2:
        return {
            "yeterli_veri": False,
            "mesaj": (
                f"Backtest icin en az 2 gecis donemi (yani en az 3 ham veri donemi) gerekli. "
                f"Mevcut gecis donemi sayisi: {len(zaman_seviyeleri)}. Ham veri donemi sayinizi "
                f"artirmadan bu ozellik anlamli sonuc veremez."
            ),
        }

    holdout_zaman = zaman_seviyeleri[-1]
    egitim_zamanlari = zaman_seviyeleri[:-1]

    egitim_df = panel_df[panel_df.index.get_level_values("time").isin(egitim_zamanlari)]
    holdout_df = panel_df[panel_df.index.get_level_values("time") == holdout_zaman]

    egitim_entities = egitim_df.index.get_level_values("entity").nunique()
    if egitim_entities < 3:
        return {"yeterli_veri": False, "mesaj": "Egitim setinde yeterli sayida DMU yok (en az 3 gerekli)."}

    try:
        egitim_panel_sonuc = run_panel_analysis(egitim_df, bagimli=bagimli, bagimsizlar=bagimsizlar)
    except Exception as e:
        return {
            "yeterli_veri": False,
            "mesaj": (
                f"Egitim seti ile panel modeli kurulamadi ({e}). Muhtemelen egitim donem sayisi "
                f"cok az (ozellikle FE modeli icin rank deficiency riski). Ham veri donemi sayinizi "
                f"artirmayi deneyin."
            ),
        }

    oneri = egitim_panel_sonuc["oneri"]
    tablo_map = {
        "pooled_robust": egitim_panel_sonuc["pooled_robust"], "pooled_clustered": egitim_panel_sonuc["pooled_clustered"],
        "fe_robust": egitim_panel_sonuc["fe_robust"], "fe_clustered": egitim_panel_sonuc["fe_clustered"],
        "re_robust": egitim_panel_sonuc["re_robust"], "re_clustered": egitim_panel_sonuc["re_clustered"],
    }
    nihai_res = tablo_map[oneri["sonuc_tablo"]]

    X_holdout = holdout_df[bagimsizlar]
    try:
        tahminler = _tahmin_et(nihai_res, X_holdout)
    except Exception as e:
        return {"yeterli_veri": False, "mesaj": f"Holdout tahmini hesaplanamadi: {e}"}

    tahmin_satirlar = []
    for entity in X_holdout.index.get_level_values("entity"):
        gercek = float(holdout_df.loc[(entity, holdout_zaman), bagimli])
        tahmin = float(tahminler.loc[(entity, holdout_zaman)])
        hata = tahmin - gercek
        yon_gercek = "Artis" if gercek > 1.0 else "Azalis"
        yon_tahmin = "Artis" if tahmin > 1.0 else "Azalis"
        tahmin_satirlar.append({
            "DMU": entity, "gercek_MI": round(gercek, 4), "tahmin_MI": round(tahmin, 4),
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
    if len(gercek_arr) >= 2 and np.std(gercek_arr) > 0 and np.std(tahmin_arr) > 0:
        pearson_r = float(np.corrcoef(gercek_arr, tahmin_arr)[0, 1])
    else:
        pearson_r = None
    yon_dogruluk = float(tahmin_df["yon_dogru_mu"].mean() * 100)

    # Naif baseline: "MI degismeyecek" (M=1) varsayimiyla tahmin -- karsilastirma noktasi
    naif_tahmin = np.ones_like(gercek_arr)
    naif_mae = float(np.mean(np.abs(naif_tahmin - gercek_arr)))

    return {
        "yeterli_veri": True,
        "egitim_zamanlari": egitim_zamanlari, "holdout_zaman": holdout_zaman,
        "nihai_baslik": oneri["sonuc_basligi"],
        "tahmin_df": tahmin_df,
        "metrikler": {
            "MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE_%": round(mape, 2),
            "Pearson_r": round(pearson_r, 4) if pearson_r is not None else None,
            "yon_dogruluk_%": round(yon_dogruluk, 1),
            "naif_baseline_MAE": round(naif_mae, 4),
            "modelin_naiften_iyi_mi": mae < naif_mae,
        },
    }
