# -*- coding: utf-8 -*-
"""
BACKTEST MODULU (tek-katli + rolling/walk-forward)
=======================================================
Panel modelinin GERCEK tahmin gucunu, DEA'yi tekrar cozmeden, regresyon
dogrulamasiyla olcer. Iki mod sunar:

  1) backtest_calistir(): TEK KATLI (leave-last-period-out) -- sadece SON
     gecisi test icin ayirir. Hizli ama TEK BIR denemeye dayanir; sonucu
     kendisi de yuksek varyansli olabilir.

  2) rolling_backtest_calistir(): COK KATLI (walk-forward / rolling-origin)
     -- MUMKUN OLAN HER gecisi sirayla test icin ayirir (o ana kadarki
     butun gecmisle egitip bir sonrakini tahmin eder), ve TUM katlarin
     ortalamasini raporlar. Bu, "bu katsayilara dayanarak yatirim kararı
     alsam, GECMISTE bu ne kadar tutarli isterdi" sorusuna -- tek bir
     denemeye degil, MUMKUN OLAN TUM denemelere dayanan -- çok daha
     güvenilir bir cevap verir. Ham veri donemi sayisi arttikca (T buyudukce)
     kat sayisi da artar ve sonuc gitgide daha anlamli hale gelir.

Her iki mod da, "MI hic degismeyecek" (M=1) varsayan naif bir baseline ile
karsilastirma yapar.
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


def _tek_kat_calistir(panel_df: pd.DataFrame, egitim_zamanlari: list, test_zamani,
                       bagimsizlar: list, bagimli: str):
    """
    TEK BIR kat icin: egitim_zamanlari ile modeli kurar, test_zamani icin
    GERCEK verilerle MI tahmin eder, gercek MI ile karsilastirir.

    Returns: dict (basarili ise) -- nihai_baslik, tahmin_df, metrikler
             veya None (egitim/tahmin basarisiz olursa -- bu kat atlanir)
    """
    egitim_df = panel_df[panel_df.index.get_level_values("time").isin(egitim_zamanlari)]
    test_df = panel_df[panel_df.index.get_level_values("time") == test_zamani]

    egitim_entities = egitim_df.index.get_level_values("entity").nunique()
    if egitim_entities < 3:
        return None

    try:
        egitim_panel_sonuc = run_panel_analysis(egitim_df, bagimli=bagimli, bagimsizlar=bagimsizlar)
    except Exception:
        return None

    oneri = egitim_panel_sonuc["oneri"]
    tablo_map = {
        "pooled_robust": egitim_panel_sonuc["pooled_robust"], "pooled_clustered": egitim_panel_sonuc["pooled_clustered"],
        "fe_robust": egitim_panel_sonuc["fe_robust"], "fe_clustered": egitim_panel_sonuc["fe_clustered"],
        "re_robust": egitim_panel_sonuc["re_robust"], "re_clustered": egitim_panel_sonuc["re_clustered"],
    }
    nihai_res = tablo_map[oneri["sonuc_tablo"]]

    X_test = test_df[bagimsizlar]
    try:
        tahminler = _tahmin_et(nihai_res, X_test)
    except Exception:
        return None

    tahmin_satirlar = []
    for entity in X_test.index.get_level_values("entity"):
        gercek = float(test_df.loc[(entity, test_zamani), bagimli])
        tahmin = float(tahminler.loc[(entity, test_zamani)])
        hata = tahmin - gercek
        # ESIK-bazli yon: MI>1.0 mi degil mi (verimlilik ONCEKI DONEME GORE
        # artti mi azaldi mi). Bu, DMU'lar ARASI goreli siralamayi OLCMEZ --
        # tum DMU'lar zaten MI>1 civarindaysa, model siralamayi tamamen
        # TERSINE cevirse bile bu metrik hala %100 cikabilir.
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

    # SIRALAMA (rank) korelasyonu: DMU'lar ARASI goreli performansi (hangisi
    # digerinden daha verimli) modelin DOGRU siralayip siralamadigini olcer --
    # esik-bazli "yon_dogruluk"tan TAMAMEN farkli bir soruya cevap verir.
    # Ornek: tum DMU'larin gercek VE tahmin degerleri MI>1 olsa bile (esikte
    # "yon_dogruluk"=100 cikar), model DMU'larin siralamasini TAM TERSINE
    # cevirmis olabilir -- bu durumda siralama_korelasyonu negatif cikar ve
    # bu yanilgiyi acikca gosterir.
    if len(gercek_arr) >= 3 and len(set(gercek_arr)) > 1 and len(set(tahmin_arr)) > 1:
        siralama_korelasyonu = float(pd.Series(gercek_arr).corr(pd.Series(tahmin_arr), method="spearman"))
    else:
        siralama_korelasyonu = None

    yon_dogruluk = float(tahmin_df["yon_dogru_mu"].mean() * 100)

    naif_tahmin = np.ones_like(gercek_arr)
    naif_mae = float(np.mean(np.abs(naif_tahmin - gercek_arr)))

    return {
        "egitim_zamanlari": list(egitim_zamanlari), "test_zamani": test_zamani,
        "nihai_baslik": oneri["sonuc_basligi"],
        "tahmin_df": tahmin_df,
        "metrikler": {
            "MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE_%": round(mape, 2),
            "Pearson_r": round(pearson_r, 4) if pearson_r is not None else None,
            "Siralama_Korelasyonu_Spearman": round(siralama_korelasyonu, 4) if siralama_korelasyonu is not None else None,
            "esik_yon_dogruluk_%": round(yon_dogruluk, 1),
            "naif_baseline_MAE": round(naif_mae, 4),
            "modelin_naiften_iyi_mi": mae < naif_mae,
        },
    }


def backtest_calistir(panel_df: pd.DataFrame, bagimsizlar: list, bagimli: str = "MI") -> dict:
    """
    TEK KATLI (leave-last-period-out) backtest -- sadece son gecisi test eder.

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

    sonuc_kati = _tek_kat_calistir(panel_df, egitim_zamanlari, holdout_zaman, bagimsizlar, bagimli)
    if sonuc_kati is None:
        return {
            "yeterli_veri": False,
            "mesaj": (
                "Egitim seti ile panel modeli kurulamadi ya da tahmin hesaplanamadi. Muhtemelen "
                "egitim donem sayisi cok az (ozellikle FE modeli icin rank deficiency riski). "
                "Ham veri donemi sayinizi artirmayi deneyin."
            ),
        }

    return {
        "yeterli_veri": True,
        "egitim_zamanlari": sonuc_kati["egitim_zamanlari"], "holdout_zaman": sonuc_kati["test_zamani"],
        "nihai_baslik": sonuc_kati["nihai_baslik"],
        "tahmin_df": sonuc_kati["tahmin_df"],
        "metrikler": sonuc_kati["metrikler"],
    }


def rolling_backtest_calistir(panel_df: pd.DataFrame, bagimsizlar: list, bagimli: str = "MI",
                               min_egitim_donemi: int = 2) -> dict:
    """
    COK KATLI (walk-forward / rolling-origin) backtest.

    Panelde mumkun olan HER gecisi sirayla test icin ayirir: ilk
    min_egitim_donemi kadar donemle model kurulup bir sonraki donem tahmin
    edilir, sonra bir donem daha egitime eklenip bir sonraki tahmin edilir,
    ve boyle devam eder (T-min_egitim_donemi kadar kat olusur). Bu, "bu
    katsayilara dayanarak yatirim kararı alsam, GECMISTE ortalama ne kadar
    haklı cikardim" sorusuna, TEK BIR denemeye degil MUMKUN OLAN TUM
    denemelere dayanan bir cevap verir.

    min_egitim_donemi: ilk kati olusturmak icin gereken asgari gecis
    donemi sayisi (varsayilan 2 -- FE modelinin bile en az bir minimum
    within-varyasyona sahip olabilmesi icin).

    Returns: dict --
      yeterli_veri (bool), mesaj (yetersizse aciklama),
      kat_sayisi, kat_detaylari (liste, her biri bir _tek_kat_calistir ciktisi),
      ortalama_metrikler (MAE/RMSE/MAPE/yon_dogruluk -- katlar arasi ORTALAMA,
                            + katlar arasi STANDART SAPMA -- tutarliligin
                            kendisinin ne kadar guvenilir oldugunu gosterir),
      naif_baseline_MAE (katlar arasi ortalama)
    """
    zaman_seviyeleri = sorted(panel_df.index.get_level_values("time").unique())
    toplam_kat_adayi = len(zaman_seviyeleri) - min_egitim_donemi
    if toplam_kat_adayi < 1:
        return {
            "yeterli_veri": False,
            "mesaj": (
                f"Rolling backtest icin en az {min_egitim_donemi + 1} gecis donemi (yani en az "
                f"{min_egitim_donemi + 2} ham veri donemi) gerekli. Mevcut gecis donemi sayisi: "
                f"{len(zaman_seviyeleri)}."
            ),
        }

    kat_detaylari = []
    for i in range(min_egitim_donemi, len(zaman_seviyeleri)):
        egitim_zamanlari = zaman_seviyeleri[:i]
        test_zamani = zaman_seviyeleri[i]
        sonuc_kati = _tek_kat_calistir(panel_df, egitim_zamanlari, test_zamani, bagimsizlar, bagimli)
        if sonuc_kati is not None:
            kat_detaylari.append(sonuc_kati)

    if not kat_detaylari:
        return {
            "yeterli_veri": False,
            "mesaj": "Hicbir kat basariyla tamamlanamadi (muhtemelen her katta model kurma hatasi olustu).",
        }

    mae_listesi = [k["metrikler"]["MAE"] for k in kat_detaylari]
    rmse_listesi = [k["metrikler"]["RMSE"] for k in kat_detaylari]
    mape_listesi = [k["metrikler"]["MAPE_%"] for k in kat_detaylari]
    yon_listesi = [k["metrikler"]["esik_yon_dogruluk_%"] for k in kat_detaylari]
    naif_listesi = [k["metrikler"]["naif_baseline_MAE"] for k in kat_detaylari]
    siralama_listesi = [k["metrikler"]["Siralama_Korelasyonu_Spearman"] for k in kat_detaylari
                         if k["metrikler"]["Siralama_Korelasyonu_Spearman"] is not None]

    ortalama_mae = float(np.mean(mae_listesi))
    ortalama_naif = float(np.mean(naif_listesi))

    return {
        "yeterli_veri": True,
        "kat_sayisi": len(kat_detaylari),
        "denenen_kat_sayisi": toplam_kat_adayi,
        "kat_detaylari": kat_detaylari,
        "ortalama_metrikler": {
            "MAE_ortalama": round(ortalama_mae, 4), "MAE_std": round(float(np.std(mae_listesi)), 4),
            "RMSE_ortalama": round(float(np.mean(rmse_listesi)), 4),
            "MAPE_ortalama_%": round(float(np.mean(mape_listesi)), 2),
            "esik_yon_dogruluk_ortalama_%": round(float(np.mean(yon_listesi)), 1),
            "esik_yon_dogruluk_std": round(float(np.std(yon_listesi)), 1),
            "Siralama_Korelasyonu_Spearman_ortalama": (
                round(float(np.mean(siralama_listesi)), 4) if siralama_listesi else None
            ),
            "naif_baseline_MAE_ortalama": round(ortalama_naif, 4),
            "modelin_naiften_iyi_mi": ortalama_mae < ortalama_naif,
            "kat_basina_naiften_iyi_sayisi": sum(1 for k in kat_detaylari if k["metrikler"]["modelin_naiften_iyi_mi"]),
        },
    }

