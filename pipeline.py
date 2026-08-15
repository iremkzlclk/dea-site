# -*- coding: utf-8 -*-
"""
PIPELINE ORKESTRASYONU
=========================
Excel -> [her donem icin DEA] -> [ardisik donemler icin Malmquist] ->
[panel veri seti olustur] -> [panel analizi]

Kullanim:
    sonuc = run_pipeline("veri.xlsx", bagimsizlar=["Girdi_SimSuresi","Cikti_Hata"])
"""
import pandas as pd
from excel_okuma import excel_oku, donemlere_ayir
from dea_module import solve_dea_period
from malmquist_module import solve_malmquist
from panel_module import run_panel_analysis


def run_pipeline(dosya_yolu, panel_degiskenler=None):
    """
    panel_degiskenler: ikinci asama panel regresyonunda kullanilacak
    sutun adlari listesi -- kullanicinin Excel'deki "diger_cols"
    havuzundan SECTIGI degiskenler. DEA girdi/ciktilariyla ORTUSEMEZ
    (ayrilabilirlik varsayimi -- Simar & Wilson, 2007).
    """
    # 1) Excel oku + dogrula
    veri = excel_oku(dosya_yolu)
    X, Y = donemlere_ayir(veri)
    donemler = veri["donem_sirali"]

    if len(donemler) < 2:
        raise ValueError("Malmquist icin en az 2 donem gerekli.")

    if panel_degiskenler is None or len(panel_degiskenler) == 0:
        raise ValueError(
            "Panel regresyonu icin en az bir cevresel degisken secilmeli "
            "(Excel'deki Girdi_/Cikti_ DISINDAKI sutunlardan)."
        )
    cakisan = set(panel_degiskenler) & set(veri["girdi_cols"] + veri["cikti_cols"])
    if cakisan:
        raise ValueError(
            f"AYRILABILIRLIK IHLALI: {cakisan} hem DEA'da hem panel "
            f"regresyonunda kullanilamaz."
        )

    # 2) Her donem icin DEA (raporlama amacli, Malmquist'ten bagimsiz calisir)
    dea_sonuclari = {}
    for d in donemler:
        dea_sonuclari[d] = solve_dea_period(X[d], Y[d])

    # 3) Ardisik donemler icin Malmquist (gecikmeli: M(k,t), t=gecisin basladigi donem)
    malmquist_df = solve_malmquist(X, Y, donemler)  # index=[DMU, donem], columns=[EC,TC,M]

    # 4) Panel veri seti olustur: her (DMU, donem) satirinda o donemin
    #    SECILEN cevresel degisken degerleri + o donemden baslayan MI
    #    degeri (son donem haric, cunku MI yok). DEA girdi/ciktilari
    #    ARTIK panel_df'e kopyalanmiyor -- ayrilabilirlik geregi panel
    #    regresyonuna hic girmiyorlar.
    df_ham = veri["df"].set_index(["Donem", "DMU"])
    panel_rows = []
    gecisli_donemler = donemler[:-1]  # MI olan donemler
    for i, d in enumerate(gecisli_donemler, start=1):
        for dmu in veri["dmu_sirali"]:
            satir = {"entity": dmu, "time": i, "donem": d}
            for col in panel_degiskenler:
                satir[col] = df_ham.loc[(d, dmu), col]
            satir["MI"] = malmquist_df.loc[(dmu, d), "M"]
            satir["EC"] = malmquist_df.loc[(dmu, d), "EC"]
            satir["TC"] = malmquist_df.loc[(dmu, d), "TC"]
            panel_rows.append(satir)

    panel_df = pd.DataFrame(panel_rows).set_index(["entity", "time"]).sort_index()

    # 5) Panel analizi -- kullanicinin sectigi cevresel degiskenlerle
    panel_sonuc = run_panel_analysis(panel_df, bagimli="MI", bagimsizlar=panel_degiskenler)

    return {
        "veri": veri,
        "X": X,
        "Y": Y,
        "dea": dea_sonuclari,
        "malmquist": malmquist_df,
        "panel_df": panel_df,
        "panel_sonuc": panel_sonuc,
    }


if __name__ == "__main__":
    import numpy as np

    # Gercekci kucuk test verisi (4 donem, 5 DMU) uret
    donemler = ["t1", "t2", "t3", "t4"]
    dmus = ["DMU1", "DMU2", "DMU3", "DMU4", "DMU5"]
    rng = np.random.default_rng(7)
    satirlar = []
    for d in donemler:
        for u in dmus:
            satirlar.append({
                "Donem": d, "DMU": u,
                "Girdi_SimSuresi": int(rng.integers(200, 450)),
                "Cikti_Prototip": int(rng.integers(1, 5)),
                "Operator_Deneyimi": int(rng.integers(1, 15)),  # cevresel (DEA disi) degisken
            })
    pd.DataFrame(satirlar).to_excel("/tmp/test_pipeline.xlsx", index=False)

    sonuc = run_pipeline("/tmp/test_pipeline.xlsx", panel_degiskenler=["Operator_Deneyimi"])
    print("DEA (t1) theta_ccr:\n", sonuc["dea"]["t1"]["theta_ccr"])
    print("\nMalmquist ozet:\n", sonuc["malmquist"])
    print("\nPanel df:\n", sonuc["panel_df"])
    print("\nSecilen model:", sonuc["panel_sonuc"]["secilen_model"])
    print("\nFE sonuc:\n", sonuc["panel_sonuc"]["fe"])
