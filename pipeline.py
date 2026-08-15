# -*- coding: utf-8 -*-
"""
PIPELINE ORKESTRASYONU
=========================
Excel -> [SECILEN girdi/ciktilarla DEA] -> [ardisik donemler icin Malmquist] ->
[SECILEN (DEA'da kullanilmayan) girdilerle panel veri seti] -> [panel analizi]

IKI AYRI SECIM ASAMASI:
  1) dea_girdiler / dea_ciktilar: Excel'deki TUM Girdi_/Cikti_ sutunlarindan,
     DEA + Malmquist hesabinda KULLANILACAK olanlar (kullanici secer).
  2) panel_girdiler: dea_girdiler'DE OLMAYAN (yani DEA'da kullanilmamis)
     Girdi_ sutunlarindan, panel regresyonunda KULLANILACAK olanlar
     (kullanici secer). Bu iki kume kesinlikle ORTUSEMEZ -- ayrilabilirlik
     varsayimini (Simar & Wilson, 2007) boylece otomatik korunur.

Kullanim:
    sonuc = run_pipeline(
        "veri.xlsx",
        dea_girdiler=["Girdi_SimSuresi", "Girdi_Maliyet"],
        dea_ciktilar=["Cikti_Hata"],
        panel_girdiler=["Girdi_OperatorDeneyimi"],   # DEA'da KULLANILMAYAN girdilerden
    )
"""
import pandas as pd
from excel_okuma import excel_oku, donemlere_ayir
from dea_module import solve_dea_period
from malmquist_module import solve_malmquist
from panel_module import run_panel_analysis


def run_pipeline(dosya_yolu, dea_girdiler=None, dea_ciktilar=None, panel_girdiler=None):
    # 1) Excel oku + dogrula (TUM Girdi_/Cikti_ sutunlari okunur)
    veri = excel_oku(dosya_yolu)
    X_tum, Y_tum = donemlere_ayir(veri)  # TUM girdi/cikti sutunlarini icerir
    donemler = veri["donem_sirali"]

    if len(donemler) < 2:
        raise ValueError("Malmquist icin en az 2 donem gerekli.")

    # --- Secim dogrulamalari ---
    if dea_girdiler is None or len(dea_girdiler) == 0:
        raise ValueError("DEA icin en az bir girdi secilmeli.")
    if dea_ciktilar is None or len(dea_ciktilar) == 0:
        raise ValueError("DEA icin en az bir cikti secilmeli.")
    if panel_girdiler is None or len(panel_girdiler) == 0:
        raise ValueError(
            "Panel regresyonu icin en az bir girdi secilmeli "
            "(DEA'da SECILMEMIS Girdi_ sutunlarindan)."
        )

    gecersiz_dea_g = set(dea_girdiler) - set(veri["girdi_cols"])
    gecersiz_dea_c = set(dea_ciktilar) - set(veri["cikti_cols"])
    if gecersiz_dea_g or gecersiz_dea_c:
        raise ValueError(f"Excel'de olmayan DEA sutunu secildi: {gecersiz_dea_g | gecersiz_dea_c}")

    cakisan = set(dea_girdiler) & set(panel_girdiler)
    if cakisan:
        raise ValueError(
            f"AYRILABILIRLIK IHLALI: {cakisan} hem DEA'da hem panelde secilemez -- "
            f"panel girdileri, DEA'da SECILMEMIS Girdi_ sutunlarindan olmalidir."
        )
    gecersiz_panel = set(panel_girdiler) - set(veri["girdi_cols"])
    if gecersiz_panel:
        raise ValueError(f"Panel icin Excel'de olmayan/Girdi_ olmayan sutun secildi: {gecersiz_panel}")

    # 2) DEA -- SADECE secilen girdi/ciktilarla, her donem icin
    X = {d: X_tum[d][dea_girdiler] for d in donemler}
    Y = {d: Y_tum[d][dea_ciktilar] for d in donemler}

    dea_sonuclari = {}
    for d in donemler:
        dea_sonuclari[d] = solve_dea_period(X[d], Y[d])

    # 3) Ardisik donemler icin Malmquist (gecikmeli: M(k,t), t=gecisin basladigi donem)
    malmquist_df = solve_malmquist(X, Y, donemler)  # index=[DMU, donem], columns=[EC,TC,M]

    # 4) Panel veri seti: her (DMU, donem) satirinda SECILEN (DEA-disi) girdi
    #    degerleri + o donemden baslayan MI degeri
    df_ham = veri["df"].set_index(["Donem", "DMU"])
    panel_rows = []
    gecisli_donemler = donemler[:-1]  # MI olan donemler
    for i, d in enumerate(gecisli_donemler, start=1):
        for dmu in veri["dmu_sirali"]:
            satir = {"entity": dmu, "time": i, "donem": d}
            for col in panel_girdiler:
                satir[col] = df_ham.loc[(d, dmu), col]
            satir["MI"] = malmquist_df.loc[(dmu, d), "M"]
            satir["EC"] = malmquist_df.loc[(dmu, d), "EC"]
            satir["TC"] = malmquist_df.loc[(dmu, d), "TC"]
            panel_rows.append(satir)

    panel_df = pd.DataFrame(panel_rows).set_index(["entity", "time"]).sort_index()

    # 6) DEA girdilerinin TUM donemler icin uzun-format (entity, time) hali --
    #    panel_df ARTIK bu girdileri icermedigi (sadece panel_girdiler var)
    #    icin, "ML Tahmin" sekmesindeki zaman-sabitlik teshisi gibi DEA
    #    girdilerine ozgu diagnostikler icin AYRICA saglanir.
    dea_girdi_panel_rows = []
    for i, d in enumerate(donemler, start=1):
        for dmu in veri["dmu_sirali"]:
            satir = {"entity": dmu, "time": i}
            for col in dea_girdiler:
                satir[col] = X[d].loc[dmu, col]
            dea_girdi_panel_rows.append(satir)
    dea_girdi_panel_df = pd.DataFrame(dea_girdi_panel_rows).set_index(["entity", "time"]).sort_index()

    # 7) Panel analizi -- kullanicinin sectigi (DEA-disi) girdilerle
    panel_sonuc = run_panel_analysis(panel_df, bagimli="MI", bagimsizlar=panel_girdiler)

    return {
        "veri": veri,
        "X": X, "Y": Y,
        "dea_girdiler": dea_girdiler, "dea_ciktilar": dea_ciktilar,
        "panel_girdiler": panel_girdiler,
        "dea": dea_sonuclari,
        "malmquist": malmquist_df,
        "panel_df": panel_df,
        "dea_girdi_panel_df": dea_girdi_panel_df,
        "panel_sonuc": panel_sonuc,
    }


if __name__ == "__main__":
    import numpy as np

    donemler = ["t1", "t2", "t3", "t4"]
    dmus = ["DMU1", "DMU2", "DMU3", "DMU4", "DMU5"]
    rng = np.random.default_rng(7)
    satirlar = []
    for d in donemler:
        for u in dmus:
            satirlar.append({
                "Donem": d, "DMU": u,
                "Girdi_SimSuresi": int(rng.integers(200, 450)),
                "Girdi_OperatorDeneyimi": int(rng.integers(1, 15)),  # DEA'da KULLANILMAYACAK
                "Cikti_Prototip": int(rng.integers(1, 5)),
            })
    pd.DataFrame(satirlar).to_excel("/tmp/test_pipeline.xlsx", index=False)

    sonuc = run_pipeline(
        "/tmp/test_pipeline.xlsx",
        dea_girdiler=["Girdi_SimSuresi"],           # DEA'da SADECE bu
        dea_ciktilar=["Cikti_Prototip"],
        panel_girdiler=["Girdi_OperatorDeneyimi"],  # panelde SADECE bu (DEA'da kullanilmayan)
    )
    print("DEA (t1) theta_ccr:\n", sonuc["dea"]["t1"]["theta_ccr"])
    print("\nMalmquist ozet:\n", sonuc["malmquist"])
    print("\nPanel df:\n", sonuc["panel_df"])
    print("\nSecilen model:", sonuc["panel_sonuc"]["secilen_model"])
