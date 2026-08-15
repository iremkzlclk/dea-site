# -*- coding: utf-8 -*-
"""
EXCEL OKUMA + DOGRULAMA MODULU
================================
Beklenen sutun yapisi (uzun format):
  Donem | DMU | Girdi_<isim1> | Girdi_<isim2> | ... | Cikti_<isim1> | ...

- "Girdi_" ile baslayan sutunlar otomatik girdi olarak alinir
- "Cikti_" ile baslayan sutunlar otomatik cikti olarak alinir
- Donem sutunu, dogal siralamayi koruyacak sekilde (t1<t2<...) sirali olmalidir
  (metin/tarih/yil/ay farketmez, ama Excel'deki SIRAYA gore t1,t2,t3... atanir
  -- bu yuzden Excel'de donemler KRONOLOJIK sirada yazilmali)
"""
import pandas as pd


class VeriDogrulamaHatasi(Exception):
    pass


def excel_oku(dosya_yolu_veya_buffer, sheet_name=0):
    df = pd.read_excel(dosya_yolu_veya_buffer, sheet_name=sheet_name)

    zorunlu = {"Donem", "DMU"}
    eksik = zorunlu - set(df.columns)
    if eksik:
        raise VeriDogrulamaHatasi(f"Excel'de eksik sutun(lar): {eksik}")

    girdi_cols = [c for c in df.columns if c.startswith("Girdi_")]
    cikti_cols = [c for c in df.columns if c.startswith("Cikti_")]

    if not girdi_cols:
        raise VeriDogrulamaHatasi("En az bir 'Girdi_...' sutunu olmali.")
    if not cikti_cols:
        raise VeriDogrulamaHatasi("En az bir 'Cikti_...' sutunu olmali.")

    # eksik hucre kontrolu
    kontrol_cols = ["Donem", "DMU"] + girdi_cols + cikti_cols
    bos = df[kontrol_cols].isna()
    if bos.any().any():
        satirlar = df[bos.any(axis=1)][["Donem", "DMU"]]
        raise VeriDogrulamaHatasi(f"Eksik hucre(ler) bulundu:\n{satirlar}")

    # negatif deger kontrolu (DEA girdi/ciktilari negatif olamaz)
    sayisal = df[girdi_cols + cikti_cols]
    if (sayisal < 0).any().any():
        raise VeriDogrulamaHatasi("Girdi/cikti sutunlarinda negatif deger bulundu.")

    # her donem-DMU kombinasyonu tek olmali
    tekrar = df.duplicated(subset=["Donem", "DMU"])
    if tekrar.any():
        raise VeriDogrulamaHatasi(f"Tekrarlanan Donem-DMU satirlari var:\n{df[tekrar][['Donem','DMU']]}")

    # her donemde ayni DMU seti olmali (dengeli panel varsayimi)
    donem_dmu_sayisi = df.groupby("Donem")["DMU"].nunique()
    if donem_dmu_sayisi.nunique() > 1:
        raise VeriDogrulamaHatasi(
            f"Donemler arasinda DMU sayisi tutarsiz (dengeli panel bekleniyor):\n{donem_dmu_sayisi}"
        )

    # donem sirasini EXCEL'DEKI ILK GORULME sirasina gore koru (kronolojik varsayim)
    donem_sirali = list(dict.fromkeys(df["Donem"]))

    # her donem icin DMU siralamasini da sabitle (ilk donemdeki sira referans)
    dmu_sirali = list(dict.fromkeys(df[df["Donem"] == donem_sirali[0]]["DMU"]))

    return {
        "df": df,
        "girdi_cols": girdi_cols,
        "cikti_cols": cikti_cols,
        "donem_sirali": donem_sirali,
        "dmu_sirali": dmu_sirali,
    }


def donemlere_ayir(veri: dict):
    """
    Her donem icin ayri X (girdi) ve Y (cikti) DataFrame'i uretir.
    Returns: X: dict{donem: DataFrame(index=DMU, columns=girdi_cols)}
             Y: dict{donem: DataFrame(index=DMU, columns=cikti_cols)}
    """
    df = veri["df"]
    X, Y = {}, {}
    for donem in veri["donem_sirali"]:
        alt = df[df["Donem"] == donem].set_index("DMU").loc[veri["dmu_sirali"]]
        X[donem] = alt[veri["girdi_cols"]]
        Y[donem] = alt[veri["cikti_cols"]]
    return X, Y


if __name__ == "__main__":
    # kucuk bir ornek Excel uretip test edelim
    import numpy as np
    donemler = ["t1", "t2", "t3"]
    dmus = ["A1", "A2", "B1"]
    satirlar = []
    rng = np.random.default_rng(42)
    for d in donemler:
        for u in dmus:
            satirlar.append({
                "Donem": d, "DMU": u,
                "Girdi_SimSuresi": rng.integers(200, 400),
                "Girdi_Maliyet": rng.integers(200, 400),
                "Cikti_Hata": rng.integers(5, 15),
            })
    ornek_df = pd.DataFrame(satirlar)
    ornek_df.to_excel("/tmp/ornek_veri.xlsx", index=False)

    veri = excel_oku("/tmp/ornek_veri.xlsx")
    print("Girdi sutunlari:", veri["girdi_cols"])
    print("Cikti sutunlari:", veri["cikti_cols"])
    print("Donem sirasi:", veri["donem_sirali"])
    print("DMU sirasi:", veri["dmu_sirali"])

    X, Y = donemlere_ayir(veri)
    print("\nt1 girdi tablosu:\n", X["t1"])
    print("\nt1 cikti tablosu:\n", Y["t1"])
