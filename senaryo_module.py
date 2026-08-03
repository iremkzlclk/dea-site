# -*- coding: utf-8 -*-
"""
GELECEK DONEM SENARYO MODULU
================================
Panel analizi sonucundaki katsayı yonleri ile degiskenlerin tarihsel "dogal
seyrini" (medyan-delta trend) ve aralarindaki coklu dogrusal iliskiyi (VIF)
birlikte kullanarak, SADECE SON DONEMIN verilerini temel alan bir "bir
sonraki donem" senaryosu uretir.

Akis (kullanicinin belirledigi sira):
  1) Ham veri -> Medyan-delta trend (dogal seyir)
     Her DMU-degisken cifti icin, tum tarihsel donemler arasindaki ardisik
     farklarin MEDYANI alinir (aykiri donemlere karsi mean'den daha dayanikli).

  2) Panel-DEA tutarlilik kontrolu -> Hedefli / Dogal ayrimi
     Panel modelinde ANLAMLI (p < HEDEFLI_ALPHA) VE DEA teorik beklentisiyle
     (Girdi -> negatif, Cikti -> pozitif) TUTARLI olan degiskenler "Hedefli"
     kabul edilir: bunlar, katsayinin isaret ettigi iyilesme yonunde
     (girdi azalt / cikti artir) deliberate olarak itilir. Anlamsiz ya da
     teoriyle celisen (muhtemelen icsel devirsellik kaynakli) degiskenler
     "Dogal" kalir: sadece kendi tarihsel medyan-delta trendini takip eder,
     hicbir deliberate mudahale yapilmaz.

  3) VIF kontrolu (hedefli <-> dogal arasi)
     VIF < 5   -> dogal degisken oldugu gibi (tam medyan-delta ile) birakilir.
     VIF >= 5  -> bu degisken panel modelindeki diger regresorlerle (hedefli
                  dahil) yuksek coklu dogrusal iliski icinde; kendi tarihsel
                  trendini TAM UYGULAMAK yaniltici olabilir (degiskenler
                  birlikte hareket ediyor olabilir). Bu durumda trend
                  DAMPING_FAKTORU ile SINIRLI EKSTRAPOLASYONLA uygulanir ve
                  ayrica bir "kisit" notu eklenir (hangi VIF degeriyle,
                  neden temkinli davranildigi raporlanir).

  4) Sinir kontrolleri
     Projeksiyon, DMU'nun kendi tarihsel min/max araligina gore makul bir
     bant disina cikamaz (asiri ekstrapolasyon onlenir); negatif deger
     DEA icin gecersiz oldugundan asla uretilmez.

  5) Duyarlilik taramasi -> Nihai senaryo
     Hedefli degiskenlerin itilme buyuklugu bir "olcek" katsayisiyla
     carpilir: Temkinli (0.5x) / Baz (1.0x) / Iyimser (1.5x). Baz senaryo
     nihai/one cikan senaryo olarak sunulur, digerleri duyarlilik araligi
     olarak yaninda gosterilir.
"""
import numpy as np
import pandas as pd

VIF_ESIGI = 5.0
HEDEFLI_ALPHA = 0.10           # panel_module.py'deki KATSAYI_ALPHA ile tutarli tutulur
DAMPING_FAKTORU = 0.5          # yuksek VIF'te dogal trendin ne kadar bastirilacagi
MIN_ADIM_ORANI = 0.02          # medyan delta ~0 ise, hedefli degisken icin nominal adim (%2)
SINIR_ALT_ORAN = 0.5           # tarihsel min'in bu oranindan asagi inilmez
SINIR_UST_ORAN = 1.5           # tarihsel max'in bu oranindan yukari cikilmaz


def _tarihsel_seri_ve_medyan_delta(X: dict, Y: dict, donemler: list, dmu: str, degisken: str,
                                    girdi_cols, cikti_cols):
    """Bir DMU'nun bir degiskeni icin tum tarihsel donemlerdeki degerleri ve
    ardisik farklarinin medyanini dondurur."""
    kaynak = X if degisken in girdi_cols else Y
    degerler = [float(kaynak[d].loc[dmu, degisken]) for d in donemler]
    farklar = np.diff(degerler)
    medyan_delta = float(np.median(farklar)) if len(farklar) else 0.0
    return degerler, medyan_delta


def hedefli_dogal_siniflandir(nihai_res, girdi_cols, cikti_cols, alpha: float = HEDEFLI_ALPHA) -> pd.DataFrame:
    """
    Panel modelindeki her bagimsiz degiskeni 'Hedefli' (anlamli + DEA
    teorisiyle tutarli -> iyilesme yonunde deliberate itilir) veya 'Dogal'
    (anlamsiz ya da teoriyle celisen -> sadece tarihsel trendini takip eder)
    olarak siniflandirir. Ayrim mantigi panel_module/yorumlama'daki
    panel_aksiyon_analizi ile ayni (celiski tanimi dahil), boylece panel
    sekmesindeki yorumla senaryo sekmesi tutarli kalir.
    """
    satirlar = []
    for degisken in nihai_res.params.index:
        if degisken == "const":
            continue
        katsayi = float(nihai_res.params[degisken])
        p = float(nihai_res.pvalues[degisken])
        anlamli = p < alpha

        if degisken in girdi_cols:
            tip, beklenen_yon = "Girdi", -1   # MI'yi iyilestirmek icin azalma beklenir
        elif degisken in cikti_cols:
            tip, beklenen_yon = "Cikti", +1   # MI'yi iyilestirmek icin artis beklenir
        else:
            tip, beklenen_yon = "Diger", 0

        gercek_yon = 1 if katsayi > 0 else -1
        celiski = anlamli and beklenen_yon != 0 and gercek_yon != beklenen_yon
        hedefli = anlamli and (not celiski) and beklenen_yon != 0
        iyilesme_yonu = beklenen_yon if hedefli else 0

        satirlar.append({
            "degisken": degisken, "tip": tip, "katsayi": round(katsayi, 5), "p_degeri": round(p, 4),
            "anlamli_mi": anlamli, "celiski": celiski,
            "siniflandirma": "Hedefli" if hedefli else "Dogal",
            "iyilesme_yonu": iyilesme_yonu,
        })
    return pd.DataFrame(satirlar).set_index("degisken")


def senaryo_olustur(X: dict, Y: dict, donemler: list, nihai_res, girdi_cols, cikti_cols,
                     vif_df: pd.DataFrame, dmu_sirali: list, olcek: float = 1.0):
    """
    Son donemin (donemler[-1]) verilerini temel alarak bir sonraki donem icin
    senaryo verisi uretir (5 asamali akisin tamami).

    olcek: hedefli degiskenlerin iyilesme adiminin buyuklugunu olcekler
           (duyarlilik taramasi icin -- Temkinli=0.5, Baz=1.0, Iyimser=1.5).

    Returns: X_next (DataFrame), Y_next (DataFrame), detay (DataFrame, index=[DMU,degisken]),
             siniflandirma (DataFrame, index=degisken)
    """
    siniflandirma = hedefli_dogal_siniflandir(nihai_res, girdi_cols, cikti_cols)
    vif_map = dict(zip(vif_df["degisken"], vif_df["VIF"]))

    son_donem = donemler[-1]
    tum_degiskenler = list(girdi_cols) + list(cikti_cols)

    X_next = X[son_donem].astype(float).copy()
    Y_next = Y[son_donem].astype(float).copy()

    detay_satirlar = []

    for dmu in dmu_sirali:
        for degisken in tum_degiskenler:
            degerler, medyan_delta = _tarihsel_seri_ve_medyan_delta(
                X, Y, donemler, dmu, degisken, girdi_cols, cikti_cols
            )
            son_deger = degerler[-1]
            vif_deger = vif_map.get(degisken, np.nan)
            yuksek_vif = (not np.isnan(vif_deger)) and (vif_deger >= VIF_ESIGI)

            is_hedefli = (degisken in siniflandirma.index) and \
                         (siniflandirma.loc[degisken, "siniflandirma"] == "Hedefli")

            if is_hedefli:
                # --- ASAMA 2: Hedefli -> iyilesme yonunde deliberate itis ---
                yon = siniflandirma.loc[degisken, "iyilesme_yonu"]
                adim = abs(medyan_delta)
                if adim < son_deger * MIN_ADIM_ORANI:
                    adim = son_deger * MIN_ADIM_ORANI  # medyan delta ~0 ise nominal adim
                yeni_deger = son_deger + yon * adim * olcek
                tip_uygulanan = "Hedefli"
                not_metni = (
                    f"Panel modelinde anlamli ve DEA teorisiyle tutarli (katsayi="
                    f"{siniflandirma.loc[degisken,'katsayi']}, p={siniflandirma.loc[degisken,'p_degeri']}); "
                    f"iyilesme yonunde ({'azalt' if yon < 0 else 'artir'}) deliberate itiliyor "
                    f"(olcek={olcek:g})."
                )
            else:
                # --- ASAMA 1 + 3: Dogal -> medyan-delta trend, VIF>=5 ise sinirli ekstrapolasyon ---
                carpan = DAMPING_FAKTORU if yuksek_vif else 1.0
                yeni_deger = son_deger + medyan_delta * carpan
                tip_uygulanan = "Dogal" + (" (VIF yuksek - sinirli ekstrapolasyon)" if yuksek_vif else "")
                if yuksek_vif:
                    not_metni = (
                        f"VIF={vif_deger:.2f} (>= {VIF_ESIGI:g}) -- diger regresorlerle (hedefli degiskenler "
                        f"dahil) yuksek coklu dogrusal iliski icinde. Tarihsel trend tam uygulanmadi; "
                        f"%{int(DAMPING_FAKTORU*100)} olcekte sinirli ekstrapolasyonla uygulandi (kisit)."
                    )
                else:
                    not_metni = "Panelde anlamsiz veya teoriyle celisen; sadece tarihsel medyan-delta trendi takip edildi."

            # --- ASAMA 4: Sinir kontrolleri ---
            hist_min, hist_max = min(degerler), max(degerler)
            alt_sinir = hist_min * SINIR_ALT_ORAN
            ust_sinir = hist_max * SINIR_UST_ORAN
            yeni_deger_sinirsiz = yeni_deger
            yeni_deger = max(alt_sinir, min(ust_sinir, yeni_deger))
            yeni_deger = max(yeni_deger, 0.0)  # DEA icin negatif deger gecersiz
            sinir_uygulandi = not np.isclose(yeni_deger, yeni_deger_sinirsiz, atol=1e-9)

            if degisken in girdi_cols:
                X_next.loc[dmu, degisken] = yeni_deger
            else:
                Y_next.loc[dmu, degisken] = yeni_deger

            detay_satirlar.append({
                "DMU": dmu, "degisken": degisken, "tip_uygulanan": tip_uygulanan,
                "son_deger": round(son_deger, 2), "medyan_delta": round(medyan_delta, 3),
                "yeni_deger": round(yeni_deger, 2),
                "degisim_yuzde": round((yeni_deger - son_deger) / son_deger * 100, 2) if son_deger else np.nan,
                "vif": round(vif_deger, 2) if not np.isnan(vif_deger) else None,
                "sinir_uygulandi": sinir_uygulandi, "not": not_metni,
            })

    detay = pd.DataFrame(detay_satirlar).set_index(["DMU", "degisken"])
    return X_next, Y_next, detay, siniflandirma


def uc_senaryo_olustur(X, Y, donemler, nihai_res, girdi_cols, cikti_cols, vif_df, dmu_sirali):
    """Temkinli (0.5x) / Baz (1.0x) / Iyimser (1.5x) uc senaryoyu birlikte uretir
    (ASAMA 5: duyarlilik taramasi)."""
    senaryolar = {}
    siniflandirma = None
    for isim, olcek in [("Temkinli", 0.5), ("Baz", 1.0), ("Iyimser", 1.5)]:
        X_n, Y_n, detay, siniflandirma = senaryo_olustur(
            X, Y, donemler, nihai_res, girdi_cols, cikti_cols, vif_df, dmu_sirali, olcek=olcek,
        )
        senaryolar[isim] = {"X": X_n, "Y": Y_n, "detay": detay, "olcek": olcek}
    return senaryolar, siniflandirma


def gelecek_donem_analizi(sonuc: dict) -> dict:
    """
    Ana orkestrasyon: pipeline.run_pipeline() ciktisini (sonuc) alir, panel
    modelinin sectigi nihai sonucu kullanarak 3 senaryo uretir, HER senaryo
    icin projeksiyon verisiyle DEA'yi tekrar cozer ve son gercek donem ile
    projeksiyon donemi arasinda Malmquist (EC/TC/M) hesaplar.

    Returns: dict -- siniflandirma, son_donem, senaryolar:
             {isim: {X, Y, detay, olcek, dea, malmquist}}
    """
    from dea_module import solve_dea_period
    from malmquist_module import solve_malmquist

    veri = sonuc["veri"]
    girdi_cols, cikti_cols = veri["girdi_cols"], veri["cikti_cols"]
    donemler = veri["donem_sirali"]
    dmu_sirali = veri["dmu_sirali"]

    p = sonuc["panel_sonuc"]
    oneri = p["oneri"]
    tablo_map = {
        "pooled_robust": p["pooled_robust"], "pooled_clustered": p["pooled_clustered"],
        "fe_robust": p["fe_robust"], "fe_clustered": p["fe_clustered"],
        "re_robust": p["re_robust"], "re_clustered": p["re_clustered"],
    }
    nihai_res = tablo_map[oneri["sonuc_tablo"]]

    senaryolar, siniflandirma = uc_senaryo_olustur(
        sonuc["X"], sonuc["Y"], donemler, nihai_res, girdi_cols, cikti_cols, p["vif"], dmu_sirali,
    )

    son_donem = donemler[-1]
    for isim, s in senaryolar.items():
        s["dea"] = solve_dea_period(s["X"], s["Y"])
        X_ext = {son_donem: sonuc["X"][son_donem], "t_gelecek": s["X"]}
        Y_ext = {son_donem: sonuc["Y"][son_donem], "t_gelecek": s["Y"]}
        s["malmquist"] = solve_malmquist(X_ext, Y_ext, [son_donem, "t_gelecek"])

    return {
        "siniflandirma": siniflandirma,
        "son_donem": son_donem,
        "nihai_res_baslik": oneri["sonuc_basligi"],
        "senaryolar": senaryolar,
    }
