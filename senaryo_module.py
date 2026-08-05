# -*- coding: utf-8 -*-
"""
GELECEK DONEM SENARYO MODULU (ceteris paribus tasarimi)
============================================================
Panel analizindeki NIHAI modelin katsayilarini kullanarak, SADECE SON
DONEMIN verilerini temel alan bir "bir sonraki donem" senaryosu uretir.

Tasarim (kullanicinin talebiyle guncellendi):
  1) Panel-Karar Degiskeni ayrimi -> Hedefli / Dogal
     SADECE GIRDI degiskenleri "Hedefli" (deliberate degistirilebilir) olabilir.
     Cikti degiskenleri HICBIR ZAMAN Hedefli olamaz -- cunku cikti (dogruluk,
     azaltilan prototip sayisi vb.) donem BASINDA karar verilebilecek bir
     degisken degildir, donem SONUNDA gozlemlenen bir SONUCTUR. Bir DMU
     "dogruluk_gunum %10 artsin" diye karar veremez; ama "simulasyon suresine
     %10 daha fazla kaynak ayirayim" diye karar VEREBILIR. Bu yuzden:
       - Girdi degiskeni + panel modelinde ANLAMLI (p < HEDEFLI_ALPHA) -> Hedefli
       - Cikti degiskeni (anlamli olsa bile) -> HER ZAMAN Dogal
       - Anlamsiz girdi degiskeni -> Dogal
     DEA teorisiyle (girdi->negatif katsayi beklentisi) TUTARLILIK ARANMAZ:
     panel katsayisinin ISARETI ne olursa olsun (teorik beklentiyle celisse
     bile) yon dogrudan bu isaretten alinir. Teoriyle celisen durumlar yine de
     'celiski' olarak ISARETLENIR (seffaflik icin), ama Hedefli olmaktan
     ALIKOYMAZ.

  2) Hedefli degiskenler -> katsayinin GERCEK isaretine gore, son donem
     degerinin SABIT BIR YUZDESI kadar degistirilir:
       yon                = katsayinin isareti (pozitif->artir, negatif->azalt)
       yeni_deger         = son_deger x (1 + yon x yuzde)
       tahmini MI etkisi  = katsayi x (yeni_deger - son_deger)   [rapor amacli]
     Yuzde SABIT tutulur -- katsayinin buyuklugu ADIMI degil, SADECE yonu ve
     raporlanan MI etkisini belirler.

  3) Digerleri (Dogal -- tum cikti degiskenleri + anlamsiz girdiler) ->
     CETERIS PARIBUS: son donem degeriyle AYNEN birakilir, hicbir sekilde
     degistirilmez. Bu degiskenlerden VIF>=5 olanlar icin -- yani hedefli
     degiskenlerle yuksek coklu dogrusal iliski icinde olanlar icin -- salt
     BILGI AMACLI bir not eklenir (gercekte hedefli degiskenle birlikte
     hareket edebilecegi hatirlatilir), ama SAYISAL olarak degistirilmezler.

  4) Sinir kontrolleri (yalnizca Hedefli degiskenler icin islevsel):
     Projeksiyon, DMU'nun kendi tarihsel min/max araligina gore makul bir
     bant disina cikamaz; negatif deger DEA icin gecersiz oldugundan asla
     uretilmez.

  5) Duyarlilik taramasi -> Nihai senaryo
     Hedefli degiskenler icin uygulanan yuzde degisim uc kademede taranir:
     %5 / %10 (Baz) / %15. Baz (%10) senaryo nihai/one cikan senaryo olarak
     sunulur, digerleri duyarlilik araligi olarak yaninda gosterilir.
"""
import numpy as np
import pandas as pd

VIF_ESIGI = 5.0
HEDEFLI_ALPHA = 0.10           # panel_module.py'deki KATSAYI_ALPHA ile tutarli tutulur
NOMINAL_ADIM = 1.0             # son_deger == 0 gibi dejenere durumda kullanilan sembolik birim
SINIR_ALT_ORAN = 0.5           # tarihsel min'in bu oranindan asagi inilmez
SINIR_UST_ORAN = 1.5           # tarihsel max'in bu oranindan yukari cikilmaz


def _tarihsel_seri(X: dict, Y: dict, donemler: list, dmu: str, degisken: str, girdi_cols, cikti_cols):
    """Bir DMU'nun bir degiskeni icin tum tarihsel donemlerdeki degerlerini dondurur
    (sinir kontrolleri ve bilgi amacli tarihsel-trend notu icin kullanilir)."""
    kaynak = X if degisken in girdi_cols else Y
    return [float(kaynak[d].loc[dmu, degisken]) for d in donemler]


def hedefli_dogal_siniflandir(nihai_res, girdi_cols, cikti_cols, alpha: float = HEDEFLI_ALPHA) -> pd.DataFrame:
    """
    Panel modelindeki her bagimsiz degiskeni 'Hedefli' (deliberate degistirilir)
    veya 'Dogal' (ceteris paribus sabit birakilir) olarak siniflandirir.

    ONEMLI: SADECE GIRDI degiskenleri Hedefli olabilir -- cikti degiskenleri
    (dogruluk, azaltilan prototip sayisi vb.) donem basinda karar verilebilecek
    degiskenler degil, donem sonunda GOZLEMLENEN sonuclardir; bu yuzden hicbir
    zaman deliberate degistirilmezler.

    Girdi bir degisken icin Hedefli sayilmasinin TEK sarti panel modelinde
    ANLAMLI olmasidir (p < alpha) -- katsayinin isareti DEA teorisiyle
    (girdi -> negatif beklentisi) TUTARLI olmak ZORUNDA DEGILDIR; yon
    dogrudan katsayinin GERCEK isaretinden alinir. Teoriyle celisen durumlar
    yine de 'celiski' sutununda ISARETLENIR (seffaflik/rapor amacli), ama bu
    durum degiskeni Hedefli olmaktan ALIKOYMAZ.
    """
    satirlar = []
    for degisken in nihai_res.params.index:
        if degisken == "const":
            continue
        katsayi = float(nihai_res.params[degisken])
        p = float(nihai_res.pvalues[degisken])
        anlamli = p < alpha

        if degisken in girdi_cols:
            tip, beklenen_yon = "Girdi", -1   # DEA teorisinin beklentisi (sadece bilgi/celiski amacli)
        elif degisken in cikti_cols:
            tip, beklenen_yon = "Cikti", +1
        else:
            tip, beklenen_yon = "Diger", 0

        gercek_yon = 1 if katsayi > 0 else -1
        celiski = anlamli and beklenen_yon != 0 and gercek_yon != beklenen_yon

        # SADECE girdi + anlamli -> Hedefli. Cikti hicbir zaman Hedefli olamaz.
        hedefli = anlamli and (tip == "Girdi")
        # Yon, DOGRUDAN katsayinin gercek isaretinden gelir (teori beklentisinden degil):
        # katsayi>0 -> degiskeni ARTIR (MI'yi artirir); katsayi<0 -> AZALT.
        iyilesme_yonu = gercek_yon if hedefli else 0

        satirlar.append({
            "degisken": degisken, "tip": tip, "katsayi": round(katsayi, 5), "p_degeri": round(p, 4),
            "anlamli_mi": anlamli, "celiski": celiski,
            "siniflandirma": "Hedefli" if hedefli else "Dogal",
            "iyilesme_yonu": iyilesme_yonu,
        })
    return pd.DataFrame(satirlar).set_index("degisken")


def senaryo_olustur(X: dict, Y: dict, donemler: list, nihai_res, girdi_cols, cikti_cols,
                     vif_df: pd.DataFrame, dmu_sirali: list, yuzde: float = 0.10):
    """
    Son donemin (donemler[-1]) verilerini temel alarak bir sonraki donem icin
    senaryo verisi uretir (ceteris paribus tasarimi).

    yuzde: Hedefli degiskenlerin son donem degerine gore ne kadar (orn. 0.10 = %10)
           degistirilecegini belirler (duyarlilik taramasi icin -- %5/%10/%15).
           Yon katsayidan gelir, BUYUKLUK bu sabit yuzdeden gelir. Dogal
           degiskenler bu parametreden ETKILENMEZ (her zaman sabit kalir).

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
            degerler = _tarihsel_seri(X, Y, donemler, dmu, degisken, girdi_cols, cikti_cols)
            son_deger = degerler[-1]
            onceki_deger = degerler[-2] if len(degerler) >= 2 else None
            vif_deger = vif_map.get(degisken, np.nan)
            yuksek_vif = (not np.isnan(vif_deger)) and (vif_deger >= VIF_ESIGI)

            is_hedefli = (degisken in siniflandirma.index) and \
                         (siniflandirma.loc[degisken, "siniflandirma"] == "Hedefli")

            if is_hedefli:
                # --- Hedefli: SADECE girdi + anlamli -> katsayinin GERCEK isareti yonunde,
                # SABIT YUZDE buyuklugunde deliberate degisim (DEA teori tutarliligi aranmaz) ---
                yon = siniflandirma.loc[degisken, "iyilesme_yonu"]
                taban = son_deger if son_deger > 0 else NOMINAL_ADIM
                yeni_deger = son_deger + yon * taban * yuzde
                katsayi = siniflandirma.loc[degisken, "katsayi"]
                celiski_mi = bool(siniflandirma.loc[degisken, "celiski"])
                tahmini_mi_etkisi = katsayi * (yeni_deger - son_deger)
                tip_uygulanan = "Hedefli"
                not_metni = (
                    f"Girdi degiskeni, panel modelinde anlamli (katsayi={katsayi}, "
                    f"p={siniflandirma.loc[degisken,'p_degeri']}); katsayinin GERCEK isareti "
                    f"yonunde ({'artir' if yon > 0 else 'azalt'}) son donem degerinin "
                    f"%{yuzde*100:g}'i kadar deliberate degistirildi. Tahmini MI etkisi "
                    f"≈ {tahmini_mi_etkisi:+.4f}."
                )
                if celiski_mi:
                    not_metni += (
                        " ⚠️ Not: bu yon, DEA teorisinin beklentisiyle (girdi icin negatif "
                        "katsayi) CELISIYOR -- yine de panel katsayisinin kendi isaretine "
                        "guvenilerek uygulandi; sonucu yorumlarken bu celiskiyi goz onunde "
                        "bulundurun (icsel devirsellik/omitted variable riski olabilir)."
                    )

                # --- Sinir kontrolleri (yalniz Hedefli icin islevsel) ---
                hist_min, hist_max = min(degerler), max(degerler)
                alt_sinir = hist_min * SINIR_ALT_ORAN
                ust_sinir = hist_max * SINIR_UST_ORAN
                yeni_deger_sinirsiz = yeni_deger
                yeni_deger = max(alt_sinir, min(ust_sinir, yeni_deger))
                yeni_deger = max(yeni_deger, 0.0)
                sinir_uygulandi = not np.isclose(yeni_deger, yeni_deger_sinirsiz, atol=1e-9)
                if sinir_uygulandi:
                    not_metni += " ⚠️ Tarihsel bant sinirlamasi nedeniyle deger kirpildi (asiri ekstrapolasyon onlendi)."
            else:
                # --- Dogal: CETERIS PARIBUS -- son donem degeriyle aynen birakilir ---
                # (butun cikti degiskenleri + anlamsiz girdiler buraya duser)
                yeni_deger = son_deger
                tahmini_mi_etkisi = 0.0
                sinir_uygulandi = False
                tip_uygulanan = "Dogal (degismedi - ceteris paribus)"
                if degisken in cikti_cols:
                    temel_not = (
                        "Cikti degiskeni -- donem basinda karar verilebilecek bir degisken "
                        "degil, donem sonunda gozlemlenen bir sonuc oldugu icin hicbir zaman "
                        "deliberate degistirilmez."
                    )
                else:
                    temel_not = (
                        "Girdi degiskeni, panelde anlamsiz -- deliberate olarak degistirilmedi."
                    )
                if yuksek_vif:
                    not_metni = (
                        f"{temel_not} Ayrica VIF={vif_deger:.2f} (>= {VIF_ESIGI:g}) -- hedefli "
                        f"degiskenler dahil diger regresorlerle yuksek coklu dogrusal iliski "
                        f"icinde; GERCEKTE hedefli degiskenle birlikte hareket edebilecegini "
                        f"unutmayin (bilgi amacli uyari, sayisal olarak dikkate alinmadi)."
                    )
                else:
                    not_metni = f"{temel_not} Son donem degeriyle ayni birakildi (ceteris paribus)."

            if degisken in girdi_cols:
                X_next.loc[dmu, degisken] = yeni_deger
            else:
                Y_next.loc[dmu, degisken] = yeni_deger

            detay_satirlar.append({
                "DMU": dmu, "degisken": degisken, "tip_uygulanan": tip_uygulanan,
                "onceki_deger": round(onceki_deger, 2) if onceki_deger is not None else None,
                "son_deger": round(son_deger, 2), "yeni_deger": round(yeni_deger, 2),
                "degisim_yuzde": round((yeni_deger - son_deger) / son_deger * 100, 2) if son_deger else np.nan,
                "tahmini_mi_etkisi": round(tahmini_mi_etkisi, 5),
                "vif": round(vif_deger, 2) if not np.isnan(vif_deger) else None,
                "sinir_uygulandi": sinir_uygulandi, "not": not_metni,
            })

    detay = pd.DataFrame(detay_satirlar).set_index(["DMU", "degisken"])
    return X_next, Y_next, detay, siniflandirma


def girdi_yon_bilgisi(nihai_res, girdi_cols, alpha: float = HEDEFLI_ALPHA) -> pd.DataFrame:
    """
    Her GIRDI degiskeni icin panel modelinden gelen BILGIYI (katsayi, p-degeri,
    anlamli mi, panelin ONERDIGI yon) dondurur. Bu, bir siniflandirma/karar
    DEGIL -- sadece kullanicinin kendi yon+yuzde secimini yaparken referans
    alabilecegi bir BILGI PANELI. Karar tamamen kullaniciya birakilir.

    onerilen_yon: katsayi>0 ise +1 (artirmak MI'yi artirir), katsayi<0 ise -1
    (azaltmak MI'yi artirir) -- DEA teorisiyle tutarlilik ARANMAZ, sadece
    katsayinin GERCEK isareti baz alinir.
    """
    satirlar = []
    for degisken in girdi_cols:
        if degisken not in nihai_res.params.index:
            continue
        katsayi = float(nihai_res.params[degisken])
        p = float(nihai_res.pvalues[degisken])
        anlamli = p < alpha
        onerilen_yon = 1 if katsayi > 0 else -1
        satirlar.append({
            "degisken": degisken, "katsayi": round(katsayi, 5), "p_degeri": round(p, 4),
            "anlamli_mi": anlamli, "onerilen_yon": onerilen_yon,
        })
    return pd.DataFrame(satirlar).set_index("degisken")


def manuel_senaryo_olustur(X: dict, Y: dict, donemler: list, girdi_cols: list, cikti_cols: list,
                            dmu_sirali: list, girdi_secimleri: dict):
    """
    Kullanicinin HER GIRDI icin kendi sectigi yon (+1 artir / -1 azalt) ve
    kendi yazdigi yuzdeyi kullanarak bir sonraki donem senaryosu uretir.

    girdi_secimleri: dict -- {girdi_adi: {"yon": 1 veya -1, "yuzde": float (orn. 0.12 = %12)}}
                     TUM girdi_cols icin bir secim icermelidir; eksik girdi
                     varsa o girdi ceteris paribus (degismez) birakilir.

    Cikti degiskenleri HER ZAMAN ceteris paribus (degismez) -- donem basinda
    karar verilebilecek degiskenler olmadigi icin (bkz. modul dokumantasyonu).

    Returns: X_next (DataFrame), Y_next (DataFrame), detay (DataFrame, index=[DMU,degisken])
    """
    son_donem = donemler[-1]
    tum_degiskenler = list(girdi_cols) + list(cikti_cols)

    X_next = X[son_donem].astype(float).copy()
    Y_next = Y[son_donem].astype(float).copy()

    detay_satirlar = []

    for dmu in dmu_sirali:
        for degisken in tum_degiskenler:
            degerler = _tarihsel_seri(X, Y, donemler, dmu, degisken, girdi_cols, cikti_cols)
            son_deger = degerler[-1]
            onceki_deger = degerler[-2] if len(degerler) >= 2 else None

            secim = girdi_secimleri.get(degisken) if degisken in girdi_cols else None

            if secim is not None:
                yon = 1 if secim["yon"] > 0 else -1
                yuzde = float(secim["yuzde"])
                taban = son_deger if son_deger > 0 else NOMINAL_ADIM
                yeni_deger = son_deger + yon * taban * yuzde
                tip_uygulanan = "Kullanici secimi"
                not_metni = (
                    f"Kullanici secimi: {'artir' if yon > 0 else 'azalt'} yonunde, son donem "
                    f"degerinin %{yuzde*100:g}'i kadar degistirildi."
                )

                # Sinir kontrolleri (asiri ekstrapolasyonu onlemek icin, kullanici secimlerinde de gecerli)
                hist_min, hist_max = min(degerler), max(degerler)
                alt_sinir = hist_min * SINIR_ALT_ORAN
                ust_sinir = hist_max * SINIR_UST_ORAN
                yeni_deger_sinirsiz = yeni_deger
                yeni_deger = max(alt_sinir, min(ust_sinir, yeni_deger))
                yeni_deger = max(yeni_deger, 0.0)
                sinir_uygulandi = not np.isclose(yeni_deger, yeni_deger_sinirsiz, atol=1e-9)
                if sinir_uygulandi:
                    not_metni += (
                        " ⚠️ Tarihsel bant sinirlamasi nedeniyle deger kirpildi "
                        "(asiri ekstrapolasyon onlendi)."
                    )
            else:
                yeni_deger = son_deger
                sinir_uygulandi = False
                if degisken in cikti_cols:
                    tip_uygulanan = "Dogal (cikti - degismez)"
                    not_metni = (
                        "Cikti degiskeni -- donem basinda karar verilebilecek bir degisken "
                        "degil, donem sonunda gozlemlenen bir sonuc oldugu icin hicbir zaman "
                        "degistirilmez (ceteris paribus)."
                    )
                else:
                    tip_uygulanan = "Dogal (secim yapilmadi)"
                    not_metni = "Bu girdi icin kullanici bir secim yapmadi -- son donem degeriyle ayni birakildi."

            if degisken in girdi_cols:
                X_next.loc[dmu, degisken] = yeni_deger
            else:
                Y_next.loc[dmu, degisken] = yeni_deger

            detay_satirlar.append({
                "DMU": dmu, "degisken": degisken, "tip_uygulanan": tip_uygulanan,
                "onceki_deger": round(onceki_deger, 2) if onceki_deger is not None else None,
                "son_deger": round(son_deger, 2), "yeni_deger": round(yeni_deger, 2),
                "degisim_yuzde": round((yeni_deger - son_deger) / son_deger * 100, 2) if son_deger else np.nan,
                "sinir_uygulandi": sinir_uygulandi, "not": not_metni,
            })

    detay = pd.DataFrame(detay_satirlar).set_index(["DMU", "degisken"])
    return X_next, Y_next, detay


def gelecek_donem_analizi_manuel(sonuc: dict, girdi_secimleri: dict) -> dict:
    """
    Ana orkestrasyon (MANUEL MOD): kullanicinin her girdi icin kendi sectigi
    yon+yuzde ile TEK BIR senaryo uretir, DEA'yi projeksiyon verisiyle
    tekrar cozer ve son gercek donem ile projeksiyon donemi arasinda
    Malmquist (EC/TC/M) hesaplar.

    girdi_secimleri: dict -- {girdi_adi: {"yon": 1/-1, "yuzde": float}}

    Returns: dict -- son_donem, girdi_bilgisi (DataFrame, panelin onerileri --
             bilgi amacli), X, Y, detay, dea, malmquist
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
    girdi_bilgisi = girdi_yon_bilgisi(nihai_res, girdi_cols)

    X_next, Y_next, detay = manuel_senaryo_olustur(
        sonuc["X"], sonuc["Y"], donemler, girdi_cols, cikti_cols, dmu_sirali, girdi_secimleri,
    )

    son_donem = donemler[-1]
    dea_sonuc = solve_dea_period(X_next, Y_next)
    X_ext = {son_donem: sonuc["X"][son_donem], "t_gelecek": X_next}
    Y_ext = {son_donem: sonuc["Y"][son_donem], "t_gelecek": Y_next}
    malmquist_sonuc = solve_malmquist(X_ext, Y_ext, [son_donem, "t_gelecek"])

    return {
        "son_donem": son_donem,
        "nihai_res_baslik": oneri["sonuc_basligi"],
        "girdi_bilgisi": girdi_bilgisi,
        "X": X_next, "Y": Y_next, "detay": detay,
        "dea": dea_sonuc, "malmquist": malmquist_sonuc,
    }
    """%5 / %10 (Baz) / %15 uc senaryoyu birlikte uretir (Hedefli degiskenler icin
    duyarlilik taramasi; Dogal degiskenler her senaryoda ayni -- ceteris paribus)."""
    senaryolar = {}
    siniflandirma = None
    for isim, yuzde in [("%5", 0.05), ("%10 (Baz)", 0.10), ("%15", 0.15)]:
        X_n, Y_n, detay, siniflandirma = senaryo_olustur(
            X, Y, donemler, nihai_res, girdi_cols, cikti_cols, vif_df, dmu_sirali, yuzde=yuzde,
        )
        senaryolar[isim] = {"X": X_n, "Y": Y_n, "detay": detay, "yuzde": yuzde}
    return senaryolar, siniflandirma


def gelecek_donem_analizi(sonuc: dict) -> dict:
    """
    Ana orkestrasyon: pipeline.run_pipeline() ciktisini (sonuc) alir, panel
    modelinin sectigi nihai sonucu kullanarak 3 senaryo (%5/%10/%15) uretir,
    HER senaryo icin projeksiyon verisiyle DEA'yi tekrar cozer ve son gercek
    donem ile projeksiyon donemi arasinda Malmquist (EC/TC/M) hesaplar.

    Returns: dict -- siniflandirma, son_donem, nihai_res_baslik, senaryolar:
             {isim: {X, Y, detay, yuzde, dea, malmquist}}
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
