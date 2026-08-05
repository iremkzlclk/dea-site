# -*- coding: utf-8 -*-
"""
GELECEK DONEM SENARYO MODULU (kullanici-kontrollu, korelasyon-farkindalikli)
================================================================================
Panel analizindeki NIHAI modelin katsayilarini BILGI olarak sunan, ama
karari tamamen KULLANICIYA birakan bir "bir sonraki donem" senaryosu uretir.

Tasarim:
  1) Karar degiskeni ayrimi: SADECE GIRDI degiskenleri deliberate
     degistirilebilir -- cunku donem basinda karar verilebilecek sey budur
     (orn. simulasyon suresine ne kadar kaynak ayrilacagi). Cikti degiskenleri
     (dogruluk, azaltilan prototip sayisi vb.) donem SONUNDA gozlemlenen
     sonuclardir, dogrudan "karar verilerek" degistirilemez.

  2) Panelin onerisi (girdi_yon_bilgisi): her girdi icin panel katsayisi,
     p-degeri ve katsayinin GERCEK isaretinden turetilen "onerilen yon"
     BILGI OLARAK sunulur. Bu bir siniflandirma/zorunluluk DEGILDIR --
     kullanici bu oneriyi izleyebilir ya da tamamen farkli bir yon+yuzde
     secebilir (manuel_senaryo_olustur, kullanicinin GERCEK secimini kullanir).

  3) Girdi degisimi (kullanici secimiyle):
       yon                = kullanicinin sectigi yon (+1 artir / -1 azalt)
       yeni_deger         = son_deger x (1 + yon x yuzde)
     yuzde de kullanicinin kendi yazdigi sayidir (sabit %5/10/15 degil).

  4) Cikti degisimi (KORELASYON-FARKINDALIKLI -- artik SAF ceteris paribus
     DEGIL): Bir girdi degistiginde, o girdiyle TARIHSEL OLARAK KORELE olan
     cikti da, aralarindaki basit dogrusal iliskiye (beta = Cov(girdi,cikti)
     / Var(girdi), panel genelindeki tum DMU-donem gozlemlerinden hesaplanir)
     ORANTILI olarak kismen hareket eder:
       cikti_degisimi = sum_over_degisen_girdiler( beta[girdi,cikti] x girdi_degisimi )
       yeni_cikti     = son_cikti + cikti_degisimi
     Boylece korelasyonu ZAYIF olan cikti pratikte sabit kalir (beta~0),
     korelasyonu GUCLU olan cikti ise girdiyle BIRLIKTE hareket eder --
     "girdi degisir, cikti sabit kalir" gibi gercekci olmayan bir varsayim
     yerine, verideki GERCEK birlikte-hareket egilimini yansitir.
     ONEMLI SINIRLAMA: bu, basit ikili (bivariate) bir yaklasimdir -- tam
     coklu-degiskenli (multivariate) bir nedensel model DEGILDIR; birden
     fazla girdi ayni anda degisirse etkiler ustuste toplanir, aralarindaki
     olasi ortak varyans ayristirilmaz. Yine de saf ceteris paribus'tan
     daha gercekci bir yaklasimdir.

  5) Sinir kontrolleri: hem girdi hem cikti icin, projeksiyon DMU'nun kendi
     tarihsel min/max araligina gore makul bir bant disina cikamaz; negatif
     ya da sifira cok yakin deger (DEA/Malmquist icin gecersiz/dejenere
     olabilir) uretilmez.
"""
import numpy as np
import pandas as pd

HEDEFLI_ALPHA = 0.10           # panel_module.py'deki KATSAYI_ALPHA ile tutarli tutulur
NOMINAL_ADIM = 1.0             # son_deger == 0 gibi dejenere durumda kullanilan sembolik birim
SINIR_ALT_ORAN = 0.5           # tarihsel min'in bu oranindan asagi inilmez
SINIR_UST_ORAN = 1.5           # tarihsel max'in bu oranindan yukari cikilmaz
MIN_POZITIF_DEGER = 0.05       # cikti/girdi icin mutlak taban (tam sifira cok yakin deger DEA/Malmquist'i bozar)


def _tarihsel_seri(X: dict, Y: dict, donemler: list, dmu: str, degisken: str, girdi_cols, cikti_cols):
    """Bir DMU'nun bir degiskeni icin tum tarihsel donemlerdeki degerlerini dondurur
    (sinir kontrolleri icin kullanilir)."""
    kaynak = X if degisken in girdi_cols else Y
    return [float(kaynak[d].loc[dmu, degisken]) for d in donemler]


def girdi_yon_bilgisi(nihai_res, girdi_cols, alpha: float = HEDEFLI_ALPHA) -> pd.DataFrame:
    """
    Her GIRDI degiskeni icin panel modelinden gelen BILGIYI (katsayi, p-degeri,
    anlamli mi, panelin ONERDIGI yon) dondurur. Bu bir siniflandirma/karar
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


def _girdi_cikti_betalari(X: dict, Y: dict, donemler: list, dmu_sirali: list,
                           girdi_cols: list, cikti_cols: list) -> dict:
    """
    Her (girdi, cikti) cifti icin, TUM DMU-donem gozlemleri havuzlanarak
    (pooled) basit ikili regresyon egimini (beta = Cov(girdi,cikti)/Var(girdi))
    ve Pearson korelasyon katsayisini hesaplar.

    Returns: dict -- {(girdi, cikti): {"beta": float, "r": float}}
    """
    kayitlar = []
    for donem in donemler:
        for dmu in dmu_sirali:
            satir = {}
            for g in girdi_cols:
                satir[g] = float(X[donem].loc[dmu, g])
            for c in cikti_cols:
                satir[c] = float(Y[donem].loc[dmu, c])
            kayitlar.append(satir)
    havuz = pd.DataFrame(kayitlar)

    betalar = {}
    for g in girdi_cols:
        var_g = havuz[g].var(ddof=1)
        for c in cikti_cols:
            if var_g and var_g > 1e-12:
                cov_gc = havuz[[g, c]].cov().iloc[0, 1]
                beta = cov_gc / var_g
                r = havuz[g].corr(havuz[c])
            else:
                beta, r = 0.0, 0.0
            betalar[(g, c)] = {"beta": float(beta) if pd.notna(beta) else 0.0,
                                "r": float(r) if pd.notna(r) else 0.0}
    return betalar


def _sinirla(yeni_deger: float, degerler: list) -> tuple:
    """Tarihsel bant + pozitiflik sinirini uygular. Returns: (sinirli_deger, sinir_uygulandi_mi)."""
    hist_min, hist_max = min(degerler), max(degerler)
    alt_sinir = max(hist_min * SINIR_ALT_ORAN, MIN_POZITIF_DEGER)
    ust_sinir = hist_max * SINIR_UST_ORAN
    sinirli = max(alt_sinir, min(ust_sinir, yeni_deger))
    sinirli = max(sinirli, MIN_POZITIF_DEGER)
    uygulandi = not np.isclose(sinirli, yeni_deger, atol=1e-9)
    return sinirli, uygulandi


def manuel_senaryo_olustur(X: dict, Y: dict, donemler: list, girdi_cols: list, cikti_cols: list,
                            dmu_sirali: list, girdi_secimleri: dict):
    """
    Kullanicinin HER GIRDI icin kendi sectigi yon (+1 artir / -1 azalt) ve
    kendi yazdigi yuzdeyi kullanarak bir sonraki donem senaryosu uretir.
    Cikti degiskenleri, degisen girdilerle olan TARIHSEL KORELASYONA gore
    KISMEN hareket eder (bkz. modul dokumantasyonu, madde 4) -- saf ceteris
    paribus degildir.

    girdi_secimleri: dict -- {girdi_adi: {"yon": 1 veya -1, "yuzde": float (orn. 0.12 = %12)}}
                     Sadece secim YAPILAN girdiler degistirilir; digerleri
                     (ve secim yapilmayan girdiler) sabit kalir.

    Returns: X_next (DataFrame), Y_next (DataFrame), detay (DataFrame, index=[DMU,degisken])
    """
    son_donem = donemler[-1]
    betalar = _girdi_cikti_betalari(X, Y, donemler, dmu_sirali, girdi_cols, cikti_cols)

    X_next = X[son_donem].astype(float).copy()
    Y_next = Y[son_donem].astype(float).copy()

    detay_satirlar = []

    for dmu in dmu_sirali:
        # --- 1. gecis: GIRDILERI kullanicinin secimine gore guncelle, delta'lari sakla ---
        girdi_deltalari = {}
        for g in girdi_cols:
            degerler = _tarihsel_seri(X, Y, donemler, dmu, g, girdi_cols, cikti_cols)
            son_deger = degerler[-1]
            secim = girdi_secimleri.get(g)

            if secim is not None:
                yon = 1 if secim["yon"] > 0 else -1
                yuzde = float(secim["yuzde"])
                taban = son_deger if son_deger > 0 else NOMINAL_ADIM
                yeni_deger_ham = son_deger + yon * taban * yuzde
                yeni_deger, sinir_uygulandi = _sinirla(yeni_deger_ham, degerler)
                girdi_deltalari[g] = yeni_deger - son_deger
                tip_uygulanan = "Kullanıcı seçimi"
                not_metni = (
                    f"Kullanıcı seçimi: {'artır' if yon > 0 else 'azalt'} yönünde, son dönem "
                    f"değerinin %{yuzde*100:g}'i kadar değiştirildi."
                )
                if sinir_uygulandi:
                    not_metni += " ⚠️ Tarihsel bant sınırlaması nedeniyle değer kırpıldı."
            else:
                yeni_deger = son_deger
                girdi_deltalari[g] = 0.0
                sinir_uygulandi = False
                tip_uygulanan = "Değişmedi (seçim yapılmadı)"
                not_metni = "Bu girdi için kullanıcı bir seçim yapmadı -- son dönem değeriyle aynı bırakıldı."

            X_next.loc[dmu, g] = yeni_deger
            detay_satirlar.append({
                "DMU": dmu, "degisken": g, "tip_uygulanan": tip_uygulanan,
                "son_deger": round(son_deger, 2), "yeni_deger": round(yeni_deger, 2),
                "degisim_yuzde": round((yeni_deger - son_deger) / son_deger * 100, 2) if son_deger else np.nan,
                "sinir_uygulandi": sinir_uygulandi, "not": not_metni,
            })

        # --- 2. gecis: CIKTILARI, degisen girdilerle olan korelasyona gore kismen hareket ettir ---
        for c in cikti_cols:
            degerler = _tarihsel_seri(X, Y, donemler, dmu, c, girdi_cols, cikti_cols)
            son_deger = degerler[-1]

            toplam_delta = 0.0
            katki_notlari = []
            for g, delta_g in girdi_deltalari.items():
                if abs(delta_g) < 1e-9:
                    continue
                b = betalar[(g, c)]["beta"]
                r = betalar[(g, c)]["r"]
                katki = b * delta_g
                if abs(katki) > 1e-9:
                    toplam_delta += katki
                    katki_notlari.append(f"{g} değişimi (r={r:.2f}, β={b:.4g}) → {katki:+.3f}")

            yeni_deger_ham = son_deger + toplam_delta
            yeni_deger, sinir_uygulandi = _sinirla(yeni_deger_ham, degerler)

            if katki_notlari:
                tip_uygulanan = "Korelasyon-bazlı kısmi hareket"
                not_metni = (
                    "Çıktı doğrudan seçilmedi (dönem sonunda gözlemlenen bir sonuç olduğu için), "
                    "ama değişen girdi(ler)le tarihsel korelasyonuna göre kısmen hareket ettirildi: "
                    + "; ".join(katki_notlari) + "."
                )
            else:
                tip_uygulanan = "Değişmedi (korelasyon ~0 veya ilgili girdi değişmedi)"
                not_metni = (
                    "Çıktı, değişen girdilerle tarihsel olarak anlamlı bir korelasyon göstermediği "
                    "(veya hiçbir ilgili girdi değişmediği) için pratikte sabit kaldı."
                )
            if sinir_uygulandi:
                not_metni += " ⚠️ Tarihsel bant sınırlaması nedeniyle değer kırpıldı."

            Y_next.loc[dmu, c] = yeni_deger
            detay_satirlar.append({
                "DMU": dmu, "degisken": c, "tip_uygulanan": tip_uygulanan,
                "son_deger": round(son_deger, 2), "yeni_deger": round(yeni_deger, 2),
                "degisim_yuzde": round((yeni_deger - son_deger) / son_deger * 100, 2) if son_deger else np.nan,
                "sinir_uygulandi": sinir_uygulandi, "not": not_metni,
            })

    detay = pd.DataFrame(detay_satirlar).set_index(["DMU", "degisken"])
    return X_next, Y_next, detay


def gelecek_donem_analizi_manuel(sonuc: dict, girdi_secimleri: dict) -> dict:
    """
    Ana orkestrasyon: kullanicinin her girdi icin kendi sectigi yon+yuzde ile
    TEK BIR senaryo uretir (cikti'lar korelasyona gore kismen hareket eder),
    DEA'yi projeksiyon verisiyle tekrar cozer ve son gercek donem ile
    projeksiyon donemi arasinda Malmquist (EC/TC/M) hesaplar.

    girdi_secimleri: dict -- {girdi_adi: {"yon": 1/-1, "yuzde": float}}

    Returns: dict -- son_donem, nihai_res_baslik, girdi_bilgisi (DataFrame,
             panelin onerileri -- bilgi amacli), X, Y, detay, dea, malmquist
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
