# -*- coding: utf-8 -*-
"""
YORUMLAMA MODULU
===================
DEA/Malmquist/Panel sayisal ciktilarini, konuyu bilmeyen bir muhendisin
anlayabilecegi, AKSIYON ODAKLI Turkce metinlere cevirir.
"""
import pandas as pd
import numpy as np


# ------------------------------------------------------------------
# 1) DEA YORUMLAMA
# ------------------------------------------------------------------
def dea_aksiyon_tablosu(dea_d: dict, X_donem: pd.DataFrame, Y_donem: pd.DataFrame, vrs=False):
    """
    Her DMU icin: mevcut theta, girdi bazinda 'azaltilabilir fazla miktar',
    cikti bazinda 'artirilabilir miktar' (slack) -- somut, birim cinsinden.
    Input-oriented CCR/BCC mantigi: hedef girdi = theta*x - slack_x
                                     hedef cikti = y + slack_y
    """
    theta = dea_d["theta_bcc"] if vrs else dea_d["theta_ccr"]
    slack_x = dea_d["slack_x_bcc"] if vrs else dea_d["slack_x_ccr"]
    slack_y = dea_d["slack_y_bcc"] if vrs else dea_d["slack_y_ccr"]

    satirlar = []
    for dmu in theta.index:
        th = theta[dmu]
        girdi_fazlasi = {}
        for r in X_donem.columns:
            fazla = X_donem.loc[dmu, r] * (1 - th) + slack_x.loc[dmu, r]
            girdi_fazlasi[r] = round(fazla, 2)
        cikti_acigi = {}
        for s in Y_donem.columns:
            acik = slack_y.loc[dmu, s]
            cikti_acigi[s] = round(acik, 2)

        satir = {"DMU": dmu, "theta": round(th, 4), "etkin_mi": th >= 0.999}
        for r, v in girdi_fazlasi.items():
            satir[f"azaltilabilir__{r}"] = v
        for s, v in cikti_acigi.items():
            satir[f"artirilabilir__{s}"] = v
        satirlar.append(satir)

    return pd.DataFrame(satirlar).set_index("DMU")


def dea_aksiyon_metni(satir: pd.Series, girdi_cols, cikti_cols) -> str:
    """Tek bir DMU'nun aksiyon tablosu satirini duz Turkce metne cevirir."""
    if satir["etkin_mi"]:
        return "Bu birim bu donemde **etkin** (θ=1.00) — mevcut kaynak kullanimi diger birimler icin referans/ornek teskil ediyor. Aksiyon gerekmiyor."

    metinler = [f"Etkin degil (θ={satir['theta']:.3f} → yaklasik %{(1-satir['theta'])*100:.1f} kaynak fazlasi var)."]
    for g in girdi_cols:
        fazla = satir.get(f"azaltilabilir__{g}", 0)
        if fazla and fazla > 0.01:
            isim = g.replace("Girdi_", "")
            metinler.append(f"- **{isim}** yaklasik **{fazla:g} birim azaltilabilir**, ayni cikti seviyesi korunarak.")
    for c in cikti_cols:
        acik = satir.get(f"artirilabilir__{c}", 0)
        if acik and acik > 0.01:
            isim = c.replace("Cikti_", "")
            metinler.append(f"- **{isim}** ayni girdilerle yaklasik **{acik:g} birim daha artirilabilir**.")
    if len(metinler) == 1:
        metinler.append("(Radyal olcekte etkin degil ama ek slack tespit edilmedi -- kucuk bir olcek ayari yeterli olabilir.)")
    return "\n".join(metinler)


# ------------------------------------------------------------------
# 2) MALMQUIST YORUMLAMA
# ------------------------------------------------------------------
def malmquist_yorum_metni(row: pd.Series) -> str:
    """EC/TC/M ucluisunu tek bir DMU-donem satiri icin yorumlar."""
    ec, tc, m = row["EC"], row["TC"], row["M"]

    if m > 1.02:
        genel = f"Verimlilik yaklasik **%{(m-1)*100:.1f} artmis**."
    elif m < 0.98:
        genel = f"Verimlilik yaklasik **%{(1-m)*100:.1f} azalmis**."
    else:
        genel = "Verimlilik pratik olarak degismemis."

    if ec > 1.02:
        ec_metin = f"Bu artisin bir kismi **etkinlik iyilesmesinden** kaynaklaniyor (EC={ec:.3f}) -- yani birim, kendi sinirina (frontier'a) daha yaklasmis, kaynak kullanimini daha iyi organize etmis."
    elif ec < 0.98:
        ec_metin = f"**Etkinlik geriledigi** icin verimlilik olumsuz etkilenmis (EC={ec:.3f}) -- birim kendi potansiyel sinirindan uzaklasmis; operasyonel/surec kaynakli bir gerileme olabilir."
    else:
        ec_metin = f"Etkinlik seviyesi pratik olarak degismemis (EC={ec:.3f})."

    if tc > 1.02:
        tc_metin = f"Ayrica **genel teknoloji/yontem sinirinda ilerleme** var (TC={tc:.3f}) -- sektordeki/sistemdeki en iyi uygulama seviyesi yukselmis (ornegin yeni bir simulasyon araci, yontem iyilestirmesi vb.)."
    elif tc < 0.98:
        tc_metin = f"Teknoloji/yontem sinirinda bir **gerileme** gorulmus (TC={tc:.3f}) -- bu genelde birime ozgu degil, tum sistemi etkileyen bir durumdur (ornegin karsilastirma seti degisti, dis kosullar kotulesti)."
    else:
        tc_metin = f"Teknoloji/yontem siniri pratik olarak sabit kalmis (TC={tc:.3f})."

    return f"{genel}\n- {ec_metin}\n- {tc_metin}"


def malmquist_donem_ortalamasi(malmquist_df: pd.DataFrame, donem_sirasi=None) -> pd.DataFrame:
    """
    Her gecis donemi (t -> t+1) icin, TUM DMU'lar uzerinden EC/TC/M ortalamasini
    (Malmquist endeksi literaturunun standart "donem ozeti" gosterimi) uretir.

    Geometrik ortalama kullanilir: EC/TC/M birer ORAN (indeks) oldugu icin
    (Fare vd. 1994'ten beri Malmquist literaturunde standart pratik) aritmetik
    ortalama yaniltici olabilir -- ornegin M=[0.5, 2.0] icin aritmetik ortalama
    1.25 "net artis" izlenimi verirken, gercekte iki donem birbirini tam
    dengeliyor (gecometrik ortalama = 1.0).

    malmquist_df: index=[DMU, donem], columns=[EC, TC, M]
    donem_sirasi: donemlerin KRONOLOJIK sirali listesi (orn. veri["donem_sirali"][:-1])
                  -- verilirse tablo bu sirada gosterilir, verilmezse alfabetik siralanir.

    Returns: DataFrame(index=donem, columns=[EC_ort, TC_ort, M_ort])
    """
    from scipy.stats import gmean

    ozet = malmquist_df.groupby(level="donem").agg(
        EC_ort=("EC", gmean),
        TC_ort=("TC", gmean),
        M_ort=("M", gmean),
    )
    if donem_sirasi is not None:
        sira = [d for d in donem_sirasi if d in ozet.index]
        ozet = ozet.loc[sira]
    return ozet.round(4)


# ------------------------------------------------------------------
# 3) PANEL YORUMLAMA (en onemli katman)
# ------------------------------------------------------------------
def panel_aksiyon_analizi(nihai_res, girdi_cols, cikti_cols, panel_df: pd.DataFrame, alpha=0.05):
    """
    Secilen nihai modelin (Pooled OLS / FE / RE, robust ya da clustered) anlamli
    katsayilarini alip, DEA girdi/cikti mantigiyla CAPRAZ KONTROL ederek
    aksiyon onerileri uretir.

    DEA mantik beklentisi:
      - Girdi_ degiskeni: MI uzerinde teorik olarak NEGATIF iliski beklenir
        (girdi arttikca, ayni cikti icin verimlilik dususu beklenir).
      - Cikti_ degiskeni: MI uzerinde teorik olarak POZITIF iliski beklenir.
    Beklentiyle CELISEN anlamli katsayilar ayrica isaretlenir (icsel devirsellik/
    coklu dogrusal baglanti kaynakli olabilir).
    """
    params = nihai_res.params
    pvalues = nihai_res.pvalues
    sonuclar = []

    for degisken in params.index:
        if degisken == "const":
            continue
        p = pvalues[degisken]
        katsayi = params[degisken]
        anlamli = p < alpha

        if degisken in girdi_cols:
            tip, beklenen_yon = "Girdi", "negatif"
        elif degisken in cikti_cols:
            tip, beklenen_yon = "Cikti", "pozitif"
        else:
            tip, beklenen_yon = "Diger", None

        gercek_yon = "pozitif" if katsayi > 0 else "negatif"
        celiski = anlamli and beklenen_yon is not None and gercek_yon != beklenen_yon

        # ortalama deger uzerinden %10'luk degisimin MI'ye etkisini hesapla (somut aksiyon)
        ort_deger = panel_df[degisken].mean()
        degisim_miktari = ort_deger * 0.10
        mi_etkisi = katsayi * degisim_miktari

        sonuclar.append({
            "degisken": degisken, "tip": tip, "katsayi": round(katsayi, 5),
            "p_degeri": round(p, 4), "anlamli_mi": anlamli,
            "beklenen_yon": beklenen_yon, "gercek_yon": gercek_yon, "celiski": celiski,
            "ort_deger": round(ort_deger, 2), "yuzde10_degisim": round(degisim_miktari, 2),
            "mi_etkisi_yuzde10": round(mi_etkisi, 4),
        })

    return pd.DataFrame(sonuclar)


def panel_aksiyon_metni(analiz_df: pd.DataFrame) -> str:
    """Panel aksiyon tablosunu duz, karar-destek metnine cevirir."""
    anlamli = analiz_df[analiz_df["anlamli_mi"]]
    if anlamli.empty:
        return ("Secilen modelde istatistiksel olarak anlamli (p<0.05) hicbir degisken bulunamadi. "
                "Bu durumda mevcut veriyle net, sayisal bir aksiyon onerisi sunmak yanlis yonlendirici olur -- "
                "ek veri/donem toplanmasi ya da farkli degisken secimi degerlendirilmelidir.")

    satirlar = []
    for _, r in anlamli.iterrows():
        isim = r["degisken"].replace("Girdi_", "").replace("Cikti_", "")
        yon_kelime = "artis" if r["mi_etkisi_yuzde10"] > 0 else "azalis"
        satirlar.append(
            f"- **{isim}** ({r['tip']}): mevcut ortalamasinin (%{'{:.1f}'.format(10)} = {r['yuzde10_degisim']:g} birim) "
            f"kadar degismesi, MI'de yaklasik **{abs(r['mi_etkisi_yuzde10']):.4f} birimlik {yon_kelime}** ile iliskilidir "
            f"(katsayi={r['katsayi']}, p={r['p_degeri']})."
        )
        if r["celiski"]:
            satirlar.append(
                f"  ⚠️ **Dikkat:** Bu iliskinin yonu ({r['gercek_yon']}), DEA'nin teorik beklentisiyle "
                f"({r['beklenen_yon']}) celisiyor. Bu, DEA girdi/ciktilarinin ayni zamanda ikinci asama "
                f"regresyonda da kullanilmasindan kaynaklanan bilinen bir icsel devirsellik (endogeneity) "
                f"sorununa isaret edebilir -- bu degiskene dayanarak dogrudan aksiyon almadan once "
                f"iliskiyi baska bir yontemle (orn. dis/harici bir kontrol degiskeniyle) dogrulaman onerilir."
            )
    return "\n".join(satirlar)
