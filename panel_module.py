# -*- coding: utf-8 -*-
"""
PANEL ANALIZI MODULU (genellestirilmis)
==========================================
Orijinal panel_analizi.py'deki mantigin BIREBIR AYNISI (Pooled OLS -> FE -> RE
-> Poolability F-testi -> Hausman -> robust/clustered SE -> karsilastirma),
sadece veri artik disaridan (Malmquist modulunun ciktisindan) geliyor,
hardcoded sozluk yok.
"""
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from linearmodels.panel import PooledOLS, PanelOLS, RandomEffects, compare
from scipy import stats as sstats

# Panel analizindeki model secimi karar noktalarinda (Poolability F-testi, Breusch-Pagan
# LM testi, Hausman testi) kullanilan ortak anlamlilik esigi. NOT: bu esik, katsayi
# anlamliligi (panel_aksiyon_analizi, yorumlama.py) icin AYRI ve %10'dur -- ikisi
# kasitli olarak farkli tutulmustur (model secimi daha sıkı, katsayi yorumu daha esnek).
ALPHA = 0.05


def hausman_test(fe_res, re_res):
    """
    Klasik Hausman testi (matris farki tabanli). NOT: 'const' (sabit terim)
    karsilastirmadan KASITLI OLARAK CIKARILIR -- standart uygulama budur
    (orn. Stata'nin hausman komutu), cunku FE ve RE'deki sabit terim kavramsal
    olarak FARKLI seyler ifade eder (FE'de DMU'lara ozgu etkilerin ortalanmis
    bir kalintisi, RE'de gercek ortak bir sabit terim). Sabit terimi dahil
    etmek, ozellikle bir degisken absorbe edilip dusuruldugunde (drop_absorbed),
    var_diff matrisinin kosegeninde NEGATIF (matematiksel olarak imkansiz bir
    varyans) degerlere ve dolayisiyla dejenere/negatif Hausman istatistigine
    yol acabilir -- bu proje sirasinda ampirik olarak dogrulanmis bir durumdur.
    """
    ortak = [v for v in fe_res.params.index if v in re_res.params.index and v != "const"]
    b_fe = fe_res.params[ortak]
    b_re = re_res.params[ortak]
    v_fe = fe_res.cov.loc[ortak, ortak]
    v_re = re_res.cov.loc[ortak, ortak]
    diff = b_fe - b_re
    var_diff = v_fe - v_re
    stat = float(diff.T @ np.linalg.pinv(var_diff.values) @ diff)
    dof = len(ortak)
    p_value = 1 - sstats.chi2.cdf(stat, dof)
    return stat, dof, p_value


def degisken_varyans_analizi(panel_df: pd.DataFrame, bagimsizlar: list) -> pd.DataFrame:
    """
    Her bagimsiz degisken icin, DMU-ici (within) varyansin TOPLAM varyansa
    oranini hesaplar -- "bu degisken ne kadar zaman-sabit" sorusuna dogrudan,
    sayisal bir cevap verir. Kullaniciyi tahmin yurutmeye (hangi degiskeni
    teker teker cikarip deneyeyim) mecbur birakmadan, TUM degiskenleri TEK
    SEFERDE gorup, en dusuk orana sahip olanlari (zaman-sabit/neredeyse
    zaman-sabit adaylarini) tespit etmesini saglar.

    Oran ~1.0'a yakinsa: degisken buyuk olcude zaman icinde degisiyor (saglam).
    Oran ~0'a yakinsa: degisken neredeyse tamamen DMU'lar arasi farkliliktan
    olusuyor, zaman icinde neredeyse hic degismiyor (riskli -- FE'de tahmin
    edilemez/dusurulur, Hausman testini bozabilir).

    Returns: DataFrame (index=degisken) -- toplam_varyans, dmu_ici_varyans,
             within_orani, durum ("Saglam" / "Dikkat" / "Zaman-sabit")
    """
    entity_seviyesi = panel_df.index.get_level_values("entity")
    satirlar = []
    for degisken in bagimsizlar:
        if degisken not in panel_df.columns:
            continue
        toplam_varyans = panel_df[degisken].var(ddof=0)
        ici_varyans_ort = panel_df.groupby(entity_seviyesi)[degisken].var(ddof=0).mean()
        oran = (ici_varyans_ort / toplam_varyans) if toplam_varyans > 1e-12 else 0.0
        oran = 0.0 if pd.isna(oran) else oran
        if oran < 0.01:
            durum = "⚠️ Zaman-sabit (ya da neredeyse)"
        elif oran < 0.10:
            durum = "🟡 Dikkat (düşük within-varyans)"
        else:
            durum = "✅ Sağlam"
        satirlar.append({
            "degisken": degisken, "toplam_varyans": round(float(toplam_varyans), 4),
            "dmu_ici_varyans": round(float(ici_varyans_ort), 4),
            "within_orani": round(float(oran), 4), "durum": durum,
        })
    return pd.DataFrame(satirlar).set_index("degisken")


def mundlak_hausman_testi(panel_df: pd.DataFrame, bagimli: str, bagimsizlar: list,
                           alpha: float = ALPHA) -> dict:
    """
    Regresyon-tabanli (Mundlak, 1978) Hausman testi -- klasik Hausman testinin
    (iki AYRI modelin kovaryans matrisini birbirinden CIKARMAKTAN kaynaklanan)
    negatif/dejenere sonuc riskini YAPISAL OLARAK ORTADAN KALDIRAN bir alternatif.

    YONTEM:
    1. Zaman icinde DEGISEN her bagimsiz degisken icin, o degiskenin DMU-
       ortalamasini (x_bar_i) hesapla. (Zaman-sabit degiskenlerin DMU-ortalamasi
       kendisiyle ozdes oldugu icin -- hep ayni sayi -- bunlar otomatik olarak
       testin disinda kalir, ayri bir filtreleme gerekmez.)
    2. RE modeline, orijinal degiskenlerin YANINA bu DMU-ortalamalarini da EK
       regresor olarak katarak "genisletilmis RE" (Mundlak/CRE -- correlated
       random effects) modelini kur.
    3. Eklenen DMU-ortalamasi katsayilarinin TOPLU olarak sifir olup olmadigini
       bir Wald testiyle sina -- bu, TEK BIR modelin (genisletilmis RE'nin)
       KENDI kovaryans matrisini kullanir, IKI FARKLI modelin matrisini
       BIRBIRINDEN CIKARMAZ. Bu yuzden istatistik, TANIM GEREGI negatif
       CIKAMAZ (gecerli bir kovaryans matrisinden turetilen bir kuadratik
       form, her zaman >= 0'dir).

    HIPOTEZLER:
    H0: DMU-ortalamalarinin katsayilari topluca sifir -> c_i (gozlemlenemeyen
        birim etkisi), regresorlerle korelasyonsuz -> RE gecerli.
    H1: en az biri sifirdan anlamli farkli -> c_i korelasyonlu -> FE tercih
        edilmeli.

    Returns: dict -- yeterli_veri, stat, dof, p_value, secilen_model,
             test_edilen_degiskenler (zaman icinde degisen, teste dahil
             edilenler), zaman_sabit_degiskenler (otomatik disarida kalanlar,
             bu degiskenler icin ayrica "sadece between-etki" uyarisi
             gosterilmelidir)
    """
    from linearmodels.panel import RandomEffects

    entity_seviyesi = panel_df.index.get_level_values("entity")

    # ONEMLI DUZELTME: mutlak esik (< 1e-10) yerine GORELI esik kullaniyoruz
    # (DMU-ici varyans / toplam varyans). Boylece sadece TAM zaman-sabit degil,
    # "neredeyse zaman-sabit" (ornegin olcum hassasiyeti/yuvarlama yuzunden
    # cok kucuk ama tam sifir olmayan degisim gosteren) degiskenler de
    # yakalanip testin disinda tutulur -- aksi halde bu tur degiskenler,
    # genisletilmis RE modelinde NEREDEYSE MUKEMMEL coklu dogrusal baglantiya
    # (DMU-ortalamasi, degiskenin kendisiyle neredeyse ozdes oldugu icin) yol
    # acip modelin "full column rank" hatasiyla COKMESINE sebep olabilir --
    # bu proje sirasinda gercek veride ampirik olarak karsilasilan bir durumdur.
    GORELI_ESIK = 0.01  # DMU-ici varyans, toplam varyansin %1'inden azsa "zaman-sabit" sayilir

    zaman_sabit_degiskenler, degisen_degiskenler = [], []
    ortalama_sutunlari = {}
    for degisken in bagimsizlar:
        if degisken not in panel_df.columns:
            continue
        toplam_varyans = panel_df[degisken].var(ddof=0)
        ici_varyans_ort = panel_df.groupby(entity_seviyesi)[degisken].var(ddof=0).mean()
        oran = (ici_varyans_ort / toplam_varyans) if toplam_varyans > 1e-12 else 0.0
        if pd.isna(oran) or oran < GORELI_ESIK:
            zaman_sabit_degiskenler.append(degisken)
        else:
            degisen_degiskenler.append(degisken)
            ortalama_ad = f"_{degisken}_dmu_ort"
            ortalama_sutunlari[ortalama_ad] = panel_df.groupby(entity_seviyesi)[degisken].transform("mean")

    if not degisen_degiskenler:
        return {
            "yeterli_veri": False,
            "mesaj": "Test edilecek zaman icinde (yeterince) degisen bagimsiz degisken bulunamadi.",
        }

    genisletilmis_df = panel_df.copy()
    for ad, deger in ortalama_sutunlari.items():
        genisletilmis_df[ad] = deger

    ortalama_adlari = list(ortalama_sutunlari.keys())
    tum_regresorler = bagimsizlar + ortalama_adlari

    y = genisletilmis_df[[bagimli]]

    def _dene(X_deneme, check_rank_kapat):
        """fit + kovaryans erisimi + istatistik hesabini TEK BLOK olarak dener --
        check_rank=False ile bile bazen .fit() basariyla dener ama SONRADAN
        .cov erisiminde (kovaryans matrisi tekil/singular oldugu icin) ayrica
        cokebiliyor -- bu yuzden ikisi de AYNI guvenlik agi icinde olmali."""
        if check_rank_kapat:
            model = RandomEffects(y, X_deneme, check_rank=False).fit()
        else:
            model = RandomEffects(y, X_deneme)
            model = model.fit()
        b_ = model.params[ortalama_adlari]
        V_ = model.cov.loc[ortalama_adlari, ortalama_adlari]
        stat_ = float(b_.T @ np.linalg.pinv(V_.values) @ b_)
        return stat_

    # RANK-EKSIKLIGI GUVENLIK AGI -- IKI KADEMELI:
    # 1) Once, goreli esigi gecen ama yine de baska degiskenlerle neredeyse
    #    collinear olan "ortalama" sutunlarini, en dusuk within-orani sirayla
    #    dusurup tekrar deniyoruz (istatistiksel olarak en temiz cozum).
    # 2) TUM ortalama sutunlari dusse bile rank sorunu devam ediyorsa (yani
    #    collinearlik, entity-mean sutunlarindan degil, HAM bagimsizlar
    #    listesinin kendisinden geliyorsa), kutuphanenin kendi onerdigi
    #    check_rank=False ile SON bir deneme yapiyoruz -- bu, fonksiyonun
    #    HICBIR KOSULDA ham bir hatayla cokmemesini garanti eder. Bu son
    #    care kullanildiginda, sonuc "check_rank_false_kullanildi" bayragiyla
    #    isaretlenir -- katsayi tahminlerinin sayisal kesinligi bu durumda
    #    garanti degildir.
    dusurulen_ekstra = []
    check_rank_false_kullanildi = False
    stat = None
    while stat is None:
        X = genisletilmis_df[tum_regresorler].assign(const=1.0)
        try:
            stat = _dene(X, check_rank_kapat=False)
            break
        except Exception as e:
            if len(ortalama_adlari) > 1:
                # En dusuk (goreli) ici-varyans oranina sahip degiskeni bul ve dusur
                oranlar = {}
                for d in degisen_degiskenler:
                    tv = panel_df[d].var(ddof=0)
                    iv = panel_df.groupby(entity_seviyesi)[d].var(ddof=0).mean()
                    oranlar[d] = (iv / tv) if tv > 1e-12 else 0.0
                en_dusuk = min(oranlar, key=oranlar.get)
                dusurulen_ekstra.append(en_dusuk)
                zaman_sabit_degiskenler.append(en_dusuk)
                degisen_degiskenler.remove(en_dusuk)
                ortalama_ad = f"_{en_dusuk}_dmu_ort"
                del ortalama_sutunlari[ortalama_ad]
                ortalama_adlari = list(ortalama_sutunlari.keys())
                tum_regresorler = bagimsizlar + ortalama_adlari
                continue
            if not ortalama_adlari:
                return {
                    "yeterli_veri": False,
                    "mesaj": "Rank eksikligi nedeniyle tum degiskenler testten cikarildi, test yapilamadi.",
                }
            if not check_rank_false_kullanildi:
                check_rank_false_kullanildi = True
                try:
                    stat = _dene(X, check_rank_kapat=True)
                    break
                except Exception as e2:
                    return {
                        "yeterli_veri": False,
                        "mesaj": f"Genisletilmis RE modeli kurulamadi (check_rank=False ile de basarisiz): {e2}",
                    }
            else:
                return {
                    "yeterli_veri": False,
                    "mesaj": f"Genisletilmis RE modeli kurulamadi (rank eksikligi cozulemedi): {e}",
                }

    dof = len(ortalama_adlari)
    p_value = 1 - sstats.chi2.cdf(stat, dof)
    secilen_model = "FE" if p_value < alpha else "RE"

    return {
        "yeterli_veri": True, "stat": round(stat, 4), "dof": dof, "p_value": round(p_value, 4),
        "secilen_model": secilen_model,
        "dusurulen_rank_eksikligi_nedeniyle": dusurulen_ekstra,
        "test_edilen_degiskenler": degisen_degiskenler,
        "zaman_sabit_degiskenler": zaman_sabit_degiskenler,
        "check_rank_false_kullanildi": check_rank_false_kullanildi,
    }


def breusch_pagan_lm_test(pooled_res, panel_data):
    """
    Dengeli panel icin Breusch-Pagan (1980) LM testi -- Pooled OLS vs Random Effects.
    H0: sigma_mu^2 = 0  (birim etkisi yok -> Pooled OLS yeterli)
    H1: sigma_mu^2 != 0 (birim etkisi var  -> en azindan RE gerekli)
    Poolability F-testinin (Pooled vs FE) tamamlayicisi: bu test Pooled vs RE'yi
    dogrudan hedefler.
    """
    resid = pooled_res.resids
    resid_df = resid.to_frame(name="e") if hasattr(resid, "to_frame") else pd.DataFrame({"e": resid})
    resid_df = resid_df.reset_index()
    entity_col = resid_df.columns[0]

    N = resid_df[entity_col].nunique()
    T_per_entity = resid_df.groupby(entity_col).size()
    T = T_per_entity.iloc[0]

    if T_per_entity.nunique() != 1:
        raise ValueError("Panel dengesiz (unbalanced) - bu basit BP-LM formulu sadece dengeli panel icin gecerlidir.")

    e_sum_sq_total = (resid_df["e"] ** 2).sum()
    entity_sums = resid_df.groupby(entity_col)["e"].sum()
    e_sum_sq_between = (entity_sums ** 2).sum()

    ratio = e_sum_sq_between / e_sum_sq_total
    lm_stat = (N * T) / (2 * (T - 1)) * (ratio - 1) ** 2
    p_value = 1 - sstats.chi2.cdf(lm_stat, df=1)

    return {"stat": float(lm_stat), "pval": float(p_value), "N": int(N), "T": int(T)}


def nihai_oneri_belirle(poolability: dict, bp_lm: dict, hausman: dict, secilen_model: str,
                         n_entities: int, alpha: float = ALPHA):
    """
    Literatur sirasina gore (Baltagi, Wooldridge) nihai model/SE onerisini belirler:
      1) Iki tamamlayici birim-etki testi:
         - Poolability F-testi (Pooled OLS vs FE)
         - Breusch-Pagan LM testi (Pooled OLS vs RE)
         Ikisinden BIRI bile H0'i reddederse (temkinli/muhafazakar yaklasim),
         DMU'lara ozgu etki oldugu kabul edilir ve panel modeline (FE/RE) gecilir.
      2) (Panel etkisi varsa) Hausman testi: FE mi RE mi? -- dejenere (negatif stat)
         sonuclarda Hausman'a guvenilmez, poolability + teorik gerekceyle karar verilmeli.
      3) SE tipi: Clustered (DMU bazinda) teorik olarak tercih edilir, AMA kume sayisi
         (=DMU sayisi) literaturde onerilen ~30-50 esiginin altindaysa (Cameron & Miller,
         2015; Cameron, Gelbach & Miller, 2008) guvenilirligi dusuktur -- bu durumda
         robust (kumeleme yapmayan) SE ile birlikte raporlanmasi ve temkinli yorumlanmasi
         onerilir. Bu, secilen model Pooled OLS de olsa FE/RE de olsa aynen gecerlidir.

    alpha: karar noktalarinda kullanilan anlamlilik esigi (varsayilan: modul-seviyesi ALPHA).
    Returns: dict -- asama1, asama2, se_onerisi, sonuc_tablo (hangi tabloyu vurgula),
                     sonuc_basligi, uyarilar (liste)
    """
    uyarilar = []
    alpha_yuzde = f"{alpha:.2f}"
    sinir_ust = alpha + 0.05  # "sinirda" bolgesinin ust siniri (raporlama amacli)

    # ASAMA 1: Pooled OLS yeterli mi? -- iki tamamlayici test birlikte
    pool_pval = poolability.get("pval")
    lm_pval = bp_lm.get("pval") if bp_lm else None

    f_reddedildi = pool_pval is not None and pool_pval < alpha
    lm_reddedildi = lm_pval is not None and lm_pval < alpha
    panel_gerekli = f_reddedildi or lm_reddedildi

    test_ozetleri = []
    if pool_pval is not None:
        test_ozetleri.append(f"Poolability F-testi (Pooled vs FE): p={pool_pval:.4f} "
                              f"({'H0 reddedildi' if f_reddedildi else 'H0 reddedilemedi'}, alpha={alpha_yuzde})")
    else:
        test_ozetleri.append("Poolability F-testi hesaplanamadi")
    if lm_pval is not None:
        test_ozetleri.append(f"Breusch-Pagan LM testi (Pooled vs RE): p={lm_pval:.4f} "
                              f"({'H0 reddedildi' if lm_reddedildi else 'H0 reddedilemedi'}, alpha={alpha_yuzde})")
    else:
        test_ozetleri.append("Breusch-Pagan LM testi hesaplanamadi (muhtemelen dengesiz panel)")

    if pool_pval is None and lm_pval is None:
        asama1 = "Ne Poolability ne de BP-LM testi hesaplanabildi -- panel etkisi oldugu varsayilarak devam edildi."
        panel_gerekli = True
    elif f_reddedildi and lm_reddedildi:
        asama1 = (f"{'; '.join(test_ozetleri)}. Her iki test de DMU'lara ozgu etki oldugunu "
                  f"gosteriyor -> Pooled OLS YETERSIZ, panel modeline (FE/RE) gecilmeli.")
    elif f_reddedildi or lm_reddedildi:
        asama1 = (f"{'; '.join(test_ozetleri)}. Testler CELISIYOR -- biri DMU etkisi oldugunu, "
                  f"digeri olmadigini soyluyor. Temkinli yaklasimla panel modeline (FE/RE) gecilmesi "
                  f"tercih edildi (iki testten en az biri etkiyi isaret ettigi icin).")
        uyarilar.append(
            "Poolability F-testi ile Breusch-Pagan LM testi FARKLI sonuclara ulasti "
            "(biri H0'i reddederken digeri reddedemedi). Bu durumda hangi teste agirlik "
            "verdiginizi raporunuzda gerekcelendirmeniz onerilir."
        )
    else:
        # her iki test de H0'i reddedemedi -- ama sinir durumu kontrolu (F-test uzerinden)
        if pool_pval is not None and pool_pval < sinir_ust:
            asama1 = (f"{'; '.join(test_ozetleri)}. Her iki test de teknik olarak Pooled OLS'in "
                      f"yeterli oldugunu gosteriyor, ANCAK Poolability p-degeri alpha={sinir_ust:.2f} "
                      f"sinirinda -- DMU'lara ozgu gizli bir fark ihtimali gozardi edilmemeli.")
            uyarilar.append(
                f"Poolability p-degeri ({pool_pval:.4f}) alpha={alpha_yuzde} ile alpha={sinir_ust:.2f} arasinda "
                f"sinirda kaliyor. Raporunuzda bu sinir durumu belirtilmelidir."
            )
        else:
            asama1 = (f"{'; '.join(test_ozetleri)}. Her iki test de DMU'lara ozgu anlamli bir etki "
                      f"tespit etmedi -> Pooled OLS YETERLI, FE/RE gereksiz karmasiklik katabilir.")
        uyarilar.append(
            "Poolability F-testi ve Breusch-Pagan LM testi Pooled OLS'in yeterli oldugunu gosteriyor; "
            "asagidaki FE/RE secimi yine de bilgi amacli gosteriliyor, ama ana sonucunuz Pooled OLS olmali."
        )

    # ASAMA 2: Hausman (sadece panel gerekliyse anlamli)
    h_stat = hausman.get("stat")
    h_pval = hausman.get("pval")
    hausman_degenere = h_stat is not None and h_stat < 0
    if hausman_degenere:
        uyarilar.append(
            f"Hausman istatistigi negatif cikti (chi2={h_stat:.4f}) -- bu, kucuk T'li panellerde "
            f"bilinen bir sinir durumdur (varyans-fark matrisi pozitif tanimli degil). Hausman "
            f"testine tek basina guvenilmemeli; secim, Poolability testi ve teorik gerekceyle "
            f"(orn. 'DMU'lara ozgu sabit farklar beklenir mi?') desteklenmelidir."
        )
        asama2 = (f"Hausman testi dejenere sonuc verdi (p={h_pval:.4f}), FE/RE arasinda net "
                  f"istatistiksel ayrim yapilamadi. Asagidaki secim ({secilen_model}) temkinli "
                  f"degerlendirilmelidir.")
    elif not panel_gerekli:
        asama2 = "Poolability testi Pooled OLS'in yeterli oldugunu gosterdigi icin bu adim ikincil onemde."
    else:
        asama2 = (f"Hausman testi (chi2={h_stat:.4f}, p={h_pval:.4f}, alpha={alpha_yuzde}) -> "
                  f"{'H0 reddedildi, FE tutarli' if h_pval < alpha else 'H0 reddedilemedi, RE tercih edilebilir (daha etkin)'}.")

    # ASAMA 3: SE tipi -- kume (DMU) sayisi yeterli mi?
    KUME_ESIGI = 30
    if n_entities >= KUME_ESIGI:
        se_onerisi = (f"DMU sayisi ({n_entities}) literaturdeki ~{KUME_ESIGI} esiginin uzerinde oldugu "
                      f"icin Clustered (DMU bazinda kumelenmis) standart hatalar guvenilir kabul edilir "
                      f"ve tercih edilmelidir.")
        se_tipi = "clustered"
    else:
        se_onerisi = (f"DMU sayisi ({n_entities}) literaturde onerilen ~{KUME_ESIGI} esiginin ALTINDA. "
                      f"Bu kadar az kume ile Clustered SE guvenilmez olabilir (varyansi oldugundan dusuk "
                      f"tahmin edip yanlislikla anlamli sonuc gosterebilir -- Cameron & Miller, 2015). "
                      f"Bu durumda Robust (heteroskedastisite-tutarli, kumelemesiz) standart hatalarin "
                      f"esas alinmasi ve Clustered sonucun sadece bilgi amacli, temkinli yorumla "
                      f"verilmesi onerilir.")
        se_tipi = "robust"
        uyarilar.append(
            f"DMU sayiniz ({n_entities}) kucuk oldugu icin Clustered standart hatalar yerine "
            f"Robust standart hatalari esas almaniz onerilir; bunu raporunuzda bir sinirlama "
            f"(limitation) olarak belirtmeniz akademik acidan dogru olur."
        )

    if panel_gerekli:
        sonuc_tablo = f"{secilen_model.lower()}_{se_tipi}"
        sonuc_basligi = f"{secilen_model} - {'Clustered' if se_tipi=='clustered' else 'Robust'} Standart Hatalar"
    else:
        sonuc_tablo = f"pooled_{se_tipi}"
        sonuc_basligi = f"Pooled OLS - {'Clustered' if se_tipi=='clustered' else 'Robust'} Standart Hatalar"

    return {
        "asama1": asama1,
        "asama2": asama2,
        "se_onerisi": se_onerisi,
        "panel_gerekli": panel_gerekli,
        "hausman_degenere": hausman_degenere,
        "se_tipi": se_tipi,
        "sonuc_tablo": sonuc_tablo,
        "sonuc_basligi": sonuc_basligi,
        "uyarilar": uyarilar,
        "alpha": alpha,
    }


def korelasyon_ve_vif_hesapla(panel_df: pd.DataFrame, bagimli: str, teshis_degiskenler: list) -> dict:
    """
    Korelasyon matrisi ve VIF'i, panel REGRESYONUNDA kullanilan bagimsiz
    degisken listesinden BAGIMSIZ olarak hesaplar. Bu, kullanicinin su
    senaryosunu desteklemek icin eklendi: panel regresyonu SADECE girdilerle
    kurulsa bile (cikti disarida birakilsa bile), korelasyon/VIF teshis
    tablolarinda cikti(lar)in HALA gorunmesini istemesi -- coklu dogrusal
    baglanti/korelasyon riskini, regresyona hangi degiskenlerin girdiginden
    BAGIMSIZ olarak izleyebilmek icin.

    teshis_degiskenler: genelde girdi_cols + cikti_cols (TAM liste) verilir --
    regresyonda kullanilan "bagimsizlar" listesinden farkli/daha genis olabilir.

    Returns: dict -- corr (DataFrame), vif (DataFrame)
    """
    gecerli = [d for d in teshis_degiskenler if d in panel_df.columns]
    corr = panel_df[gecerli + [bagimli]].corr()
    Xc = add_constant(panel_df[gecerli])
    vif_data = pd.DataFrame()
    vif_data["degisken"] = Xc.columns
    vif_data["VIF"] = [variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])]
    return {"corr": corr, "vif": vif_data}


def within_korelasyon_hesapla(panel_df: pd.DataFrame, bagimli: str, teshis_degiskenler: list) -> dict:
    """
    HAM (seviye) korelasyon, DMU'lar arasi seviye farklarini (between) VE her
    DMU'nun kendi zaman ici degisimini (within) KARISTIRIR. Ama FE (Sabit
    Etkiler) modeli SADECE within varyasyonu kullanir -- her degiskenden
    kendi DMU-ortalamasi CIKARILDIKTAN SONRAKI (demeaned) veriyle calisir.

    Bu fonksiyon, AYNI degisken setinin within-donusturulmus (DMU-ortalamasi
    cikarilmis) halindeki korelasyonunu hesaplar. Eger bu korelasyonlar HAM
    korelasyondan BELIRGIN SEKILDE dusukse, bu, degiskenlerin DMU'lar arasi
    seviye farklarinda (orn. "hangi DMU daha gelismis") birbirine cok bagli
    olsa da, zaman icindeki DEGISIMLERININ (FE'nin gercekte kullandigi bilgi)
    birbirinden daha BAGIMSIZ oldugu anlamina gelir -- yani FE modeli, ham
    tabloya bakarak dusunulenden DAHA IYI ayirt ediyor olabilir.

    NOT: Bu SADECE bir TESHIS/rapor tablosudur -- hicbir modelleme kararini
    (hangi degiskenlerin secildigini, hangi tahmincinin kullanildigini)
    OTOMATIK OLARAK degistirmez.

    Returns: dict -- corr_within (DataFrame), corr_ham (DataFrame, kiyaslama
             icin), ortalama_dusus (ham - within korelasyonlarinin, mutlak
             degerce, ortalama farki -- pozitifse within DAHA DUSUK demektir)
    """
    gecerli = [d for d in teshis_degiskenler if d in panel_df.columns]
    kolonlar = gecerli + [bagimli]

    corr_ham = panel_df[kolonlar].corr()

    # Within donusum: her DMU (entity) icin kendi ortalamasini cikar
    df_within = panel_df[kolonlar].groupby(level="entity").transform(lambda s: s - s.mean())
    corr_within = df_within.corr()

    # Sadece bagimsizlar arasindaki (bagimli haric) ikili karsilastirmalarin
    # ortalama mutlak farkini hesapla -- "within genel olarak dusuk mu" sorusu icin
    farklar = []
    for i in range(len(gecerli)):
        for j in range(i + 1, len(gecerli)):
            ham_r = abs(corr_ham.iloc[i, j])
            within_r = abs(corr_within.iloc[i, j])
            if pd.notna(ham_r) and pd.notna(within_r):
                farklar.append(ham_r - within_r)
    ortalama_dusus = float(np.mean(farklar)) if farklar else None

    return {"corr_within": corr_within, "corr_ham": corr_ham, "ortalama_dusus": ortalama_dusus}


def ridge_izi_hesapla(panel_df: pd.DataFrame, bagimli: str, bagimsizlar: list) -> dict:
    """
    Ridge REGRESYON IZI (ridge trace): ceza parametresi (alpha) ADIM ADIM
    artirilirken her degiskenin (standardize edilmis) katsayisinin nasil
    degistigini gosterir. Bu, coklu dogrusal baglantidan kaynaklanan
    KIRILGAN/YAPAY katsayilari tespit etmenin klasik yontemidir
    (Hoerl & Kennard, 1970): eger bir degiskenin katsayisi ceza artarken
    IsARET DEGISTIRIYORSA ya da capraz sicriyorsa, bu, o katsayinin
    OLS/FE'deki degerinin buyuk olcude coklu dogrusal baglantidan
    kaynaklanan bir yapaylik oldugunu gosterir -- gercek bir katsayi ceza
    altinda YUMUSAKCA sifira yaklasir, isaret degistirmez.

    WITHIN (FE-ozgu) veri uzerinde calisir -- yani her degiskenden once
    kendi DMU-ortalamasi cikarilir -- boylece FE modelindeki (tartisilan)
    katsayilarla ADIL bir kiyaslama yapilir (Pooled/ham veri degil).

    Returns: dict -- iz_df (index=log10(alpha), columns=bagimsizlar,
             values=standardize edilmis ridge katsayisi),
             ols_katsayilar (dict, alpha=0'a karsilik gelen -- yani
             cezasiz -- referans katsayilar, ayni within/standardize
             olcekte)
    """
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.preprocessing import StandardScaler

    gecerli = [d for d in bagimsizlar if d in panel_df.columns]
    df_within = panel_df[gecerli + [bagimli]].groupby(level="entity").transform(lambda s: s - s.mean())

    X = df_within[gecerli].values
    y = df_within[bagimli].values

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    ols_ref = LinearRegression(fit_intercept=False).fit(X_std, y)
    ols_katsayilar = dict(zip(gecerli, ols_ref.coef_))

    log_alphalar = np.linspace(-3, 4, 60)  # alpha = 10^-3 .. 10^4
    satirlar = []
    for la in log_alphalar:
        alpha = 10 ** la
        ridge = Ridge(alpha=alpha, fit_intercept=False).fit(X_std, y)
        satir = {"log10_alpha": round(float(la), 3)}
        for i, deg in enumerate(gecerli):
            satir[deg] = ridge.coef_[i]
        satirlar.append(satir)

    iz_df = pd.DataFrame(satirlar).set_index("log10_alpha")
    return {"iz_df": iz_df, "ols_katsayilar": ols_katsayilar}


def _panel_test_seti_gecerli_mi(panel_df: pd.DataFrame, bagimli: str, bagimsizlar: list):
    """
    Verilen bagimsizlar seti icin FE, RE, Hausman, Poolability (F-testi) VE
    BP-LM testlerinin HEPSININ hesaplanabilir ve GECERLI (dejenere olmayan)
    olup olmadigini kontrol eder. Herhangi biri EXCEPTION firlatirsa ya da
    Hausman istatistigi dejenere (negatif/≈0) cikarsa GECERSIZ sayilir --
    zaman-sabit/neredeyse-sabit bir girdi genelde bu testlerden EN AZ BIRINI
    (cogunlukla Hausman'i, ama bazen Poolability F-testini ya da BP-LM'yi de)
    sayisal olarak imkansiz hale getirir.

    Returns: (gecerli_mi: bool, h_stat: float veya None)
    """
    try:
        y = panel_df[bagimli]
        X = panel_df[bagimsizlar]
        Xc = add_constant(X)
        res_fe = PanelOLS(y, X, entity_effects=True, drop_absorbed=True).fit()
        res_re = RandomEffects(y, Xc).fit()
        h_stat, h_dof, h_pval = hausman_test(res_fe, res_re)
        if h_stat < 0 or abs(h_stat) < 1e-6:
            return False, None

        res_pooled = PooledOLS(y, Xc).fit()
        _ = res_fe.f_pooled            # Poolability F-testi hesaplanabiliyor mu
        _ = breusch_pagan_lm_test(res_pooled, panel_df)  # BP-LM hesaplanabiliyor mu
        return True, h_stat
    except Exception:
        return False, None


def hausman_dejenerelik_giderici(panel_df: pd.DataFrame, bagimli: str, bagimsizlar: list) -> dict:
    """
    Zaman-sabit filtrelemesi YAPILDIKTAN SONRA bile, Hausman testi (ya da
    Poolability F-testi, ya da BP-LM testi) hala negatif/dejenere/hesaplanamaz
    cikabilir -- bunun sebebi zaman-sabitlik olmayabilir (orn. iki zaman-
    icinde-degisen degiskenin birbirine cok yakin hareket etmesi de ayni
    sorunu yaratabilir). Bu fonksiyon, BU DURUMU AYRICA cozer: bagimsizlar
    listesindeki degiskenleri TEKER TEKER cikarip TUM testleri (Hausman +
    Poolability + BP-LM) her seferinde yeniden hesaplar, ve HANGI DEGISKENIN
    cikarilmasi TUM testleri gecerli hale getiriyorsa (aday gecerliler
    arasinda en yuksek Hausman chi2'sini verıyorsa) o degiskeni KALICI
    olarak modelden cikarir. Bu islem, hala sorun devam ettigi surece
    TEKRARLANIR (birden fazla sorunlu degisken olabilir).

    Returns: dict -- bagimsizlar (indirgenmis liste), cikarilan_degiskenler
             (bu adimda -- zaman-sabitlikten degil, test-dejenereligi
             yuzunden -- cikarilan degiskenler), son_hausman (stat, dof, pval)
    """
    cikarilan_degiskenler = []
    guvenlik_sayaci = 0

    while len(bagimsizlar) > 1 and guvenlik_sayaci < 10:
        guvenlik_sayaci += 1
        gecerli_mi, _ = _panel_test_seti_gecerli_mi(panel_df, bagimli, bagimsizlar)
        if gecerli_mi:
            y = panel_df[bagimli]
            X = panel_df[bagimsizlar]
            Xc = add_constant(X)
            res_fe = PanelOLS(y, X, entity_effects=True, drop_absorbed=True).fit()
            res_re = RandomEffects(y, Xc).fit()
            h_stat, h_dof, h_pval = hausman_test(res_fe, res_re)
            return {
                "bagimsizlar": bagimsizlar, "cikarilan_degiskenler": cikarilan_degiskenler,
                "son_hausman": {"stat": h_stat, "dof": h_dof, "pval": h_pval},
            }

        y = panel_df[bagimli]

        dejenere_mi = True  # gecerli_mi False ise buraya geldik

        # Dejenere -- her degiskeni sirayla cikarip hangisinin TUM testleri
        # (Hausman + Poolability + BP-LM) gecerli hale getirdigini bul.
        en_iyi_aday, en_iyi_stat = None, None
        for aday in bagimsizlar:
            deneme_liste = [v for v in bagimsizlar if v != aday]
            if not deneme_liste:
                continue
            gecerli_aday_mi, stat_d = _panel_test_seti_gecerli_mi(panel_df, bagimli, deneme_liste)
            if gecerli_aday_mi:
                if en_iyi_stat is None or stat_d > en_iyi_stat:
                    en_iyi_aday, en_iyi_stat = aday, stat_d

        if en_iyi_aday is None:
            # Hicbir tek-degisken cikarma sorunu cozmuyor -- daha fazla
            # ugrasmadan, mevcut (dejenere) sonucla dur.
            try:
                X = panel_df[bagimsizlar]
                Xc = add_constant(X)
                res_fe = PanelOLS(y, X, entity_effects=True, drop_absorbed=True).fit()
                res_re = RandomEffects(y, Xc).fit()
                h_stat, h_dof, h_pval = hausman_test(res_fe, res_re)
                son_hausman = {"stat": h_stat, "dof": h_dof, "pval": h_pval}
            except Exception as e:
                son_hausman = {"hata": str(e)}
            return {
                "bagimsizlar": bagimsizlar, "cikarilan_degiskenler": cikarilan_degiskenler,
                "son_hausman": son_hausman,
            }

        cikarilan_degiskenler.append(en_iyi_aday)
        bagimsizlar = [v for v in bagimsizlar if v != en_iyi_aday]

    # Dongu bitti (tek degisken kaldi ya da guvenlik siniri asildi) --
    # son durumu hesaplayip dondur.
    y = panel_df[bagimli]
    X = panel_df[bagimsizlar]
    Xc = add_constant(X)
    try:
        res_fe = PanelOLS(y, X, entity_effects=True, drop_absorbed=True).fit()
        res_re = RandomEffects(y, Xc).fit()
        h_stat, h_dof, h_pval = hausman_test(res_fe, res_re)
        son_hausman = {"stat": h_stat, "dof": h_dof, "pval": h_pval}
    except Exception as e:
        son_hausman = {"hata": str(e)}

    return {
        "bagimsizlar": bagimsizlar, "cikarilan_degiskenler": cikarilan_degiskenler,
        "son_hausman": son_hausman,
    }


def coklu_baglanti_rank_kontrolu(X: pd.DataFrame) -> dict:
    """
    Xc (sabit terim + bagimsizlar) sutunlarinin TAM RANKA sahip olup olmadigini
    kontrol eder. Rank eksikse (mukemmel/near-mukemmel coklu dogrusal baglanti --
    orn. secilen iki panel girdisi birbirinin neredeyse birebir kati), hangi
    degisken CIFTLERININ sorumlu oldugunu (yuksek |korelasyon| ciftleri
    uzerinden) tespit edip acik bir Turkce teshis uretir. Bu, linearmodels'in
    ham Ingilizce "exog does not have full column rank" hatasindan ONCE
    devreye girip run_panel_analysis icinde yakalanip actionable bir
    ValueError'a cevrilir.

    Returns: dict -- sorunlu_mu (bool), ve sorunlu ise: rank, beklenen_rank,
             supheli_ciftler (liste of (degisken1, degisken2, |korelasyon|))
    """
    Xc = add_constant(X)
    rank = np.linalg.matrix_rank(Xc.values)
    tam_rank = Xc.shape[1]
    if rank >= tam_rank:
        return {"sorunlu_mu": False}

    korelasyon = X.corr().abs()
    kor_arr = korelasyon.to_numpy(copy=True)
    np.fill_diagonal(kor_arr, 0.0)
    kolonlar = list(korelasyon.columns)
    supheli_ciftler = []
    for i in range(len(kolonlar)):
        for j in range(i + 1, len(kolonlar)):
            r = kor_arr[i, j]
            if pd.notna(r) and r > 0.995:
                supheli_ciftler.append((kolonlar[i], kolonlar[j], round(float(r), 4)))

    return {
        "sorunlu_mu": True,
        "rank": int(rank),
        "beklenen_rank": int(tam_rank),
        "supheli_ciftler": supheli_ciftler,
    }


def run_panel_analysis(panel_df: pd.DataFrame, bagimli: str, bagimsizlar: list, alpha: float = ALPHA):
    """
    panel_df: index=['entity','time'], sutunlarda bagimli + bagimsizlar bulunmali
    alpha: TUM karar noktalarinda (Poolability, BP-LM, Hausman, katsayi anlamliligi)
           kullanilan ortak anlamlilik esigi (varsayilan: modul-seviyesi ALPHA).

    ONEMLI: Zaman icinde (neredeyse) hic degismeyen bagimsiz degiskenler, ANALIZIN
    EN BASINDA, HANGI MODEL (Pooled/FE/RE) SECILIRSE SECILSIN hicbir tabloda
    GORUNMEYECEK sekilde TAMAMEN cikarilir -- eskiden bu degiskenler sadece
    Hausman testinden cikarilip, RE secilirse "between-etki" olarak nihai
    tabloda GORUNMEYE DEVAM EDIYORDU; kullanici bunu (RE her secildiginde
    degiskenin geri gelmesini) istenmeyen bir davranis olarak degerlendirdi.
    Artik ML Tahmin sekmesindeki yapisal engelle AYNI felsefe: bu degiskenler
    modelin HICBIR asamasina (Pooled/FE/RE/Hausman/Mundlak/nihai tablo) hic
    girmiyor, sadece "zaman_sabit_tamamen_disarida" listesinde raporlaniyor.

    Returns: dict -- corr, vif, pooled, fe, re, poolability, hausman,
                     secilen_model, robust, clustered, comparison, n_entities, oneri,
                     zaman_sabit_tamamen_disarida (bu calistirmada TUM analizden
                     cikarilan degisken adlari)
    """
    varyans_on_kontrol = degisken_varyans_analizi(panel_df, bagimsizlar)
    zaman_sabit_tamamen_disarida = list(
        varyans_on_kontrol[varyans_on_kontrol["within_orani"] < 0.01].index
    )
    bagimsizlar = [v for v in bagimsizlar if v not in zaman_sabit_tamamen_disarida]
    if not bagimsizlar:
        raise ValueError(
            "Tum bagimsiz degiskenler zaman icinde (neredeyse) sabit -- panel analizi "
            "icin hicbir kullanilabilir degisken kalmadi."
        )

    # Zaman-sabitlik filtrelemesinden SONRA, hala dejenere (negatif/≈0) bir
    # Hausman sonucu cikiyorsa (orn. iki zaman-icinde-degisen degiskenin
    # birbirine cok yakin hareket etmesinden kaynaklanan bir sorun), suclu
    # degiskeni tespit edip AYRICA modelden cikar -- bu da ML Tahmin ve
    # nihai tablo dahil TUM analize yansir.
    dejenerelik_giderici_sonuc = None
    if len(bagimsizlar) > 1:
        dejenerelik_giderici_sonuc = hausman_dejenerelik_giderici(panel_df, bagimli, bagimsizlar)
        if dejenerelik_giderici_sonuc["cikarilan_degiskenler"]:
            bagimsizlar = dejenerelik_giderici_sonuc["bagimsizlar"]
    hausman_dejenerelik_nedeniyle_cikarilan = (
        dejenerelik_giderici_sonuc["cikarilan_degiskenler"] if dejenerelik_giderici_sonuc else []
    )

    y = panel_df[bagimli]
    X = panel_df[bagimsizlar]
    Xc = add_constant(X)

    # Coklu dogrusal baglanti / rank eksikligi kontrolu -- linearmodels'in ham
    # "exog does not have full column rank" hatasindan ONCE, sorumlu degisken
    # ciftini gosteren acik bir Turkce teshis uretir.
    rank_kontrol = coklu_baglanti_rank_kontrolu(X)
    if rank_kontrol["sorunlu_mu"]:
        if rank_kontrol["supheli_ciftler"]:
            cift_metni = "; ".join(
                f"{a} ile {b} (|r|={r})" for a, b, r in rank_kontrol["supheli_ciftler"]
            )
            detay = f"Suphelenilen coklu dogrusal baglanti: {cift_metni}."
        else:
            detay = (
                "Belirgin bir ikili yuksek korelasyon bulunamadi -- sorun 3+ "
                "degiskenin BIRLIKTE tam dogrusal bir kombinasyon olusturmasindan "
                "kaynaklaniyor olabilir."
            )
        raise ValueError(
            f"Secilen panel girdileri arasinda mukemmel/near-mukemmel coklu "
            f"dogrusal baglanti var (rank={rank_kontrol['rank']}, "
            f"beklenen={rank_kontrol['beklenen_rank']}) -- model tahmin "
            f"edilemiyor. {detay} Bu degiskenlerden birini panel secim "
            f"kutusundan cikarip tekrar deneyin."
        )

    n_entities = panel_df.index.get_level_values("entity").nunique()

    corr = panel_df[bagimsizlar + [bagimli]].corr()

    vif_data = pd.DataFrame()
    vif_data["degisken"] = Xc.columns
    vif_data["VIF"] = [variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])]

    res_pooled = PooledOLS(y, Xc).fit()
    res_fe = PanelOLS(y, X, entity_effects=True, drop_absorbed=True).fit()
    res_re = RandomEffects(y, Xc).fit()

    res_pooled_robust = PooledOLS(y, Xc).fit(cov_type="robust")
    res_pooled_clustered = PooledOLS(y, Xc).fit(cov_type="clustered", cluster_entity=True)

    poolability = {}
    try:
        f_pool = res_fe.f_pooled
        poolability = {"stat": f_pool.stat, "pval": f_pool.pval,
                        "sonuc": "FE tercih edilmelidir" if f_pool.pval < alpha else "Pooled OLS yeterlidir"}
    except Exception as e:
        poolability = {"hata": str(e)}

    try:
        bp_lm = breusch_pagan_lm_test(res_pooled, panel_df)
    except Exception as e:
        bp_lm = {"hata": str(e)}

    # NOT: Zaman-sabit degiskenler artik fonksiyonun EN BASINDA (yukarida)
    # tum bagimsizlar listesinden zaten cikarildigi icin, res_fe/res_re zaten
    # SADECE zaman icinde yeterince degisen degiskenlerle kurulmus durumda --
    # Hausman testi icin ayrica bir indirgeme yapmaya gerek yok.
    h_stat, h_dof, h_pval = hausman_test(res_fe, res_re)
    secilen_model = "FE" if h_pval < alpha else "RE"
    hausman = {"stat": h_stat, "dof": h_dof, "pval": h_pval}

    # Regresyon-tabanli (Mundlak) Hausman testi -- klasik testin dejenere/negatif
    # cikma riskini yapisal olarak tasimayan bir dogrulama/alternatif.
    mundlak = mundlak_hausman_testi(panel_df, bagimli, bagimsizlar, alpha=alpha)

    res_fe_robust = PanelOLS(y, X, entity_effects=True, drop_absorbed=True).fit(cov_type="robust")
    res_fe_clustered = PanelOLS(y, X, entity_effects=True, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
    res_re_robust = RandomEffects(y, Xc).fit(cov_type="robust")
    res_re_clustered = RandomEffects(y, Xc).fit(cov_type="clustered", cluster_entity=True)

    if secilen_model == "FE":
        res_robust, res_clustered = res_fe_robust, res_fe_clustered
    else:
        res_robust, res_clustered = res_re_robust, res_re_clustered

    comparison = compare({"Pooled OLS": res_pooled, "FE": res_fe, "RE": res_re})

    oneri = nihai_oneri_belirle(poolability, bp_lm, hausman, secilen_model, n_entities, alpha=alpha)

    return {
        "corr": corr,
        "vif": vif_data,
        "pooled": res_pooled,
        "fe": res_fe,
        "re": res_re,
        "poolability": poolability,
        "bp_lm": bp_lm,
        "hausman": hausman,
        "mundlak_hausman": mundlak,
        "varyans_analizi": varyans_on_kontrol,
        "zaman_sabit_tamamen_disarida": zaman_sabit_tamamen_disarida,
        "hausman_dejenerelik_nedeniyle_cikarilan": hausman_dejenerelik_nedeniyle_cikarilan,
        "secilen_model": secilen_model,
        "robust": res_robust,
        "clustered": res_clustered,
        "pooled_robust": res_pooled_robust, "pooled_clustered": res_pooled_clustered,
        "fe_robust": res_fe_robust, "fe_clustered": res_fe_clustered,
        "re_robust": res_re_robust, "re_clustered": res_re_clustered,
        "comparison": comparison,
        "n_entities": n_entities,
        "oneri": oneri,
        "alpha": alpha,
    }


def leave_one_out_kararlilik(panel_df: pd.DataFrame, bagimli: str, bagimsizlar: list,
                              sonuc_tablo: str, alpha: float = 0.10) -> dict:
    """
    Leave-one-DMU-out katsayi kararlilik testi.

    Nihai modeli (oneri["sonuc_tablo"] neyse ayni model tipi + SE tipiyle),
    panelden SIRAYLA BIR DMU cikararak N kere yeniden tahmin eder. Bir
    iliskinin tek bir DMU'nun surukledigi bir yapaylik mi, yoksa genel bir
    orunek mi oldugunu anlamanin standart bir yolu: katsayinin ISARETI,
    hangi DMU cikarilirsa cikarilsin ayni kaliyor mu diye bakmaktir.

    sonuc_tablo: panel_sonuc["oneri"]["sonuc_tablo"] degeri, orn. "pooled_robust",
                 "fe_clustered", "re_robust" -- hangi model+SE kombinasyonunun
                 tekrar tekrar tahmin edilecegini belirler.
    alpha: katsayi "hala anlamli mi" sayimi icin esik (varsayilan 0.10, app.py'deki
           "Katsayilarin Verimlilige Etkisi" tablosuyla tutarli).

    Returns: dict -- yeterli_veri, model_tipi, se_tipi, ozet_df (index=degisken:
             tam_ornek_katsayi, min/max/std_katsayi, yon_tutarliligi_%,
             anlamli_kalan_%, kararlilik: Saglam/Orta/Kirilgan),
             detay_df (index=[degisken,cikarilan_dmu]: katsayi, p_degeri),
             basarisiz_dmular (cikarilinca model kurulamayan DMU'lar)
    """
    model_tipi, se_tipi = sonuc_tablo.split("_", 1)
    entities = list(panel_df.index.get_level_values("entity").unique())

    if len(entities) < 4:
        return {
            "yeterli_veri": False,
            "mesaj": f"Leave-one-out kararlilik testi icin en az 4 DMU gerekli (mevcut: {len(entities)}).",
        }

    def _fit(alt_df: pd.DataFrame):
        y = alt_df[bagimli]
        X = alt_df[bagimsizlar]
        cluster_kw = {"cluster_entity": True} if se_tipi == "clustered" else {}
        if model_tipi == "pooled":
            Xc = add_constant(X)
            return PooledOLS(y, Xc).fit(cov_type=se_tipi, **cluster_kw)
        elif model_tipi == "fe":
            return PanelOLS(y, X, entity_effects=True, drop_absorbed=True).fit(cov_type=se_tipi, **cluster_kw)
        else:  # "re"
            Xc = add_constant(X)
            return RandomEffects(y, Xc).fit(cov_type=se_tipi, **cluster_kw)

    tam_model = _fit(panel_df)
    tam_katsayilar = tam_model.params.drop("const", errors="ignore")

    kayitlar_by_deg = {deg: [] for deg in bagimsizlar}
    basarisiz = []
    for cikarilan in entities:
        alt_df = panel_df[panel_df.index.get_level_values("entity") != cikarilan]
        try:
            res = _fit(alt_df)
        except Exception:
            basarisiz.append(cikarilan)
            continue
        for deg in bagimsizlar:
            if deg in res.params.index:
                kayitlar_by_deg[deg].append({
                    "cikarilan_dmu": cikarilan,
                    "katsayi": float(res.params[deg]),
                    "p_degeri": float(res.pvalues[deg]),
                })

    ozet_satirlari = []
    detay_satirlari = []
    for deg in bagimsizlar:
        kayitlar = kayitlar_by_deg[deg]
        if not kayitlar or deg not in tam_katsayilar.index:
            continue
        tam_katsayi = float(tam_katsayilar[deg])
        tam_yon = 1 if tam_katsayi > 0 else -1
        katsayilar = [k["katsayi"] for k in kayitlar]
        p_degerleri = [k["p_degeri"] for k in kayitlar]
        ayni_yon_sayisi = sum(1 for k in katsayilar if (1 if k > 0 else -1) == tam_yon)
        anlamli_kalan_sayisi = sum(1 for p in p_degerleri if p < alpha)
        yon_tutarliligi = ayni_yon_sayisi / len(katsayilar) * 100

        if ayni_yon_sayisi == len(katsayilar):
            kararlilik = "Saglam"
        elif yon_tutarliligi >= 80:
            kararlilik = "Orta"
        else:
            kararlilik = "Kirilgan"

        ozet_satirlari.append({
            "degisken": deg, "tam_ornek_katsayi": round(tam_katsayi, 5),
            "min_katsayi": round(min(katsayilar), 5), "max_katsayi": round(max(katsayilar), 5),
            "std_katsayi": round(float(np.std(katsayilar)), 5),
            "yon_tutarliligi_%": round(yon_tutarliligi, 1),
            "anlamli_kalan_%": round(anlamli_kalan_sayisi / len(p_degerleri) * 100, 1),
            "kararlilik": kararlilik,
        })
        for k in kayitlar:
            detay_satirlari.append({"degisken": deg, **k})

    ozet_df = pd.DataFrame(ozet_satirlari).set_index("degisken")
    detay_df = pd.DataFrame(detay_satirlari).set_index(["degisken", "cikarilan_dmu"])

    return {
        "yeterli_veri": True,
        "model_tipi": model_tipi, "se_tipi": se_tipi,
        "ozet_df": ozet_df, "detay_df": detay_df,
        "basarisiz_dmular": basarisiz,
    }


def aciklayicilik_analizi(panel_df: pd.DataFrame, nihai_res, girdi_cols: list, cikti_cols: list,
                           bagimli: str = "MI") -> dict:
    """
    Panel Analizi'ndeki NIHAI modelin (Pooled/FE/RE, hangisi secildiyse) R^2'sini,
    her bagimsiz degiskenin PAYINA ayirir -- "hangi degisken MI'yi EN IYI
    aciklyor" sorusuna dogrudan, gorsel bir cevap vermek icin.

    YONTEM: Pratt'in Goreli Onem Olcusu (Pratt, 1987) -- regresyon
    literaturunde standart, basit bir teknik:
        Pratt_i = standart_katsayi_i * r_i
    burada standart_katsayi_i = ham_katsayi_i * (std(X_i) / std(y)) ve
    r_i = X_i ile y arasindaki Pearson korelasyonu. Bu olcunun onemli bir
    ozelligi: TUM degiskenlerin Pratt degerlerinin TOPLAMI, modelin R^2'sine
    ESIT olur (OLS icin tam, FE/RE icin yaklasik) -- yani "R^2'nin ne kadari
    hangi degiskenden geliyor" sorusuna DOGRUDAN, TUTARLI bir cevap verir.

    ONEMLI AYRIM -- IKI FARKLI "YON" KAVRAMI KARISTIRILMAMALI:
    1) katsayi_yonu: degiskenin HAM REGRESYON KATSAYISININ isareti -- yani
       "bu degiskeni artirmak MI'yi artirir mi azaltir mi" sorusunun CEVABI.
       Bu, Panel Analizi sekmesindeki "gercek_yon" ile HER ZAMAN AYNIDIR.
    2) katki_turu: PRATT DEGERININ isareti -- degiskenin R^2'ye YAPICI
       (Uyumlu) mu yoksa BASTIRICI (Suppression) mi katki yaptigi. Bu,
       katsayi_yonu ile AYNI OLMAK ZORUNDA DEGILDIR: eger degiskenin ham
       korelasyonu ile regresyon katsayisi ZIT isaretliyse (genelde diger
       degiskenlerle guclu korelasyon -- yuksek VIF -- oldugunda olur),
       Pratt degeri katsayinin isaretinden FARKLI cikabilir. Bu durumda
       degisken bir "bastirici" (suppressor) olarak isaretlenir -- yani
       modele KENDI etkisinden cok, DIGER degiskenlerin kestirimini
       netlestirmek icin katkida bulunuyor demektir.
    ONCEKI SURUMDE bu iki kavram YANLISLIKLA TEK bir "yon" alaninda
    birlestirilmisti -- bu, Panel Analizi sekmesiyle CELISEN sonuclar
    gosterebiliyordu (bir degisken katsayisi negatif oldugu halde burada
    "pozitif" gorunebiliyordu). Artik ayristirildi.

    Returns: dict -- yeterli_veri, r_kare (nihai modelin R^2'si),
             toplam_pratt (Pratt degerlerinin toplami, R^2'ye yakin olmali --
             tutarlilik kontrolu icin), tablo (DataFrame: degisken, tip,
             katsayi, katsayi_yonu, pratt_degeri, pay_yuzde (mutlak, pasta
             icin), katki_turu)
    """
    tum_degiskenler = [d for d in nihai_res.params.index if d != "const"]
    if not tum_degiskenler:
        return {"yeterli_veri": False, "mesaj": "Modelde bagimsiz degisken bulunamadi."}

    std_y = panel_df[bagimli].std()
    if not std_y or std_y < 1e-12:
        return {"yeterli_veri": False, "mesaj": "Bagimli degiskende (MI) varyasyon yok."}

    satirlar = []
    for degisken in tum_degiskenler:
        if degisken not in panel_df.columns:
            continue
        ham_katsayi = float(nihai_res.params[degisken])
        std_x = panel_df[degisken].std()
        r = panel_df[degisken].corr(panel_df[bagimli])
        if pd.isna(r) or not std_x or std_x < 1e-12:
            pratt = 0.0
        else:
            standart_katsayi = ham_katsayi * (std_x / std_y)
            pratt = standart_katsayi * float(r)
        tip = "Girdi" if degisken in girdi_cols else ("Çıktı" if degisken in cikti_cols else "Diğer")
        katsayi_yonu = "Pozitif (MI'yi artırır)" if ham_katsayi >= 0 else "Negatif (MI'yi azaltır)"
        satirlar.append({
            "degisken": degisken, "tip": tip, "katsayi": round(ham_katsayi, 6),
            "katsayi_yonu": katsayi_yonu, "pratt_degeri": round(pratt, 5),
        })

    tablo = pd.DataFrame(satirlar)
    toplam_mutlak = tablo["pratt_degeri"].abs().sum()
    if toplam_mutlak < 1e-12:
        tablo["pay_yuzde"] = 0.0
    else:
        tablo["pay_yuzde"] = (tablo["pratt_degeri"].abs() / toplam_mutlak * 100).round(2)
    tablo["katki_turu"] = tablo["pratt_degeri"].apply(
        lambda v: "Uyumlu (doğrudan katkı)" if v >= 0 else "⚠️ Bastırıcı (suppression)"
    )
    tablo = tablo.sort_values("pay_yuzde", ascending=False).reset_index(drop=True)

    r_kare = float(getattr(nihai_res, "rsquared", float("nan")))

    return {
        "yeterli_veri": True,
        "r_kare": round(r_kare, 4) if pd.notna(r_kare) else None,
        "toplam_pratt": round(float(tablo["pratt_degeri"].sum()), 4),
        "tablo": tablo,
    }
