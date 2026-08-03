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
    ortak = [v for v in fe_res.params.index if v in re_res.params.index]
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


def run_panel_analysis(panel_df: pd.DataFrame, bagimli: str, bagimsizlar: list, alpha: float = ALPHA):
    """
    panel_df: index=['entity','time'], sutunlarda bagimli + bagimsizlar bulunmali
    alpha: TUM karar noktalarinda (Poolability, BP-LM, Hausman, katsayi anlamliligi)
           kullanilan ortak anlamlilik esigi (varsayilan: modul-seviyesi ALPHA).
    Returns: dict -- corr, vif, pooled, fe, re, poolability, hausman,
                     secilen_model, robust, clustered, comparison, n_entities, oneri
    """
    y = panel_df[bagimli]
    X = panel_df[bagimsizlar]
    Xc = add_constant(X)

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

    h_stat, h_dof, h_pval = hausman_test(res_fe, res_re)
    secilen_model = "FE" if h_pval < alpha else "RE"
    hausman = {"stat": h_stat, "dof": h_dof, "pval": h_pval}

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
