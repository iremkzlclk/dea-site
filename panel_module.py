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


def nihai_oneri_belirle(poolability: dict, hausman: dict, secilen_model: str, n_entities: int):
    """
    Literatur sirasina gore (Baltagi, Wooldridge) nihai model/SE onerisini belirler:
      1) Poolability F-testi: Pooled OLS yeterli mi, yoksa birey/DMU etkileri var mi?
      2) (Panel etkisi varsa) Hausman testi: FE mi RE mi? -- dejenere (negatif stat)
         sonuclarda Hausman'a guvenilmez, poolability + teorik gerekceyle karar verilmeli.
      3) SE tipi: Clustered (DMU bazinda) teorik olarak tercih edilir, AMA kume sayisi
         (=DMU sayisi) literaturde onerilen ~30-50 esiginin altindaysa (Cameron & Miller,
         2015; Cameron, Gelbach & Miller, 2008) guvenilirligi dusuktur -- bu durumda
         robust (kumeleme yapmayan) SE ile birlikte raporlanmasi ve temkinli yorumlanmasi
         onerilir.
    Returns: dict -- asama1, asama2, se_onerisi, sonuc_tablo (hangi tabloyu vurgula),
                     sonuc_basligi, uyarilar (liste)
    """
    uyarilar = []

    # ASAMA 1: Pooled OLS yeterli mi?
    pool_pval = poolability.get("pval")
    if pool_pval is None:
        asama1 = "Poolability testi hesaplanamadi -- panel etkisi oldugu varsayilarak devam edildi."
        panel_gerekli = True
    elif pool_pval < 0.05:
        asama1 = (f"Poolability F-testi (p={pool_pval:.4f}) H0'i reddediyor -> DMU'lara ozgu "
                   f"sabit/rastgele etkiler mevcut, Pooled OLS YETERSIZ. Panel modeline (FE/RE) gecilmeli.")
        panel_gerekli = True
    elif pool_pval < 0.10:
        asama1 = (f"Poolability F-testi (p={pool_pval:.4f}) alpha=0.05'te H0'i reddedemiyor -> teknik "
                   f"olarak Pooled OLS yeterli sayilir. ANCAK p-degeri alpha=0.10 sinirinda -- DMU'lara "
                   f"ozgu gizli bir fark ihtimali gozardi edilmemeli, sonuclar temkinli yorumlanmali.")
        panel_gerekli = False
        uyarilar.append(
            f"Poolability p-degeri ({pool_pval:.4f}) alpha=0.05 ile alpha=0.10 arasinda sinirda kaliyor. "
            f"Pooled OLS teknik olarak yeterli sayilsa da, DMU'lara ozgu gizli etkiler tamamen "
            f"disliyor demek degildir -- raporunuzda bu sinir durumu belirtmeniz onerilir."
        )
    else:
        asama1 = (f"Poolability F-testi (p={pool_pval:.4f}) H0'i reddedemiyor -> DMU'lara ozgu "
                   f"anlamli bir etki tespit edilemedi. Bu durumda literatur Pooled OLS'in "
                   f"YETERLI oldugunu, FE/RE'nin gereksiz karmasiklik katabilecegini soyler.")
        panel_gerekli = False
        uyarilar.append(
            "Poolability testine gore Pooled OLS yeterli görünüyor; asagidaki FE/RE secimi "
            "yine de bilgi amacli gosteriliyor, ama ana sonucunuz Pooled OLS olmali."
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
        asama2 = (f"Hausman testi (chi2={h_stat:.4f}, p={h_pval:.4f}) -> "
                  f"{'H0 reddedildi, FE tutarli' if h_pval < 0.05 else 'H0 reddedilemedi, RE tercih edilebilir (daha etkin)'}.")

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
        sonuc_tablo = "pooled"
        sonuc_basligi = "Pooled OLS"

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
    }


def run_panel_analysis(panel_df: pd.DataFrame, bagimli: str, bagimsizlar: list):
    """
    panel_df: index=['entity','time'], sutunlarda bagimli + bagimsizlar bulunmali
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

    poolability = {}
    try:
        f_pool = res_fe.f_pooled
        poolability = {"stat": f_pool.stat, "pval": f_pool.pval,
                        "sonuc": "FE tercih edilmelidir" if f_pool.pval < 0.05 else "Pooled OLS yeterlidir"}
    except Exception as e:
        poolability = {"hata": str(e)}

    h_stat, h_dof, h_pval = hausman_test(res_fe, res_re)
    secilen_model = "FE" if h_pval < 0.05 else "RE"
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

    oneri = nihai_oneri_belirle(poolability, hausman, secilen_model, n_entities)

    return {
        "corr": corr,
        "vif": vif_data,
        "pooled": res_pooled,
        "fe": res_fe,
        "re": res_re,
        "poolability": poolability,
        "hausman": hausman,
        "secilen_model": secilen_model,
        "robust": res_robust,
        "clustered": res_clustered,
        "fe_robust": res_fe_robust, "fe_clustered": res_fe_clustered,
        "re_robust": res_re_robust, "re_clustered": res_re_clustered,
        "comparison": comparison,
        "n_entities": n_entities,
        "oneri": oneri,
    }
