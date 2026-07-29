# -*- coding: utf-8 -*-
"""
DEA + MALMQUIST + PANEL ANALIZI - WEB ARAYUZU
================================================
Calistirmak icin: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import io
from excel_okuma import excel_oku, donemlere_ayir, VeriDogrulamaHatasi
from pipeline import run_pipeline


def reg_params_table(res):
    """linearmodels regresyon sonucunu tek bir tidy DataFrame'e cevirir."""
    ci = res.conf_int()
    ci.columns = ["Alt CI (%95)", "Ust CI (%95)"]
    tablo = pd.DataFrame({
        "Katsayi": res.params,
        "Std. Hata": res.std_errors,
        "T-istatistigi": res.tstats,
        "P-degeri": res.pvalues,
    }).join(ci)
    return tablo.round(4)


def reg_meta_table(res):
    """R-kare, gozlem sayisi, F-istatistigi gibi ozet bilgileri tek satirlik tabloya cevirir."""
    return pd.DataFrame({
        "R-kare": [round(res.rsquared, 4)],
        "Gozlem sayisi": [int(res.nobs)],
        "F-istatistigi": [round(res.f_statistic.stat, 4)],
        "F p-degeri": [round(res.f_statistic.pval, 4)],
    })


def comparison_tables(comp):
    """Model karsilastirma nesnesini (Pooled/FE/RE) katsayi ve t-istatistigi tablolarina cevirir."""
    katsayi = comp.params.round(4)
    tstat = comp.tstats.round(3)
    ozet = pd.DataFrame({"R-kare": comp.rsquared}).join(comp.f_statistic).round(4)
    return katsayi, tstat, ozet

st.set_page_config(page_title="DEA + Malmquist + Panel Analizi", layout="wide")
st.title("DEA + Gecikmeli Malmquist + Panel Veri Analizi")

with st.expander("Excel sablonu nasil olmali?", expanded=False):
    st.markdown("""
    Sutunlar (uzun format, her satir bir Donem-DMU kombinasyonu):

    | Donem | DMU | Girdi_... | Cikti_... |
    |---|---|---|---|

    - `Girdi_` ile baslayan sutunlar otomatik **girdi**, `Cikti_` ile baslayanlar **cikti** olarak algilanir
    - Istediginiz kadar Girdi_/Cikti_ sutunu ekleyebilirsiniz
    - Donemler Excel'de **kronolojik sirada** olmali (t1, t2, t3... veya 2022, 2023, 2024...)
    - Her donemde ayni DMU seti bulunmali (dengeli panel)
    """)
    ornek = pd.DataFrame({
        "Donem": ["t1", "t1", "t2", "t2"],
        "DMU": ["A1", "A2", "A1", "A2"],
        "Girdi_SimSuresi": [382, 334, 400, 310],
        "Cikti_Hata": [14, 10, 12, 9],
    })
    st.dataframe(ornek, width='stretch')

uploaded = st.file_uploader("Excel dosyanizi yukleyin (.xlsx)", type=["xlsx"])

if uploaded is not None:
    try:
        veri_onizleme = excel_oku(uploaded)
        st.success(
            f"Veri okundu: {len(veri_onizleme['dmu_sirali'])} DMU, "
            f"{len(veri_onizleme['donem_sirali'])} donem, "
            f"{len(veri_onizleme['girdi_cols'])} girdi, {len(veri_onizleme['cikti_cols'])} cikti."
        )

        tum_secenekler = veri_onizleme["girdi_cols"] + veri_onizleme["cikti_cols"]
        bagimsizlar = st.multiselect(
            "Panel regresyonunda bagimsiz degisken olarak kullanilacak sutunlar",
            options=tum_secenekler, default=tum_secenekler,
        )

        if st.button("Analizi Calistir", type="primary"):
            with st.spinner("DEA -> Malmquist -> Panel analizi calistiriliyor..."):
                uploaded.seek(0)
                sonuc = run_pipeline(uploaded, bagimsizlar=bagimsizlar)

            st.session_state["sonuc"] = sonuc

    except VeriDogrulamaHatasi as e:
        st.error(f"Veri dogrulama hatasi: {e}")
    except Exception as e:
        st.error(f"Beklenmeyen hata: {e}")

if "sonuc" in st.session_state:
    sonuc = st.session_state["sonuc"]
    tab_dea, tab_malmquist, tab_panel = st.tabs(["DEA Sonuclari", "Malmquist Sonuclari", "Panel Analizi"])

    with tab_dea:
        donem_sec = st.selectbox("Donem sec", options=sonuc["veri"]["donem_sirali"], key="dea_donem")
        dea_d = sonuc["dea"][donem_sec]
        c1, c2 = st.columns(2)
        with c1:
            st.write("**CCR (theta)**")
            st.dataframe(dea_d["theta_ccr"].round(4))
        with c2:
            st.write("**BCC (theta)**")
            st.dataframe(dea_d["theta_bcc"].round(4))
        st.write("**Olcek Etkinligi**")
        st.dataframe(dea_d["olcek_etkinligi"].round(4))

        st.write("---")
        st.write("**Slack Degerleri (Asama 2 - Maks Slack Modeli)**")
        c3, c4 = st.columns(2)
        with c3:
            st.write("*CCR - Girdi Slack*")
            st.dataframe(dea_d["slack_x_ccr"].round(4), width='stretch')
            st.write("*CCR - Cikti Slack*")
            st.dataframe(dea_d["slack_y_ccr"].round(4), width='stretch')
        with c4:
            st.write("*BCC - Girdi Slack*")
            st.dataframe(dea_d["slack_x_bcc"].round(4), width='stretch')
            st.write("*BCC - Cikti Slack*")
            st.dataframe(dea_d["slack_y_bcc"].round(4), width='stretch')

        st.write("---")
        st.write("**Referans (Peer) Agirliklari - Lambda**")
        c5, c6 = st.columns(2)
        with c5:
            st.write("*CCR lambda (satir=peer, sutun=degerlendirilen DMU)*")
            st.dataframe(dea_d["lambda_ccr"].round(4), width='stretch')
        with c6:
            st.write("*BCC lambda (satir=peer, sutun=degerlendirilen DMU)*")
            st.dataframe(dea_d["lambda_bcc"].round(4), width='stretch')

        dea_csv_buf = io.StringIO()
        pd.concat({
            "theta_ccr": dea_d["theta_ccr"], "theta_bcc": dea_d["theta_bcc"],
            "olcek_etkinligi": dea_d["olcek_etkinligi"],
        }, axis=1).to_csv(dea_csv_buf)
        st.download_button(f"{donem_sec} DEA ozet sonuclarini CSV indir", dea_csv_buf.getvalue(),
                            file_name=f"dea_ozet_{donem_sec}.csv", key="dea_csv_dl")

    with tab_malmquist:
        st.write("**EC / TC / M degerleri (ardisik donem gecisleri)**")
        st.dataframe(sonuc["malmquist"].round(4), width='stretch')

        csv_buf = io.StringIO()
        sonuc["malmquist"].to_csv(csv_buf)
        st.download_button("Malmquist sonuclarini CSV indir", csv_buf.getvalue(),
                            file_name="malmquist_sonuclari.csv")

    with tab_panel:
        p = sonuc["panel_sonuc"]
        st.write("**Korelasyon Matrisi**")
        st.dataframe(p["corr"].round(3))
        st.write("**VIF**")
        st.dataframe(p["vif"].round(3))

        st.write(f"**Poolability F-testi:** stat={p['poolability'].get('stat', 'NA'):.4f}, "
                 f"p={p['poolability'].get('pval', 'NA'):.4f} -> {p['poolability'].get('sonuc','')}")
        st.write(f"**Hausman testi:** chi2={p['hausman']['stat']:.4f}, dof={p['hausman']['dof']}, "
                 f"p={p['hausman']['pval']:.4f}")
        st.write(f"**Secilen model:** {p['secilen_model']}")

        oneri = p["oneri"]
        st.write(f"DMU (kume) sayisi: **{p['n_entities']}**")

        with st.container(border=True):
            st.markdown("**Karar sureci (literatur sirasiyla):**")
            st.write(f"1) {oneri['asama1']}")
            st.write(f"2) {oneri['asama2']}")
            st.write(f"3) {oneri['se_onerisi']}")

        for uyari in oneri["uyarilar"]:
            st.warning(uyari)

        st.success(
            f"**Raporunuz icin asil bakmaniz gereken tablo: \"{oneri['sonuc_basligi']}\".** "
            f"Diger tablolar (Pooled OLS, FE, RE ve alternatif SE tipleri) bu sonuca nasil "
            f"ulasildigini gosteren ara adimlardir."
        )

        st.write("**Pooled OLS**")
        st.dataframe(reg_meta_table(p["pooled"]), width='stretch', hide_index=True)
        st.dataframe(reg_params_table(p["pooled"]), width='stretch')

        st.write("**Fixed Effects (FE)**")
        st.dataframe(reg_meta_table(p["fe"]), width='stretch', hide_index=True)
        st.dataframe(reg_params_table(p["fe"]), width='stretch')

        st.write("**Random Effects (RE)**")
        st.dataframe(reg_meta_table(p["re"]), width='stretch', hide_index=True)
        st.dataframe(reg_params_table(p["re"]), width='stretch')

        tablo_map = {
            "pooled": ("Pooled OLS", p["pooled"]),
            "fe_robust": ("FE - Robust Standart Hatalar", p["fe_robust"]),
            "fe_clustered": ("FE - Clustered Standart Hatalar", p["fe_clustered"]),
            "re_robust": ("RE - Robust Standart Hatalar", p["re_robust"]),
            "re_clustered": ("RE - Clustered Standart Hatalar", p["re_clustered"]),
        }
        nihai_baslik, nihai_res = tablo_map[oneri["sonuc_tablo"]]

        st.write("---")
        st.markdown(f"### ⭐ NIHAI SONUC: {nihai_baslik}")
        if oneri["sonuc_tablo"] != "pooled":
            st.dataframe(reg_meta_table(nihai_res), width='stretch', hide_index=True)
        st.dataframe(reg_params_table(nihai_res), width='stretch')
        st.write("---")

        st.write("**Alternatif SE tipleriyle karsilastirma (bilgi amacli):**")
        if oneri["panel_gerekli"]:
            c_r, c_c = st.columns(2)
            with c_r:
                st.write(f"*{p['secilen_model']} - Robust*")
                st.dataframe(reg_params_table(p["robust"]), width='stretch')
            with c_c:
                st.write(f"*{p['secilen_model']} - Clustered*")
                st.dataframe(reg_params_table(p["clustered"]), width='stretch')
        else:
            st.write(f"*(Bilgi amacli) {p['secilen_model']} modeli - Robust Standart Hatalar*")
            st.dataframe(reg_params_table(p["robust"]), width='stretch')

        st.write("**Model Karsilastirma (Pooled OLS vs FE vs RE)**")
        katsayi_tablo, tstat_tablo, ozet_tablo = comparison_tables(p["comparison"])
        st.write("*Katsayilar*")
        st.dataframe(katsayi_tablo, width='stretch')
        st.write("*T-istatistikleri*")
        st.dataframe(tstat_tablo, width='stretch')
        st.write("*Model ozet istatistikleri*")
        st.dataframe(ozet_tablo, width='stretch')
