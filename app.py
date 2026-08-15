# -*- coding: utf-8 -*-
"""
DEA + MALMQUIST + PANEL ANALIZI - WEB ARAYUZU
================================================
Calistirmak icin: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import altair as alt
import streamlit.components.v1 as components
import io
from excel_okuma import excel_oku, donemlere_ayir, VeriDogrulamaHatasi
from dea_module import min_dmu_kontrolu
from panel_module import leave_one_out_kararlilik, aciklayicilik_analizi, degisken_varyans_analizi, run_panel_analysis, korelasyon_ve_vif_hesapla
from pipeline import run_pipeline
from backtest_module import backtest_calistir, rolling_backtest_calistir
from ml_module import gelecek_donem_dea_senaryo
from yorumlama import (
    malmquist_yorum_metni,
    malmquist_donem_ortalamasi,
    dea_aksiyon_tablosu,
    dea_aksiyon_metni,
    panel_aksiyon_analizi,
    panel_aksiyon_metni,
)


def reg_params_table(res):
    """linearmodels regresyon sonucunu tek bir tidy DataFrame'e cevirir.
    Cok kucuk p-degerleri (<0.0001), 4 ondalige yuvarlaninca '0.0000' gorunup
    'gercekten sifir mi' izlenimi yaratmasin diye akademik yazimin standardi
    olan '<0.0001' bicimiyle gosterilir (p GERCEKTE sifir degildir, sadece
    cok kucuktur)."""
    ci = res.conf_int()
    ci.columns = ["Alt CI (%95)", "Ust CI (%95)"]
    tablo = pd.DataFrame({
        "Katsayi": res.params,
        "Std. Hata": res.std_errors,
        "T-istatistigi": res.tstats,
        "P-degeri": res.pvalues,
    }).join(ci)
    tablo = tablo.round(4)
    tablo["P-degeri"] = res.pvalues.apply(
        lambda p: "<0.0001" if p < 0.0001 else f"{p:.4f}"
    )
    return tablo


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


def _katsayi_renklendir(row):
    """Panel etki tablosunda mi_etkisi_yuzde10 isaretine gore satiri renklendirir."""
    if row["mi_etkisi_yuzde10"] > 0:
        renk = "background-color: #d4f7d4"  # yesilimsi -- pozitif etki
    elif row["mi_etkisi_yuzde10"] < 0:
        renk = "background-color: #f7d4d4"  # kirmizimsi -- negatif etki
    else:
        renk = ""
    return [renk] * len(row)


def excel_indirme_verisi(df_or_obj) -> bytes:
    """
    Bir DataFrame'i indirilebilir Excel (.xlsx) bytes'ina cevirir.

    Onceden CSV kullaniliyordu, ama Turkce karakterler (ü, ı, ş, ç, ğ, ö) CSV'de
    Excel tarafindan yanlis kodlamayla acilip bozuk gorunuyordu (orn. "Türbin" ->
    "TÃ¼rbin"), ve Turkce Windows'ta virgul yerine noktali virgul beklendigi icin
    sutunlar tek bir hucreye sikisip kaliyordu. Native Excel formati (.xlsx) bu iki
    sorunu da tamamen ortadan kaldirir -- kodlama/ayirac belirsizligi olmadan
    doğrudan duzgun acilir.
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_or_obj.to_excel(writer, sheet_name="Sonuc")
    return buf.getvalue()


EXCEL_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


st.set_page_config(page_title="ArGe Verimlilik Analiz Platformu", page_icon="📐", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --kurumsal-lacivert: #1F3A5F;
        --kurumsal-lacivert-koyu: #142943;
        --kurumsal-altin: #C9A227;
        --kurumsal-acik-gri: #F7F8FA;
        --kurumsal-kenar: #E2E5EA;
    }

    /* Genel govde metni -- temiz, kurumsal sans-serif */
    html, body, [class*="css"]  {
        font-size: 18px !important;
        font-family: 'Inter', 'Segoe UI', sans-serif !important;
    }

    /* Basliklar -- lacivert renk + zarif serif font */
    h1, h2, h3 {
        font-family: 'Playfair Display', Georgia, serif !important;
        color: var(--kurumsal-lacivert) !important;
    }
    h1 { font-size: 2.4rem !important; padding-bottom: 0.4rem; border-bottom: 3px solid var(--kurumsal-altin); }
    h2 { font-size: 1.9rem !important; }
    h3 { font-size: 1.6rem !important; }
    h4 { font-size: 1.35rem !important; color: var(--kurumsal-lacivert-koyu) !important; }
    h5 { font-size: 1.2rem !important; }

    /* st.caption ile yazilan aciklama metinleri -- varsayilan cok kucuk kaliyor */
    [data-testid="stCaptionContainer"], .stCaption, small {
        font-size: 1.05rem !important;
        line-height: 1.5 !important;
        color: #555 !important;
    }

    /* st.metric -- kart gorunumu (hafif golge, altin aksan cizgisi) */
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--kurumsal-kenar);
        border-left: 4px solid var(--kurumsal-altin);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 6px rgba(31,58,95,0.06);
    }
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        color: var(--kurumsal-lacivert) !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1.05rem !important;
        color: #555 !important;
    }

    /* Tablo (st.dataframe) icindeki yazi */
    [data-testid="stDataFrame"] * {
        font-size: 1.05rem !important;
    }

    /* Sekme (tab) basliklari -- secili sekme lacivert+altin vurgulu */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 2px solid var(--kurumsal-kenar);
    }
    .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"] {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        padding: 12px 20px !important;
        color: #666 !important;
    }
    .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"] div,
    .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"] span {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: var(--kurumsal-lacivert) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: var(--kurumsal-altin) !important;
        height: 3px !important;
    }

    /* Butonlar -- kurumsal lacivert (Streamlit'in varsayilan kirmizisi yerine) */
    .stButton button[kind="primary"], .stButton button[kind="primaryFormSubmit"] {
        background-color: var(--kurumsal-lacivert) !important;
        border-color: var(--kurumsal-lacivert) !important;
        font-size: 1.1rem !important;
        padding: 0.6rem 1.4rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .stButton button[kind="primary"]:hover {
        background-color: var(--kurumsal-lacivert-koyu) !important;
        border-color: var(--kurumsal-lacivert-koyu) !important;
    }
    .stButton button[kind="secondary"], .stDownloadButton button {
        color: var(--kurumsal-lacivert) !important;
        border-color: var(--kurumsal-lacivert) !important;
        font-size: 1.05rem !important;
        padding: 0.55rem 1.2rem !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    .stButton button[kind="secondary"]:hover, .stDownloadButton button:hover {
        background-color: var(--kurumsal-acik-gri) !important;
        border-color: var(--kurumsal-lacivert-koyu) !important;
        color: var(--kurumsal-lacivert-koyu) !important;
    }

    /* Girdi kutulari, radio, selectbox etiketleri */
    .stRadio label, .stSelectbox label, .stNumberInput label, .stTextInput label {
        font-size: 1.05rem !important;
    }

    /* Genisletilebilir (expander) kutulari -- kart gorunumu */
    [data-testid="stExpander"] {
        border: 1px solid var(--kurumsal-kenar) !important;
        border-radius: 10px !important;
        background: white;
    }
    .streamlit-expanderHeader, [data-testid="stExpander"] summary {
        font-size: 1.15rem !important;
        color: var(--kurumsal-lacivert) !important;
        font-weight: 600 !important;
    }

    /* Bilgi/uyari/basari kutulari -- yumusatilmis kurumsal tonlar */
    [data-testid="stAlertContentInfo"], [data-testid="stAlertContentSuccess"],
    [data-testid="stAlertContentWarning"], [data-testid="stAlertContentError"] {
        font-size: 1.1rem !important;
    }
    div[data-baseweb="notification"] {
        border-radius: 10px !important;
    }

    /* st.container(border=True) kutulari -- hafif golgeli kart gorunumu */
    div[data-testid="stVerticalBlockBorderWrapper"] > div[style*="border"] {
        border-radius: 10px !important;
        box-shadow: 0 1px 4px rgba(31,58,95,0.05);
    }
</style>
""", unsafe_allow_html=True)

st.title("ArGe Verimlilik Analiz Platformu")
st.markdown(
    "<p style='font-size:1.2rem; color:#666; margin-top:-0.09rem; font-family:Inter,sans-serif;'>"
    "DEA · Malmquist Endeksi · Panel Veri Analizi</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## ⚙️ Kontrol Paneli")
    st.caption("Veri yükleyin, ayarları yapın ve analizi başlatın. Sonuçlar sağdaki sekmelerde görünecek.")

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
        st.dataframe(ornek, use_container_width=True)

    uploaded = st.file_uploader("Excel dosyanizi yukleyin (.xlsx)", type=["xlsx"])

    if uploaded is not None:
        try:
            veri_onizleme = excel_oku(uploaded)
            st.success(
                f"Veri okundu: {len(veri_onizleme['dmu_sirali'])} DMU, "
                f"{len(veri_onizleme['donem_sirali'])} donem, "
                f"{len(veri_onizleme['girdi_cols'])} girdi, {len(veri_onizleme['cikti_cols'])} cikti."
            )

            # --- ASAMA 1: DEA + Malmquist'te kullanilacak girdi/ciktilar ---
            st.markdown("#### 1️⃣ DEA + Malmquist'te kullanilacak degiskenler")
            dea_girdiler = st.multiselect(
                "DEA girdileri (Girdi_...)",
                options=veri_onizleme["girdi_cols"],
                default=veri_onizleme["girdi_cols"],
                key="dea_girdi_secim",
            )
            dea_ciktilar = st.multiselect(
                "DEA ciktilari (Cikti_...)",
                options=veri_onizleme["cikti_cols"],
                default=veri_onizleme["cikti_cols"],
                key="dea_cikti_secim",
            )

            # --- Minimum DMU sayisi kontrolu -- SECILEN girdi/cikti sayisina gore ---
            if dea_girdiler and dea_ciktilar:
                kontrol = min_dmu_kontrolu(
                    len(dea_girdiler), len(dea_ciktilar), len(veri_onizleme["dmu_sirali"]),
                )
                if kontrol["seviye"] == "yeterli":
                    st.info(
                        f"ℹ️ DMU sayınız ({kontrol['n_dmu']}), literatürdeki yaygın kural olan "
                        f"n ≥ max(girdi×çıktı, 3×(girdi+çıktı)) = **{kontrol['onerilen_siki']}** eşiğini "
                        f"karşılıyor ({kontrol['n_girdi']} girdi × {kontrol['n_cikti']} çıktı için)."
                    )
                elif kontrol["seviye"] == "asgari":
                    st.warning(
                        f"⚠️ DMU sayınız ({kontrol['n_dmu']}), literatürdeki daha sıkı kuralı "
                        f"(n ≥ {kontrol['onerilen_siki']} = max(girdi×çıktı, 3×(girdi+çıktı))) karşılamıyor, "
                        f"ancak daha gevşek asgari kuralın (n ≥ 2×(girdi+çıktı) = {kontrol['onerilen_gevsek']}) "
                        f"üzerinde. Modelin ayrım gücü (discriminatory power) sınırlı olabilir — etkin DMU "
                        f"sayısı orantısız yüksek çıkabilir."
                    )
                else:
                    st.error(
                        f"🚨 DMU sayınız ({kontrol['n_dmu']}), literatürdeki asgari kuralın bile "
                        f"(n ≥ 2×(girdi+çıktı) = {kontrol['onerilen_gevsek']}) altında kalıyor "
                        f"({kontrol['n_girdi']} girdi, {kontrol['n_cikti']} çıktı için). Bu durumda DEA'nın "
                        f"ayrım gücü ciddi şekilde zayıflar; çoğu DMU yapay olarak 'etkin' çıkabilir. Girdi/çıktı "
                        f"sayısını azaltmayı ya da (mümkünse) DMU sayısını artırmayı değerlendirin."
                    )

            # --- ASAMA 2: Panel analizinde kullanilacak girdiler ---
            # Havuz SADECE 1. asamada SECILMEMIS Girdi_ sutunlarindan olusur --
            # boylece ayni girdi hem DEA'da hem panelde asla kullanilamaz
            # (ayrilabilirlik varsayimi -- Simar & Wilson, 2007 -- otomatik korunur).
            st.markdown("#### 2️⃣ Panel analizinde kullanilacak girdiler")
            panel_havuzu = [g for g in veri_onizleme["girdi_cols"] if g not in dea_girdiler]
            if not panel_havuzu:
                st.error(
                    "🚨 Tum Girdi_ sutunlari 1. asamada DEA icin secildi -- panel "
                    "analizinde kullanilacak girdi kalmadi. Panel icin en az bir "
                    "girdiyi 1. asamadan CIKARIP burada secilebilir hale getirin."
                )
                panel_girdiler = []
            else:
                st.caption(
                    f"Sadece 1. asamada SECILMEYEN girdiler burada listelenir "
                    f"(DEA'da kullanilanlar: {', '.join(dea_girdiler) if dea_girdiler else '—'})."
                )
                panel_girdiler = st.multiselect(
                    "Panel regresyonu girdileri",
                    options=panel_havuzu,
                    default=panel_havuzu,
                    key="panel_girdi_secim",
                )

            calistirma_hazir = bool(dea_girdiler and dea_ciktilar and panel_girdiler)
            if st.button("Analizi Calistir", type="primary", disabled=not calistirma_hazir):
                with st.spinner("DEA -> Malmquist -> Panel analizi calistiriliyor..."):
                    uploaded.seek(0)
                    sonuc = run_pipeline(
                        uploaded,
                        dea_girdiler=dea_girdiler,
                        dea_ciktilar=dea_ciktilar,
                        panel_girdiler=panel_girdiler,
                    )

                st.session_state["sonuc"] = sonuc
                # Onceki dosyaya ait alt-sekme sonuclarini temizle -- aksi halde farkli bir
                # DMU/donem setine sahip yeni bir dosya yuklendiginde, eski sonuclar (gelecek
                # senaryosu, backtest, kararlilik testi) yeni DMU/degisken listesiyle uyusmayip
                # KeyError'a yol acabilir (secim kutulari yeni veriden, sonuc eskisinden gelir).
                for eski_anahtar in ["backtest", "rolling_backtest", "kararlilik", "ml_sonuc"]:
                    st.session_state.pop(eski_anahtar, None)

        except VeriDogrulamaHatasi as e:
            st.error(f"Veri dogrulama hatasi: {e}")
        except Exception as e:
            st.error(f"Beklenmeyen hata: {e}")

if "sonuc" not in st.session_state:
    st.info(
        "👈 Başlamak için soldaki kontrol panelinden bir Excel dosyası yükleyip "
        "**Analizi Çalıştır**'a basın. Sonuçlar burada, sekmeler halinde görünecek."
    )


if "sonuc" in st.session_state:
    sonuc = st.session_state["sonuc"]
    # DIKKAT: bunlar Excel'deki TUM Girdi_/Cikti_ sutunlari degil, kullanicinin
    # 1. asamada DEA icin SECTIGI alt kumedir -- asagidaki tum etiketleme/
    # kararlilik/VIF islemleri SADECE gercekten DEA'da kullanilanlara gore yapilmali.
    girdi_cols = sonuc["dea_girdiler"]
    cikti_cols = sonuc["dea_ciktilar"]
    panel_girdiler = sonuc["panel_girdiler"]

    tab_dea, tab_malmquist, tab_panel, tab_aciklayici, tab_backtest, tab_ml = st.tabs(
        ["DEA Sonuclari", "Malmquist Sonuclari", "Panel Analizi", "Açıklayıcılık",
         "Backtest (Model Dogrulama)", "ML Tahmin"]
    )

    with tab_dea:
        st.markdown("### 📊 DEA (Veri Zarflama Analizi) Sonuçları")
        st.markdown(
            "Bu bölüm, her **DMU**'nun (değerlendirilen birimin — örn. bir parça/süreç) kendi "
            "kaynaklarını (girdi) ne kadar etkin kullanarak çıktı ürettiğini gösterir. Sistem, "
            "seçtiğiniz dönemdeki her DMU'yu, **benzer girdi/çıktı yapısına sahip diğer DMU'larla** "
            "karşılaştırıp bir *etkinlik skoru* (theta) üretir. Amaç: hangi birimlerin zaten en iyi "
            "performansı gösterdiğini (diğerlerine referans/örnek olan DMU'lar) ve etkin olmayan "
            "birimlerin ne kadar iyileşme potansiyeli taşıdığını ortaya koymaktır."
        )
        donem_sec = st.selectbox("Donem sec", options=sonuc["veri"]["donem_sirali"], key="dea_donem")
        dea_d = sonuc["dea"][donem_sec]
        n_dmu_donem = len(sonuc["veri"]["dmu_sirali"])

        # --- Minimum DMU sayisi kontrolu (secilen donem baglaminda tekrar goster) ---
        kontrol = min_dmu_kontrolu(len(girdi_cols), len(cikti_cols), n_dmu_donem)
        if kontrol["seviye"] == "yeterli":
            st.caption(
                f"✅ DMU sayısı ({kontrol['n_dmu']}) literatür eşiğini (n ≥ {kontrol['onerilen_siki']}) karşılıyor."
            )
        elif kontrol["seviye"] == "asgari":
            st.caption(
                f"⚠️ DMU sayısı ({kontrol['n_dmu']}) sadece asgari kuralı (n ≥ {kontrol['onerilen_gevsek']}) "
                f"karşılıyor, siki kurali (n ≥ {kontrol['onerilen_siki']}) karsilamiyor -- ayrim gucu sinirli olabilir."
            )
        else:
            st.caption(
                f"🚨 DMU sayısı ({kontrol['n_dmu']}) asgari kuralin (n ≥ {kontrol['onerilen_gevsek']}) altinda."
            )

        st.caption(
            "**Theta (θ)** her DMU'nun etkinlik skorudur: θ=1.00 -> DMU tam etkin (kendi kaynaklarini "
            "en iyi kullanan referans birimlerden biri); θ<1.00 -> DMU'nun mevcut ciktiyi ayni "
            "kaynaklarin (θ) oraninda azaltarak da uretebilecegi anlamina gelir (ne kadar dusukse o "
            "kadar fazla kaynak israfi var demektir). **CCR** sabit olcege gore getiri varsayar; "
            "**BCC** olcek buyuklugunun etkisini de hesaba katar (daha esnek, θ_BCC ≥ θ_CCR)."
        )
        c1, c2 = st.columns(2)
        with c1:
            st.write("**CCR (theta)**")
            st.dataframe(dea_d["theta_ccr"].round(4))
        with c2:
            st.write("**BCC (theta)**")
            st.dataframe(dea_d["theta_bcc"].round(4))
        st.write("**Olcek Etkinligi**")
        st.caption(
            "Olcek Etkinligi = θ_CCR / θ_BCC. 1.00'e yakinsa DMU'nun buyuklugu (olcegi) uygun; "
            "1.00'den belirgin sekilde dusukse DMU ya cok kucuk ya da cok buyuk olcekte calisiyor "
            "olabilir (optimal olcekten uzak)."
        )
        st.dataframe(dea_d["olcek_etkinligi"].round(4))

        st.write("---")
        st.write("**Slack Degerleri (Asama 2 - Maks Slack Modeli)**")
        st.caption(
            "Theta, tum girdileri/ciktilari AYNI ORANDA kucultup/buyuterek bulunan etkinlik skorudur. "
            "Ama bazen theta uygulandiktan sonra bile HALA fazladan azaltilabilecek belirli bir girdi "
            "ya da HALA artirilabilecek belirli bir cikti kalabilir -- iste bu ek miktara **slack** "
            "(gevseklik) denir. Girdi slack'i: theta kadar kucultuldukten sonra o girdiden hala ne "
            "kadar azaltilabilecegi. Cikti slack'i: o ciktidan hala ne kadar daha artirilabilecegi."
        )
        c3, c4 = st.columns(2)
        with c3:
            st.write("*CCR - Girdi Slack*")
            st.dataframe(dea_d["slack_x_ccr"].round(4), use_container_width=True)
            st.write("*CCR - Cikti Slack*")
            st.dataframe(dea_d["slack_y_ccr"].round(4), use_container_width=True)
        with c4:
            st.write("*BCC - Girdi Slack*")
            st.dataframe(dea_d["slack_x_bcc"].round(4), use_container_width=True)
            st.write("*BCC - Cikti Slack*")
            st.dataframe(dea_d["slack_y_bcc"].round(4), use_container_width=True)

        st.write("---")
        st.markdown("### 📋 Yorum: Etkin Olmayan DMU'lar Nasıl Etkin Hale Gelir?")
        vrs_secim = st.radio(
            "Hangi model uzerinden yorumlansin?", ["CCR", "BCC"],
            horizontal=True, key="dea_yorum_model",
        )
        vrs = (vrs_secim == "BCC")

        X_donem = sonuc["X"][donem_sec]
        Y_donem = sonuc["Y"][donem_sec]
        aksiyon_tablosu = dea_aksiyon_tablosu(dea_d, X_donem, Y_donem, vrs=vrs)
        st.dataframe(aksiyon_tablosu, use_container_width=True)

        etkin_olmayanlar = aksiyon_tablosu[~aksiyon_tablosu["etkin_mi"]]
        if etkin_olmayanlar.empty:
            st.success(f"Bu donemde ({vrs_secim} modeline gore) tum DMU'lar etkin (theta=1.00).")
        else:
            dmu_sec = st.selectbox(
                "Etkin olmayan bir DMU secin", options=list(etkin_olmayanlar.index), key="dea_yorum_dmu",
            )
            st.markdown(dea_aksiyon_metni(aksiyon_tablosu.loc[dmu_sec], girdi_cols, cikti_cols))

        dea_ozet_df = pd.concat({
            "theta_ccr": dea_d["theta_ccr"], "theta_bcc": dea_d["theta_bcc"],
            "olcek_etkinligi": dea_d["olcek_etkinligi"],
        }, axis=1)
        st.download_button(
            f"{donem_sec} DEA ozet sonuclarini Excel indir", excel_indirme_verisi(dea_ozet_df),
            file_name=f"dea_ozet_{donem_sec}.xlsx", mime=EXCEL_MIME, key="dea_csv_dl",
        )

    with tab_malmquist:
        st.markdown("### 📈 Malmquist Verimlilik Endeksi Sonuçları")
        st.markdown(
            "DEA sekmesi size TEK BİR dönem için 'şu an ne kadar etkin' bilgisini veriyordu. Bu "
            "bölüm ise bunu **zaman içine** taşır: bir DMU'nun bir dönemden sonrakine geçerken "
            "verimliliğinin arttığını mı azaldığını mı, ve bu değişimin ne kadarının *kendi "
            "performansını iyileştirmesinden* (EC), ne kadarının *genel teknoloji/yöntem "
            "ilerlemesinden* (TC) kaynaklandığını gösterir."
        )
        st.write("**EC / TC / M degerleri (ardisik donem gecisleri)**")
        st.caption(
            "**EC (Etkinlik Degisimi):** DMU'nun kendi potansiyel sinirina (frontier) ne kadar "
            "yaklastigi/uzaklastigi -- yonetim/operasyonel iyilesme sinyalidir. **TC (Teknoloji "
            "Degisimi):** en iyi uygulama sinirinin kendisinin zaman icinde ne kadar ilerledigi -- "
            "genelde arac/yontem degisikliginden kaynaklanir, tum DMU'lari ayni sekilde etkiler. "
            "**M (Malmquist Endeksi = EC × TC):** toplam verimlilik degisimi. M>1 -> verimlilik "
            "artmis; M<1 -> azalmis; M=1 -> degisim yok."
        )
        st.dataframe(sonuc["malmquist"].round(4), use_container_width=True)

        st.write("---")
        st.write("**Donem Bazinda Ortalama EC / TC / M (tum DMU'lar uzerinden, geometrik ortalama)**")
        gecisli_donemler = sonuc["veri"]["donem_sirali"][:-1]
        donem_ort = malmquist_donem_ortalamasi(sonuc["malmquist"], donem_sirasi=gecisli_donemler)
        st.dataframe(donem_ort, use_container_width=True)
        st.caption(
            "Not: Ortalama, geometrik ortalama olarak hesaplanmistir -- EC/TC/M birer indeks (oran) "
            "oldugu icin bu, Malmquist literaturunde standart pratiktir (aritmetik ortalama yanıltıcı olabilir)."
        )

        st.write("---")
        st.markdown("### 📈 Yorum")
        malmquist_df = sonuc["malmquist"]
        secim = st.selectbox(
            "Birim / dönem geçişi seç", options=list(malmquist_df.index),
            format_func=lambda x: f"{x[0]}  ({x[1]} → sonraki dönem)", key="malmquist_yorum_sec",
        )
        st.markdown(malmquist_yorum_metni(malmquist_df.loc[secim]))

        st.download_button(
            "Malmquist sonuclarini Excel indir", excel_indirme_verisi(sonuc["malmquist"]),
            file_name="malmquist_sonuclari.xlsx", mime=EXCEL_MIME, key="malmquist_csv_dl",
        )

    with tab_panel:
        st.markdown("### 📐 Panel Veri Analizi Sonuçları")
        st.markdown(
            "Bu bölüm, kullanıcının seçtiği **çevresel (DEA dışı) değişkenler** ile "
            "Malmquist verimlilik endeksi (MI) arasındaki **istatistiksel ilişkiyi** inceler. Amaç: "
            "'hangi değişkenler verimliliği artırıyor/azaltıyor, ne yönde ve ne kadar güçlü' sorusuna "
            "cevap vermektir. Bunun için önce en uygun istatistiksel modeli (Pooled OLS / Sabit "
            "Etkiler / Rastgele Etkiler) seçer, sonra bu modelin ne kadar güvenilir olduğunu (katsayı "
            "kararlılığı gibi ek testlerle) sınar."
        )
        p = sonuc["panel_sonuc"]
        # bagimsizlar: panel regresyonunda GERCEKTEN kullanilan degisken listesi
        # (nihai modelin kendi parametrelerinden dinamik olarak alinir --
        # kullanicinin secim kutusundan girdigi liste ile birebir ayni olmali).
        panel_bagimsizlar = [v for v in p["pooled"].params.index if v != "const"]
        teshis_vif = korelasyon_ve_vif_hesapla(sonuc["panel_df"], "MI", panel_bagimsizlar)

        st.write("**Korelasyon Matrisi**")
        st.caption(
            "Degiskenler arasindaki ikili (pairwise) dogrusal iliski. 1.0'a "
            "yakin (pozitif ya da negatif) degerler, iki degiskenin neredeyse "
            "AYNI bilgiyi tasidigini gosterir -- VIF tablosundaki yuksek "
            "degerlerin HANGI degisken CIFTINDEN kaynaklandigini burada "
            "gorebilirsiniz. |r| ≥ 0.8 genellikle dikkat esigi kabul edilir."
        )
        korelasyon_goster = teshis_vif["corr"].round(3)

        def _kor_renk(v):
            if pd.isna(v):
                return ""
            # matplotlib GEREKTIRMEYEN, elle kirmizi(-1)/beyaz(0)/mavi(+1) gradyani
            v = max(-1.0, min(1.0, float(v)))
            if v >= 0:
                r, g, b = int(255 - 155 * v), int(255 - 155 * v), 255
            else:
                r, g, b = 255, int(255 + 155 * v), int(255 + 155 * v)
            return f"background-color: rgb({r},{g},{b})"

        try:
            styler = korelasyon_goster.style.map(_kor_renk)   # pandas >= 2.1
        except AttributeError:
            styler = korelasyon_goster.style.applymap(_kor_renk)  # pandas < 2.1 (eski surumler)
        st.dataframe(styler, use_container_width=True)

        st.write("**VIF**")
        st.caption(
            "VIF (Variance Inflation Factor): bir degiskenin, MODELDEKI DIGER degiskenler tarafindan "
            "ne kadar 'aciklanabildigini' gosterir. VIF=1 -> digerleriyle hic ortusmuyor (ideal); "
            "VIF≥5 literaturde genellikle 'coklu dogrusal baglanti sorunu var' esigi olarak kabul "
            "edilir -- bu durumda o degiskenin KENDI etkisini digerlerinden ayirt etmek zorlasir. "
            "(Bu tablo, panel regresyonundan (zaman-sabit değişkenler nedeniyle) çıkarılmış "
            "olabilecek girdiler dahil, TÜM girdileri gösterir -- teşhis amaçlıdır.)"
        )
        st.dataframe(teshis_vif["vif"].round(3))

        st.caption(
            "Asagidaki uc test, panel veriye hangi modelin (Pooled OLS / Sabit Etkiler-FE / Rastgele "
            "Etkiler-RE) en uygun oldugunu SIRAYLA belirler:"
        )
        st.caption(
            "**1) Poolability F-testi:** DMU'lar arasinda sabit bir fark var mi, yoksa hepsi ayni "
            "davranisi mi sergiliyor? (p<0.05 ise fark var -> Pooled OLS yetersiz)"
        )
        st.write(f"**Poolability F-testi (Pooled vs FE):** stat={p['poolability'].get('stat', 'NA'):.4f}, "
                 f"p={p['poolability'].get('pval', 'NA'):.4f} -> {p['poolability'].get('sonuc','')}")
        st.caption(
            "**2) Breusch-Pagan LM testi:** Ayni soruyu (DMU'lara ozgu bir etki var mi) farkli bir "
            "istatistiksel yontemle sinar -- Poolability testinin tamamlayicisidir."
        )
        if "pval" in p.get("bp_lm", {}):
            st.write(f"**Breusch-Pagan LM testi (Pooled vs RE):** stat={p['bp_lm']['stat']:.4f}, "
                     f"p={p['bp_lm']['pval']:.4f} (N={p['bp_lm']['N']}, T={p['bp_lm']['T']})")
        else:
            st.write(f"**Breusch-Pagan LM testi:** hesaplanamadi ({p.get('bp_lm', {}).get('hata', 'bilinmeyen hata')})")
        st.caption(
            "**3) Hausman testi:** Eger DMU'lara ozgu bir etki varsa, bu etki aciklayici degiskenlerle "
            "iliskili mi degil mi sorusunu sinar -> Sabit Etkiler (FE) mi, Rastgele Etkiler (RE) mi "
            "daha uygun oldugunu belirler (p<0.05 -> FE, p≥0.05 -> RE tercih edilebilir)."
        )
        st.write(f"**Hausman testi:** chi2={p['hausman']['stat']:.4f}, dof={p['hausman']['dof']}, "
                 f"p={p['hausman']['pval']:.4f}")
        st.write(f"**Hausman'a göre FE/RE seçimi:** {p['secilen_model']}")

        gercek_nihai_tablo = p["oneri"]["sonuc_tablo"]  # "pooled_robust"/"pooled_clustered"/"fe_..."/"re_..."
        gercek_nihai_tip = (
            "Pooled OLS" if gercek_nihai_tablo.startswith("pooled")
            else ("FE" if gercek_nihai_tablo.startswith("fe") else "RE")
        )
        if gercek_nihai_tip != p["secilen_model"]:
            st.info(
                f"ℹ️ **Dikkat:** Hausman testi FE/RE arasında **{p['secilen_model']}**'yi işaret etti, "
                f"ama bu adım **ikincil** -- Poolability/BP-LM testleri (1. adım) 'Pooled OLS zaten "
                f"yeterli' dediği için, aşağıdaki **⭐ NİHAİ SONUÇ tablosu {gercek_nihai_tip}** olarak "
                f"belirlendi. Bu iki satır çelişmiyor -- sadece FARKLI adımların sonuçları; kararınızı "
                f"her zaman **⭐ NİHAİ SONUÇ** başlıklı tabloya göre verin."
            )

        if p.get("zaman_sabit_tamamen_disarida"):
            st.info(
                f"ℹ️ **{', '.join(p['zaman_sabit_tamamen_disarida'])}** zaman içinde (neredeyse) "
                f"hiç değişmediği için, analizin **tamamından** (Pooled OLS, FE, RE, Hausman, "
                f"⭐ NİHAİ SONUÇ tablosu dahil) **tamamen çıkarıldı** -- hangi model seçilirse "
                f"seçilsin bu değişken(ler) hiçbir tabloda görünmeyecek. Verinizi silmenize gerek "
                f"kalmadan, elle çıkarıp tekrar denediğinizdeki temiz sonuç burada otomatik elde edildi."
            )

        if p.get("hausman_dejenerelik_nedeniyle_cikarilan"):
            st.warning(
                f"⚠️ **{', '.join(p['hausman_dejenerelik_nedeniyle_cikarilan'])}** zaman-sabit "
                f"OLMADIĞI HALDE, Hausman testini negatif/dejenere hale getirdiği tespit edildi "
                f"(muhtemelen başka bir değişkenle çok yakın hareket ediyor). Bu değişken(ler), "
                f"testi geçerli/güvenilir hale getirmek için analizin **tamamından** otomatik "
                f"olarak çıkarıldı -- hiçbir tabloda görünmeyecek."
            )

        if p["hausman"]["stat"] < 0 or (p["hausman"]["stat"] < 1e-6 and p["hausman"]["pval"] > 0.999):
            st.warning(
                "⚠️ Klasik Hausman istatistiği negatif ya da dejenere (≈0) çıktı -- bu, genelde "
                "zaman içinde hiç değişmeyen bir girdi/çıktı değişkeninin (ya da başka bir sayısal "
                "istikrarsızlığın) testi bozduğunun işareti. Aşağıdaki Mundlak testine güvenin."
            )

        mh = p.get("mundlak_hausman", {})
        with st.expander("🔬 Alternatif: Mundlak (regresyon-tabanlı) Hausman testi"):
            st.caption(
                "Klasik Hausman testi, iki AYRI modelin (FE ve RE) kovaryans matrisini birbirinden "
                "çıkararak çalışır -- bu çıkarma işlemi, özellikle zaman-sabit değişkenler varsa, "
                "matematiksel olarak negatif/dejenere bir sonuç üretebilir. Bu alternatif test "
                "(Mundlak, 1978), TEK BİR modelin (RE'nin, DMU-ortalamaları eklenerek genişletilmiş "
                "hali) kendi kovaryans matrisini kullanır -- bu yüzden **hiçbir zaman negatif çıkamaz**, "
                "matematiksel olarak garantilidir. Sonuçları farklıysa, bu teste öncelik verin."
            )

            st.markdown("##### Değişken Varyans Analizi — hangi değişken ne kadar zaman-sabit?")
            st.caption(
                "**within_orani** sütunu: bu değişkenin DMU-içi (zaman içi) varyansının, toplam "
                "varyansına oranı. 1'e yakınsa değişken zaman içinde gerçekten değişiyor (sağlam); "
                "0'a yakınsa neredeyse tamamen DMU'lar arası farktan oluşuyor, zaman içinde neredeyse "
                "hiç değişmiyor (Hausman testini bozma riski taşır)."
            )
            va = p.get("varyans_analizi")
            if va is not None and not va.empty:
                st.dataframe(va, use_container_width=True)

            if not mh.get("yeterli_veri"):
                st.error(mh.get("mesaj", "Mundlak testi çalıştırılamadı."))
            else:
                st.write(f"**Mundlak testi:** chi2={mh['stat']:.4f}, dof={mh['dof']}, p={mh['p_value']:.4f}")
                st.write(f"**Bu teste göre seçilen model:** {mh['secilen_model']}")
                if mh["zaman_sabit_degiskenler"]:
                    st.warning(
                        f"⚠️ **Zaman içinde hiç değişmeyen (ya da neredeyse) değişken(ler)** tespit "
                        f"edildi: {', '.join(mh['zaman_sabit_degiskenler'])}. Bu değişken(ler) testin "
                        f"dışında tutuldu (matematiksel olarak güvenilir test edilemezler) -- panel "
                        f"katsayı tablosunda anlamlı çıksalar bile, bu sadece DMU'lar arası "
                        f"farklılaşmayı (between-etki) yansıtır; tek bir DMU'nun bu değişkeni "
                        f"değiştirmesinin etkisine dair kanıt yoktur, senaryo/yatırım analizlerinde "
                        f"kullanılmamalıdır."
                    )
                if mh.get("dusurulen_rank_eksikligi_nedeniyle"):
                    st.info(
                        f"ℹ️ Ek olarak şu değişken(ler), diğer değişkenlerle çok yakın hareket "
                        f"ettiği için (rank eksikliği riski) testten çıkarıldı: "
                        f"{', '.join(mh['dusurulen_rank_eksikligi_nedeniyle'])}."
                    )
                if mh["secilen_model"] != p["secilen_model"]:
                    st.error(
                        f"⚠️ İki test **farklı model** öneriyor (klasik: {p['secilen_model']}, "
                        f"Mundlak: {mh['secilen_model']}) -- klasik Hausman testi güvenilmez "
                        f"olabilir, Mundlak testinin sonucuna öncelik verin."
                    )

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

        st.caption(
            "Asagidaki uc tablo, ayni veriye uc farkli model varsayimiyla (Pooled/FE/RE) bakar. "
            "Her tabloda: **Katsayi** = o degiskenin MI uzerindeki tahmini etkisi (yon ve buyukluk); "
            "**P-degeri** < 0.05 (bazi yerlerde 0.10) ise etki istatistiksel olarak anlamli kabul "
            "edilir; **Alt/Ust CI** ise katsayinin %95 guvenle icinde olduğu araligi gosterir. "
            "Hangisinin 'dogru' oldugunu yukaridaki 3 test (Poolability, BP-LM, Hausman) belirler."
        )
        st.write("**Pooled OLS**")
        st.dataframe(reg_meta_table(p["pooled"]), use_container_width=True, hide_index=True)
        st.dataframe(reg_params_table(p["pooled"]), use_container_width=True)

        st.write("**Fixed Effects (FE)**")
        st.dataframe(reg_meta_table(p["fe"]), use_container_width=True, hide_index=True)
        st.dataframe(reg_params_table(p["fe"]), use_container_width=True)

        st.write("**Random Effects (RE)**")
        st.dataframe(reg_meta_table(p["re"]), use_container_width=True, hide_index=True)
        st.dataframe(reg_params_table(p["re"]), use_container_width=True)

        tablo_map = {
            "pooled_robust": ("Pooled OLS - Robust Standart Hatalar", p["pooled_robust"]),
            "pooled_clustered": ("Pooled OLS - Clustered Standart Hatalar", p["pooled_clustered"]),
            "fe_robust": ("FE - Robust Standart Hatalar", p["fe_robust"]),
            "fe_clustered": ("FE - Clustered Standart Hatalar", p["fe_clustered"]),
            "re_robust": ("RE - Robust Standart Hatalar", p["re_robust"]),
            "re_clustered": ("RE - Clustered Standart Hatalar", p["re_clustered"]),
        }
        nihai_baslik, nihai_res = tablo_map[oneri["sonuc_tablo"]]

        st.write("---")
        st.markdown(f"### ⭐ NIHAI SONUC: {nihai_baslik}")
        st.dataframe(reg_meta_table(nihai_res), use_container_width=True, hide_index=True)
        st.dataframe(reg_params_table(nihai_res), use_container_width=True)
        st.write("---")

        # --- Katsayilarin verimlilige (MI) etkisi -- yon ve buyukluk gosterimi ---
        st.markdown("### 🎯 Katsayıların Verimliliğe (MI) Etkisi — Yön ve Büyüklük")
        KATSAYI_ALPHA = 0.10  # katsayi anlamliligi icin ayri esik -- model secimi testlerinden (p['alpha']) bagimsiz
        st.caption(
            f"Nihai (yukaridaki '⭐ NIHAI SONUC') modeldeki her degiskenin, ortalama degerinin %10 "
            f"degismesi durumunda MI uzerindeki tahmini etkisi. Anlamlilik esigi: p<{KATSAYI_ALPHA:.2f}. "
            f"Yesil = pozitif (verimlilik artisi), kirmizi = negatif (verimlilik azalisi). Bu degiskenler "
            f"DEA girdi/ciktisi olmadigi (cevresel/harici oldugu) icin DEA teorisinden kaynaklanan bir "
            f"yon beklentisi uygulanmaz; asagidaki tablo sadece katsayinin kendi isaretini ve "
            f"anlamliligini gosterir."
        )
        analiz_df = panel_aksiyon_analizi(
            nihai_res, girdi_cols, cikti_cols, sonuc["panel_df"], alpha=KATSAYI_ALPHA,
        )
        st.dataframe(
            analiz_df.style.apply(_katsayi_renklendir, axis=1),
            use_container_width=True, hide_index=True,
        )

        anlamli_grafik = analiz_df[analiz_df["anlamli_mi"]].set_index("degisken")["mi_etkisi_yuzde10"]
        if not anlamli_grafik.empty:
            st.write(f"*Anlamli (p<{KATSAYI_ALPHA:.2f}) degiskenlerin %10'luk degisim etkisi -- gorsel yon:*")
            st.bar_chart(anlamli_grafik)

        st.markdown(panel_aksiyon_metni(analiz_df))
        st.write("---")

        st.markdown("### 🔒 Katsayı Kararlılığı (Leave-One-DMU-Out)")
        st.caption(
            "Nihai modeldeki her degiskeni, panelden SIRAYLA BIR DMU cikararak yeniden tahmin eder. "
            "Bir iliskinin genel bir orunek mi, yoksa tek bir DMU'nun surukledigi bir yapaylik mi "
            "oldugunu anlamanin standart bir yolu budur: katsayinin ISARETI, hangi DMU cikarilirsa "
            "cikarilsin ayni kaliyor mu diye bakilir. Isaret hicbir cikarmada degismiyorsa iliski "
            "**Saglam**; bir DMU cikarilinca isaret degisiyorsa iliski o DMU'ya asiri bagimli, yani "
            "**Kirilgan** olabilir."
        )
        if st.button("Kararlilik Testini Calistir", key="kararlilik_btn"):
            with st.spinner("Panel modeli her DMU icin sirayla cikarilarak yeniden tahmin ediliyor..."):
                panel_bagimsizlar = [v for v in p["pooled"].params.index if v != "const"]
                st.session_state["kararlilik"] = leave_one_out_kararlilik(
                    sonuc["panel_df"], "MI", panel_bagimsizlar, oneri["sonuc_tablo"],
                )

        if "kararlilik" in st.session_state:
            kr = st.session_state["kararlilik"]
            if not kr["yeterli_veri"]:
                st.error(kr["mesaj"])
            else:
                def _kararlilik_renklendir(row):
                    if row["kararlilik"] == "Saglam":
                        renk = "background-color: #d4f7d4"
                    elif row["kararlilik"] == "Kirilgan":
                        renk = "background-color: #f7d4d4"
                    else:
                        renk = "background-color: #fff3cd"
                    return [renk] * len(row)

                st.dataframe(
                    kr["ozet_df"].reset_index().style.apply(_kararlilik_renklendir, axis=1),
                    use_container_width=True, hide_index=True,
                )
                if kr["basarisiz_dmular"]:
                    st.warning(f"Su DMU'lar cikarildiginda model kurulamadi (atlandi): {kr['basarisiz_dmular']}")

                kirilgan_degiskenler = kr["ozet_df"][kr["ozet_df"]["kararlilik"] == "Kirilgan"]
                if not kirilgan_degiskenler.empty:
                    st.warning(
                        f"⚠️ Su degiskenlerin katsayi isareti, en az bir DMU cikarilinca degisiyor: "
                        f"**{', '.join(kirilgan_degiskenler.index)}**. Bu degiskenlere dayanan yorumlar "
                        f"(ozellikle 'Gelecek Verimlilik Tahmini' sekmesindeki 'Hedefli' siniflandirmasi) "
                        f"temkinli degerlendirilmelidir -- iliski tek bir DMU'nun etkisiyle surukleniyor olabilir."
                    )
                else:
                    st.success(
                        "✅ Butun degiskenlerin katsayi isareti, hicbir DMU tek basina cikarilinca "
                        "degismiyor -- iliskiler bu acidan saglam gorunuyor."
                    )

                with st.expander("Detay: Her DMU cikarildiginda katsayilar"):
                    st.dataframe(kr["detay_df"].reset_index(), use_container_width=True, hide_index=True)

                st.download_button(
                    "Kararlilik testi ozetini Excel indir", excel_indirme_verisi(kr["ozet_df"]),
                    file_name="kararlilik_ozet.xlsx", mime=EXCEL_MIME, key="kararlilik_csv_dl",
                )
        st.write("---")

        st.write("**Alternatif SE tipleriyle karsilastirma (bilgi amacli):**")
        if oneri["panel_gerekli"]:
            c_r, c_c = st.columns(2)
            with c_r:
                st.write(f"*{p['secilen_model']} - Robust*")
                st.dataframe(reg_params_table(p["robust"]), use_container_width=True)
            with c_c:
                st.write(f"*{p['secilen_model']} - Clustered*")
                st.dataframe(reg_params_table(p["clustered"]), use_container_width=True)
        else:
            st.write("*(Bilgi amacli, Hausman'in FE/RE arasindan sectigi model)* "
                      f"{p['secilen_model']} - Robust Standart Hatalar*")
            st.dataframe(reg_params_table(p["robust"]), use_container_width=True)

        st.write(f"*(Bilgi amacli) Pooled OLS - Robust vs Clustered*")
        c_pr, c_pc = st.columns(2)
        with c_pr:
            st.write("*Pooled OLS - Robust*")
            st.dataframe(reg_params_table(p["pooled_robust"]), use_container_width=True)
        with c_pc:
            st.write("*Pooled OLS - Clustered*")
            st.dataframe(reg_params_table(p["pooled_clustered"]), use_container_width=True)

        st.write("**Model Karsilastirma (Pooled OLS vs FE vs RE)**")
        katsayi_tablo, tstat_tablo, ozet_tablo = comparison_tables(p["comparison"])
        st.write("*Katsayilar*")
        st.dataframe(katsayi_tablo, use_container_width=True)
        st.write("*T-istatistikleri*")
        st.dataframe(tstat_tablo, use_container_width=True)
        st.write("*Model ozet istatistikleri*")
        st.dataframe(ozet_tablo, use_container_width=True)

    with tab_backtest:
        st.markdown("### 🎯 Backtest — Panel Modelinin Gercek Tahmin Gucu")
        st.markdown("""
        Bu bolum, **"Gelecek Verimlilik Tahmini"** sekmesindeki senaryo mekanizmasindan
        BAGIMSIZ bir dogrulamadir. DEA'yi tekrar cozmez; sadece panel modelinin kendi
        basina ne kadar isabetli oldugunu, gecmiste zaten bildiginiz bir donem uzerinde
        test eder ("leave-last-period-out"):

        1. Panel modeli, **son gecis donemi HARIC** butun gecmis donemlerle yeniden kurulur
           (ayni Pooled/FE/RE + Hausman + robust/clustered karar zinciri uygulanir).
        2. Bu egitilen modelin katsayilariyla, son gecisin **gercek** (varsayimsal degil)
           girdi/cikti degerleri kullanilarak MI tahmin edilir.
        3. Tahmin, o gecisin **gercekten gerceklesen** MI degeriyle karsilastirilir.
        4. Sonuc, "MI hep aynen kalir" varsayimina dayanan naif bir tahminle de kiyaslanir
           -- modeliniz bu naif tahminden daha iyi mi, sorusuna somut bir cevap verir.

        ⚠️ En az 3 ham veri donemi (yani en az 2 gecis donemi) gerektirir.
        """)

        if st.button("Backtest Calistir", type="primary", key="backtest_calistir_btn"):
            with st.spinner("Panel modeli son donem haric egitiliyor ve dogrulama yapiliyor..."):
                bagimsizlar_bt = [v for v in p["pooled"].params.index if v != "const"]
                st.session_state["backtest"] = backtest_calistir(sonuc["panel_df"], bagimsizlar_bt)

        if "backtest" in st.session_state:
            bt = st.session_state["backtest"]

            if not bt["yeterli_veri"]:
                st.error(bt["mesaj"])
            else:
                st.write("---")
                st.caption(
                    f"Egitim donemleri (gecis indeksi): **{bt['egitim_zamanlari']}** — "
                    f"Test (holdout) donemi: **{bt['holdout_zaman']}** — "
                    f"Egitimde secilen model: **{bt['nihai_baslik']}**"
                )

                m = bt["metrikler"]
                st.markdown(
                    "**Bu sayılar ne anlama geliyor?** Modelinizin, geçmişte zaten bildiğiniz "
                    "bir dönemi (yukarıda belirtilen test dönemini) ne kadar isabetli tahmin "
                    "ettiğini ölçüyoruz. \"MI\" verimlilik endeksinizdir (1.0 = değişim yok, "
                    "1.10 = %10 artış, 0.90 = %10 azalış gibi düşünülebilir)."
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(
                    "Ortalama hata (MAE)", m["MAE"],
                    help=(
                        "Tahminin gerçek değerden ORTALAMA ne kadar saptığı. Örnek: MAE=0.10 "
                        "ise tahminleriniz gerçek MI'den ortalama 0.10 birim (kabaca %10 puan "
                        "gibi düşünebilirsiniz) uzak çıkıyor demektir. Ne kadar düşükse o kadar iyi."
                    ),
                )
                c2.metric(
                    "Büyük hata cezası (RMSE)", m["RMSE"],
                    help=(
                        "MAE'ye benzer, ama büyük sapmaları daha ağır cezalandırır. RMSE, MAE'den "
                        "belirgin şekilde büyükse, bazı tahminlerde tek tük büyük kaçışlar var demektir "
                        "(çoğu tahmin iyi ama birkaçı ciddi yanlış)."
                    ),
                )
                c3.metric(
                    "Yüzdesel hata (MAPE)", f"%{m['MAPE_%']}",
                    help=(
                        "Hatayı yüzde olarak ifade eder. Örnek: MAPE=%14 ise tahminleriniz "
                        "ortalama olarak gerçek degerin %14'u kadar sapiyor demektir. Yuzde "
                        "oldugu icin yorumlamasi en kolay olan olcuttur."
                    ),
                )
                c4.metric(
                    "Yön doğruluğu", f"%{m['yon_dogruluk_%']}",
                    help=(
                        "Modelin, verimliligin ARTACAGINI mi AZALACAGINI mi dogru tahmin ettigi "
                        "DMU'larin orani. %50 = yazi-tura (rastgele tahminle ayni seviye); %80+ "
                        "gibi yuksek bir oran, model buyuklugu tam tutturamasa bile YONU guvenilir "
                        "sekilde yakaladigini gosterir -- bu, pratikte en isinize yarayacak sayidir."
                    ),
                )

                if m["Pearson_r"] is not None:
                    st.caption(
                        f"**Gerçek ile tahmin ne kadar birlikte hareket ediyor:** {m['Pearson_r']} "
                        f"(-1 ile +1 arası bir sayı. +1'e yakınsa tahminleriniz gerçek değerlerle "
                        f"aynı yönde ve orantılı değişiyor demektir; 0'a yakınsa aralarında bir "
                        f"ilişki yok demektir.)"
                    )

                if m["modelin_naiften_iyi_mi"]:
                    st.success(
                        f"✅ Modelinizin ortalama hatasi (MAE={m['MAE']}), \"MI hic degismez\" varsayan "
                        f"naif tahminin hatasindan (MAE={m['naif_baseline_MAE']}) **daha dusuk**. "
                        f"Panel modeliniz naif tahminden daha iyi performans gosteriyor."
                    )
                else:
                    st.warning(
                        f"⚠️ Modelinizin ortalama hatasi (MAE={m['MAE']}), \"MI hic degismez\" varsayan "
                        f"naif tahminin hatasindan (MAE={m['naif_baseline_MAE']}) **daha yuksek veya esit**. "
                        f"Bu, panel modelinizin bu veri setinde ek bir tahmin gucu katmadigini gosteriyor -- "
                        f"sonuclari (ozellikle Gelecek Verimlilik Tahmini sekmesindekileri) yorumlarken "
                        f"temkinli olun."
                    )

                if m["yon_dogruluk_%"] < 60:
                    st.warning(
                        f"⚠️ Yon dogruluk orani (%{m['yon_dogruluk_%']}) yazi-tura (%50) seviyesine yakin -- "
                        f"model, verimliligin artacagini mi azalacagini mi dogru tahmin etmekte de "
                        f"zorlaniyor olabilir."
                    )

                st.write("---")
                st.markdown("#### DMU Bazinda Gercek vs Tahmin")

                def _yon_renklendir(row):
                    renk = "background-color: #d4f7d4" if row["yon_dogru_mu"] else "background-color: #f7d4d4"
                    return [renk] * len(row)

                st.dataframe(
                    bt["tahmin_df"].reset_index().style.apply(_yon_renklendir, axis=1),
                    use_container_width=True, hide_index=True,
                )

                st.caption(
                    "Yesil satirlar: modelin verimlilik ARTIS/AZALIS yonunu dogru tahmin ettigi DMU'lar. "
                    "Kirmizi satirlar: yonun yanlis tahmin edildigi DMU'lar."
                )

                st.download_button(
                    "Backtest sonuclarini Excel indir", excel_indirme_verisi(bt["tahmin_df"]),
                    file_name="backtest_sonuclari.xlsx", mime=EXCEL_MIME, key="backtest_csv_dl",
                )

        st.write("---")
        st.markdown("### 🔁 Rolling Backtest (Tum Mumkun Gecisler)")
        st.markdown("""
        Yukaridaki tek-katli test, SADECE SON gecisi degerlendirir -- bu, TEK BIR
        denemeye dayandigi icin kendisi de yuksek varyansli olabilir. Bu bolum,
        **MUMKUN OLAN HER gecisi** sirayla test icin ayirip (o ana kadarki tum
        gecmisle egitip bir sonraki gercek gecisi tahmin ederek) sonuclarin
        **ortalamasini ve kat-kat DAGILIMINI** gosterir.

        Bu, "bu katsayilara dayanarak (orn. maliyeti X kadar azaltirsam verimlilik
        Y kadar artar diyerek) yatirim kararı alsam, GECMISTE bu ne kadar
        TUTARLI/HAKLI cikardim" sorusuna -- tek bir ornege degil, elinizdeki TUM
        gecmise dayanan -- cok daha guvenilir bir cevap verir. Ham veri donem
        sayiniz arttikca kat sayisi da artar ve bu sonuc gitgide daha anlamli
        hale gelir (T=4 ile sadece 1 kat mumkunken, T=8 ile 5 kat elde edilir).

        ⚠️ En az 4 ham veri donemi (yani en az 3 gecis donemi, 2'si egitim +
        1'i test) gerektirir.
        """)

        if st.button("Rolling Backtest Calistir", type="primary", key="rolling_backtest_btn"):
            with st.spinner("Mumkun olan her gecis icin model kurulup dogrulama yapiliyor (bu biraz surebilir)..."):
                bagimsizlar_rbt = [v for v in p["pooled"].params.index if v != "const"]
                st.session_state["rolling_backtest"] = rolling_backtest_calistir(
                    sonuc["panel_df"], bagimsizlar_rbt,
                )

        if "rolling_backtest" in st.session_state:
            rbt = st.session_state["rolling_backtest"]

            if not rbt["yeterli_veri"]:
                st.error(rbt["mesaj"])
            else:
                st.write("---")
                st.caption(
                    f"Basariyla tamamlanan kat sayisi: **{rbt['kat_sayisi']}** / denenen: "
                    f"{rbt['denenen_kat_sayisi']}. Her kat, o ana kadarki tum gecmisle egitilip "
                    f"bir sonraki GERCEK gecisi tahmin eder."
                )

                om = rbt["ortalama_metrikler"]
                st.markdown(
                    "**Bu sayılar ne anlama geliyor?** Aşağıdaki değerler, tek bir denemeye değil, "
                    "yukarıda listelenen tüm katların ORTALAMASINA dayanır -- yani tek seferlik "
                    "şans/şanssızlık yerine, modelinizin genel eğilimini gösterir."
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(
                    "Ortalama hata (MAE)", om["MAE_ortalama"],
                    help=(
                        f"Tahminin gercek degerden ORTALAMA ne kadar saptigi (katlar arasi ortalama). "
                        f"Ne kadar dusukse o kadar iyi. Katlar arasi degiskenlik (std): {om['MAE_std']} "
                        f"-- bu sayi buyukse, model bazi donemlerde iyi bazilarinda kotu tahmin yapiyor demektir."
                    ),
                )
                c2.metric(
                    "Büyük hata cezası (RMSE)", om["RMSE_ortalama"],
                    help="MAE'ye benzer, ama buyuk sapmalari daha agir cezalandirir (katlar arasi ortalama).",
                )
                c3.metric(
                    "Yüzdesel hata (MAPE)", f"%{om['MAPE_ortalama_%']}",
                    help="Hatayi yuzde olarak ifade eder (katlar arasi ortalama). Yorumlamasi en kolay olcut.",
                )
                c4.metric(
                    "Yön doğruluğu", f"%{om['yon_dogruluk_ortalama_%']}",
                    help=(
                        f"Modelin verimliligin artacagini mi azalacagini mi dogru tahmin ettigi orani "
                        f"(katlar arasi ortalama). %50=yazi-tura seviyesi. Katlar arasi degiskenlik (std): "
                        f"%{om['yon_dogruluk_std']} -- bu sayi buyukse (orn. >15), modelin tutarliligina "
                        f"tek bir katla degil, ancak bu ortalamayla guvenebilirsiniz."
                    ),
                )

                if om["modelin_naiften_iyi_mi"]:
                    st.success(
                        f"✅ Katlar arasi ortalama MAE ({om['MAE_ortalama']}), naif tahminin ortalama "
                        f"hatasindan ({om['naif_baseline_MAE_ortalama']}) **daha dusuk**. "
                        f"{rbt['kat_sayisi']} kattan **{om['kat_basina_naiften_iyi_sayisi']}**'inde model "
                        f"naiften daha iyi cikti."
                    )
                else:
                    st.warning(
                        f"⚠️ Katlar arasi ortalama MAE ({om['MAE_ortalama']}), naif tahminin ortalama "
                        f"hatasindan ({om['naif_baseline_MAE_ortalama']}) **daha yuksek**. "
                        f"{rbt['kat_sayisi']} kattan sadece **{om['kat_basina_naiften_iyi_sayisi']}**'inde "
                        f"model naiften daha iyi cikti. 'Maliyeti X azaltirsam verimlilik Y artar' turu "
                        f"yatirim tavsiyelerini bu sonuca gore TEMKINLI verin."
                    )

                if om["yon_dogruluk_std"] > 15:
                    st.warning(
                        f"⚠️ Yon dogrulugu katlar arasinda buyuk farklilik gosteriyor (std=%{om['yon_dogruluk_std']}) "
                        f"-- modelin tutarliligi donemden doneme degisken, tek bir katla yargilamak yaniltici olurdu."
                    )

                st.write("---")
                st.markdown("#### Kat Bazinda Detay")
                kat_ozet_satirlari = []
                for k in rbt["kat_detaylari"]:
                    kat_ozet_satirlari.append({
                        "Egitim donemleri": str(k["egitim_zamanlari"]), "Test donemi": str(k["test_zamani"]),
                        "MAE": k["metrikler"]["MAE"], "Naif MAE": k["metrikler"]["naif_baseline_MAE"],
                        "Naiften iyi mi": k["metrikler"]["modelin_naiften_iyi_mi"],
                        "Yon Dogruluk (%)": k["metrikler"]["yon_dogruluk_%"],
                    })
                kat_ozet_df = pd.DataFrame(kat_ozet_satirlari)
                st.dataframe(kat_ozet_df, use_container_width=True, hide_index=True)
                st.bar_chart(kat_ozet_df.set_index("Test donemi")["Yon Dogruluk (%)"])

                st.download_button(
                    "Rolling backtest kat ozetini Excel indir", excel_indirme_verisi(kat_ozet_df),
                    file_name="rolling_backtest_ozet.xlsx", mime=EXCEL_MIME, key="rolling_backtest_csv_dl",
                )



    with tab_ml:
        st.markdown("### 🔮 Gelecek Dönem Tahmini")
        st.markdown("""
        **Bu sayfa ne yapar?** Her girdi için **Artır/Azalt** seçip bir yüzde yazarsınız
        ("bu girdiyi %X artırırsam/azaltırsam"). Sistem bu değişikliklerle **yeni, varsayımsal
        bir "sonraki dönem" verisi kurar** -- girdileriniz sizin seçtiğiniz kadar değişir, çıktılar
        ise (doğruluk, prototip sayısı vb.) geçmiş verinizdeki girdi-çıktı ilişkisine göre **otomatik
        olarak** hareket eder (sabit tutulmaz).

        Sonra bu yeni dönem ile **son gerçek döneminiz** arasında **gerçekten DEA + Malmquist
        endeksi çözülür** -- yani sonuç, bir istatistiksel tahmin değil, **DEA'nın kendi
        optimizasyonunun doğrudan çıktısı.** Bu, Malmquist Sonuçları sekmenizdeki hesaplamayla
        **aynı yöntemi** kullanır, sadece "sonraki dönem" gerçek değil, sizin senaryonuz.
        """)

        st.write("---")
        st.markdown("#### 🎛️ Girdilerinizi Ayarlayın")
        st.caption(
            "Her girdi için yön (Artır/Azalt) seçip yüzdeyi yazın. Çıktılar (doğruluk, "
            "prototip sayısı vb.) burada elle seçilemez -- girdi değişikliğinize göre "
            "otomatik olarak hesaplanır, çünkü bunlar dönem sonunda ortaya çıkan sonuçlardır."
        )

        # Zaman-sabit girdilerde Artir/Azalt'i yapisal olarak devre disi birak (ML Tahmin
        # sekmesindeki ayni guvenlik onlemi -- degisken_varyans_analizi kullanarak).
        varyans_analizi_ml = degisken_varyans_analizi(sonuc["panel_df"], girdi_cols)
        zaman_sabit_girdiler_ml = set(
            varyans_analizi_ml[varyans_analizi_ml["within_orani"] < 0.01].index
        )

        girdi_yuzdeleri = {}
        for g in girdi_cols:
            with st.container(border=True):
                st.write(f"**{g}**")
                zaman_sabit_mi = g in zaman_sabit_girdiler_ml
                if zaman_sabit_mi:
                    st.error(
                        "⛔ Bu girdi, zaman içinde neredeyse hiç değişmiyor (DMU'lar arası "
                        "farklı ama her DMU kendi içinde sabit kalıyor). Bu tür girdiler için "
                        "'bunu artırırsam/azaltırsam ne olur' sorusuna dair hiçbir gerçek "
                        "veri kanıtı yok -- bu yüzden Artır/Azalt seçeneği burada **devre "
                        "dışı bırakıldı**."
                    )
                cc1, cc2 = st.columns([1, 1])
                with cc1:
                    yon_secim = st.radio(
                        "Yön", ["Değiştirme", "Artır", "Azalt"], index=0,
                        key=f"ml_yon_{g}", horizontal=True, disabled=zaman_sabit_mi,
                    )
                with cc2:
                    if yon_secim != "Değiştirme" and not zaman_sabit_mi:
                        yuzde_deger = st.number_input(
                            "Yüzde (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0,
                            key=f"ml_yuzde_{g}",
                        )
                    else:
                        yuzde_deger = 0.0
                if zaman_sabit_mi:
                    girdi_yuzdeleri[g] = 0.0
                elif yon_secim == "Artır":
                    girdi_yuzdeleri[g] = yuzde_deger / 100.0
                elif yon_secim == "Azalt":
                    girdi_yuzdeleri[g] = -yuzde_deger / 100.0
                else:
                    girdi_yuzdeleri[g] = 0.0

        hesapla_tiklandi = st.button("Senaryoyu Hesapla (DEA + Malmquist)", type="primary", key="ml_dea_hesapla_btn")

        if hesapla_tiklandi:
            if all(v == 0 for v in girdi_yuzdeleri.values()):
                st.info("Henüz bir girdi için Artır/Azalt seçmediniz -- yukarıdan seçim yapın.")
            else:
                with st.spinner("Yeni dönem verisi kuruluyor ve DEA + Malmquist çözülüyor..."):
                    try:
                        sonuc_dea = gelecek_donem_dea_senaryo(sonuc, girdi_cols, cikti_cols, girdi_yuzdeleri)
                        st.session_state["ml_dea_sonuc"] = {"veri": sonuc_dea, "hata": None}
                    except Exception as e:
                        st.session_state["ml_dea_sonuc"] = {"hata": str(e)}

        if "ml_dea_sonuc" in st.session_state:
            paket = st.session_state["ml_dea_sonuc"]
            if paket.get("hata"):
                st.error(f"❌ Hesaplanamadı. Hata detayı (bunu paylaşırsanız hemen düzeltebilirim):\n\n`{paket['hata']}`")
            else:
                r = paket["veri"]
                st.write("---")
                st.markdown("#### 📈 Bu Senaryonun Etkisi")

                degisim = r["degisim_yuzde"]
                if degisim > 0.5:
                    st.success(f"## ✅ Verimlilik değişimi: %{degisim:+.1f} (artış)")
                elif degisim < -0.5:
                    st.warning(f"## ⚠️ Verimlilik değişimi: %{degisim:+.1f} (azalış)")
                else:
                    st.info(f"## Verimlilik değişimi: %{degisim:+.1f} (pratikte değişim yok)")

                c1, c2, c3 = st.columns(3)
                c1.metric("Etkinlik Değişimi (EC)", r["ortalama_EC"])
                c2.metric("Sınır/Teknoloji Değişimi (TC)", r["ortalama_TC"])
                c3.metric("Toplam Verimlilik (M = EC×TC)", r["ortalama_M"])
                st.caption(
                    "**EC:** senaryonuzdaki DMU'ların, o anki en iyi pratik sınırına göre "
                    "etkinliği ne kadar değişti. **TC:** sınırın (frontier) kendisi ne kadar "
                    "kaydı. **M:** ikisinin çarpımı, toplam verimlilik değişimi -- Malmquist "
                    "Sonuçları sekmenizdeki M ile aynı anlama gelir."
                )

                with st.expander("DMU (proje) bazında detay (EC / TC / M)"):
                    st.dataframe(r["detay_df"], use_container_width=True)
                    st.download_button(
                        "Detayı Excel indir", excel_indirme_verisi(r["detay_df"]),
                        file_name="gelecek_donem_dea_senaryo.xlsx", mime=EXCEL_MIME, key="ml_dea_detay_dl",
                    )

                with st.expander("Oluşturulan senaryo verisi (girdi + çıktı)"):
                    st.caption(
                        "Bu, DEA'ya verilen GERÇEK sayılar -- girdiler sizin seçtiğiniz yüzdeyle "
                        "değişti, çıktılar geçmiş girdi-çıktı ilişkisine göre otomatik hesaplandı."
                    )
                    st.write("**Girdi (senaryo):**")
                    st.dataframe(r["X_senaryo"], use_container_width=True)
                    st.write("**Çıktı (senaryo, otomatik hesaplanan):**")
                    st.dataframe(r["Y_senaryo"], use_container_width=True)
    with tab_aciklayici:
        st.markdown("### 📊 Açıklayıcılık Analizi — Hangi Değişken En İyi Açıklıyor?")
        st.markdown("""
        Panel Analizi sekmesinde gördüğünüz **R²** (modelin MI'deki değişimin ne kadarını
        açıkladığı), tek bir sayı olarak veriliyordu. Bu sekme, o tek sayıyı **her girdi ve
        çıktının kendi payına** ayırıyor -- yani *"R²'nin ne kadarı hangi değişkenden geliyor"*
        sorusuna, bir pasta grafiğiyle görsel bir cevap veriyor.

        **Yöntem:** Her değişken için, katsayısının büyüklüğü ile MI ile olan korelasyonu
        birlikte değerlendirilerek bir "katkı payı" hesaplanıyor (istatistikte **Pratt'in
        Göreli Önem Ölçüsü** olarak bilinen, standart bir teknik). Bu payların toplamı,
        modelin R²'sine yakın çıkar -- yani bu, R²'yi parçalara ayırmanın tutarlı bir yolu.

        ⚠️ **Not:** Değişkenler birbiriyle güçlü korele ise (Panel Analizi'ndeki VIF
        tablosuna bakın), bir değişkenin payı **negatif** çıkabilir -- bu, o değişkenin
        TEK BAŞINA değil, DİĞERLERİYLE BİRLİKTE bir rol oynadığı anlamına gelir. Pasta
        grafiğindeki dilim BÜYÜKLÜĞÜ her zaman mutlak değeri gösterir; yön (pozitif/negatif)
        ayrıca tablo ve renklerde belirtilir.
        """)

        p_ac = sonuc["panel_sonuc"]
        oneri_ac = p_ac["oneri"]
        tablo_map_ac = {
            "pooled_robust": p_ac["pooled_robust"], "pooled_clustered": p_ac["pooled_clustered"],
            "fe_robust": p_ac["fe_robust"], "fe_clustered": p_ac["fe_clustered"],
            "re_robust": p_ac["re_robust"], "re_clustered": p_ac["re_clustered"],
        }
        nihai_res_ac = tablo_map_ac[oneri_ac["sonuc_tablo"]]

        # BILINCLI KARAR: Aciklayicilik sekmesi SADECE resmi (girdi-yalnizca) modeli
        # kullanir -- eskiden ciktiyi da ekleyen ayri bir "teshis" modeli vardi, ama
        # bu, bastirma etkisi (suppression) yuzunden YANILTICI cikiyordu: bir girdinin
        # "cikti ile birlikteyken" gorunen payi, o girdinin GERCEK (tek basina) acikla-
        # ma gucunden cok daha buyuk cikabiliyordu -- ayni degisken icin IKI FARKLI
        # sayi gostermek, tam da onlemeye calistigimiz kafa karisikligini yaratiyordu.
        # Bu yuzden pasta grafigi HER ZAMAN resmi modelle birebir tutarlidir.
        aciklama_sonuc = aciklayicilik_analizi(sonuc["panel_df"], nihai_res_ac, girdi_cols, cikti_cols)

        if aciklama_sonuc.get("r_kare") is not None and aciklama_sonuc["r_kare"] < 0.20:
            st.caption(
                f"ℹ️ **Not:** Bu modelin R² değeri (~{aciklama_sonuc['r_kare']:.2f}) görece "
                f"düşüktür. Bunun başlıca sebebi, modele dahil edilmeyen çıktı değişken(ler)inin "
                f"MI'deki değişimin büyük bir kısmını taşımasıdır -- ancak çıktı(lar), girdilerle "
                f"birlikte kullanıldığında bastırma etkisi (suppression) yaratabildiği için, "
                f"kasıtlı olarak modelin dışında tutulmuştur."
            )

        if not aciklama_sonuc["yeterli_veri"]:
            st.error(aciklama_sonuc["mesaj"])
        else:
            c1, c2 = st.columns(2)
            c1.metric("Modelin R² değeri (Panel Analizi)", aciklama_sonuc["r_kare"])
            c2.metric("Payların toplamı (kontrol -- R²'ye yakın olmalı)", aciklama_sonuc["toplam_pratt"])

            tablo_ac = aciklama_sonuc["tablo"]
            en_iyi = tablo_ac.iloc[0]
            st.success(
                f"✅ **En iyi açıklayan değişken: {en_iyi['degisken']}** ({en_iyi['tip']}) -- "
                f"toplam açıklayıcılığın **%{en_iyi['pay_yuzde']:.1f}**'ini tek başına oluşturuyor. "
                f"Katsayı yönü: **{en_iyi['katsayi_yonu']}** (bu, Panel Analizi sekmesindeki katsayı "
                f"yönüyle her zaman birebir aynıdır)."
            )
            if "⚠️" in str(en_iyi["katki_turu"]):
                st.warning(
                    f"⚠️ Bu değişken bir **bastırıcı (suppressor)** olarak işaretlendi -- yani R²'ye "
                    f"katkısı, kendi ham korelasyonuyla değil, diğer değişkenlerle birlikte ortaya "
                    f"çıkıyor. Yüksek pay göstermesi, 'MI'yi güçlü şekilde artırıyor/azaltıyor' anlamına "
                    f"gelmez -- sadece modele istatistiksel olarak katkı payı büyük demektir. Panel "
                    f"Analizi sekmesindeki VIF tablosuna bakarak bu değişkenin başka hangi değişkenle "
                    f"güçlü korele olduğunu kontrol edin."
                )

            st.write("---")

            # Kurumsal/zarif renk paleti (koyu lacivert -> altın -> bordo -> petrol yesili)
            klas_paleti = ["#1F3A5F", "#C9A227", "#7B2D26", "#2E6F72", "#5B4B8A", "#8C8C8C"]
            renk_skala = alt.Scale(range=klas_paleti)

            tablo_ac_gorsel = tablo_ac.copy()
            tablo_ac_gorsel["etiket_metni"] = tablo_ac_gorsel["pay_yuzde"].map(lambda x: f"%{x:.1f}")

            taban_kat = alt.Chart(tablo_ac_gorsel).encode(
                theta=alt.Theta("pay_yuzde:Q", stack=True, sort="descending"),
                color=alt.Color(
                    "degisken:N", scale=renk_skala,
                    legend=alt.Legend(
                        title="Değişken", labelFontSize=20, titleFontSize=22,
                        symbolSize=320, labelLimit=420, orient="bottom",
                        columns=1, columnPadding=30, rowPadding=12, direction="vertical",
                        titleAnchor="start", padding=15, titlePadding=10,
                    ),
                ),
                order=alt.Order("pay_yuzde:Q", sort="descending"),
                tooltip=[
                    alt.Tooltip("degisken:N", title="Değişken"),
                    alt.Tooltip("tip:N", title="Tip"),
                    alt.Tooltip("pay_yuzde:Q", title="Pay (%)", format=".1f"),
                    alt.Tooltip("katsayi_yonu:N", title="Katsayı Yönü"),
                    alt.Tooltip("katki_turu:N", title="Katkı Türü"),
                ],
            )
            pasta_dilimleri = taban_kat.mark_arc(
                innerRadius=130, outerRadius=270, stroke="white", strokeWidth=3, cornerRadius=3,
            )
            yuzde_etiketleri = taban_kat.mark_text(
                radius=305, fontSize=21, fontWeight="bold", color="#333333",
            ).encode(text=alt.Text("etiket_metni:N"))

            pasta = (pasta_dilimleri + yuzde_etiketleri).properties(
                width="container", height=750,
                padding={"top": 55, "bottom": 15, "left": 40, "right": 40},
                title=alt.TitleParams(
                    text="Değişkenlerin R²'ye Katkı Payı", subtitle="(mutlak payla ölçeklenmiştir)",
                    fontSize=24, subtitleFontSize=15, font="Georgia", color="#1F3A5F",
                    subtitleColor="#666666", anchor="middle", offset=20,
                ),
            ).configure_view(strokeWidth=0).configure(background="transparent")

            # ONCE her zaman calisan (kanitlanmis) native Streamlit render'i gosteriyoruz --
            # bu, sayfanin ASLA bos kalmamasini garanti eder.
            st.altair_chart(pasta, use_container_width=True)

            # Ek olarak, SVG (zoom'da bulanıklaşmayan) bir versiyonu da katlanır bir
            # kutuda deniyoruz -- basarisiz olursa (CDN/iframe sorunu) net bir hata
            # mesaji gosterir, sayfa SESSIZCE bos KALMAZ.
            with st.expander("🔍 Daha keskin (SVG) versiyonu dene (isteğe bağlı, deneysel)"):
                st.caption(
                    "Yukarıdaki grafik zaten çalışıyor -- bu, sadece aşırı yakınlaştırmada "
                    "daha keskin kalabilecek bir alternatif. Yüklenmezse yukarıdaki grafiği kullanın."
                )
                grafik_json = pasta.to_json()
                html_kod = f"""
                <div id="svg-pasta-grafik" style="width:100%; min-height:750px;"></div>
                <div id="svg-hata-mesaji" style="color:#b00020; font-family:sans-serif; display:none;"></div>
                <script src="https://cdn.jsdelivr.net/npm/vega@5.30.0"></script>
                <script src="https://cdn.jsdelivr.net/npm/vega-lite@5.20.1"></script>
                <script src="https://cdn.jsdelivr.net/npm/vega-embed@6.26.0"></script>
                <script>
                  document.addEventListener("DOMContentLoaded", function () {{
                    if (typeof vegaEmbed === "undefined") {{
                      document.getElementById("svg-hata-mesaji").style.display = "block";
                      document.getElementById("svg-hata-mesaji").innerText =
                        "⚠️ Grafik kütüphanesi yüklenemedi (CDN erişim sorunu olabilir). Lütfen yukarıdaki normal grafiği kullanın.";
                      return;
                    }}
                    vegaEmbed("#svg-pasta-grafik", {grafik_json}, {{"renderer": "svg", "actions": false}})
                      .catch(function (hata) {{
                        document.getElementById("svg-hata-mesaji").style.display = "block";
                        document.getElementById("svg-hata-mesaji").innerText =
                          "⚠️ Grafik yüklenemedi: " + hata.message + " -- Lütfen yukarıdaki normal grafiği kullanın.";
                      }});
                  }});
                </script>
                """
                components.html(html_kod, height=820, scrolling=False)

            st.markdown("##### Detay Tablo")
            st.caption(
                "**katsayi_yonu:** bu değişkeni artırmak MI'yi artırır mı azaltır mı (Panel Analizi "
                "ile birebir tutarlı). **katki_turu:** bu değişkenin R²'ye katkısı 'uyumlu' (kendi "
                "başına anlamlı) mı, yoksa 'bastırıcı' (sadece diğer değişkenlerle birlikte) mi."
            )
            st.dataframe(
                tablo_ac[["degisken", "tip", "katsayi", "katsayi_yonu", "pay_yuzde", "katki_turu", "pratt_degeri"]],
                use_container_width=True, hide_index=True,
            )

            st.download_button(
                "Açıklayıcılık tablosunu Excel indir", excel_indirme_verisi(tablo_ac),
                file_name="aciklayicilik_analizi.xlsx", mime=EXCEL_MIME, key="aciklayicilik_csv_dl",
            )
