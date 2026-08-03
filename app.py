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
from dea_module import min_dmu_kontrolu
from panel_module import leave_one_out_kararlilik
from pipeline import run_pipeline
from senaryo_module import gelecek_donem_analizi
from backtest_module import backtest_calistir
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

        # --- Minimum DMU sayisi kontrolu (yukleme aninda erken uyari) ---
        kontrol = min_dmu_kontrolu(
            len(veri_onizleme["girdi_cols"]), len(veri_onizleme["cikti_cols"]),
            len(veri_onizleme["dmu_sirali"]),
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
            # Onceki dosyaya ait alt-sekme sonuclarini temizle -- aksi halde farkli bir
            # DMU/donem setine sahip yeni bir dosya yuklendiginde, eski sonuclar (gelecek
            # senaryosu, backtest, kararlilik testi) yeni DMU/degisken listesiyle uyusmayip
            # KeyError'a yol acabilir (secim kutulari yeni veriden, sonuc eskisinden gelir).
            for eski_anahtar in ["gelecek", "backtest", "kararlilik"]:
                st.session_state.pop(eski_anahtar, None)

    except VeriDogrulamaHatasi as e:
        st.error(f"Veri dogrulama hatasi: {e}")
    except Exception as e:
        st.error(f"Beklenmeyen hata: {e}")

if "sonuc" in st.session_state:
    sonuc = st.session_state["sonuc"]
    girdi_cols = sonuc["veri"]["girdi_cols"]
    cikti_cols = sonuc["veri"]["cikti_cols"]

    tab_dea, tab_malmquist, tab_panel, tab_gelecek, tab_backtest = st.tabs(
        ["DEA Sonuclari", "Malmquist Sonuclari", "Panel Analizi", "Gelecek Verimlilik Tahmini",
         "Backtest (Model Dogrulama)"]
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
            st.dataframe(dea_d["slack_x_ccr"].round(4), width='stretch')
            st.write("*CCR - Cikti Slack*")
            st.dataframe(dea_d["slack_y_ccr"].round(4), width='stretch')
        with c4:
            st.write("*BCC - Girdi Slack*")
            st.dataframe(dea_d["slack_x_bcc"].round(4), width='stretch')
            st.write("*BCC - Cikti Slack*")
            st.dataframe(dea_d["slack_y_bcc"].round(4), width='stretch')

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
        st.dataframe(aksiyon_tablosu, width='stretch')

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
        st.dataframe(sonuc["malmquist"].round(4), width='stretch')

        st.write("---")
        st.write("**Donem Bazinda Ortalama EC / TC / M (tum DMU'lar uzerinden, geometrik ortalama)**")
        gecisli_donemler = sonuc["veri"]["donem_sirali"][:-1]
        donem_ort = malmquist_donem_ortalamasi(sonuc["malmquist"], donem_sirasi=gecisli_donemler)
        st.dataframe(donem_ort, width='stretch')
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
            "Bu bölüm, DMU'ların girdi/çıktı değerleri (örn. SİMÜLASYON_SÜRESİ, MALİYET) ile "
            "Malmquist verimlilik endeksi (MI) arasındaki **istatistiksel ilişkiyi** inceler. Amaç: "
            "'hangi değişkenler verimliliği artırıyor/azaltıyor, ne yönde ve ne kadar güçlü' sorusuna "
            "cevap vermektir. Bunun için önce en uygun istatistiksel modeli (Pooled OLS / Sabit "
            "Etkiler / Rastgele Etkiler) seçer, sonra bu modelin ne kadar güvenilir olduğunu (katsayı "
            "kararlılığı gibi ek testlerle) sınar."
        )
        p = sonuc["panel_sonuc"]
        st.write("**Korelasyon Matrisi**")
        st.caption(
            "Degiskenler arasindaki DOGRUSAL iliskinin yonu ve gucu (-1 ile +1 arasi). +1'e yakin: "
            "birlikte artiyorlar; -1'e yakin: biri artarken digeri azaliyor; 0'a yakin: aralarinda "
            "dogrusal bir iliski yok. Bagimsiz degiskenler arasindaki YUKSEK korelasyonlar (orn. "
            "|r|>0.8), asagidaki VIF testinde de goreceginiz coklu dogrusal baglanti riskine isaret eder."
        )
        st.dataframe(p["corr"].round(3))
        st.write("**VIF**")
        st.caption(
            "VIF (Variance Inflation Factor): bir degiskenin, MODELDEKI DIGER degiskenler tarafindan "
            "ne kadar 'aciklanabildigini' gosterir. VIF=1 -> digerleriyle hic ortusmuyor (ideal); "
            "VIF≥5 literaturde genellikle 'coklu dogrusal baglanti sorunu var' esigi olarak kabul "
            "edilir -- bu durumda o degiskenin KENDI etkisini digerlerinden ayirt etmek zorlasir."
        )
        st.dataframe(p["vif"].round(3))

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

        st.caption(
            "Asagidaki uc tablo, ayni veriye uc farkli model varsayimiyla (Pooled/FE/RE) bakar. "
            "Her tabloda: **Katsayi** = o degiskenin MI uzerindeki tahmini etkisi (yon ve buyukluk); "
            "**P-degeri** < 0.05 (bazi yerlerde 0.10) ise etki istatistiksel olarak anlamli kabul "
            "edilir; **Alt/Ust CI** ise katsayinin %95 guvenle icinde olduğu araligi gosterir. "
            "Hangisinin 'dogru' oldugunu yukaridaki 3 test (Poolability, BP-LM, Hausman) belirler."
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
        st.dataframe(reg_meta_table(nihai_res), width='stretch', hide_index=True)
        st.dataframe(reg_params_table(nihai_res), width='stretch')
        st.write("---")

        # --- Katsayilarin verimlilige (MI) etkisi -- yon ve buyukluk gosterimi ---
        st.markdown("### 🎯 Katsayıların Verimliliğe (MI) Etkisi — Yön ve Büyüklük")
        KATSAYI_ALPHA = 0.10  # katsayi anlamliligi icin ayri esik -- model secimi testlerinden (p['alpha']) bagimsiz
        st.caption(
            f"Nihai (yukaridaki '⭐ NIHAI SONUC') modeldeki her degiskenin, ortalama degerinin %10 "
            f"degismesi durumunda MI uzerindeki tahmini etkisi. Anlamlilik esigi: p<{KATSAYI_ALPHA:.2f}. "
            f"Yesil = pozitif (verimlilik artisi), kirmizi = negatif (verimlilik azalisi). Yon ayrica "
            f"DEA'nin teorik beklentisiyle (Girdi -> negatif, Cikti -> pozitif) karsilastirilir; celisen "
            f"anlamli katsayilar asagida ayrica isaretlenir."
        )
        analiz_df = panel_aksiyon_analizi(
            nihai_res, girdi_cols, cikti_cols, sonuc["panel_df"], alpha=KATSAYI_ALPHA,
        )
        st.dataframe(
            analiz_df.style.apply(_katsayi_renklendir, axis=1),
            width='stretch', hide_index=True,
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
                st.session_state["kararlilik"] = leave_one_out_kararlilik(
                    sonuc["panel_df"], "MI", girdi_cols + cikti_cols, oneri["sonuc_tablo"],
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
                    width='stretch', hide_index=True,
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
                    st.dataframe(kr["detay_df"].reset_index(), width='stretch', hide_index=True)

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
                st.dataframe(reg_params_table(p["robust"]), width='stretch')
            with c_c:
                st.write(f"*{p['secilen_model']} - Clustered*")
                st.dataframe(reg_params_table(p["clustered"]), width='stretch')
        else:
            st.write("*(Bilgi amacli, Hausman'in FE/RE arasindan sectigi model)* "
                      f"{p['secilen_model']} - Robust Standart Hatalar*")
            st.dataframe(reg_params_table(p["robust"]), width='stretch')

        st.write(f"*(Bilgi amacli) Pooled OLS - Robust vs Clustered*")
        c_pr, c_pc = st.columns(2)
        with c_pr:
            st.write("*Pooled OLS - Robust*")
            st.dataframe(reg_params_table(p["pooled_robust"]), width='stretch')
        with c_pc:
            st.write("*Pooled OLS - Clustered*")
            st.dataframe(reg_params_table(p["pooled_clustered"]), width='stretch')

        st.write("**Model Karsilastirma (Pooled OLS vs FE vs RE)**")
        katsayi_tablo, tstat_tablo, ozet_tablo = comparison_tables(p["comparison"])
        st.write("*Katsayilar*")
        st.dataframe(katsayi_tablo, width='stretch')
        st.write("*T-istatistikleri*")
        st.dataframe(tstat_tablo, width='stretch')
        st.write("*Model ozet istatistikleri*")
        st.dataframe(ozet_tablo, width='stretch')

    with tab_gelecek:
        st.markdown("### 🔮 Gelecek Dönem Verimlilik Tahmini")
        st.markdown("""
        Bu bolum, panel analizindeki **nihai model** katsayilarini kullanarak, **sadece
        son donemin verilerini** temel alan bir "bir sonraki donem" senaryosu uretir.
        Izlenen yontem (**ceteris paribus** -- sadece kanit oldugu degiskenler degisir,
        gerisi sabit tutulur):

        1. **Karar degiskeni ayrimi → Hedefli / Dogal:** SADECE GIRDI degiskenleri
           "Hedefli" olabilir -- cunku donem basinda karar verebileceginiz sey
           girdidir (orn. simulasyon suresi, maliyet), cikti (dogruluk, azaltilan
           prototip sayisi) donem SONUNDA gozlemlenen bir sonuctur, onceden karar
           verilemez. Girdi, panel modelinde ANLAMLIYSA (p<0.10) Hedefli sayilir --
           DEA teorisiyle (girdi→negatif katsayi beklentisi) TUTARLI olmasi
           ARANMAZ; yon dogrudan katsayinin GERCEK isaretinden alinir. Teoriyle
           celisen durumlar yine de ⚠️ isaretlenir (seffaflik icin), ama Hedefli
           olmaktan alikonmaz.
        2. **Hedefli degiskenler:** katsayinin GERCEK isaretine gore (artir/azalt),
           son donem degerinin %5 / %10 / %15'i kadar degistirilir; bu hareketin
           MI uzerindeki tahmini etkisi de (katsayi x degisim) ayrica raporlanir.
        3. **Digerleri (Dogal -- tum cikti degiskenleri + anlamsiz girdiler):**
           ceteris paribus -- son donem degeriyle AYNEN birakilir, hicbir sekilde
           degistirilmez. VIF≥5 olanlar icin (hedefli degiskenlerle yuksek
           korelasyon) sadece bilgi amacli bir uyari eklenir.
        4. **Sinir kontrolleri:** Hedefli degiskenlerin projeksiyonu, DMU'nun kendi
           tarihsel araligina gore makul bir bant disina cikamaz; negatif deger
           asla uretilmez.
        5. **Duyarlilik taramasi:** %5 / %10 (Baz) / %15 uc senaryo birlikte uretilir;
           Baz senaryo nihai sonuc olarak one cikarilir.
        """)

        if st.button("Gelecek Donem Senaryosunu Hesapla", type="primary", key="gelecek_hesapla"):
            with st.spinner("3 senaryo icin son donem verisi olusturuluyor, DEA ve Malmquist tekrar cozuluyor..."):
                st.session_state["gelecek"] = gelecek_donem_analizi(sonuc)

        if "gelecek" in st.session_state:
            gelecek = st.session_state["gelecek"]

            st.write("---")
            st.markdown("#### 1-2) Hedefli / Dogal Siniflandirma")
            st.caption(
                f"Nihai model: **{gelecek['nihai_res_baslik']}**. Son gercek donem: **{gelecek['son_donem']}** "
                f"(senaryo bu donemin verisini baz alir)."
            )
            st.dataframe(
                gelecek["siniflandirma"].reset_index(),
                width='stretch', hide_index=True,
            )

            hedefli_sayi = (gelecek["siniflandirma"]["siniflandirma"] == "Hedefli").sum()
            if hedefli_sayi == 0:
                st.info(
                    "Bu modelde 'Hedefli' (anlamli bir GIRDI) degisken bulunamadi -- cikti "
                    "degiskenleri zaten hicbir zaman Hedefli olamiyor, ve hicbir girdi de panelde "
                    "anlamli cikmadi. Tum degiskenler ceteris paribus sabit kaliyor (%5/%10/%15 "
                    "senaryolari bu nedenle birbirinin ayni cikacaktir)."
                )

            st.write("---")
            st.markdown("#### 5) Senaryo Karsilastirma (Duyarlilik Taramasi)")
            karsilastirma_satirlar = []
            for isim, s in gelecek["senaryolar"].items():
                karsilastirma_satirlar.append({
                    "Senaryo": isim, "Yuzde": f"%{s['yuzde']*100:g}",
                    "Ortalama EC": round(s["malmquist"]["EC"].mean(), 4),
                    "Ortalama TC": round(s["malmquist"]["TC"].mean(), 4),
                    "Ortalama M": round(s["malmquist"]["M"].mean(), 4),
                })
            karsilastirma_df = pd.DataFrame(karsilastirma_satirlar).set_index("Senaryo")
            st.dataframe(karsilastirma_df, width='stretch')
            st.bar_chart(karsilastirma_df["Ortalama M"])

            st.write("---")
            senaryo_sec = st.selectbox(
                "Detayli incelemek icin senaryo secin", options=list(gelecek["senaryolar"].keys()),
                index=1, key="gelecek_senaryo_sec",  # varsayilan: Baz
            )
            s = gelecek["senaryolar"][senaryo_sec]

            st.markdown(f"#### Senaryo Detayi: {senaryo_sec} (hedefli degisim yuzdesi=%{s['yuzde']*100:g})")
            dmu_sec_gelecek = st.selectbox(
                "DMU sec (senaryo detay tablosu icin)", options=sonuc["veri"]["dmu_sirali"],
                key="gelecek_dmu_sec",
            )
            detay_dmu_listesi = s["detay"].index.get_level_values("DMU").unique()
            if dmu_sec_gelecek not in detay_dmu_listesi:
                st.warning(
                    "Bu senaryo sonucu farkli bir veri setine ait gorunuyor (DMU eslesmiyor). "
                    "Lutfen 'Gelecek Donem Senaryosunu Hesapla' butonuna tekrar basin."
                )
            else:
                st.dataframe(s["detay"].loc[dmu_sec_gelecek], width='stretch')

            st.write("---")
            st.markdown("#### Projeksiyon Donemi DEA Sonuclari")
            c1, c2 = st.columns(2)
            with c1:
                st.write("**CCR (theta)**")
                st.dataframe(s["dea"]["theta_ccr"].round(4), width='stretch')
            with c2:
                st.write("**BCC (theta)**")
                st.dataframe(s["dea"]["theta_bcc"].round(4), width='stretch')
            st.write("**Olcek Etkinligi**")
            st.dataframe(s["dea"]["olcek_etkinligi"].round(4), width='stretch')

            st.write("---")
            st.markdown(f"#### Malmquist: {gelecek['son_donem']} → Projeksiyon Donemi")
            st.dataframe(s["malmquist"].round(4), width='stretch')

            st.markdown("##### Yorum")
            malmquist_yorum_sec = st.selectbox(
                "Yorumlanacak DMU", options=list(s["malmquist"].index.get_level_values("DMU")),
                key="gelecek_malmquist_yorum_dmu",
            )
            satir = s["malmquist"].xs(malmquist_yorum_sec, level="DMU").iloc[0]
            st.markdown(malmquist_yorum_metni(satir))

            st.caption(
                "⚠️ Bu tahmin, gecmis trend ve panel model katsayilarina dayanan bir SENARYO'dur, "
                "kesin bir ongoru degildir. Ozellikle DMU sayisi az oldugunda ve/veya panelde anlamli "
                "'Hedefli' degisken bulunamadigi durumlarda temkinli yorumlanmalidir."
            )

            st.download_button(
                f"{senaryo_sec} senaryo detay tablosunu Excel indir", excel_indirme_verisi(s["detay"]),
                file_name=f"gelecek_senaryo_{senaryo_sec.lower()}.xlsx", mime=EXCEL_MIME, key="gelecek_csv_dl",
            )

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
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("MAE", m["MAE"])
                c2.metric("RMSE", m["RMSE"])
                c3.metric("MAPE (%)", m["MAPE_%"])
                c4.metric("Yon Dogruluk (%)", m["yon_dogruluk_%"])

                if m["Pearson_r"] is not None:
                    st.write(f"**Pearson korelasyonu (gercek vs tahmin):** {m['Pearson_r']}")

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
                    width='stretch', hide_index=True,
                )

                st.caption(
                    "Yesil satirlar: modelin verimlilik ARTIS/AZALIS yonunu dogru tahmin ettigi DMU'lar. "
                    "Kirmizi satirlar: yonun yanlis tahmin edildigi DMU'lar."
                )

                st.download_button(
                    "Backtest sonuclarini Excel indir", excel_indirme_verisi(bt["tahmin_df"]),
                    file_name="backtest_sonuclari.xlsx", mime=EXCEL_MIME, key="backtest_csv_dl",
                )


