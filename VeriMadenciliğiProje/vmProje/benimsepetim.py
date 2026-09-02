import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="BenimSepetim", page_icon="🛒", layout="wide")

st.markdown("""
<style>
/* Clean Modern E-Commerce Theme */
.stApp {
    background-color: #f9fafb;
    color: #111827;
}
.main-title {
    font-size: 42px; font-weight: 800; color: #4338ca; margin-bottom: 0;
}
.subtitle {
    font-size: 16px; color: #6b7280; margin-top: 5px;
}
div.stButton>button {
    background-color: #4f46e5; color: white !important;
    border-radius: 8px; border: none; font-weight: 600;
}
div.stButton>button:hover {
    background-color: #4338ca; border: none;
}
.price {
    font-size: 20px; font-weight: bold; color: #e11d48;
}
.small-text {
    font-size: 13px; color: #6b7280;
}
.pink-box { background:#fdf2f8; color:#831843; padding:15px; border-radius:10px; border-left:4px solid #f472b6; margin-bottom:10px; }
.success-box { background:#f0fdf4; color:#14532d; padding:15px; border-radius:10px; border-left:4px solid #4ade80; margin-bottom:10px; }
.warning-box { background:#fffbeb; color:#78350f; padding:15px; border-radius:10px; border-left:4px solid #fbbf24; margin-bottom:10px; }
.info-box { background:#eff6ff; color:#1e3a8a; padding:15px; border-radius:10px; border-left:4px solid #60a5fa; margin-bottom:10px; }
.metric-box { background:#ffffff; border:1px solid #e5e7eb; border-radius:12px; padding:16px; text-align:center; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
.metric-value { font-size:24px; font-weight:bold; color:#4338ca; }
.metric-label { font-size:13px; color:#6b7280; font-weight:bold;}
.dark-box { background:#1f2937; color:white; padding:20px; border-radius:12px; margin-bottom:12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
.dark-box h2, .dark-box h3, .dark-box p, .dark-box b { color: white !important; }
.dark-line { border-top:1px solid #374151; margin:12px 0; }
</style>
""", unsafe_allow_html=True)

def tl_format(tutar):
    return f"{tutar:,.2f} TL".replace(",","X").replace(".",",").replace("X",".")

def kolon_bul(df, olasi_isimler):
    for isim in olasi_isimler:
        if isim in df.columns:
            return isim
    temiz = {c.strip().lower(): c for c in df.columns}
    for isim in olasi_isimler:
        if isim.strip().lower() in temiz:
            return temiz[isim.strip().lower()]
    return None

@st.cache_data
def veri_setini_yukle(dosya_yolu="fis_veriseti_guncel_660_temiz.csv"):
    df = pd.read_csv(dosya_yolu)
    fis_col = kolon_bul(df,["Fiş ID","Fis ID","FişID","FisID","fis_id","FIS_ID"])
    urun_col = kolon_bul(df,["Ürün Adı","Urun Adi","Urun Adı","Ürün","Urun","urun_adi","URUN_ADI"])
    fiyat_col = kolon_bul(df,["Birim Fiyat","Birim Fiyat (TL)","BirimFiyat","Fiyat","fiyat","BIRIM_FIYAT"])
    adet_col = kolon_bul(df,["Adet","adet","Miktar","miktar"])
    eksikler = []
    if fis_col is None: eksikler.append("Fiş ID")
    if urun_col is None: eksikler.append("Ürün Adı")
    if fiyat_col is None: eksikler.append("Birim Fiyat")
    if adet_col is None: eksikler.append("Adet")
    if eksikler:
        raise ValueError("CSV dosyasında şu sütunlar bulunamadı: "+", ".join(eksikler)+f"\\nMevcut sütunlar: {list(df.columns)}")
    df = df.rename(columns={fis_col:"fis_id", urun_col:"urun_adi", fiyat_col:"birim_fiyat", adet_col:"adet"})
    df["urun_adi"] = df["urun_adi"].astype(str).str.strip()
    df["birim_fiyat"] = pd.to_numeric(df["birim_fiyat"], errors="coerce")
    df["adet"] = pd.to_numeric(df["adet"], errors="coerce")
    df = df.dropna(subset=["fis_id","urun_adi","birim_fiyat","adet"])
    df = df[(df["urun_adi"]!="") & (df["birim_fiyat"]>0) & (df["adet"]>0)]
    return df

@st.cache_data
def urun_listesi_olustur(df):
    urun_df = df.groupby("urun_adi").agg(fiyat=("birim_fiyat","mean"), satis_adedi=("adet","sum"), fis_sayisi=("fis_id","nunique")).reset_index()
    urun_df = urun_df.rename(columns={"urun_adi":"ad"})
    urun_df["id"] = range(1, len(urun_df)+1)
    urun_df["kategori"] = "Market Ürünü"
    return urun_df[["id","ad","kategori","fiyat","satis_adedi","fis_sayisi"]].sort_values("ad").reset_index(drop=True)

@st.cache_data
def apriori_kurallarini_olustur(df, min_support=0.01, min_confidence=0.20):
    sepet_matrisi = df.groupby(["fis_id","urun_adi"])["adet"].sum().unstack().fillna(0) > 0
    frequent_itemsets = apriori(sepet_matrisi, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return pd.DataFrame()
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        return pd.DataFrame()
    return rules.sort_values(by=["lift","confidence","support"], ascending=False).reset_index(drop=True)

try:
    veri_df = veri_setini_yukle()
    urun_df = urun_listesi_olustur(veri_df)
except Exception as e:
    st.error("Veri seti yüklenirken hata oluştu.")
    st.exception(e)
    st.stop()

with st.sidebar:
    st.header("Analiz Ayarları")
    st.write("Apriori algoritması için minimum değerleri buradan ayarlayabilirsin.")
    min_support = st.slider("Minimum Support",0.001,0.100,0.010,0.001,format="%.3f")
    min_confidence = st.slider("Minimum Confidence",0.05,0.90,0.20,0.05)
    st.caption("Support düşük olursa daha fazla kural çıkar. Çok yüksek olursa kural çıkmayabilir.")

kurallar_df = apriori_kurallarini_olustur(veri_df, min_support=min_support, min_confidence=min_confidence)

if "sepet" not in st.session_state:
    st.session_state.sepet = {}
if "indirimli_urunler" not in st.session_state:
    st.session_state.indirimli_urunler = {}

EMOJILER = {
    "Ekmek": "🍞", "Peynir": "🧀", "Süt": "🥛", "Çay": "☕", "Yumurta": "🥚",
    "Tavuk": "🍗", "Makarna": "🍝", "Yoğurt": "🥣", "Deterjan": "🧼", 
    "Şampuan": "🧴", "Şeker": "🍬", "Diş Macunu": "🪥", "Limon": "🍋",
    "Domates": "🍅", "Salatalık": "🥒", "Sıvı Yağ": "🌻"
}
def urun_emojisi(ad):
    for kelime, emoji in EMOJILER.items():
        if kelime.lower() in ad.lower(): return emoji
    return "🛍️"

def urun_bilgisi_getir(urun_adi):
    sonuc = urun_df[urun_df["ad"] == urun_adi]
    if sonuc.empty:
        return None
    return sonuc.iloc[0].to_dict()

def sepete_ekle(urun_adi, indirim_orani=0, kural_bilgisi=None):
    urun = urun_bilgisi_getir(urun_adi)
    if urun is None:
        return
    if urun_adi in st.session_state.sepet:
        st.session_state.sepet[urun_adi]["adet"] += 1
    else:
        st.session_state.sepet[urun_adi] = {"fiyat":float(urun["fiyat"]), "kategori":urun["kategori"], "adet":1}
    if indirim_orani > 0:
        st.session_state.indirimli_urunler[urun_adi] = {"indirim":indirim_orani, "kural":kural_bilgisi}

def sepetten_azalt(urun_adi):
    if urun_adi in st.session_state.sepet:
        st.session_state.sepet[urun_adi]["adet"] -= 1
        if st.session_state.sepet[urun_adi]["adet"] <= 0:
            del st.session_state.sepet[urun_adi]
            st.session_state.indirimli_urunler.pop(urun_adi, None)

def sepetten_sil(urun_adi):
    st.session_state.sepet.pop(urun_adi, None)
    st.session_state.indirimli_urunler.pop(urun_adi, None)

def sepet_urun_adlari():
    return list(st.session_state.sepet.keys())

def ara_toplam_hesapla():
    return sum(bilgi["fiyat"]*bilgi["adet"] for bilgi in st.session_state.sepet.values())

def urun_indirimi_hesapla():
    toplam = 0
    for urun_adi, indirim_bilgisi in st.session_state.indirimli_urunler.items():
        if urun_adi in st.session_state.sepet:
            bilgi = st.session_state.sepet[urun_adi]
            toplam += bilgi["fiyat"] * bilgi["adet"] * indirim_bilgisi["indirim"] / 100
    return toplam

kampanyalar = [
    {"ad":"Kahvaltı Paketi","urunler":["Ekmek","Peynir","Çay (500g)"],"indirim":20},
    {"ad":"Salata Paketi","urunler":["Domates","Salatalık","Limon"],"indirim":15},
    {"ad":"Kişisel Bakım Paketi","urunler":["Şampuan","Diş Macunu","Deodorant"],"indirim":18},
    {"ad":"Akşam Yemeği Paketi","urunler":["Makarna","Sıvı Yağ"],"indirim":10},
]

def kampanya_indirimi_hesapla():
    toplam = 0
    aktif = []
    mevcut = sepet_urun_adlari()
    for kampanya in kampanyalar:
        urunler_var = all(urun in urun_df["ad"].values for urun in kampanya["urunler"])
        tamam = all(urun in mevcut for urun in kampanya["urunler"])
        if urunler_var and tamam:
            kampanya_toplami = sum(st.session_state.sepet[urun]["fiyat"]*st.session_state.sepet[urun]["adet"] for urun in kampanya["urunler"])
            indirim_tutari = kampanya_toplami * kampanya["indirim"] / 100
            toplam += indirim_tutari
            aktif.append({"ad":kampanya["ad"],"indirim":kampanya["indirim"],"indirim_tutari":indirim_tutari})
    return toplam, aktif

def eksik_kampanyalari_getir():
    mevcut = sepet_urun_adlari()
    sonuc = []
    for kampanya in kampanyalar:
        kampanya_urunleri = [urun for urun in kampanya["urunler"] if urun in urun_df["ad"].values]
        if not kampanya_urunleri:
            continue
        eslesenler = [urun for urun in kampanya_urunleri if urun in mevcut]
        eksikler = [urun for urun in kampanya_urunleri if urun not in mevcut]
        if eslesenler and eksikler:
            sonuc.append({"ad":kampanya["ad"],"indirim":kampanya["indirim"],"eslesenler":eslesenler,"eksikler":eksikler})
    return sonuc

def uygun_onerileri_getir():
    mevcut = set(sepet_urun_adlari())
    uygun = []
    if kurallar_df.empty or not mevcut:
        return uygun
    for _, row in kurallar_df.iterrows():
        kosul = set(row["antecedents"])
        onerilenler = set(row["consequents"])
        if kosul.issubset(mevcut) and not onerilenler.intersection(mevcut):
            for urun in onerilenler:
                indirim = min(max(round(float(row["lift"])*5),5),25)
                uygun.append({"kosul":list(kosul),"onerilen":urun,"indirim":indirim,"support":float(row["support"]),"confidence":float(row["confidence"]),"lift":float(row["lift"]),"aciklama":"Bu öneri, sepetinize göre düzenlenir."})
    benzersiz = {}
    for oneri in uygun:
        urun = oneri["onerilen"]
        if urun not in benzersiz or oneri["lift"] > benzersiz[urun]["lift"]:
            benzersiz[urun] = oneri
    return sorted(benzersiz.values(), key=lambda x:(x["lift"],x["confidence"]), reverse=True)[:6]

st.markdown('<h1 class="main-title">🛒 BenimSepetim</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Akıllı alışveriş sepeti ve ürün öneri sistemi.</p>', unsafe_allow_html=True)
st.write("")

with st.container(border=True):
    st.subheader("Veri Seti Özeti")
    m1,m2,m3,m4 = st.columns(4)
    toplam_fis = veri_df["fis_id"].nunique()
    toplam_urun_cesidi = veri_df["urun_adi"].nunique()
    toplam_satis_satiri = len(veri_df)
    toplam_ciro = (veri_df["birim_fiyat"] * veri_df["adet"]).sum()
    for col, value, label in [(m1,toplam_fis,"Toplam Fiş"),(m2,toplam_urun_cesidi,"Ürün Çeşidi"),(m3,toplam_satis_satiri,"Satış Satırı"),(m4,tl_format(toplam_ciro),"Toplam Ciro")]:
        with col:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)
    st.write("")
    if kurallar_df.empty:
        st.warning("Bu support/confidence değerleriyle birliktelik kuralı oluşmadı. Sol menüden minimum support veya confidence değerini düşürmeyi dene.")
    else:
        st.markdown(f'<div class="info-box">Apriori algoritması sonucunda <b>{len(kurallar_df)}</b> adet birliktelik kuralı oluşturuldu. Öneriler bu kurallardaki <b>support</b>, <b>confidence</b> ve <b>lift</b> değerlerine göre yapılmaktadır.</div>', unsafe_allow_html=True)

sol, sag = st.columns([2,1])
with sol:
    with st.container(border=True):
        st.subheader("Ürünler")
        arama = st.text_input("Ürün ara", placeholder="Örneğin: ekmek, süt, çay...")
        kategoriler = ["Tümü"] + sorted(urun_df["kategori"].unique().tolist())
        secili_kategori = st.selectbox("Kategori seç", kategoriler)
        filtreli_df = urun_df.copy()
        if arama:
            def kucult(metin):
                if not isinstance(metin, str): return ""
                return metin.replace("I", "ı").replace("İ", "i").lower()
            aranan = kucult(arama)
            filtreli_df = filtreli_df[filtreli_df["ad"].apply(lambda x: aranan in kucult(x))]
        if secili_kategori != "Tümü":
            filtreli_df = filtreli_df[filtreli_df["kategori"] == secili_kategori]
        if filtreli_df.empty:
            st.info("Aramana uygun ürün bulunamadı.")
        else:
            urun_kolonlari = st.columns(3)
            for index, row in filtreli_df.reset_index(drop=True).iterrows():
                with urun_kolonlari[index % 3]:
                    with st.container(border=True):
                        emoji = urun_emojisi(row['ad'])
                        st.markdown(f"### {emoji} {row['ad']}")
                        st.markdown(f"<p class='small-text'>{row['kategori']} · {int(row['fis_sayisi'])} fişte geçti</p>", unsafe_allow_html=True)
                        st.markdown(f"<p class='small-text'>Toplam satış adedi: {row['satis_adedi']:.0f}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p class='price'>{tl_format(row['fiyat'])}</p>", unsafe_allow_html=True)
                        if st.button("Sepete Ekle", key=f"ekle_{row['id']}"):
                            sepete_ekle(row["ad"])
                            st.rerun()

with sag:
    with st.container(border=True):
        st.subheader("Sepetim")
        if len(st.session_state.sepet) == 0:
            st.info("Sepetin şu anda boş. Ürün ekleyerek başlayabilirsin.")
        else:
            for urun_adi, bilgi in list(st.session_state.sepet.items()):
                adet, fiyat, toplam = bilgi["adet"], bilgi["fiyat"], bilgi["adet"]*bilgi["fiyat"]
                emoji = urun_emojisi(urun_adi)
                st.markdown(f"**{emoji} {urun_adi}**")
                st.markdown(f"{adet} adet x {tl_format(fiyat)} = **{tl_format(toplam)}**")
                if urun_adi in st.session_state.indirimli_urunler:
                    indirim_bilgisi = st.session_state.indirimli_urunler[urun_adi]
                    st.markdown(f'<div class="success-box">Bu ürüne %{indirim_bilgisi["indirim"]} öneri indirimi uygulanıyor.</div>', unsafe_allow_html=True)
                c1,c2,c3 = st.columns(3)
                with c1:
                    if st.button("➖", key=f"azalt_{urun_adi}"):
                        sepetten_azalt(urun_adi); st.rerun()
                with c2:
                    if st.button("➕", key=f"arttir_{urun_adi}"):
                        sepete_ekle(urun_adi); st.rerun()
                with c3:
                    if st.button("🗑️", key=f"sil_{urun_adi}"):
                        sepetten_sil(urun_adi); st.rerun()
                st.divider()
        ara_toplam = ara_toplam_hesapla()
        urun_indirimi = urun_indirimi_hesapla()
        kampanya_indirimi, aktif_kampanyalar = kampanya_indirimi_hesapla()
        toplam_indirim = urun_indirimi + kampanya_indirimi
        genel_toplam = ara_toplam - toplam_indirim
        st.markdown(f'''<div class="dark-box"><h3>Sepet Özeti</h3><p>Ara Toplam: <b>{tl_format(ara_toplam)}</b></p><p>Öneri İndirimi: <b>- {tl_format(urun_indirimi)}</b></p><p>Paket İndirimi: <b>- {tl_format(kampanya_indirimi)}</b></p><div class="dark-line"></div><h2>Toplam: {tl_format(genel_toplam)}</h2></div>''', unsafe_allow_html=True)
        if st.button("Sepeti Temizle"):
            st.session_state.sepet = {}; st.session_state.indirimli_urunler = {}; st.rerun()

st.write("")
with st.container(border=True):
    st.subheader("Akıllı Ürün Önerileri")
    uygun_oneriler = uygun_onerileri_getir()
    if len(st.session_state.sepet) == 0:
        st.info("Ürün eklediğinde burada veri setinden çıkarılan kurallara göre öneriler görünecek.")
    elif kurallar_df.empty:
        st.warning("Apriori sonucunda uygun kural bulunamadığı için öneri oluşturulamıyor. Minimum support veya confidence değerini düşürmeyi deneyebilirsin.")
    elif len(uygun_oneriler) == 0:
        st.success("Sepetindeki ürünlere göre şu anda yeni öneri bulunmuyor. Farklı ürünler ekleyerek yeni öneriler oluşturabilirsin.")
    else:
        oneri_kolonlari = st.columns(2)
        for i, oneri in enumerate(uygun_oneriler):
            with oneri_kolonlari[i % 2]:
                emoji = urun_emojisi(oneri["onerilen"])
                st.markdown(f'''<div class="pink-box"><b>{emoji} {oneri["onerilen"]}</b> ürününü öneriyoruz.<br><br><span>{oneri["aciklama"]}</span><br><br><b>Koşul ürünler:</b> {", ".join(oneri["kosul"])}<br><b>Support:</b> {oneri["support"]:.3f}<br><b>Confidence:</b> {oneri["confidence"]:.3f}<br><b>Lift:</b> {oneri["lift"]:.3f}<br><br><b>Özel indirim: %{oneri["indirim"]}</b></div>''', unsafe_allow_html=True)
                if st.button(f"{oneri['onerilen']} ürününü indirimli ekle", key=f"oneri_{oneri['onerilen']}"):
                    kural_bilgisi = {"support":oneri["support"],"confidence":oneri["confidence"],"lift":oneri["lift"],"kosul":oneri["kosul"]}
                    sepete_ekle(oneri["onerilen"], indirim_orani=oneri["indirim"], kural_bilgisi=kural_bilgisi)
                    st.rerun()

with st.container(border=True):
    st.subheader("Paket Kampanyaları")
    eksik_kampanyalar = eksik_kampanyalari_getir()
    if len(st.session_state.sepet) == 0:
        st.info("Sepetine ürün eklediğinde paket kampanyaları burada görünür.")
    else:
        if aktif_kampanyalar:
            st.markdown("#### Aktif Olan Kampanyalar")
            for kampanya in aktif_kampanyalar:
                st.markdown(f'<div class="success-box"><b>{kampanya["ad"]}</b> tamamlandı.<br>%{kampanya["indirim"]} indirim uygulandı.<br>Kazanç: <b>{tl_format(kampanya["indirim_tutari"])}</b></div>', unsafe_allow_html=True)
        if eksik_kampanyalar:
            st.markdown("#### Tamamlanabilecek Kampanyalar")
            for kampanya in eksik_kampanyalar:
                st.markdown(f'<div class="warning-box"><b>{kampanya["ad"]}</b> için %{kampanya["indirim"]} indirim fırsatı var.<br>Sepetinde olanlar: <b>{", ".join(kampanya["eslesenler"])}</b><br>Eksik ürünler: <b>{", ".join(kampanya["eksikler"])}</b></div>', unsafe_allow_html=True)
                for eksik_urun in kampanya["eksikler"]:
                    if st.button(f"{eksik_urun} ürününü ekle", key=f"kampanya_{kampanya['ad']}_{eksik_urun}"):
                        sepete_ekle(eksik_urun); st.rerun()
        if not aktif_kampanyalar and not eksik_kampanyalar:
            st.info("Şu an uygun paket kampanyası bulunmuyor.")

with st.container(border=True):
    st.subheader("Apriori ile Oluşturulan Birliktelik Kuralları")
    if kurallar_df.empty:
        st.warning("Seçilen support ve confidence değerleriyle kural oluşturulamadı.")
    else:
        tablo = kurallar_df.copy()
        tablo["Koşul Ürünler"] = tablo["antecedents"].apply(lambda x: ", ".join(list(x)))
        tablo["Önerilen Ürünler"] = tablo["consequents"].apply(lambda x: ", ".join(list(x)))
        tablo = tablo[["Koşul Ürünler","Önerilen Ürünler","support","confidence","lift"]].head(20)
        tablo = tablo.rename(columns={"support":"Support","confidence":"Confidence","lift":"Lift"})
        st.dataframe(tablo, use_container_width=True)

with st.container(border=True):
    st.subheader("Bu Uygulama Hakkında")
    st.write("""
Bu uygulama, proje kapsamında oluşturulan market fişi veri setini kullanarak çalışır.
Önce CSV dosyasındaki fişler okunur ve her fiş bir alışveriş sepeti olarak kabul edilir.
Daha sonra Apriori algoritması ile sık birlikte görülen ürünler bulunur.

Apriori sonucunda support, confidence ve lift değerlerine sahip birliktelik kuralları oluşturulur.
Kullanıcı sepete ürün eklediğinde sistem bu kuralları kontrol eder ve uygun ürün önerilerini gösterir.
Önerilen ürün sepete eklendiğinde ürüne özel indirim uygulanır.

Bu nedenle sistem yalnızca manuel bir alışveriş sepeti değildir; veri setine dayalı çalışan basit bir akıllı öneri sistemi örneğidir.
""")
