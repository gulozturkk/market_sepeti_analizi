import streamlit as st
import pandas as pd
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(#web sayfasının tarayıcı sekmesindeki ayarları yapıyor
    page_title="Market Sepeti Analizi",
    page_icon="🛒",
    layout="wide"
)

#Yardımcı Fonksiyonlar
def sepet_matris_olustur(veriseti):
    # Bu fonksiyon, ham veri setini Apriori algoritmasının kullanabileceği True/False formatlı sepet matrisine dönüştürür.
    sepet_matrisi = veriseti.groupby(['Fiş ID', 'Ürün Adı'])['Adet'].sum().unstack(fill_value=0)
    sepet_matrisi = sepet_matrisi.map(lambda x: True if x > 0 else False)
    return sepet_matrisi


#Yan menü
st.sidebar.title("🛒 Market Sepeti Analizi")
st.sidebar.markdown("---")

st.sidebar.subheader("📂 Veri Seti")
yuklenen_dosya = st.sidebar.file_uploader("CSV dosyası yükle", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Apriori Parametreleri")

min_destek = st.sidebar.slider("Minimum Destek (Support)", 0.01, 0.5, 0.05, 0.01)
min_guven = st.sidebar.slider("Minimum Güven (Confidence)", 0.1, 1.0, 0.3, 0.05)
min_kaldirac = st.sidebar.slider("Minimum Kaldıraç (Lift)", 1.0, 10.0, 1.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("👩‍💻 Proje Grubu")
st.sidebar.info("210710023 - Şeyda ÇORUH\n\n250710093 - Gül ÖZTÜRK")


#Başlık
st.title("🛒 Market Sepeti Analizi")
st.write("Ürün birlikteliklerinin incelenmesi için Apriori algoritması kullanılmıştır.")
st.markdown("---")

#Veri Yükleme
if yuklenen_dosya:#kullanıcı verisetini csv formatında yüklediyse
    sepet_verisi = pd.read_csv(yuklenen_dosya)
else:#dosa yüklenmemişse
    try:
        sepet_verisi = pd.read_csv("fis_veriseti_guncel_660_temiz.csv")
        st.info("📌 Varsayılan veri seti yüklendi: fis_veriseti_guncel_660_temiz.csv")
    except:
        st.error("❌ Veri seti bulunamadı. Sol panelden CSV dosyası yükleyin.")
        st.stop()


#Genel İstatistikler
st.subheader("📊 Genel İstatistikler")

# Veri setindeki eşsiz fiş sayısı, ürün çeşidi, işlem yoğunluğu ve toplam ciro değerleri hesaplanıyor
toplam_fis_sayisi = sepet_verisi['Fiş ID'].nunique()
toplam_urun_cesidi = sepet_verisi['Ürün Adı'].nunique()
toplam_satir_sayisi = len(sepet_verisi)
fis_basi_ortalama_urun = round(len(sepet_verisi) / toplam_fis_sayisi, 2)
toplam_ciro = sepet_verisi['Toplam Tutar (TL)'].sum()

metrik_kolon1, metrik_kolon2, metrik_kolon3, metrik_kolon4, metrik_kolon5 = st.columns(5)
metrik_kolon1.metric("🧾 Toplam Fiş", f"{toplam_fis_sayisi:,}")
metrik_kolon2.metric("📦 Ürün Çeşidi", f"{toplam_urun_cesidi}")
metrik_kolon3.metric("📋 Toplam Satır", f"{toplam_satir_sayisi:,}")
metrik_kolon4.metric("🛍️ Ort. Ürün/Fiş", f"{fis_basi_ortalama_urun}")
metrik_kolon5.metric("💰 Toplam Ciro", f"{toplam_ciro:,.0f} ₺")

st.markdown("---")


# Satış Analizi
st.subheader("📈 Satış Yoğunluğu Analizi")

# Her bir ürün için toplam satış adetini ve ciroyu gruplayarak ürün bazlı satış performans özeti oluşturur.
urun_satis_ozeti = sepet_verisi.groupby('Ürün Adı').agg(
    İşlem_Sayısı=('Adet', 'count'),#ürünün kaç fişte yer aldığı
    Toplam_Adet=('Adet', 'sum'),#ürünün kaç tane satılğığı
    Toplam_Ciro=('Toplam Tutar (TL)', 'sum')
).reset_index().sort_values('İşlem_Sayısı', ascending=False)#tabloyu en çok fişte yer alan üründen en az fişte yer alan ürüne göre sıralar

sol_kolon, sag_kolon = st.columns(2)

with sol_kolon:
    grafik_en_cok_satilan = px.bar(#sütün grafiği oluşturuluyo
        urun_satis_ozeti.head(10),
        x='İşlem_Sayısı', y='Ürün Adı',
        orientation='h',#bar grafiği oluşturuluyor ürün isimleri uzunsa daha kolay okunur
        title="En Çok Satılan 10 Ürün",
        color_discrete_sequence=["#2563eb"]#mavi
    )
    grafik_en_cok_satilan.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(grafik_en_cok_satilan, use_container_width=True)

with sag_kolon:
    grafik_en_az_satilan = px.bar(
        urun_satis_ozeti.tail(10).sort_values('İşlem_Sayısı'),
        x='İşlem_Sayısı', y='Ürün Adı',
        orientation='h',
        title="En Az Satılan 10 Ürün",
        color_discrete_sequence=["#dc2626"]#kırmızı
    )
    grafik_en_az_satilan.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(grafik_en_az_satilan, use_container_width=True)

# Pasta grafik
grafik_ciro_pastasi = px.pie(
    urun_satis_ozeti.nlargest(10, 'Toplam_Ciro'),
    values='Toplam_Ciro',
    names='Ürün Adı',
    title="En Yüksek Cirolu 10 Ürün"
)
st.plotly_chart(grafik_ciro_pastasi, use_container_width=True)

with st.expander("📋 Tüm Ürün Satış Tablosunu Görüntüle"):
    st.dataframe(urun_satis_ozeti, use_container_width=True)

st.markdown("---")


# Apriori Analizi
st.subheader("🔗 Apriori — Birliktelik Kuralları Analizi")
st.write(f"**Parametreler:** Destek ≥ {min_destek} | Güven ≥ {min_guven} | Kaldıraç ≥ {min_kaldirac}")

with st.spinner("Apriori algoritması çalışıyor..."):
    try:
        sepet_matrisi = sepet_matris_olustur(sepet_verisi)
        # Sepet matrisi üzerinden, belirlenen minimum destek (support) eşiğini geçen sık alınan öğe kümelerini bulur.
        sik_ogeler_kumesi = apriori(sepet_matrisi, min_support=min_destek, use_colnames=True)

        if sik_ogeler_kumesi.empty:
            st.warning("⚠️ Bu parametrelerle sık öğe kümesi bulunamadı. Destek değerini düşürün.")
            st.stop()

        # Bulunan sık öğe kümelerinden yola çıkarak, belirlenen güven ve kaldıraç (lift) eşiklerini sağlayan birliktelik (öneri) kurallarını türetir.
        birliktelik_kurallari    = association_rules(sik_ogeler_kumesi, metric="confidence", min_threshold=min_guven)
        
        birliktelik_kurallari = birliktelik_kurallari[birliktelik_kurallari['lift'] >= min_kaldirac].sort_values('lift', ascending=False)

        metrik_sol, metrik_orta, metrik_sag = st.columns(3)
        metrik_sol.metric("🔍 Sık Öğe Kümesi", len(sik_ogeler_kumesi))
        metrik_orta.metric("📐 Birliktelik Kuralı", len(birliktelik_kurallari))
        metrik_sag.metric("🚀 En Yüksek Lift", round(birliktelik_kurallari['lift'].max(), 3) if not birliktelik_kurallari.empty else "-")

        st.markdown("---")

        if not birliktelik_kurallari.empty:
            kurallar_tablosu = birliktelik_kurallari[['antecedents','consequents','support','confidence','lift']].copy()
            
            kurallar_tablosu['antecedents'] = kurallar_tablosu['antecedents'].apply(lambda x: ', '.join(list(x)))
            
            kurallar_tablosu['consequents'] = kurallar_tablosu['consequents'].apply(lambda x: ', '.join(list(x)))
            
            kurallar_tablosu.columns = ['Eğer', 'Öyleyse', 'Destek', 'Güven', 'Kaldıraç']
            kurallar_tablosu = kurallar_tablosu.reset_index(drop=True)#tablonun satır numaralrını sıfırdan başlayacak şekilde düzenler

            st.subheader("📊 Birliktelik Kuralları Tablosu")
            st.dataframe(
                kurallar_tablosu.style.format({'Destek': '{:.4f}', 'Güven': '{:.4f}', 'Kaldıraç': '{:.4f}'}),
                use_container_width=True,
                height=400
            )

            st.subheader("📉 Güven vs Kaldıraç Dağılımı")
            dagilim_verisi = kurallar_tablosu.copy()
            dagilim_verisi['Kural'] = dagilim_verisi['Eğer'] + ' → ' + dagilim_verisi['Öyleyse']
            
            grafik_dagilim = px.scatter(#her bir kuralı grafikte baloncuk şeklinde gösterir
                dagilim_verisi,
                x='Güven',
                y='Kaldıraç',
                color='Kaldıraç',
                size='Destek',
                size_max=20,
                hover_name='Kural',
                hover_data={'Güven': ':.3f', 'Kaldıraç': ':.3f', 'Destek': ':.3f'},
                color_continuous_scale='Blues',
                title="Güven (Confidence) - Kaldıraç (Lift) Dağılımı",
                height=450
            )
            grafik_dagilim.update_layout(
                xaxis_title="Güven (Confidence)",
                yaxis_title="Kaldıraç (Lift)",
            )
            st.plotly_chart(grafik_dagilim, use_container_width=True)

            st.subheader("🏅 En Güçlü 15 Birliktelik Kuralı")
            ilk_15_kural = kurallar_tablosu.head(15).copy()
            ilk_15_kural['Kural'] = ilk_15_kural['Eğer'] + ' → ' + ilk_15_kural['Öyleyse']
            
            grafik_en_guclu_kurallar = px.bar(#sütun grafiği oluşturuluyor
                ilk_15_kural, x='Kaldıraç', y='Kural',
                orientation='h',
                title="En Yüksek Kaldıraçlı 15 Kural",
                color_discrete_sequence=["#059669"]#koyu yeşil
            )
            grafik_en_guclu_kurallar.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
            st.plotly_chart(grafik_en_guclu_kurallar, use_container_width=True)
            
            # Strateji Önerileri
            st.markdown("---")
            st.subheader("💡 Veriye Dayalı Satış Stratejileri")
            st.info("Aşağıdaki stratejiler, yukarıda bulunan en güçlü birliktelik kurallarına göre otomatik olarak üretilmiştir. Çapraz satış fırsatlarını değerlendirmek için kullanılabilir.")
            
            for index, row in ilk_15_kural.head(3).iterrows():
                eger = row['Eğer']
                oyleyse = row['Öyleyse']
                guven = row['Güven'] * 100
                st.success(f"**Strateji {index+1} (Çapraz Satış):** Müşterilerin **{eger}** aldığında **{oyleyse}** alma olasılığı **%{guven:.1f}**. Bu ürünler yan yana dizilebilir veya '{eger} alana {oyleyse} %10 indirimli' kampanyası yapılabilir.")
                
            en_az_satan = urun_satis_ozeti.tail(1)['Ürün Adı'].values[0]
            st.warning(f"**Strateji (Promosyon):** **{en_az_satan}** mağazada en az satılan ürün durumunda. Satışını artırmak için kasa arkası fırsat ürünü olarak sunulabilir veya popüler ürünlerin yanında promosyonlu satılabilir.")

        else:
            st.warning("⚠️ Seçilen parametrelerle kural bulunamadı. Güven veya Kaldıraç değerini düşürün.")

    except Exception as e:
        st.error(f"Hata: {e}")

st.markdown("---")
st.caption("Market Sepeti Analizi | Şeyda ÇORUH & Gül ÖZTÜRK | Veri Madenciliği Projesi 2026")
