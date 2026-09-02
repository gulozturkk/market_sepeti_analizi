***Akıllı Alışveriş Sepeti ve Ürün Öneri Sistemi (BenimSepetim)
Atatürk Üniversitesi Mühendislik Fakültesi Yazılım Mühendisliği Bölümü Veri Madenciliği dersi kapsamında geliştirilen bu proje; gerçek market fişlerinden elde edilen verileri kullanarak Apriori algoritması ile sepet analizi yapan ve kullanıcı sepetine göre anlık akıllı ürün önerileri sunan web tabanlı bir uygulamadır.

***Projenin Amacı ve Kapsamı
Bu proje, veri madenciliği sonuçlarını yalnızca statik bir analiz raporu olarak bırakmayıp, etkileşimli bir e-ticaret/market sepeti simülasyonuna dönüştürmeyi amaçlar:

Otomatik Veri İşleme: Fiş veri setinden benzersiz ürün listesini, ortalama fiyatları ve satış adetlerini otomatik türetir.

Apriori Algoritması: Ürünler arasındaki gizli birliktelik ilişkilerini çıkarır.

Akıllı Öneri ve İndirim: Kullanıcının sepetindeki ürünlere göre Support, Confidence ve Lift metriklerini hesaplayarak en uygun ürünü indirimli şekilde önerir.

Paket Kampanyaları: Tamamlayıcı ürün grupları için çapraz satış indirimleri sunar.

***Kullanılan Teknolojiler ve Kütüphaneler
Projede kullanılan ana teknolojiler ve alt yapı bileşenleri:

Python: Mantıksal işlemler, veri işleme ve algoritma yönetimi.

Streamlit: Hızlı ve etkileşimli web arayüzü geliştirme.

Pandas: Veri seti analizi, gruplama ve sepet matrisi oluşturma.

Mlxtend: Apriori algoritması ve birliktelik kuralı çıkarımı.

CSS: Modern ve kullanıcı dostu arayüz tasarımı.

***Kurulum ve Çalıştırma
Projeyi bilgisayarınızda çalıştırmak için terminale sırasıyla şu komutları yazabilirsiniz:

git clone https://github.com/gulozturkk/pazar_sepeti_analizi.git

cd pazar_sepeti_analizi

pip install streamlit pandas mlxtend

streamlit run vmProje/benimsepetim.py
