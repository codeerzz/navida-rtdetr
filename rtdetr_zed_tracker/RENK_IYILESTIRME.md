# Duba Renk Tespitinde Yapılan İyileştirmeler (Basit Anlatım)

Bu belge, robotun kameradan gördüğü dubaların rengini (kırmızı / yeşil / siyah)
daha güvenilir tahmin etmesi için yaptığımız değişiklikleri, teknik bilgisi
olmayan biri de anlayabilsin diye sade bir dille anlatıyor.

Önce en önemli şeyi söyleyelim: **hangi rengin hangi sayılara karşılık geldiğini
belirleyen, gerçek kamerada elle kalibre edilmiş değerlere hiç dokunmadık.**
Sadece robotun bu değerleri "okuma biçimini" iyileştirdik.

---

## Kısaca Ne Yaptık?

Dört farklı iyileştirme ekledik. Hepsi robotun kamerasından gelen görüntüye
otomatik olarak uygulanıyor, elle bir şey yapmaya gerek yok. İstenirse dördü de
tek tek kapatılabilir (ayarlardan), ya da eski haline tamamen dönülebilir --
eski kodun tamamı, yeni kodun hemen yanında yorum satırı olarak duruyor.

1. **Görüntüyü önce düzeltiyoruz** (beyaz dengesi + kontrast)
2. **Güneş yansımasını (parlamayı) devre dışı bırakıyoruz**
3. **Kırmızı için "ikinci bir kontrol" ekledik** (kayıp bir kırmızıyı kurtarmak için)
4. **Yeşil için "iki yöntem de aynı fikirde olmalı" kuralı ekledik** (yanlış yeşil demeyi azaltmak için)

Aşağıda her birini tek tek açıklıyoruz.

---

## 1. Görüntüyü Önce Düzeltiyoruz

Deniz üstünde çekilen görüntüler genelde hafif mavi-yeşile çalar -- suyun ve
gökyüzünün rengi kameranın "beyaz dengesini" etkiler, bu da kırmızı rengi
soluk/donuk gösterebilir. Ayrıca karanlık ya da gölgeli anlarda renkler
birbirine yakınlaşıp ayırt edilmesi zorlaşabilir.

Bunu düzeltmek için görüntü, renk tespitine girmeden hemen önce iki adımdan
geçiyor:

- **Beyaz dengesi düzeltmesi**: Görüntüdeki genel renk kaymasını (mavi-yeşil
  baskınlığını) otomatik olarak dengeler, böylece kırmızı gerçek kırmızılığına
  daha yakın görünür.
- **Kontrast artırma** (sadece parlaklıkta, renklerde değil): Karanlık
  bölgelerdeki ayrıntıyı biraz daha görünür hale getirir, renk bilgisine
  dokunmadan.

Bu adım, tüm karede (fotoğrafın tamamında) yapılıyor, çünkü "ortalama rengin ne
olması gerektiğini" anlamak için sadece dubanın kendisine değil, çevresine de
bakmak gerekiyor.

---

## 2. Güneş Yansımasını (Parlamayı) Devre Dışı Bırakıyoruz

Güneş suya ya da dubanın kendisine çok parlak bir şekilde vurduğunda, o
noktadaki piksel neredeyse bembeyaz görünür ve gerçek rengini kaybeder. Eskiden
bu aşırı parlak noktalar da renk kararına dahil ediliyordu ve yanlış renk
seçilmesine katkıda bulunabiliyordu.

Şimdi: bir nokta çok parlaksa (neredeyse beyaza yakınsa), o nokta renk
kararından tamamen çıkarılıyor -- sanki hiç orada değilmiş gibi. Böylece sadece
gerçek renk bilgisi taşıyan pikseller karar veriyor.

---

## 3. Kırmızı İçin "İkinci Bir Kontrol" (Kurtarma Amaçlı)

Ana renk tespiti yöntemimiz, bazı durumlarda gerçekten kırmızı olan bir dubayı
"emin olamadım" diye işaretleyebiliyordu -- özellikle parlamanın kırmızıyı biraz
soldurduğu durumlarda.

Bunun için ikinci, bağımsız bir renk okuma yöntemi (farklı bir matematiksel
bakış açısı) devreye giriyor: eğer ana yöntem "emin değilim" derse ya da zaten
"kırmızı" diyorsa, bu ikinci yöntem de kırmızıyı destekliyorsa, karar kırmızı
olarak onaylanıyor ya da güven puanı artırılıyor. İkinci yöntem kırmızı
görmüyorsa hiçbir şey değişmiyor -- sadece ek destek olarak çalışıyor, asla
tek başına karar vermiyor.

Yani bu adım sadece kırmızıyı **kurtarmaya** yarıyor, hiçbir zaman yanlışlıkla
kırmızı **uydurmuyor**.

---

## 4. Yeşil İçin "İki Yöntem de Aynı Fikirde Olmalı" Kuralı

Yeşilde ise ters bir sorun vardı: bazen su rengi, yosun ya da yansımalar,
ana yöntemin gözünde yanlışlıkla "yeşil" gibi görünebiliyordu -- gerçekte
ortada yeşil bir duba yokken.

Bunu azaltmak için: ana yöntem "yeşil" derse, ikinci (bağımsız) yöntemin de
aynı fikirde olması isteniyor. İkisi de yeşil diyorsa karar yeşil olarak
onaylanıyor. Ama ikinci yöntem yeşili DESTEKLEMİYORSA, o karar iptal ediliyor
-- ana yöntemin "yeşil" demesi tek başına artık yeterli değil.

İptal edildiğinde robot otomatik olarak "belirsiz" demiyor; önce "acaba başka
bir renk mi olabilir" diye bakıyor (örneğin aslında kırmızı ya da siyah olan
bir dubaya yanlışlıkla yeşil dendiyse, gerçek rengi hâlâ yakalayabiliyor).
Hiçbir renk yeterince güçlü değilse ancak o zaman "belirsiz" diyor.

Bu kural, gerçek fotoğraflar üzerinde test edildi: yeşil için yanlış-pozitif
(yani "orada yeşil duba yokken yeşil demek") oranını ciddi şekilde düşürdü,
kırmızı ve siyahın doğruluğuna dokunmadı.

---

## Değiştirmediğimiz Şey

`color_ranges.yaml` dosyasındaki kalibre edilmiş sayılara (hangi renk aralığının
"kırmızı", hangisinin "yeşil", hangisinin "siyah" sayılacağını belirleyen
değerler) **hiç dokunmadık**. Bu değerler gerçek kamerayla elle ölçülüp
ayarlanmıştı; biz sadece bu değerlerin nasıl **kullanıldığını** iyileştirdik.

---

## Nasıl Geri Alınır?

Her iyileştirme ayrı ayrı açılıp kapatılabilir (kod değiştirmeden, sadece ayar
değeriyle). Ayrıca her değişikliğin YANINDA, eski (değişiklik öncesi) kodun
tamamı yorum satırı (`#`) olarak duruyor -- silinmedi. Bir sorun çıkarsa:

- Sadece bir özelliği kapatmak için: ilgili ayarı `false` yapmak yeterli.
- Tamamen eski davranışa dönmek için: yorumdaki eski kod açılıp yeni kod
  kapatılabilir.

---

## Nerede Test Edildi?

Bu dört iyileştirme, önce ayrı bir test aracıyla (`scripts/test_buoy_folder.py`)
gerçek duba fotoğrafları ve elle işaretlenmiş (hangi kutunun hangi renk olduğu
belirtilmiş) örnekler üzerinde denendi, sonuçları ölçüldü, ancak **iyi sonuç
verdiği kanıtlandıktan sonra** gerçek robot koduna (`color_classifier.py` ve
`color_classification_node.py`) taşındı. Bu belgedeki dört madde, o testlerde
doğruluğu artırdığı ya da en azından bozmadığı görülen değişikliklerdir.
