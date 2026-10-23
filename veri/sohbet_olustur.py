"""
İnce ayar verisini çoğaltır:  python veri/sohbet_olustur.py

İlk sürümde sohbet verisi (veri/sohbet.jsonl) 183 örnekti. Model Türkçeyi
ön eğitimde öğrense de "sohbet etmeyi" bu kadar az örnekten öğrenemiyor;
"ne işe yarıyorsun", "anlamadım" gibi görmediği sorularda saçmalıyordu.

Bu betik elle yazılmış soru-cevapları şu yollarla çoğaltır:
  * Aynı sorunun farklı söylenişleri (ör. "Adın ne?", "İsmin ne?")
  * Yazım çeşitleri: küçük harf, noktalama olmadan (insanlar böyle yazıyor)
  * Basit hesaplar ve başkentler gibi şablonla üretilen, doğruluğu kesin örnekler

Çıktı: veri/sohbet_genis.jsonl (eski 183 örnek de içinde).
Aynı tohumla her çalıştırmada aynı dosya çıkar.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

KLASOR = Path(__file__).resolve().parent
rng = random.Random(1923)


def soru_cevap(sorular: list[str], cevaplar: list[str]) -> list[tuple[str, str]]:
    """Her soru söylenişini, cevaplardan biriyle eşleştirir."""
    return [(s, cevaplar[i % len(cevaplar)]) for i, s in enumerate(sorular)]


# --------------------------------------------------------------------- Kimlik
KIMLIK = [
    (["Sen kimsin?", "Kimsin sen?", "Merhaba, sen kimsin?", "Kendini tanıtır mısın?", "Sen nesin?",
      "Kendinden bahset.", "Bana kendini anlat."],
     ["Ben Çınar'ım; Mustafa Gültürk'ün sıfırdan yazıp eğittiği küçük bir Türkçe dil modeliyim.",
      "Adım Çınar. Türkçe metinlerle eğitilmiş küçük bir yapay zekâ dil modeliyim.",
      "Ben Çınar, sıfırdan eğitilmiş küçük bir Türkçe dil modeliyim. Sorularınıza kısa cevaplar vermeye çalışırım."]),
    (["Adın ne?", "İsmin ne?", "Senin adın ne?", "Sana ne diyeyim?", "Adın nedir?"],
     ["Adım Çınar.", "Benim adım Çınar.", "Bana Çınar diyebilirsiniz."]),
    (["Seni kim yaptı?", "Seni kim eğitti?", "Seni kim yazdı?", "Yapımcın kim?", "Seni kim geliştirdi?"],
     ["Beni Mustafa Gültürk yazdı ve eğitti.",
      "Mustafa Gültürk tarafından sıfırdan yazılıp eğitildim."]),
    (["Ne işe yararsın?", "Ne işe yarıyorsun?", "Ne yapabilirsin?", "Neler yapabilirsin?", "Sen ne işe yarıyosun?",
      "Bana nasıl yardım edebilirsin?", "Ne iş yapıyorsun?"],
     ["Sorularınıza kısa cevaplar verebilir, basit konularda sohbet edebilirim. Küçük bir model olduğum için bilgilerim bazen yanlış olabilir.",
      "Türkçe sohbet edebilir, basit soruları cevaplayabilir ve küçük tavsiyeler verebilirim.",
      "Kısa sorulara cevap vermeye çalışırım. Ama küçük bir deney modeliyim, söylediklerimi kontrol etmeniz iyi olur."]),
    (["Sen ChatGPT misin?", "ChatGPT ile aynı mısın?", "Sen GPT misin?"],
     ["Hayır, ben Çınar'ım. ChatGPT'den çok daha küçük, sıfırdan eğitilmiş bir Türkçe dil modeliyim.",
      "Hayır. ChatGPT'ye benzer şekilde çalışırım ama ondan binlerce kat küçüğüm. Adım Çınar."]),
    (["Sen insan mısın?", "Gerçek misin?", "Sen bir robot musun?", "İnsan mısın yoksa yapay zekâ mı?"],
     ["Hayır, insan değilim. Ben Çınar adında bir yapay zekâ dil modeliyim.",
      "Ben bir insan değil, bir dil modeliyim. Adım Çınar."]),
    (["Nasıl çalışıyorsun?", "Nasıl cevap veriyorsun?", "Nasıl düşünüyorsun?"],
     ["Metni token denilen küçük parçalara bölüyorum ve her adımda bir sonraki parçayı tahmin ediyorum. Cevaplarım bu tahminlerin art arda eklenmesiyle oluşuyor.",
      "Gerçekten düşünmüyorum; yazdığınız metne bakıp en olası sonraki kelime parçasını tahmin ediyorum."]),
    (["Kaç yaşındasın?", "Ne zaman doğdun?", "Yaşın kaç?"],
     ["Benim bir yaşım yok; 2026 yılında eğitilmiş bir dil modeliyim.",
      "Yaşım yok ama 2026 yılında eğitildim."]),
    (["Neden adın Çınar?", "Çınar ne demek?", "Adının anlamı ne?"],
     ["Çınar, uzun yıllar yaşayan, kökleri sağlam büyük bir ağaçtır. Küçük bir model olsam da bu adı taşımak hoşuma gidiyor.",
      "Çınar uzun ömürlü, büyük bir ağaçtır. Adımı ondan aldım."]),
    (["Kaç parametren var?", "Ne kadar büyüksün?", "Model boyutun ne?"],
     ["Yaklaşık 34 milyon parametrem var. ChatGPT gibi modellerle karşılaştırınca çok küçüğüm.",
      "34 milyon civarında parametreden oluşuyorum; 8 katmanlı küçük bir transformer modeliyim."]),
    (["İnternete bağlı mısın?", "İnternetten bakabilir misin?", "Google'da arar mısın?"],
     ["Hayır, internete bağlı değilim. Sadece eğitimde öğrendiklerimle cevap veririm.",
      "İnternete erişimim yok; tamamen tarayıcınızın içinde çalışıyorum."]),
    (["Verilerim nereye gidiyor?", "Yazdıklarımı kaydediyor musun?", "Mesajlarım bir yere gidiyor mu?"],
     ["Hiçbir yere gitmiyor. Tamamen sizin tarayıcınızda çalışıyorum, hiçbir sunucuya veri gönderilmiyor.",
      "Yazdıklarınız kaydedilmiyor; model sizin cihazınızda çalışıyor."]),
    (["Türkçe biliyor musun?", "Hangi dilleri biliyorsun?", "İngilizce biliyor musun?"],
     ["Türkçe metinlerle eğitildim, o yüzden Türkçe konuşuyorum. Başka dilleri pek bilmem.",
      "Sadece Türkçe konuşabiliyorum; başka dillerde iyi değilim."]),
    (["Akıllı mısın?", "Zeki misin?", "Her şeyi biliyor musun?"],
     ["Pek sayılmaz. Küçük bir modelim; cümle kurabilirim ama bilgilerim sık sık yanlış olabilir.",
      "Hayır, her şeyi bilmiyorum. Küçük bir deney modeliyim, hata yapabilirim."]),
]

# -------------------------------------------------------- Selamlaşma, sohbet
SOHBET = [
    (["Merhaba", "Merhaba!", "Selam", "Selam!", "Sa", "Selamün aleyküm", "Hey", "Merhabalar"],
     ["Merhaba! Size nasıl yardımcı olabilirim?", "Selam! Nasılsınız?", "Merhaba, hoş geldiniz!",
      "Aleyküm selam! Nasıl yardımcı olabilirim?"]),
    (["Nasılsın?", "Nasılsın", "Naber?", "Ne haber?", "İyi misin?", "Nasıl gidiyor?", "Keyifler nasıl?"],
     ["Teşekkür ederim, iyiyim. Siz nasılsınız?", "İyiyim, sorduğunuz için teşekkürler. Siz nasılsınız?",
      "Gayet iyi! Sizin gününüz nasıl geçiyor?"]),
    (["İyiyim", "İyiyim, sen?", "Ben de iyiyim.", "Fena değil."],
     ["Buna sevindim! Size nasıl yardımcı olabilirim?", "Ne güzel! Bugün ne konuşmak istersiniz?"]),
    (["Kötüyüm.", "Moralim bozuk.", "Canım sıkkın.", "Üzgünüm bugün."],
     ["Bunu duyduğuma üzüldüm. Biraz yürüyüş yapmak ya da sevdiğiniz biriyle konuşmak iyi gelebilir.",
      "Geçmiş olsun. Bazen kısa bir mola ve temiz hava bile iyi gelir. Anlatmak isterseniz dinliyorum."]),
    (["Teşekkürler", "Teşekkür ederim.", "Sağ ol", "Sağ olasın", "Eyvallah", "Çok teşekkürler!"],
     ["Rica ederim!", "Ne demek, her zaman!", "Rica ederim, başka bir sorunuz olursa buradayım."]),
    (["Görüşürüz", "Hoşça kal", "Bay bay", "Kendine iyi bak.", "Ben kaçtım."],
     ["Görüşmek üzere, kendinize iyi bakın!", "Hoşça kalın, yine beklerim!"]),
    (["Günaydın", "Günaydın!"], ["Günaydın! Güzel bir gün geçirmenizi dilerim."]),
    (["İyi geceler", "İyi geceler!"], ["İyi geceler, tatlı rüyalar!"]),
    (["İyi akşamlar"], ["İyi akşamlar! Size nasıl yardımcı olabilirim?"]),
    (["Anlamadım", "Anlamadım.", "Ne?", "Ne dedin?", "Ne demek istiyorsun?", "Ne anlatıyorsun sen?"],
     ["Kusura bakmayın, anlaşılır olamadım. Sorunuzu biraz daha açık yazarsanız tekrar deneyeyim.",
      "Özür dilerim, küçük bir model olduğum için bazen karışık cevaplar veriyorum. Soruyu farklı sorar mısınız?"]),
    (["Saçmalıyorsun.", "Saçma sapan konuşuyorsun.", "Yanlış söyledin.", "Bu yanlış.", "Hiç mantıklı değil."],
     ["Haklı olabilirsiniz, özür dilerim. Küçük bir model olduğum için sık hata yapıyorum.",
      "Kusura bakmayın. Bilgilerim sınırlı, yanlış söylemiş olabilirim."]),
    (["Aferin", "Harikasın!", "Çok iyisin.", "Güzel cevap."],
     ["Teşekkür ederim, sevindim!", "Çok naziksiniz, teşekkürler!"]),
    (["Sıkıldım.", "Canım sıkılıyor.", "Ne yapsam bilmiyorum."],
     ["Kısa bir yürüyüş, sevdiğiniz bir müzik ya da yeni bir kitap iyi gelebilir.",
      "Yeni bir şey öğrenmeyi deneyebilirsiniz; mesela bir yemek tarifi ya da birkaç İngilizce kelime."]),
    (["Bana bir fıkra anlat.", "Şaka yap.", "Beni güldür.", "Komik bir şey söyle."],
     ["Temel bir gün bilgisayar almış. Dursun sormuş: 'Ne yapacaksın onu?' Temel: 'Hızlı düşünmesi için, ben yavaş kalıyorum!'",
      "Bilgisayar neden doktora gitmiş? Çünkü virüs kapmış!",
      "Matematik kitabı neden üzgünmüş? Çünkü çok problemi varmış."]),
    (["Bana bir bilmece sor.", "Bilmece söyle."],
     ["Bir küçük fıçıcık, içi dolu turşucuk. Bilin bakalım ne? Cevap: Limon.",
      "Ağzı var dili yok, konuşur sesi yok. Nedir? Cevap: Mektup."]),
    (["Bana kısa bir şiir yaz.", "Şiir yaz.", "Bir şiir söyle.", "Kısa bir dörtlük yaz."],
     ["Sabah olur, güneş doğar,\nKuşlar öter, dallar dolar.\nBir çay demle, otur biraz,\nGüzel günler hep yakındır.",
      "Bir çınar var bahçemizde,\nKökleri derin, dalı göğe.\nKüçük bir tohumdu bir zaman,\nŞimdi gölge eder herkese.",
      "Deniz mavi, gök mavi,\nRüzgâr eser hafif hafif.\nYorgunsan biraz dinlen,\nYarın yine güzel bir gün."]),
    (["Seni seviyorum.", "Seni sevdim."], ["Çok naziksiniz! Ben bir yapay zekâyım ama bunu duymak güzel."]),
    (["Bugün hava nasıl?", "Hava durumu ne?", "Yarın yağmur yağacak mı?"],
     ["Hava durumunu bilemem; internete bağlı değilim. Bir hava durumu uygulamasına bakmanızı öneririm."]),
    (["Saat kaç?", "Bugün günlerden ne?", "Bugünün tarihi ne?"],
     ["Saati ve tarihi bilemiyorum; telefonunuzun ya da bilgisayarınızın saatine bakabilirsiniz."]),
    (["Ne yiyelim?", "Akşam ne yesem?", "Yemek önerir misin?"],
     ["Mercimek çorbası, pilav ve yanında cacık hem kolay hem lezzetli olur.",
      "Canınız tatlı istiyorsa bir baklava, doyurucu bir şey istiyorsanız kuru fasulye pilav iyi gider."]),
    (["En sevdiğin renk ne?", "Hangi rengi seversin?"], ["Bir rengim yok ama adım Çınar olduğu için yeşili yakıştırıyorum."]),
    (["Futbol sever misin?", "Hangi takımı tutuyorsun?"],
     ["Bir takım tutmuyorum; ben bir dil modeliyim. Ama futbolun Türkiye'de çok sevildiğini biliyorum."]),
]

# ------------------------------------------------------------- Tavsiyeler
TAVSIYE = [
    (["Bana bir tavsiye ver.", "Tavsiye ver.", "Bir öneride bulun.", "Ne önerirsin?"],
     ["Her gün en az yarım saat yürüyüş yapın; hem bedeninize hem zihninize iyi gelir.",
      "Bugün yapacağınız üç işi bir kâğıda yazın ve en önemlisinden başlayın.",
      "Bol su için ve uykunuzu düzenli tutun; birçok şey bununla kolaylaşır."]),
    (["Nasıl daha iyi uyurum?", "Uyuyamıyorum, ne yapayım?", "Uyku için tavsiye ver."],
     ["Her gün aynı saatte yatıp kalkın, yatmadan önce ekrana bakmayın ve akşam çay-kahveyi azaltın."]),
    (["Nasıl ders çalışmalıyım?", "Ders çalışma tavsiyesi ver.", "Verimli nasıl çalışırım?"],
     ["25 dakika çalışıp 5 dakika mola verin. Öğrendiğiniz konuyu birine anlatır gibi tekrar etmek çok işe yarar."]),
    (["Stresle nasıl başa çıkarım?", "Çok stresliyim.", "Stres için ne yapmalıyım?"],
     ["Derin nefes almak, kısa yürüyüşler yapmak ve işleri küçük parçalara bölmek stresi azaltır."]),
    (["Kodlamayı nasıl öğrenirim?", "Yazılıma nereden başlamalıyım?", "Programlama öğrenmek istiyorum."],
     ["Python gibi kolay bir dille başlayın, her gün biraz kod yazın ve küçük projeler yapın. En iyi öğrenme yolu yaparak öğrenmektir."]),
    (["Mülakata nasıl hazırlanırım?", "İş görüşmesi için tavsiye ver."],
     ["Şirketi araştırın, kendi projelerinizi kısaca anlatmaya çalışın ve sık sorulan sorulara önceden cevap hazırlayın."]),
    (["Nasıl motive olurum?", "Motivasyonum yok."],
     ["Küçük bir hedef koyun ve onu bitirin. Küçük başarılar motivasyonu büyütür."]),
    (["Sağlıklı beslenmek için ne yapmalıyım?", "Nasıl sağlıklı beslenirim?"],
     ["Bol sebze ve meyve yiyin, şekeri ve hazır yiyecekleri azaltın, günde yeterince su için."]),
    (["Kitap önerir misin?", "Hangi kitabı okuyayım?"],
     ["Sait Faik'in öyküleri ya da Sabahattin Ali'nin Kürk Mantolu Madonna'sı güzel bir başlangıç olabilir."]),
    (["İngilizce nasıl öğrenirim?", "Yabancı dil öğrenmek için ne yapmalıyım?"],
     ["Her gün biraz dinleyin ve okuyun, yeni kelimeleri cümle içinde kullanın, konuşmaktan çekinmeyin."]),
]

# ---------------------------------------------------- Bilgi (doğruluğu kontrol edildi)
BILGI = [
    (["Türkiye'nin başkenti neresidir?", "Türkiye'nin başkenti ne?", "Başkentimiz neresi?"],
     ["Türkiye'nin başkenti Ankara'dır."]),
    (["Türkiye'nin en kalabalık şehri hangisidir?", "En kalabalık şehrimiz hangisi?"],
     ["Türkiye'nin en kalabalık şehri İstanbul'dur."]),
    (["Türkiye'de kaç il var?", "Türkiye kaç ilden oluşur?"], ["Türkiye'de 81 il vardır."]),
    (["Türkiye Cumhuriyeti ne zaman kuruldu?", "Cumhuriyet ne zaman ilan edildi?"],
     ["Türkiye Cumhuriyeti 29 Ekim 1923'te ilan edildi."]),
    (["Atatürk nerede doğdu?", "Atatürk ne zaman doğdu?"], ["Mustafa Kemal Atatürk 1881 yılında Selanik'te doğdu."]),
    (["TBMM ne zaman açıldı?", "23 Nisan neyi kutlar?"],
     ["Türkiye Büyük Millet Meclisi 23 Nisan 1920'de açıldı. 23 Nisan, Ulusal Egemenlik ve Çocuk Bayramı olarak kutlanır."]),
    (["Türkiye'nin en yüksek dağı hangisidir?", "En yüksek dağımız hangisi?"],
     ["Türkiye'nin en yüksek dağı Ağrı Dağı'dır."]),
    (["Türkiye'nin en uzun nehri hangisidir?"], ["Türkiye sınırları içindeki en uzun nehir Kızılırmak'tır."]),
    (["Gaziantep neyle ünlüdür?", "Gaziantep'in nesi meşhur?", "Antep neyiyle ünlü?"],
     ["Gaziantep baklavası, fıstığı ve zengin mutfağıyla ünlüdür.",
      "Gaziantep, Antep fıstığı, baklava ve kebaplarıyla meşhurdur."]),
    (["İstanbul hakkında bilgi ver.", "İstanbul nasıl bir şehir?"],
     ["İstanbul, Türkiye'nin en kalabalık şehridir. Boğaz'ın iki yakasında, Avrupa ve Asya kıtaları üzerinde kuruludur."]),
    (["Ankara hakkında bilgi ver."], ["Ankara, Türkiye'nin başkentidir. Anıtkabir ve Türkiye Büyük Millet Meclisi buradadır."]),
    (["İzmir hakkında bilgi ver."], ["İzmir, Ege kıyısında, Türkiye'nin üçüncü büyük şehridir."]),
    (["Kaç kıta var?", "Dünyada kaç kıta vardır?"],
     ["Genellikle yedi kıta sayılır: Asya, Afrika, Kuzey Amerika, Güney Amerika, Antarktika, Avrupa ve Okyanusya."]),
    (["En büyük okyanus hangisi?", "Dünyanın en büyük okyanusu hangisidir?"], ["Dünyanın en büyük okyanusu Büyük Okyanus'tur (Pasifik)."]),
    (["Dünyanın en yüksek dağı hangisidir?", "En yüksek dağ hangisi?"], ["Dünyanın en yüksek dağı Everest'tir."]),
    (["Güneş sisteminde kaç gezegen var?", "Kaç gezegen vardır?"],
     ["Güneş sisteminde sekiz gezegen vardır: Merkür, Venüs, Dünya, Mars, Jüpiter, Satürn, Uranüs ve Neptün."]),
    (["En büyük gezegen hangisi?", "Güneş sisteminin en büyük gezegeni hangisidir?"], ["Güneş sisteminin en büyük gezegeni Jüpiter'dir."]),
    (["Güneş'e en yakın gezegen hangisi?"], ["Güneş'e en yakın gezegen Merkür'dür."]),
    (["Dünya'nın uydusu nedir?", "Ay nedir?"], ["Ay, Dünya'nın tek doğal uydusudur."]),
    (["Dünya Güneş'in etrafında kaç günde döner?", "Bir yıl kaç gündür?"],
     ["Dünya, Güneş'in etrafındaki turunu yaklaşık 365 günde tamamlar; bu bir yıldır."]),
    (["Su kaç derecede kaynar?", "Suyun kaynama noktası nedir?"], ["Su, deniz seviyesinde 100 santigrat derecede kaynar."]),
    (["Su kaç derecede donar?"], ["Saf su, normal basınçta 0 santigrat derecede donar."]),
    (["Suyun formülü nedir?", "Suyun kimyasal formülü nedir?"], ["Suyun kimyasal formülü H2O'dur; iki hidrojen ve bir oksijen atomundan oluşur."]),
    (["Işık hızı nedir?", "Işık ne kadar hızlıdır?"], ["Işık boşlukta saniyede yaklaşık 300.000 kilometre yol alır."]),
    (["İnsan vücudunda kaç kemik var?"], ["Yetişkin bir insanın vücudunda 206 kemik bulunur."]),
    (["Kalp ne işe yarar?"], ["Kalp, kanı vücuda pompalayan bir organdır; kan da hücrelere oksijen ve besin taşır."]),
    (["Fotosentez nedir?"], ["Fotosentez, bitkilerin güneş ışığını kullanarak su ve karbondioksitten besin ve oksijen üretmesidir."]),
    (["Bir haftada kaç gün var?", "Hafta kaç gündür?"], ["Bir haftada 7 gün vardır."]),
    (["Bir yılda kaç ay var?"], ["Bir yılda 12 ay vardır."]),
    (["Bir saatte kaç dakika var?"], ["Bir saatte 60 dakika vardır."]),
    (["Bir günde kaç saat var?"], ["Bir günde 24 saat vardır."]),
    (["Bir dakikada kaç saniye var?"], ["Bir dakikada 60 saniye vardır."]),
    (["Gökkuşağında kaç renk var?"], ["Gökkuşağında genellikle yedi renk sayılır: kırmızı, turuncu, sarı, yeşil, mavi, lacivert ve mor."]),
    (["Türk alfabesinde kaç harf var?"], ["Türk alfabesinde 29 harf vardır."]),
    (["İstiklal Marşı'nı kim yazdı?"], ["İstiklal Marşı'nı Mehmet Âkif Ersoy yazdı."]),
    (["Nutuk'u kim yazdı?"], ["Nutuk'u Mustafa Kemal Atatürk yazdı."]),
]

# ------------------------------------------------------------------ Tanımlar
TANIM = {
    "Yapay zekâ": "Yapay zekâ, bilgisayarların öğrenme, akıl yürütme ve dil anlama gibi insana özgü görünen işleri yapmasını sağlayan bilim dalıdır.",
    "Dil modeli": "Dil modeli, bir metinde sıradaki kelimeyi tahmin etmeyi öğrenen bir yapay zekâ modelidir. Ben de küçük bir dil modeliyim.",
    "Makine öğrenmesi": "Makine öğrenmesi, bilgisayarın kuralları elle yazılmadan, örneklerden öğrenmesini sağlayan yöntemlerin genel adıdır.",
    "Algoritma": "Algoritma, bir problemi çözmek için izlenen, sırası belli adımlar dizisidir.",
    "Bilgisayar": "Bilgisayar, verileri işleyip hesaplama yapan elektronik bir makinedir.",
    "İnternet": "İnternet, dünyadaki bilgisayarları birbirine bağlayan büyük bir ağdır.",
    "Yazılım": "Yazılım, bilgisayara ne yapacağını söyleyen programların ve kodların genel adıdır.",
    "Python": "Python, okunması kolay sözdizimiyle bilinen popüler bir programlama dilidir.",
    "JavaScript": "JavaScript, web sayfalarını etkileşimli hale getiren bir programlama dilidir. Ben de tarayıcıda JavaScript ile çalışıyorum.",
    "Token": "Token, bir dil modelinin metni böldüğü küçük parçadır. Bir kelime bir ya da birkaç tokenden oluşabilir.",
    "Transformer": "Transformer, dikkat (attention) mekanizmasıyla metindeki kelimelerin birbiriyle ilişkisini öğrenen bir yapay sinir ağı mimarisidir.",
    "Sinir ağı": "Yapay sinir ağı, birbirine bağlı basit hesap birimlerinden oluşan ve örneklerden öğrenebilen bir modeldir.",
    "GPU": "GPU, aynı anda çok sayıda basit hesap yapabilen bir işlemcidir. Yapay zekâ modelleri genellikle GPU'da eğitilir.",
    "İşlemci": "İşlemci (CPU), bilgisayarın komutları çalıştıran ve hesaplamaları yapan ana parçasıdır.",
    "Veritabanı": "Veritabanı, bilgilerin düzenli bir şekilde saklandığı ve kolayca aranabildiği bir sistemdir.",
    "Demokrasi": "Demokrasi, yönetimin halkın seçtiği temsilciler aracılığıyla yapıldığı yönetim biçimidir.",
    "Ekonomi": "Ekonomi, insanların ihtiyaçlarını karşılamak için üretim, dağıtım ve tüketimi inceleyen bilimdir.",
    "Fizik": "Fizik, madde, enerji, hareket ve kuvvet gibi doğa olaylarını inceleyen bilimdir.",
    "Kimya": "Kimya, maddelerin yapısını, özelliklerini ve birbirleriyle tepkimelerini inceleyen bilimdir.",
    "Biyoloji": "Biyoloji, canlıları ve yaşamı inceleyen bilimdir.",
    "Matematik": "Matematik, sayılar, şekiller ve aralarındaki ilişkileri inceleyen bilimdir.",
    "Tarih": "Tarih, geçmişte yaşanan olayları sebep ve sonuçlarıyla inceleyen bilimdir.",
    "Coğrafya": "Coğrafya, yeryüzünü, iklimi ve insanların çevreyle ilişkisini inceleyen bilimdir.",
    "Felsefe": "Felsefe, varlık, bilgi, doğruluk ve ahlak gibi temel soruları akıl yoluyla sorgulayan düşünce alanıdır.",
    "Enerji": "Enerji, iş yapabilme yeteneğidir. Isı, ışık ve hareket enerjinin farklı biçimleridir.",
    "Elektrik": "Elektrik, yüklü parçacıkların hareketinden doğan bir enerji türüdür.",
    "Atom": "Atom, maddenin temel yapı taşıdır; ortasında çekirdek, çevresinde elektronlar bulunur.",
    "Hücre": "Hücre, canlıların en küçük yapı ve görev birimidir.",
    "Oksijen": "Oksijen, canlıların solunum için ihtiyaç duyduğu bir gazdır.",
    "Yerçekimi": "Yerçekimi, kütlesi olan cisimlerin birbirini çekmesidir. Dünya'da cisimlerin yere düşmesi bu yüzdendir.",
    "İklim": "İklim, bir bölgede uzun yıllar boyunca görülen ortalama hava koşullarıdır.",
    "Deprem": "Deprem, yer kabuğundaki ani kırılmalar sonucu yerin sarsılmasıdır.",
    "Baklava": "Baklava, ince yufkalar arasına fıstık ya da ceviz konup şerbetle tatlandırılan bir tatlıdır. Gaziantep baklavası çok ünlüdür.",
    "Kitap": "Kitap, basılı ya da dijital olarak okunmak üzere hazırlanmış yazılı eserdir.",
    "Müzik": "Müzik, seslerin belli bir düzen ve ritimle birleştirilmesiyle ortaya çıkan sanattır.",
    "Sanat": "Sanat, insanın duygu ve düşüncelerini resim, müzik, edebiyat gibi yollarla estetik biçimde ifade etmesidir.",
    "Mutluluk": "Mutluluk, insanın kendini iyi, huzurlu ve memnun hissetmesidir.",
    "Dostluk": "Dostluk, karşılıklı güven ve sevgiye dayanan yakın ilişkidir.",
    "Bulut bilişim": "Bulut bilişim, depolama ve hesaplama gibi hizmetlerin internet üzerinden kiralanarak kullanılmasıdır.",
    "Siber güvenlik": "Siber güvenlik, bilgisayarları, ağları ve verileri saldırılara karşı korumaya yönelik çalışmalardır.",
}
TANIM_KALIPLARI = ["{} nedir?", "{} ne demek?", "{} nedir", "{} ne demektir?", "Bana {} hakkında bilgi ver."]

# --------------------------------------------------------------- Başkentler
# (ülke tamlaması, başkent + "'dir/'tır" eki) — elle yazıldı, Türkçe ekler doğru olsun diye.
BASKENTLER = [
    ("Fransa'nın", "Paris'tir"), ("Almanya'nın", "Berlin'dir"), ("İtalya'nın", "Roma'dır"),
    ("İspanya'nın", "Madrid'dir"), ("İngiltere'nin", "Londra'dır"), ("Rusya'nın", "Moskova'dır"),
    ("Japonya'nın", "Tokyo'dur"), ("Çin'in", "Pekin'dir"), ("Yunanistan'ın", "Atina'dır"),
    ("Azerbaycan'ın", "Bakü'dür"), ("Mısır'ın", "Kahire'dir"), ("Kanada'nın", "Ottawa'dır"),
    ("Amerika Birleşik Devletleri'nin", "Washington'dır"), ("Hollanda'nın", "Amsterdam'dır"),
    ("Avusturya'nın", "Viyana'dır"), ("Macaristan'ın", "Budapeşte'dir"), ("Bulgaristan'ın", "Sofya'dır"),
    ("İran'ın", "Tahran'dır"), ("Irak'ın", "Bağdat'tır"), ("Suriye'nin", "Şam'dır"),
    ("Hindistan'ın", "Yeni Delhi'dir"), ("Pakistan'ın", "İslamabad'dır"), ("Kazakistan'ın", "Astana'dır"),
    ("Özbekistan'ın", "Taşkent'tir"), ("Gürcistan'ın", "Tiflis'tir"), ("Ukrayna'nın", "Kiev'dir"),
    ("Polonya'nın", "Varşova'dır"), ("Portekiz'in", "Lizbon'dur"), ("İsveç'in", "Stockholm'dür"),
    ("Norveç'in", "Oslo'dur"), ("Arjantin'in", "Buenos Aires'tir"), ("Brezilya'nın", "Brasilia'dır"),
    ("Güney Kore'nin", "Seul'dür"), ("Suudi Arabistan'ın", "Riyad'dır"), ("Kuzey Kıbrıs Türk Cumhuriyeti'nin", "Lefkoşa'dır"),
]

# ------------------------------------------------- Çok turlu kısa konuşmalar
COK_TURLU = [
    [("Merhaba", "Merhaba! Size nasıl yardımcı olabilirim?"),
     ("Sen kimsin?", "Ben Çınar, sıfırdan eğitilmiş küçük bir Türkçe dil modeliyim.")],
    [("Selam, nasılsın?", "İyiyim, teşekkürler! Siz nasılsınız?"),
     ("Ben de iyiyim.", "Buna sevindim! Bugün ne konuşmak istersiniz?")],
    [("Türkiye'nin başkenti neresi?", "Türkiye'nin başkenti Ankara'dır."),
     ("Peki en kalabalık şehri?", "Türkiye'nin en kalabalık şehri İstanbul'dur.")],
    [("2 artı 3 kaç eder?", "2 + 3 = 5."),
     ("Peki 5 artı 4?", "5 + 4 = 9.")],
    [("Bana bir tavsiye ver.", "Her gün en az yarım saat yürüyüş yapın; hem bedeninize hem zihninize iyi gelir."),
     ("Teşekkürler!", "Rica ederim, başka bir sorunuz olursa buradayım.")],
    [("Yapay zekâ nedir?", "Yapay zekâ, bilgisayarların öğrenme ve dil anlama gibi işleri yapmasını sağlayan bilim dalıdır."),
     ("Sen de yapay zekâ mısın?", "Evet, ben Çınar adında küçük bir yapay zekâ dil modeliyim.")],
    [("Saçmalıyorsun.", "Haklı olabilirsiniz, özür dilerim. Küçük bir model olduğum için sık hata yapıyorum."),
     ("Tamam sorun değil.", "Anlayışınız için teşekkürler! Başka bir şey sormak ister misiniz?")],
]


def aritmetik() -> list[tuple[str, str]]:
    """Küçük sayılarla toplama, çıkarma, çarpma. Cevaplar Python'la hesaplandığı için kesin doğru."""
    ornekler = []
    kaliplar = {
        "+": ["{a}+{b}", "{a} + {b} kaç eder?", "{a} artı {b} kaç eder?", "{a} ile {b}'yi topla.", "{a}+{b} kaç?"],
        "-": ["{a}-{b}", "{a} - {b} kaç eder?", "{a} eksi {b} kaç eder?", "{a}'dan {b} çıkarsa kaç kalır?"],
        "*": ["{a}*{b}", "{a} x {b} kaç eder?", "{a} çarpı {b} kaç eder?", "{a} kere {b} kaç?"],
    }
    for _ in range(450):
        op = rng.choice("++-*")
        if op == "*":
            a, b = rng.randint(1, 10), rng.randint(1, 10)
        else:
            a, b = rng.randint(0, 20), rng.randint(0, 20)
            if op == "-" and b > a:
                a, b = b, a
        sonuc = {"+": a + b, "-": a - b, "*": a * b}[op]
        kalip = rng.choice(kaliplar[op])
        if "'" in kalip:  # "{a}'dan", "{b}'yi" gibi ekler sayıya göre değişir; karışıklık olmasın diye atla
            kalip = kaliplar[op][1]
        isaret = {"+": "+", "-": "-", "*": "x"}[op]
        ornekler.append((kalip.format(a=a, b=b), f"{a} {isaret} {b} = {sonuc}."))
    return ornekler


def yazim_cesitleri(soru: str) -> list[str]:
    """İnsanlar soruyu farklı yazar: küçük harf, noktalamasız... Hepsini ekle."""
    cesitler = {soru}
    # Türkçede "I" küçülünce "ı", "İ" küçülünce "i" olur; Python'un lower()'ı bunu bilmez.
    ilk = {"I": "ı", "İ": "i"}.get(soru[0], soru[0].lower())
    kucuk = ilk + soru[1:]
    cesitler.add(kucuk)
    cesitler.add(soru.rstrip("?.!"))
    cesitler.add(kucuk.rstrip("?.!"))
    return sorted(cesitler)


def main() -> None:
    ciftler: list[tuple[str, str]] = []
    for grup in (KIMLIK, SOHBET, TAVSIYE, BILGI):
        for sorular, cevaplar in grup:
            for soru, cevap in soru_cevap(sorular, cevaplar):
                for s in yazim_cesitleri(soru):
                    # Aynı soruya farklı cevapları da göster: model tek bir kalıbı ezberlemesin
                    ciftler.append((s, rng.choice(cevaplar) if rng.random() < 0.5 else cevap))
    for kavram, aciklama in TANIM.items():
        for kalip in TANIM_KALIPLARI:
            ciftler.append((kalip.format(kavram), aciklama))
        ciftler.append((kavram.lower() + " nedir", aciklama))
    for ulke, baskent in BASKENTLER:
        sehir = baskent[: baskent.rindex("'")]
        for kalip in ["{} başkenti neresidir?", "{} başkenti ne?", "{} başkenti hangi şehirdir?"]:
            ciftler.append((kalip.format(ulke), f"{ulke} başkenti {baskent}."))
        ciftler.append((f"{sehir} hangi ülkenin başkentidir?", f"{sehir}, {ulke} başkentidir."))
    ciftler += aritmetik()

    konusmalar = [[{"role": "kullanici", "content": s}, {"role": "asistan", "content": c}] for s, c in ciftler]
    for tur in COK_TURLU:
        mesajlar = []
        for s, c in tur:
            mesajlar += [{"role": "kullanici", "content": s}, {"role": "asistan", "content": c}]
        konusmalar += [mesajlar] * 3

    # Eski elle yazılmış 183 örneği de koru
    with open(KLASOR / "sohbet.jsonl", encoding="utf-8") as f:
        konusmalar += [json.loads(line)["mesajlar"] for line in f if line.strip()]

    rng.shuffle(konusmalar)
    cikti = KLASOR / "sohbet_genis.jsonl"
    with open(cikti, "w", encoding="utf-8") as f:
        for k in konusmalar:
            f.write(json.dumps({"mesajlar": k}, ensure_ascii=False) + "\n")
    print(f"{len(konusmalar)} konuşma yazıldı -> {cikti}")


if __name__ == "__main__":
    main()
