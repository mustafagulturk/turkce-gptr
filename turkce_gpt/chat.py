"""
Sohbet şablonu ve ince ayar için etiket (label) maskesi.

Ön eğitimli model sadece "metnin devamını tahmin etmeyi" bilir. Ona sohbet
etmeyi öğretmek için konuşmaları sabit bir kalıba koyarız:

    <|kullanici|>Merhaba!<|asistan|>Merhaba, nasıl yardımcı olabilirim?<|son|>

Çok turlu bir konuşmada bu parçalar art arda eklenir:

    <|kullanici|>soru 1<|asistan|>cevap 1<|son|><|kullanici|>soru 2<|asistan|>cevap 2<|son|>

Sohbet sırasında (ince ayardan sonra) model'e şu metni veririz:

    <|kullanici|>soru<|asistan|>

Model "<|asistan|>"dan sonrasını tamamlar ve "<|son|>" ürettiğinde dururuz.

Etiket maskesi neden gerekli?
    İnce ayarda modelin KULLANICI gibi konuşmayı değil, ASİSTAN gibi cevap
    vermeyi öğrenmesini isteriz. Bu yüzden kayıp (loss) sadece asistanın
    cevabı ve cevabı bitiren "<|son|>" tokeni üzerinden hesaplanır. Diğer
    konumların etiketini -100 yaparız; PyTorch'un cross_entropy fonksiyonu
    (ignore_index=-100) bu konumları hesaba katmaz.

    "<|son|>"u öğrenmek çok önemlidir: model ne zaman susacağını buradan öğrenir.
    Bunu öğrenmeyen model cevabın sonunda durmaz, saçmalamaya devam eder.
"""

from __future__ import annotations

from .tokenizer import ASSISTANT, END_OF_TEXT, USER, BPETokenizer

# Kayıp hesabında yok sayılacak konumların etiketi (PyTorch'un varsayılanı).
IGNORE_INDEX = -100

# Mesajlardaki "role" alanının alabileceği değerler.
ROLE_USER = "kullanici"
ROLE_ASSISTANT = "asistan"


def _check_role(role: str) -> None:
    if role not in (ROLE_USER, ROLE_ASSISTANT):
        raise ValueError(f"Bilinmeyen rol '{role}'. 'kullanici' veya 'asistan' olmalı.")


def format_prompt(messages: list[dict]) -> str:
    """Konuşma geçmişini modelin tamamlayacağı metne çevirir.

    Sona "<|asistan|>" eklenir; yani model sıradaki asistan cevabını yazar.

    Örnek:
        format_prompt([{"role": "kullanici", "content": "Selam"}])
        -> "<|kullanici|>Selam<|asistan|>"

    DİKKAT: Bu fonksiyon okunabilir bir METİN döndürür. Metni tokenlere
    çevirirken allow_special=True kullanırsanız, kullanıcının yazdığı
    "<|asistan|>" yazısı da özel token sanılır. Güvenli kodlama için
    encode_prompt() fonksiyonunu kullanın.
    """
    parts = []
    for msg in messages:
        _check_role(msg["role"])
        if msg["role"] == ROLE_USER:
            parts.append(USER + msg["content"])
        else:
            parts.append(ASSISTANT + msg["content"] + END_OF_TEXT)
    parts.append(ASSISTANT)
    return "".join(parts)


def encode_conversation(tokenizer: BPETokenizer, messages: list[dict]) -> tuple[list[int], list[int]]:
    """Bir konuşmayı token numaralarına ve eğitim etiketlerine çevirir.

    Dönen iki liste AYNI uzunluktadır ve konum konum eşleşir:
        ids[i]    : i. token
        labels[i] : ids[i] öğrenilecekse ids[i], öğrenilmeyecekse -100

    Yani labels "bu token öğrenilecek mi?" sorusunun cevabıdır. Modelin girdisi
    ve hedefi arasındaki bir kaydırmayı (girdi = ids[:-1], hedef = labels[1:])
    eğitim kodu yapar; böylece model i. konumda (i+1). tokeni tahmin etmeyi öğrenir.

    Örnek ("U"=<|kullanici|>, "A"=<|asistan|>, "S"=<|son|>):
        tokenler : U  se  lam  A  mer  haba  S
        etiket   : -  -   -    -  mer  haba  S     ("-" = -100)

    Kullanıcı ve asistan metinleri allow_special=False ile kodlanır: metnin
    içinde "<|asistan|>" yazsa bile bu sıradan karakterler olarak kalır ve
    konuşmanın yapısını bozamaz (bir tür "prompt injection" koruması).
    """
    user_id = tokenizer.special_id(USER)
    assistant_id = tokenizer.special_id(ASSISTANT)
    end_id = tokenizer.special_id(END_OF_TEXT)

    ids: list[int] = []
    labels: list[int] = []
    for msg in messages:
        _check_role(msg["role"])
        content_ids = tokenizer.encode(msg["content"], allow_special=False)
        if msg["role"] == ROLE_USER:
            # Kullanıcı kısmının hiçbiri öğrenilmez: başlık tokeni + metin.
            chunk = [user_id] + content_ids
            ids += chunk
            labels += [IGNORE_INDEX] * len(chunk)
        else:
            # "<|asistan|>" başlığı öğrenilmez (onu her zaman biz ekleriz);
            # cevap metni ve cevabı bitiren "<|son|>" öğrenilir.
            ids += [assistant_id] + content_ids + [end_id]
            labels += [IGNORE_INDEX] + content_ids + [end_id]
    return ids, labels


def encode_prompt(tokenizer: BPETokenizer, messages: list[dict]) -> list[int]:
    """format_prompt() ile aynı şablonu doğrudan, GÜVENLİ biçimde tokenlere çevirir.

    Özel tokenleri kendimiz ekleriz; kullanıcı metni allow_special=False ile
    kodlanır. Sohbet sırasında modele verilecek girdi budur.
    """
    ids, _ = encode_conversation(tokenizer, messages)
    return ids + [tokenizer.special_id(ASSISTANT)]
