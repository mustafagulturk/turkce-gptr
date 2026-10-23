"""
Komut satırı girişi:  python -m turkce_gpt <komut> [seçenekler]

Komutlar:
    veri         Metni indirir, temizler, tokenizer eğitir, token dosyalarını yazar
    egit         Modeli sıfırdan eğitir (ön eğitim)
    ince-ayar    Ön eğitimli modeli soru-cevap örnekleriyle sohbete alıştırır
    degerlendir  Modelin doğrulama verisindeki başarısını ölçer
    uret         Verilen başlangıçtan metin üretir
    sohbet       Modelle etkileşimli sohbet
    web-aktar    Modeli tarayıcıda çalışacak biçime çevirir (web/ sohbet sayfası için)

Her komutun ayrıntısı için:  python -m turkce_gpt <komut> --help
"""

import sys


def main() -> None:
    # Modülleri sadece gerektiğinde içe aktarıyoruz; böylece "--help" bile
    # PyTorch yüklenmeden hızlıca açılır.
    commands = {
        "veri": ("turkce_gpt.data", "cli"),
        "egit": ("turkce_gpt.train", "cli"),
        "ince-ayar": ("turkce_gpt.finetune", "cli"),
        "degerlendir": ("turkce_gpt.evaluate", "cli"),
        "uret": ("turkce_gpt.interactive", "generate_cli"),
        "sohbet": ("turkce_gpt.interactive", "chat_cli"),
        "web-aktar": ("turkce_gpt.export_web", "cli"),
    }
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(__doc__)
        sys.exit(0 if len(sys.argv) < 2 else 1)

    import importlib

    module_name, func_name = commands[sys.argv[1]]
    func = getattr(importlib.import_module(module_name), func_name)
    func(sys.argv[2:])


if __name__ == "__main__":
    main()
