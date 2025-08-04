# 🤖 LeanCart Global RAG Chatbot

Bu proje, **Retrieval-Augmented Generation (RAG)** teknolojisini kullanarak LeanCart Global admin panel dokümantasyonundan soru-cevap yapabilen akıllı chatbot sistemidir.

## ✨ Özellikler

- **🤖 Akıllı Soru-Cevap**: Admin panel konularında uzman asistan
- **📁 Doküman İşleme**: TXT dosyalarından öğrenme
- **🧠 Vektör Tabanlı Arama**: FAISS ile hızlı bilgi erişimi
- **🔗 OpenAI Entegrasyonu**: GPT-3.5-turbo ile doğal dil işleme
- **💻 Modern Web Arayüzü**: Flask ile kullanıcı dostu interface
- **� Demo Sayfası**: Kolay entegrasyon gösterimi
- **⚡ Hızlı Başlangıç**: Tek komutla çalıştırma

## 📦 Kurulum

### Hızlı Başlangıç
```bash
# 1. Depoyu klonlayın veya indirin
cd RagAIChatbot

# 2. Sanal ortam oluşturun ve aktive edin
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# veya
.venv\Scripts\activate     # Windows

# 3. Bağımlılıkları yükleyin
pip install -r requirements.txt

# 4. Çevre değişkenlerini ayarlayın
cp .env.example .env
# .env dosyasında OPENAI_API_KEY'i ayarlayın

# 5. Sistemi başlatın
python web_app.py
```

### Otomatik Kurulum (start.sh)
```bash
./start.sh
```

## 🌐 Kullanım

Sistem başladıktan sonra:
- **Ana Chatbot**: http://localhost:8086
- **Demo Sayfası**: http://localhost:8086/demo

## 📁 Proje Yapısı

```
RagAIChatbot/
├── web_app.py              # 🌐 Ana Flask uygulaması
├── demo.html               # � Demo sayfası
├── requirements.txt        # 📦 Bağımlılıklar
├── start.sh               # � Başlangıç scripti
├── .env                   # � Çevre değişkenleri
├── data/
│   └── train.txt          # � Eğitim verisi
├── src/                   # � Kaynak kod
│   ├── core/              # 🎯 Çekirdek modüller
│   ├── services/          # � Servisler
│   ├── storage/           # 💾 Veri depolama
│   ├── utils/             # 🛠️ Yardımcı araçlar
│   └── rag_runner.py      # 🤖 Ana RAG sistemi
└── tests/                 # 🧪 Test dosyaları
```

## 🎯 Örnek Kullanım

### Soru Örnekleri
- "Ürün nasıl eklenir?"
- "Kampanya nasıl oluşturulur?"
- "Kategori düzenleme nasıl yapılır?"
- "Müşteri bilgileri nasıl güncellenir?"
- "Sipariş durumu nasıl değiştirilir?"

### API Kullanımı
```bash
curl -X POST http://localhost:8086/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Admin panelde ürün nasıl eklenir?"}'
```

## 🔧 Konfigürasyon

### .env Dosyası
```env
OPENAI_API_KEY=your_openai_api_key_here
DATA_DIR=./data
EMBEDDINGS_DIR=./embeddings
DEBUG=True
```

## 🛠️ Geliştirme

### Test Çalıştırma
```bash
python -m pytest tests/
```

### Yeni Doküman Ekleme
1. `data/` klasörüne `.txt` dosyanızı ekleyin
2. Sistemi yeniden başlatın
3. Sistem otomatik olarak yeni dökümanları işleyecek

## 🔍 Nasıl Çalışır?

1. **Doküman Yükleme**: `data/train.txt` dosyası okunur
2. **Metin Bölütleme**: Büyük metin küçük parçalara bölünür
3. **Embedding Oluşturma**: Her parça sayısal vektöre çevrilir
4. **Vektör Depolama**: FAISS ile hızlı arama için indekslenir
5. **Soru Geldiğinde**: En alakalı parçalar bulunur
6. **Cevap Üretimi**: OpenAI GPT ile doğal cevap oluşturulur

## � Sorun Giderme

### Yaygın Hatalar
- **API Key Hatası**: `.env` dosyasında `OPENAI_API_KEY` kontrolü
- **Port Çakışması**: Başka bir uygulama 8086 portunu kullanıyor olabilir
- **Import Hataları**: `requirements.txt` bağımlılıklarını kontrol edin

### Yardım
Sistem çalışmıyorsa:
1. Virtual environment aktif mi? (`source .venv/bin/activate`)
2. Tüm paketler yüklenmiş mi? (`pip install -r requirements.txt`)
3. `.env` dosyası doğru mu?
4. Terminal'de hata mesajları var mı?

## � Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

---

**LeanCart Global RAG Chatbot** - Akıllı Admin Panel Asistanı 🤖✨