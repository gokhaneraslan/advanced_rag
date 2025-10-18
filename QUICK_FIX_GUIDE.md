# 🚨 Quick Fix Guide for CI/CD Errors

## Problem: CI/CD Pipeline Failing

Eğer GitHub Actions'da şu hataları görüyorsan:

```
❌ Run Black (code formatter check) - exit code 123
❌ Run isort (import sorting check) - exit code 1  
❌ Run flake8 (linting) - exit code 1
❌ Run pylint (code analysis) - exit code 30
❌ test_error_handling - Failed: DID NOT RAISE ValueError
```

## ✅ Hızlı Çözüm (3 Adım)

### Adım 1: Otomatik Düzeltme

```bash
# Yöntem A: Makefile kullan (önerilen)
make format

# Yöntem B: Manuel
black src/ tests/ *.py
isort src/ tests/ *.py

# Yöntem C: Script kullan
chmod +x scripts/fix_code_quality.sh
./scripts/fix_code_quality.sh
```

### Adım 2: Test Et

```bash
# Testleri çalıştır
make test

# Coverage ile test
make test-cov

# Format check
make format-check

# Linting check
make lint
```

### Adım 3: Commit & Push

```bash
git add .
git commit -m "Fix code quality issues"
git push
```

---

## 🔍 Her Hatanın Detaylı Açıklaması

### 1. Black (exit code 123) - "Code format hatası"

**Ne demek?** Kodun formatı Black standardına uymuyor.

**Örnek sorunlar:**
- Satır uzunluğu 120'den fazla
- Fonksiyon parametreleri yanlış hizalanmış
- Boşluk kullanımı standart dışı

**Çözüm:**
```bash
black src/ tests/ *.py
```

**Ne yapar?** Kodu otomatik olarak düzeltir.

---

### 2. isort (exit code 1) - "Import sırası hatası"

**Ne demek?** Import'lar doğru sıralanmamış.

**Doğru sıralama:**
```python
# 1. Standard library
import os
import sys

# 2. Third-party
from langchain import ...
from fastapi import ...

# 3. Local
from src.chains import ...
```

**Çözüm:**
```bash
isort src/ tests/ *.py
```

---

### 3. flake8 (exit code 1) - "Style hatası"

**Ne demek?** PEP8 standartlarına uyulmuyor.

**Yaygın sorunlar:**
- Kullanılmayan import'lar: `F401`
- Undefined name: `F821`
- Fazla boşluk: `E203`, `W503`
- Satır uzunluğu: `E501`

**Çözüm:** Flake8 otomatik düzeltme yapmaz, manuel düzeltmen gerek:

```bash
# Hataları gör
flake8 src/ tests/ *.py

# Genelde black + isort çözer
black . && isort .
```

---

### 4. pylint (exit code 30) - "Code quality hatası"

**Ne demek?** Kod kalitesi düşük (best practices ihlali).

**Yaygın sorunlar:**
- Çok fazla argument: `R0913`
- Çok uzun fonksiyon: `R0915`
- Missing docstring: `C0111`
- Snake_case ihlali: `C0103`

**Çözüm:** Uyarıları oku ve düzelt, veya config'de disable et:

```bash
# Skoru gör
pylint src/ --exit-zero

# pyproject.toml'da disable ettik çoğu uyarıyı
```

---

### 5. test_error_handling Failed

**Ne demek?** Test, `ValueError` bekliyor ama kod raise etmiyor.

**Sorun:**
```python
split_text([], method="invalid_method")  # Boş liste -> ValueError yok
```

**Çözüm:**
```python
# Dummy document ekle
dummy_doc = [Document(page_content="test", metadata={})]
split_text(dummy_doc, method="invalid_method")  # Şimdi ValueError raise eder
```

**Fix'lendi:** ✅ `test_integration.py` güncellendi

---

## 🛠️ Kalıcı Çözüm: Pre-commit Hooks

Gelecekte otomatik fix için:

```bash
# Pre-commit kur
pip install pre-commit

# .pre-commit-config.yaml oluştur
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/psf/black
    rev: 24.10.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 7.1.1
    hooks:
      - id: flake8
EOF

# Aktive et
pre-commit install

# Her commit'te otomatik çalışır
git commit -m "test"  # -> Otomatik black + isort + flake8
```

---

## 📋 Checklist (Push Öncesi)

Her push öncesi şunu çalıştır:

```bash
# ✅ Format
make format

# ✅ Test
make test

# ✅ Lint check
make lint

# ✅ Security check (opsiyonel)
make security

# ✅ Commit
git add .
git commit -m "Your message"
git push
```

---

## 🤔 Sık Sorulan Sorular

### Q: "Black vs flake8 çakışıyor, hangisini dinlemeliyim?"

**A:** Black öncelikli! Black'i çalıştır, sonra flake8'i ignore et. Config'de `E203, W503` zaten ignore edildi.

### Q: "Pylint skorumu nasıl yükseltebilirim?"

**A:** 
1. Docstring ekle (fonksiyon açıklamaları)
2. Uzun fonksiyonları böl
3. Magic number'ları constant'a çevir
4. Type hint ekle

### Q: "CI'da geçsin ama local'de düzeltmek istemiyorum?"

**A:** CI config'de `continue-on-error: true` var, build fail olmaz. Ama düzeltmek best practice!

---

## 🎯 Özet

```bash
# Tek komutla tüm düzeltmeleri yap
make format && make test

# Eğer çalışırsa
git add . && git commit -m "Fix code quality" && git push

# CI artık yeşil olmalı ✅
```

**Herhangi bir sorun varsa:**
1. `make clean` çalıştır
2. `make install` çalıştır  
3. `make format` çalıştır
4. `make test` çalıştır
5. Hala sorun varsa issue aç!