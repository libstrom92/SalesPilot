# SalesPilot - Rösttranskriptionssystem

Ett realtidsrösttranskriptionssystem som använder Whisper och talaridentifiering med fokus på säljsamtal.

## Snabbstart

1. Klona detta repository
2. Dubbelklicka på `setup.bat` för att installera nödvändiga paket och skapa den virtuella miljön
3. Dubbelklicka på `activate_env.bat` och följ instruktionerna
4. Välj alternativ 2 för att köra ljuddiagnostik om det är första gången du använder systemet
5. Välj alternativ 1 för att starta transkriptionssystemet

## Detaljerad installation

### Förutsättningar
- Python 3.9 eller senare
- Git
- Internetanslutning för nedladdning av modeller

### Steg för steg

1. Skapa en Python virtuell miljö (hanteras automatiskt av setup.bat eller activate_env.bat):
```bash
python -m venv myenv
source myenv/bin/activate  # På Linux/Mac
myenv\Scripts\activate     # På Windows
```

2. Installera beroenden:
```bash
pip install -r requirements.txt
```

3. Konfigurera dina inställningar i .env (kommer att skapas automatiskt):
   - Skaffa din Hugging Face token från https://huggingface.co/settings/tokens
   - Ange `HF_AUTH_TOKEN` i .env-filen
   - Justera ljudenhet och andra inställningar efter behov

4. Ladda ner modeller (hanteras automatiskt vid första körning):
   - Whisper transkriptionsmodell kommer att laddas ner automatiskt
   - Se [MODELS.md](MODELS.md) för detaljer om modellfiler och anpassning

## Diagnostikverktyg

Systemet innehåller flera diagnostikverktyg för att hjälpa dig med konfiguration och felsökning:

- `test_audio.bat` - Menydriven ljuddiagnostiksvit
- `hitta_enhet.py` - Lista tillgängliga ljudenheter
- `volym_visualiserare.py` - Realtidsövervakning av ljudnivå
- `test_audio_input.py` - Snabb test av ljudinmatning
- `test_audio_setup.py` - Omfattande systemkontroll

## Konfiguration

### .env-inställningar
Nyckelvärden i `.env`:
- `HF_AUTH_TOKEN`: Din Hugging Face API-token (obligatorisk)
- `AUDIO_DEVICE_ID`: ID för din ljudinmatningsenhet (standard: 2)
- `SAMPLE_RATE`: Ljudsamplingsfrekvens (standard: 16000)
- `WHISPER_MODEL`: Whisper-modellstorlek (standard: medium)
- `COMPUTE_TYPE`: Beräkningstyp (standard: int8)
- `WHISPER_LANGUAGE`: Transkriptionsspråk (standard: sv)

### Ramverk och strukturer
Du kan anpassa systemets beteende genom att redigera JSON-filerna i `frameworks/` mappen:
- `default.json`: Standardinställningar för transkription
- `fast_realtime.json`: Optimerad för snabbhet
- `high_quality.json`: Optimerad för kvalitet

## Funktioner

- Realtidstranskription med Whisper
- Talaridentifiering
- Brusreducering med GPU-acceleration när tillgänglig
- Övervakning och varningar för ljudnivå
- Samtalshistorik med automatisk sparning
- Omfattande loggsystem
- Felåterställning och återförsöksmekanismer
- Säljanalysstöd med AI

## Felsökning

Om du upplever problem:

1. Kör `test_audio.bat` och följ diagnostikstegen
2. Kontrollera loggarna i `logs`-katalogen
3. Verifiera dina ljudenhetsinställningar med `hitta_enhet.py`
4. Övervaka ljudnivåer med `volym_visualiserare.py`

## Arkitektur

- `main_server.py`: WebSocket-server och samordning av ljudbearbetning
- `audio_processor.py`: Kärnlogik för ljudbearbetning och transkription
- `transcribe_live.py`: Huvudingångspunkt för transkriptionsservern
- `audio_monitor.py`: Övervakning och visualisering av ljudnivå
- `logging_config.py`: Centraliserad loggkonfiguration
- `gpt_analyzer.py`: AI-analys av transkriberade samtal

## Utveckling

- Loggar lagras i `logs/transcription.log` med rotation
- Samtalshistorik sparas i `conversation_logs/`
- Enhetstester finns tillgängliga i test_audio_processor.py
- Kör tester med `run_tests.bat`