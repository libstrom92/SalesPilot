# my-transcribe-app

En modern React + TypeScript-app för realtidsvisning och styrning av transkribering.

## Funktion
- Visar transkriberad text i realtid från backend (Python, t.ex. transcribe_live.py)
- Kan styra backend: starta/stoppa server och transkribering via WebSocket-kommandon
- Har logg- och debugpanel för felsökning och utveckling

## Krav
- Node.js & npm (för frontend)
- Python 3.9+ (för backend)
- Backend måste vara igång och lyssna på samma port som frontend försöker ansluta till (default ws://localhost:8000)
- Alla Python-beroenden installerade (se ../requirements.txt)

## Installation & start

### 1. Installera och starta backend

```sh
# (I projektroten, där requirements.txt finns)
pip install -r requirements.txt
python transcribe_live.py
# eller
python main_server.py
# eller
run_transcribe_live.bat
```

### 2. Installera och starta frontend

```sh
cd my-transcribe-app
npm install
npm run dev
```

Öppna http://localhost:5173/ i webbläsaren.

## Användning
- Använd knapparna för att starta/stoppa server och transkribering.
- Transkriberad text visas i huvudfönstret.
- Loggpanelen visar händelser och status.
- Debugpanelen (sidofält till höger) visar alla WebSocket-meddelanden och kan minimeras/visas.

## Felsökning
- Om ingen text visas: kontrollera att backend är igång och att porten stämmer.
- Använd debugpanelen för att se exakt vad som skickas/tas emot och var det ev. stannar.
- Kontrollera terminalen där backend körs för felmeddelanden.

## Rensa bort gammalt
- All gammal malltext från Vite/React är borttagen.
- Endast denna README gäller för appens verkliga funktion.

## Vidareutveckling
- Lägg till triggers, mallförslag eller användarstöd i React-appen.
- Utöka backend för fler kommandon eller statusmeddelanden.

---

**Senast uppdaterad: 2025-04-28**
