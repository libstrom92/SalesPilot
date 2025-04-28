# Transkriberingssystem – Projektöversikt

## Starta hela systemet

Starta allt med ett enda kommando:

```
start_all.bat
```

Detta öppnar backend (main_server.py) och frontend (React/Vite) i separata terminalfönster. Backend väljer automatiskt en ledig port och frontend ansluter automatiskt till rätt port.

- Ingen manuell port-synk behövs.
- Debugpanel och logg finns i webappen.

## Krav
- Python 3.9+ och installerade beroenden (`pip install -r requirements.txt`)
- Node.js & npm (för frontend)
- Aktivera Python-venv med `activate_env.bat` om du kör script separat

## Legacy-script
Följande script är nu legacy och behövs normalt inte:
- starta_server.bat
- stoppa_server.bat
- run_transcribe_live.bat
- run.bat
- test_voice.bat
- test_audio.bat
- test_audio_input.py
- test_audio_output.wav
- ...

Använd endast dessa om du felsöker eller vill köra delar av systemet separat.

## Felsökning
- Om frontend inte hittar backend: kontrollera att websocket_port.txt finns och att backend är igång.
- Om porten är upptagen: backend väljer automatiskt en ny port och skriver den till websocket_port.txt.
- Kontrollera debugpanelen i webappen och loggar i backend-fönstret.

## Vidareutveckling
- Lägg till triggers, mallförslag eller användarstöd i React-appen.
- Utöka backend för fler kommandon eller statusmeddelanden.

---

**Senast uppdaterad: 2025-04-28**