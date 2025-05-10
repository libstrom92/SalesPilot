# Realtids Transkribering & AI-analys

## Arkitektur
- `main_server.py`: WebSocket-server, ljudinmatning, transkribering, AI-analys
- `streaming_processor.py`: Whisper/Faster-Whisper, domänprompt
- `gpt_analyzer.py`: AI-feedback, block- och kontextanalys (GPT/HuggingFace)
- `my-transcribe-app/`: React-UI, live-transkription, AI-paneler

## Flöde
1. Ljud → text (Whisper)
2. Text → AI-feedback (Realtime, Block, Kontext)
3. WebSocket → Frontend
4. UI visar transkription, AI-insikter, logg/debug

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
- Lägg till dialektanalys, triggers, mallförslag
- Integrera fler svenska modeller (Hugging Face)
- Bygg ut UI-komponenter för CRM/affärsstöd

## TODO
- [ ] Lägg till dialektanalys i `gpt_analyzer.py`
- [ ] Integrera Hugging Face-modeller för svenska
- [ ] Skapa fler UI-komponenter för AI-insikter
- [ ] Lägg till mallförslag och action-knappar i frontend
- [ ] Förbättra dokumentation och kodkommentarer
// step 0 done – next: kontrollera modulstruktur och kommentarer

---

**Senast uppdaterad: 2025-04-28**