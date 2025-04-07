import asyncio
import websockets
import json

WS_URL = "ws://localhost:9091"

async def listen():
    async with websockets.client.connect(WS_URL) as websocket:
        print(f"🟢 Ansluten till {WS_URL} – väntar på data...\n")
        try:
            while True:
                message = await websocket.recv()
                try:
                    data = json.loads(message)
                    transcription = data.get("transcription", "")
                    summary = data.get("summary", "")
                    diarization = data.get("diarization", "")

                    print("🗣️ Transkription:", transcription)
                    print("🧠 Summering:   ", summary)
                    print("👥 Diarization: ", diarization)
                    print("-" * 60)

                except json.JSONDecodeError:
                    print("⚠️ Ogiltigt JSON-meddelande:", message)

        except websockets.exceptions.ConnectionClosed:
            print("🔌 Anslutningen stängdes.")

if __name__ == "__main__":
    asyncio.run(listen())
