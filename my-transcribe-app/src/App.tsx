import { useEffect, useRef, useState } from "react";

function App() {
  const [transcript, setTranscript] = useState<string>("");
  const [log, setLog] = useState<string[]>([]);
  const [status, setStatus] = useState<string>("Stoppad");
  const [debug, setDebug] = useState<string[]>([]);
  const [showDebug, setShowDebug] = useState<boolean>(false);
  const ws = useRef<WebSocket | null>(null);
  const [wsPort, setWsPort] = useState<number | null>(null);
  const [aiFeedback, setAiFeedback] = useState<any>(null);
  const [blockAnalysis, setBlockAnalysis] = useState<any>(null);
  const [contextAnalysis, setContextAnalysis] = useState<any>(null);

  // Lägg till loggning för att spåra port och WebSocket-status
  useEffect(() => {
    console.log("Hämtar WebSocket-port...");
    fetch("/websocket_port.txt")
      .then(res => res.text())
      .then(portStr => {
        console.log("Port hämtad:", portStr);
        const port = parseInt(portStr.trim(), 10);
        if (!isNaN(port)) {
          setWsPort(port);
          console.log("WebSocket-port inställd på:", port);
        } else {
          console.error("Ogiltig port:", portStr);
        }
      })
      .catch(err => console.error("Fel vid hämtning av WebSocket-port:", err));
  }, []);

  // Initiera WebSocket när porten är känd
  useEffect(() => {
    if (!wsPort) return;
    ws.current = new WebSocket(`ws://localhost:${wsPort}`);
    ws.current.onopen = () => {
      setLog((l) => [...l, "WebSocket ansluten"]);
      setStatus("Stoppad");
      setDebug((d) => [...d, `[UI->WS] Anslutning öppnad (port ${wsPort})`]);
    };
    ws.current.onerror = (error) => {
      setLog((l) => [...l, "WebSocket fel: " + JSON.stringify(error)]);
      setDebug((d) => [...d, `[WS->UI] FEL: ${JSON.stringify(error)}`]);
    };
    ws.current.onmessage = (event) => {
      setDebug((d) => [...d, `[WS->UI] Mottaget: ${event.data}`]);
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'pong') {
          setLog((l) => [...l, `Pong från backend (${data.timestamp})`]);
          setDebug((d) => [...d, `[WS->UI] Pong mottaget: ${JSON.stringify(data)}`]);
        }
        if (data.type === 'block_analysis') {
          setBlockAnalysis(data.block_result);
          setLog((l) => [...l, "Blockanalys mottagen"]);
        }
        if (data.type === 'context_analysis') {
          setContextAnalysis(data.context_result);
          setLog((l) => [...l, "Kontextanalys mottagen"]);
        }
        if (data.transcript) {
          setTranscript((prev) => prev + data.transcript);
          setLog((l) => [...l, "Transkribering: " + data.transcript]);
        } else if (data.text) {
          setTranscript((prev) => prev + data.text);
          setLog((l) => [...l, "Transkribering (text): " + data.text]);
        }
        if (data.ai_feedback) {
          setAiFeedback(data.ai_feedback);
          setLog((l) => [...l, "AI-feedback mottagen"]);
        }
        if (data.status) {
          setStatus(data.status);
          setLog((l) => [...l, "Status: " + data.status]);
        }
        if (data.error) {
          setStatus("Fel");
          setLog((l) => [...l, "Fel: " + data.error]);
          setDebug((d) => [...d, `[WS->UI] FEL: ${data.error}`]);
        }
      } catch (e) {
        setLog((l) => [...l, "Oväntat meddelande: " + event.data]);
        setDebug((d) => [...d, `[WS->UI] FEL vid JSON.parse: ${event.data}`]);
      }
    };
    ws.current.onclose = () => {
      setLog((l) => [...l, "WebSocket stängd"]);
      setDebug((d) => [...d, `[WS->UI] Anslutning stängd`]);
    };
    return () => ws.current?.close();
  }, [wsPort]);

  // Skicka kommando till backend
  const sendCommand = (command: string) => {
    if (ws.current && ws.current.readyState === 1) {
      const msg = JSON.stringify({ command });
      ws.current.send(msg);
      setLog((l) => [...l, `Skickade kommando: ${command}`]);
      setDebug((d) => [...d, `[UI->WS] Skickat: ${msg}`]);
    }
  };

  // Skicka ping och visa pong i loggen
  const sendPing = () => {
    if (ws.current && ws.current.readyState === 1) {
      ws.current.send(JSON.stringify({ command: "ping" }));
      setLog((l) => [...l, "Skickade ping till backend"]);
    }
  };

  const testAudio = async () => {
    try {
      const response = await fetch("/test-audio");
      if (!response.ok) {
        throw new Error("Failed to fetch test audio");
      }
      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      audio.play();
    } catch (error) {
      console.error("Error testing audio:", error);
    }
  };

  return (
    <div style={{ display: 'flex', maxWidth: 1100, margin: "2rem auto", fontFamily: "sans-serif", position: 'relative' }}>
      <div style={{ flex: 1, minWidth: 0 }}>
        <h1>Live Transkribering</h1>
        <div style={{ marginBottom: 16 }}>
          <button onClick={() => sendCommand("start_server")}>Starta server</button>
          <button onClick={() => sendCommand("stop_server")} style={{ marginLeft: 8 }}>Stoppa server</button>
          <button onClick={() => sendCommand("start_transcription")} style={{ marginLeft: 8 }}>Starta transkribering</button>
          <button onClick={() => sendCommand("stop_transcription")} style={{ marginLeft: 8 }}>Stoppa transkribering</button>
          <button onClick={sendPing} style={{ marginLeft: 8 }}>Ping backend</button>
          <button onClick={testAudio} style={{ marginLeft: 8 }}>Test Audio</button>
        </div>
        <div style={{ marginBottom: 16 }}>
          <b>Status:</b> {status}
        </div>
        <div style={{ background: "#fffbe6", padding: 24, minHeight: 180, borderRadius: 12, border: '2px solid #f4b400', marginBottom: 24 }}>
          <h2 style={{ marginTop: 0, color: '#b8860b' }}>Transkriberad text</h2>
          <pre style={{ whiteSpace: "pre-wrap", fontSize: 18, color: '#222', margin: 0 }}>{transcript || 'Ingen text mottagen ännu.'}</pre>
          {aiFeedback && (
            <div style={{ marginTop: 16, background: '#e3f2fd', padding: 12, borderRadius: 8, color: '#1565c0' }}>
              <b>AI-feedback:</b>
              <pre style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{JSON.stringify(aiFeedback, null, 2)}</pre>
            </div>
          )}
        </div>
        {blockAnalysis && (
          <div style={{ background: '#e8f5e9', padding: 16, borderRadius: 8, border: '2px solid #388e3c', marginBottom: 24 }}>
            <h2 style={{ margin: 0, color: '#388e3c' }}>Blockanalys (AI)</h2>
            <pre style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{JSON.stringify(blockAnalysis, null, 2)}</pre>
          </div>
        )}
        {contextAnalysis && (
          <div style={{ background: '#fff3e0', padding: 16, borderRadius: 8, border: '2px solid #f57c00', marginBottom: 24 }}>
            <h2 style={{ margin: 0, color: '#f57c00' }}>Sektions-/Kontextanalys (AI)</h2>
            <pre style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{JSON.stringify(contextAnalysis, null, 2)}</pre>
          </div>
        )}
        <h2>Logg</h2>
        <ul style={{ background: "#222", color: "#fff", padding: 12, borderRadius: 8, fontSize: 14 }}>
          {log.map((row, i) => (
            <li key={i}>{row}</li>
          ))}
        </ul>
      </div>
      {/* Sidofält för debug, nu alltid längst till höger och position: fixed */}
      <div style={{
        position: 'fixed',
        top: 0,
        right: 0,
        height: '100vh',
        width: showDebug ? 340 : 32,
        transition: 'width 0.2s',
        zIndex: 1000,
        background: showDebug ? '#111' : 'rgba(17,17,17,0.3)',
        boxShadow: showDebug ? '-2px 0 8px #0006' : 'none',
        borderLeft: showDebug ? '2px solid #222' : 'none',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'stretch',
      }}>
        <button
          onClick={() => setShowDebug((v) => !v)}
          style={{
            position: 'absolute',
            left: 0,
            top: 0,
            zIndex: 2,
            width: 32,
            height: 32,
            background: '#111',
            color: '#0f0',
            border: 'none',
            borderRadius: 4,
            cursor: 'pointer',
            opacity: showDebug ? 1 : 0.5
          }}
          title={showDebug ? 'Minimera debug' : 'Visa debug'}
        >
          {showDebug ? '⮜' : '⮞'}
        </button>
        {showDebug && (
          <div style={{
            background: "#111",
            color: "#0f0",
            padding: 12,
            borderRadius: 8,
            fontSize: 13,
            height: '100%',
            overflowY: 'auto',
            marginLeft: 32,
            marginTop: 32
          }}>
            <h2 style={{ color: '#0f0', fontSize: 16, marginTop: 0 }}>Debug</h2>
            <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
              {debug.map((row, i) => (
                <li key={i}>{row}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;