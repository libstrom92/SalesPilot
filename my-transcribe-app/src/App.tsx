import { useEffect, useRef, useState } from "react";

function App() {
  const [transcript, setTranscript] = useState<string>("");
  const [log, setLog] = useState<string[]>([]);
  const [status, setStatus] = useState<string>("Stoppad");
  const [debug, setDebug] = useState<string[]>([]);
  const [showDebug, setShowDebug] = useState<boolean>(false);
  const ws = useRef<WebSocket | null>(null);
  const [wsPort, setWsPort] = useState<number | null>(null);

  // Hämta port från websocket_port.txt vid start
  useEffect(() => {
    fetch("/websocket_port.txt")
      .then(res => res.text())
      .then(portStr => {
        const port = parseInt(portStr.trim(), 10);
        if (!isNaN(port)) setWsPort(port);
        else setLog(l => [...l, "Kunde inte läsa WebSocket-port från websocket_port.txt"]);
      })
      .catch(() => setLog(l => [...l, "Kunde inte läsa websocket_port.txt"]));
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
        if (data.transcript) {
          setTranscript((prev) => prev + data.transcript);
          setLog((l) => [...l, "Transkribering: " + data.transcript]);
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

  return (
    <div style={{ display: 'flex', maxWidth: 1100, margin: "2rem auto", fontFamily: "sans-serif", position: 'relative' }}>
      <div style={{ flex: 1, minWidth: 0 }}>
        <h1>Live Transkribering</h1>
        <div style={{ marginBottom: 16 }}>
          <button onClick={() => sendCommand("start_server")}>Starta server</button>
          <button onClick={() => sendCommand("stop_server")} style={{ marginLeft: 8 }}>Stoppa server</button>
          <button onClick={() => sendCommand("start_transcription")} style={{ marginLeft: 8 }}>Starta transkribering</button>
          <button onClick={() => sendCommand("stop_transcription")} style={{ marginLeft: 8 }}>Stoppa transkribering</button>
        </div>
        <div style={{ marginBottom: 16 }}>
          <b>Status:</b> {status}
        </div>
        <div style={{ background: "#f4f4f4", padding: 16, minHeight: 120, borderRadius: 8 }}>
          <pre style={{ whiteSpace: "pre-wrap" }}>{transcript}</pre>
        </div>
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