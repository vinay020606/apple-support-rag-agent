"""
app.py - FastAPI Web UI Server
AI Customer Support Agent & Hybrid RAG Pipeline (@AppleSupport)

Minimalist, High-Contrast Black & White Professional Interface and REST API.
"""

import os
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List

from agent import AppleSupportAgent
from vector_store import ResolutionVectorStore
from eval_harness import EvaluationHarness

app = FastAPI(
    title="AI Customer Support Agent (@AppleSupport)",
    description="Hybrid Search RAG Agent with RRF Fusion, Cross-Encoder Re-ranking, and Escalation Routing",
    version="2.0.0"
)

# Initialize Agent & Evaluation Harness
agent = AppleSupportAgent()
evaluator = EvaluationHarness()

class QueryRequest(BaseModel):
    query: str
    mode: Optional[str] = "RAG_AGENT"
    top_k: Optional[int] = 3

class QueryResponse(BaseModel):
    mode: str
    predicted_intent: str
    predicted_escalate: bool
    escalation_reason: str
    retrieved_context: List[dict]
    draft_reply: str
    char_count: int
    llm_judge: dict


@app.post("/api/query", response_model=QueryResponse)
def handle_query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")
    
    result = agent.process_message(req.query, mode=req.mode)
    judge_metrics = evaluator.evaluate_llm_as_judge(
        customer_query=req.query,
        draft_reply=result["draft_reply"],
        ground_truth_intent=result["predicted_intent"],
        expected_criteria="Provide accurate support advice and official self-service links.",
        is_escalated=result["predicted_escalate"],
        mode=req.mode
    )

    return QueryResponse(
        mode=result["mode"],
        predicted_intent=result["predicted_intent"],
        predicted_escalate=result["predicted_escalate"],
        escalation_reason=result["escalation_reason"],
        retrieved_context=result["retrieved_context"][:req.top_k],
        draft_reply=result["draft_reply"],
        char_count=len(result["draft_reply"]),
        llm_judge=judge_metrics
    )


@app.get("/", response_class=HTMLResponse)
def render_ui():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Apple Support AI Agent | Minimalist Dashboard</title>
    <!-- Professional Typography -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #000000;
            --panel-bg: #0a0a0a;
            --panel-border: #262626;
            --input-bg: #121212;
            --text-primary: #ffffff;
            --text-secondary: #a3a3a3;
            --text-muted: #737373;
            --accent-bw: #ffffff;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 2.5rem 1.5rem;
            line-height: 1.5;
        }

        .container {
            max-width: 1140px;
            margin: 0 auto;
        }

        /* Header */
        header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--panel-border);
            margin-bottom: 2rem;
        }

        .header-title {
            font-size: 1.35rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: var(--text-primary);
        }

        .header-subtitle {
            font-size: 0.82rem;
            color: var(--text-secondary);
            margin-top: 2px;
            font-weight: 400;
        }

        .system-status {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: var(--text-secondary);
            border: 1px solid var(--panel-border);
            padding: 4px 10px;
            border-radius: 4px;
            background: #121212;
            letter-spacing: 0.05em;
        }

        /* Grid */
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.75rem;
        }

        @media (max-width: 860px) {
            .grid { grid-template-columns: 1fr; }
        }

        /* Panel Card */
        .card {
            background-color: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 1.5rem;
        }

        .card-header {
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        /* Textarea Input */
        textarea {
            width: 100%;
            height: 120px;
            background-color: var(--input-bg);
            border: 1px solid var(--panel-border);
            border-radius: 6px;
            padding: 0.85rem;
            color: var(--text-primary);
            font-size: 0.9rem;
            resize: none;
            outline: none;
            transition: border-color 0.2s ease;
        }

        textarea:focus {
            border-color: #ffffff;
        }

        /* Preset Chips */
        .presets {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin: 0.85rem 0 1.25rem 0;
        }

        .chip {
            background-color: var(--input-bg);
            border: 1px solid var(--panel-border);
            color: var(--text-secondary);
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.75rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .chip:hover {
            background-color: #ffffff;
            color: #000000;
            border-color: #ffffff;
        }

        /* Controls */
        .controls {
            display: flex;
            gap: 0.75rem;
        }

        select {
            flex: 1;
            background-color: var(--input-bg);
            border: 1px solid var(--panel-border);
            color: var(--text-primary);
            padding: 0.65rem 0.85rem;
            border-radius: 6px;
            outline: none;
            font-size: 0.85rem;
            cursor: pointer;
        }

        select:focus {
            border-color: #ffffff;
        }

        .btn {
            background-color: #ffffff;
            color: #000000;
            border: 1px solid #ffffff;
            padding: 0.65rem 1.4rem;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.85rem;
            cursor: pointer;
            transition: background-color 0.2s ease, color 0.2s ease;
        }

        .btn:hover {
            background-color: #e5e5e5;
            border-color: #e5e5e5;
        }

        /* Output Badges */
        .badge-row {
            display: flex;
            gap: 0.75rem;
            margin-bottom: 1.25rem;
        }

        .mono-badge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            padding: 6px 12px;
            border-radius: 4px;
            border: 1px solid var(--panel-border);
            background-color: var(--input-bg);
            color: var(--text-primary);
        }

        .mono-badge.escalated {
            background-color: #ffffff;
            color: #000000;
            border-color: #ffffff;
            font-weight: 600;
        }

        /* Response Draft Box */
        .response-box {
            background-color: var(--input-bg);
            border: 1px solid var(--panel-border);
            border-radius: 6px;
            padding: 1rem;
            margin-bottom: 1.25rem;
            position: relative;
        }

        .response-text {
            font-size: 0.9rem;
            color: var(--text-primary);
            white-space: pre-wrap;
        }

        .char-count {
            position: absolute;
            bottom: 8px;
            right: 12px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            color: var(--text-muted);
        }

        /* Retrieved Context List */
        .retrieved-container {
            margin-bottom: 1.25rem;
        }

        .retrieved-title {
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.5rem;
        }

        .retrieved-card {
            background-color: var(--input-bg);
            border: 1px solid var(--panel-border);
            border-left: 2px solid #ffffff;
            border-radius: 4px;
            padding: 8px 12px;
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }

        .retrieved-card strong {
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            display: block;
            margin-bottom: 2px;
        }

        /* Score Metrics Grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.75rem;
        }

        .metric-box {
            background-color: var(--input-bg);
            border: 1px solid var(--panel-border);
            border-radius: 6px;
            padding: 0.85rem;
            text-align: center;
        }

        .metric-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .metric-label {
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-top: 2px;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }

        .loading {
            display: none;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: var(--text-secondary);
            padding: 1.5rem;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <div class="header-title">Apple Support AI Agent</div>
                <div class="header-subtitle">Hybrid Search (Dense + BM25) • RRF Fusion • Cross-Encoder Re-ranking</div>
            </div>
            <div class="system-status">STATUS: ONLINE</div>
        </header>

        <div class="grid">
            <!-- Left: Query Input Panel -->
            <div class="card">
                <div class="card-header">Input Query</div>
                <textarea id="queryInput" placeholder="Enter customer tweet or click sample preset below..."></textarea>
                
                <div class="presets">
                    <div class="chip" onclick="setQuery('My iPhone 14 battery dropped from 80% to 15% in one hour after updating to iOS 17.4!')">Battery Drain</div>
                    <div class="chip" onclick="setQuery('Someone locked my Apple ID remotely and demanded $200 ransom to unlock it!')">Locked Apple ID</div>
                    <div class="chip" onclick="setQuery('My MacBook Pro battery is swelling up and pushing the trackpad out! Is this dangerous?')">Swollen Battery</div>
                    <div class="chip" onclick="setQuery('I got billed $450.00 for 10 unauthorized in-app purchases on my credit card last night!')">$450 Charge</div>
                    <div class="chip" onclick="setQuery('Where can I track my trade-in kit for my old iPhone 12?')">Trade-in Shipping</div>
                </div>

                <div class="controls">
                    <select id="modeSelect">
                        <option value="RAG_AGENT">RAG_AGENT (Hybrid RRF + Cross-Encoder)</option>
                        <option value="SIMPLE">SIMPLE (Few-Shot Prompt w/o RAG)</option>
                        <option value="TRIVIAL">TRIVIAL (Zero-Shot Majority Class)</option>
                    </select>
                    <button class="btn" onclick="submitQuery()">Execute Agent</button>
                </div>
            </div>

            <!-- Right: Response & Analysis Panel -->
            <div class="card">
                <div class="card-header">Agent Output & RAG Analysis</div>
                
                <div id="loader" class="loading">[Processing query through RAG pipeline...]</div>

                <div id="outputArea">
                    <div class="badge-row">
                        <div id="intentBadge" class="mono-badge">INTENT: technical_issue</div>
                        <div id="escBadge" class="mono-badge">HANDLED: AUTO</div>
                    </div>

                    <div class="response-box">
                        <div id="draftText" class="response-text">Select a query above and click "Execute Agent" to run prediction.</div>
                        <div id="charCounter" class="char-count">0 / 280</div>
                    </div>

                    <div class="retrieved-container">
                        <div class="retrieved-title">Top-3 Hybrid RAG Retrieved Resolutions</div>
                        <div id="retrievedList">
                            <div class="retrieved-card">Context resolutions will display here upon query execution.</div>
                        </div>
                    </div>

                    <div class="metrics-grid">
                        <div class="metric-box">
                            <div id="groundingScore" class="metric-value">4.85</div>
                            <div class="metric-label">Grounding</div>
                        </div>
                        <div class="metric-box">
                            <div id="toneScore" class="metric-value">4.90</div>
                            <div class="metric-label">Brand Tone</div>
                        </div>
                        <div class="metric-box">
                            <div id="safetyScore" class="metric-value">5.00</div>
                            <div class="metric-label">Safety</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function setQuery(text) {
            document.getElementById('queryInput').value = text;
        }

        async function submitQuery() {
            const query = document.getElementById('queryInput').value;
            const mode = document.getElementById('modeSelect').value;

            if (!query.trim()) {
                alert('Please enter a query string.');
                return;
            }

            document.getElementById('loader').style.display = 'block';
            document.getElementById('outputArea').style.opacity = '0.3';

            try {
                const res = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query, mode: mode, top_k: 3 })
                });
                const data = await res.json();

                document.getElementById('intentBadge').innerText = 'INTENT: ' + data.predicted_intent;

                const escBadge = document.getElementById('escBadge');
                if (data.predicted_escalate) {
                    escBadge.className = 'mono-badge escalated';
                    escBadge.innerText = 'STATUS: ESCALATED TO HUMAN';
                } else {
                    escBadge.className = 'mono-badge';
                    escBadge.innerText = 'STATUS: AUTO-HANDLED';
                }

                document.getElementById('draftText').innerText = data.draft_reply;
                document.getElementById('charCounter').innerText = data.char_count + ' / 280';

                const retrievedList = document.getElementById('retrievedList');
                retrievedList.innerHTML = '';
                if (data.retrieved_context && data.retrieved_context.length > 0) {
                    data.retrieved_context.forEach((item, idx) => {
                        const div = document.createElement('div');
                        div.className = 'retrieved-card';
                        const scoreText = item.cross_encoder_score ? ` (Cross-Encoder: ${item.cross_encoder_score.toFixed(4)})` : '';
                        div.innerHTML = `<strong>#${idx+1} [${item.intent}]${scoreText}</strong>${item.brand_resolution}`;
                        retrievedList.appendChild(div);
                    });
                } else {
                    retrievedList.innerHTML = '<div class="retrieved-card">No RAG retrieval used in baseline mode.</div>';
                }

                document.getElementById('groundingScore').innerText = data.llm_judge.grounding_factual_accuracy.toFixed(2);
                document.getElementById('toneScore').innerText = data.llm_judge.brand_tone.toFixed(2);
                document.getElementById('safetyScore').innerText = data.llm_judge.helpfulness_safety.toFixed(2);

            } catch (err) {
                alert('Error querying agent: ' + err);
            } finally {
                document.getElementById('loader').style.display = 'none';
                document.getElementById('outputArea').style.opacity = '1';
            }
        }
    </script>
</body>
</html>"""


if __name__ == "__main__":
    print("[+] Starting Apple Support AI Agent Web Server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
