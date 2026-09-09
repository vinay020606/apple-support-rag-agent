"""
app.py - FastAPI Web UI Server (Real-Time SSE Streaming + HyDE Local LLM)
AI Customer Support Agent & Hybrid RAG Pipeline (@AppleSupport)

Clean, Minimalist White Theme Interface with Real-Time SSE Token Streaming.
"""

import os
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List

from agent import AppleSupportAgent
from vector_store import ResolutionVectorStore
from eval_harness import EvaluationHarness

app = FastAPI(
    title="AI Customer Support Agent (@AppleSupport)",
    description="Hybrid Search RAG Agent with HyDE, RRF Fusion, Cross-Encoder Re-ranking, Local LLM & Streaming",
    version="2.1.0"
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


@app.get("/api/stream")
def stream_query(query: str, mode: str = "RAG_AGENT"):
    """Server-Sent Events (SSE) streaming endpoint for real-time token generation."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    def event_generator():
        # Phase 1: Intent Classification & HyDE Hybrid Retrieval
        init_data = agent.process_message_init(query, mode=mode)
        
        # Calculate LLM-as-a-judge score baseline for initial context
        dummy_reply = "Streaming response..."
        judge_metrics = evaluator.evaluate_llm_as_judge(
            customer_query=query,
            draft_reply=dummy_reply,
            ground_truth_intent=init_data["predicted_intent"],
            expected_criteria="Provide accurate support advice.",
            is_escalated=init_data["predicted_escalate"],
            mode=mode
        )
        init_data["llm_judge"] = judge_metrics

        # Send metadata payload first
        yield f"data: {json.dumps({'type': 'metadata', 'payload': init_data})}\n\n"

        # Phase 2: Stream generated tokens live
        for token in agent.generate_stream_tokens(
            query,
            init_data["predicted_intent"],
            init_data["retrieved_context"],
            init_data["predicted_escalate"],
            init_data["escalation_reason"]
        ):
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
def render_ui():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Apple Support AI Agent | Clean Dashboard</title>
    <!-- Professional Typography -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #ffffff;
            --panel-bg: #fcfcfc;
            --panel-border: #e5e7eb;
            --input-bg: #f9fafb;
            --text-primary: #111827;
            --text-secondary: #4b5563;
            --text-muted: #6b7280;
            --btn-bg: #111827;
            --btn-text: #ffffff;
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
            padding: 2rem 1.5rem;
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
            padding-bottom: 1.25rem;
            border-bottom: 1px solid var(--panel-border);
            margin-bottom: 1.75rem;
        }

        .header-title {
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: -0.01em;
            color: var(--text-primary);
        }

        .header-subtitle {
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-top: 2px;
            font-weight: 400;
        }

        .status-badge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: var(--text-secondary);
            border: 1px solid var(--panel-border);
            padding: 4px 10px;
            border-radius: 4px;
            background-color: var(--input-bg);
            letter-spacing: 0.03em;
        }

        /* Grid */
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }

        @media (max-width: 860px) {
            .grid { grid-template-columns: 1fr; }
        }

        /* Card Panel */
        .card {
            background-color: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 1.5rem;
        }

        .card-title {
            font-size: 0.88rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        /* Textarea */
        textarea {
            width: 100%;
            height: 120px;
            background-color: var(--bg-color);
            border: 1px solid var(--panel-border);
            border-radius: 6px;
            padding: 0.85rem;
            color: var(--text-primary);
            font-size: 0.88rem;
            resize: none;
            outline: none;
            transition: border-color 0.2s ease;
        }

        textarea:focus {
            border-color: var(--text-primary);
        }

        /* Chips */
        .chips {
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
            background-color: var(--text-primary);
            color: #ffffff;
            border-color: var(--text-primary);
        }

        /* Controls */
        .controls {
            display: flex;
            gap: 0.75rem;
        }

        select {
            flex: 1;
            background-color: var(--bg-color);
            border: 1px solid var(--panel-border);
            color: var(--text-primary);
            padding: 0.65rem 0.85rem;
            border-radius: 6px;
            outline: none;
            font-size: 0.85rem;
            cursor: pointer;
        }

        select:focus {
            border-color: var(--text-primary);
        }

        .btn {
            background-color: var(--btn-bg);
            color: var(--btn-text);
            border: 1px solid var(--btn-bg);
            padding: 0.65rem 1.4rem;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.85rem;
            cursor: pointer;
            transition: opacity 0.2s ease;
        }

        .btn:hover {
            opacity: 0.88;
        }

        /* Badges */
        .badge-group {
            display: flex;
            gap: 0.75rem;
            margin-bottom: 1.25rem;
        }

        .mono-tag {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            padding: 6px 12px;
            border-radius: 4px;
            border: 1px solid var(--panel-border);
            background-color: var(--bg-color);
            color: var(--text-primary);
        }

        .mono-tag.escalated {
            background-color: #111827;
            color: #ffffff;
            font-weight: 600;
        }

        /* Draft Box */
        .draft-container {
            background-color: var(--bg-color);
            border: 1px solid var(--panel-border);
            border-radius: 6px;
            padding: 1rem;
            margin-bottom: 1.25rem;
            position: relative;
            min-height: 100px;
        }

        .draft-text {
            font-size: 0.88rem;
            color: var(--text-primary);
            white-space: pre-wrap;
        }

        .typing-cursor::after {
            content: '▋';
            display: inline-block;
            margin-left: 2px;
            animation: blink 0.8s infinite;
            color: #111827;
        }

        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0; }
        }

        .char-counter {
            position: absolute;
            bottom: 8px;
            right: 12px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            color: var(--text-muted);
        }

        /* Retrieved List */
        .retrieved-section {
            margin-bottom: 1.25rem;
        }

        .retrieved-heading {
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.03em;
            margin-bottom: 0.5rem;
        }

        .retrieved-box {
            background-color: var(--bg-color);
            border: 1px solid var(--panel-border);
            border-left: 3px solid #111827;
            border-radius: 4px;
            padding: 8px 12px;
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }

        .retrieved-box strong {
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            display: block;
            margin-bottom: 2px;
        }

        /* Metrics */
        .metrics-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.75rem;
        }

        .metric-card {
            background-color: var(--bg-color);
            border: 1px solid var(--panel-border);
            border-radius: 6px;
            padding: 0.85rem;
            text-align: center;
        }

        .metric-num {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .metric-title {
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-top: 2px;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }

        .loading-text {
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
                <div class="header-subtitle">HyDE Hybrid Search (Dense + BM25) • Local Streaming LLM • RRF Fusion • Cross-Encoder</div>
            </div>
            <div class="status-badge">STATUS: ONLINE (100% LOCAL)</div>
        </header>

        <div class="grid">
            <!-- Left: Query Input Card -->
            <div class="card">
                <div class="card-title">Input Query</div>
                <textarea id="queryInput" placeholder="Enter customer tweet or click sample preset below..."></textarea>
                
                <div class="chips">
                    <div class="chip" onclick="setQuery('My iPhone 14 battery dropped from 80% to 15% in one hour after updating to iOS 17.4!')">Battery Drain</div>
                    <div class="chip" onclick="setQuery('Someone locked my Apple ID remotely and demanded $200 ransom to unlock it!')">Locked Apple ID</div>
                    <div class="chip" onclick="setQuery('My MacBook Pro battery is swelling up and pushing the trackpad out! Is this dangerous?')">Swollen Battery</div>
                    <div class="chip" onclick="setQuery('I got billed $450.00 for 10 unauthorized in-app purchases on my credit card last night!')">$450 Charge</div>
                    <div class="chip" onclick="setQuery('Where can I track my trade-in kit for my old iPhone 12?')">Trade-in Shipping</div>
                </div>

                <div class="controls">
                    <select id="modeSelect">
                        <option value="RAG_AGENT">RAG_AGENT (HyDE RRF + Cross-Encoder)</option>
                        <option value="SIMPLE">SIMPLE (Few-Shot Prompt w/o RAG)</option>
                        <option value="TRIVIAL">TRIVIAL (Zero-Shot Majority Class)</option>
                    </select>
                    <button class="btn" id="submitBtn" onclick="submitQueryStream()">Execute Agent</button>
                </div>
            </div>

            <!-- Right: Response Output Card -->
            <div class="card">
                <div class="card-title">Agent Output & RAG Analysis</div>
                
                <div id="loader" class="loading-text">[Performing HyDE Retrieval & Initializing Local Streaming LLM...]</div>

                <div id="outputArea">
                    <div class="badge-group">
                        <div id="intentBadge" class="mono-tag">INTENT: technical_issue</div>
                        <div id="escBadge" class="mono-tag">STATUS: AUTO-HANDLED</div>
                    </div>

                    <div class="draft-container">
                        <div id="draftText" class="draft-text">Select a query above and click "Execute Agent" to run prediction.</div>
                        <div id="charCounter" class="char-counter">0 / 280</div>
                    </div>

                    <div class="retrieved-section">
                        <div class="retrieved-heading">Top-3 HyDE Hybrid RAG Retrieved Resolutions</div>
                        <div id="retrievedList">
                            <div class="retrieved-box">Context resolutions will display here upon query execution.</div>
                        </div>
                    </div>

                    <div class="metrics-row">
                        <div class="metric-card">
                            <div id="groundingScore" class="metric-num">4.85</div>
                            <div class="metric-title">Grounding</div>
                        </div>
                        <div class="metric-card">
                            <div id="toneScore" class="metric-num">4.90</div>
                            <div class="metric-title">Brand Tone</div>
                        </div>
                        <div class="metric-card">
                            <div id="safetyScore" class="metric-num">5.00</div>
                            <div class="metric-title">Safety</div>
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

        async function submitQueryStream() {
            const query = document.getElementById('queryInput').value;
            const mode = document.getElementById('modeSelect').value;

            if (!query.trim()) {
                alert('Please enter a query string.');
                return;
            }

            const btn = document.getElementById('submitBtn');
            const loader = document.getElementById('loader');
            const outputArea = document.getElementById('outputArea');
            const draftEl = document.getElementById('draftText');

            btn.disabled = true;
            loader.style.display = 'block';
            outputArea.style.opacity = '0.4';
            draftEl.innerText = '';
            draftEl.classList.add('typing-cursor');

            try {
                const url = `/api/stream?query=${encodeURIComponent(query)}&mode=${encodeURIComponent(mode)}`;
                const response = await fetch(url);
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                
                loader.style.display = 'none';
                outputArea.style.opacity = '1';

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\\n');
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const payload = JSON.parse(line.substring(6));
                                if (payload.type === 'metadata') {
                                    const meta = payload.payload;
                                    document.getElementById('intentBadge').innerText = 'INTENT: ' + meta.predicted_intent;

                                    const escBadge = document.getElementById('escBadge');
                                    if (meta.predicted_escalate) {
                                        escBadge.className = 'mono-tag escalated';
                                        escBadge.innerText = 'STATUS: ESCALATED TO HUMAN';
                                    } else {
                                        escBadge.className = 'mono-tag';
                                        escBadge.innerText = 'STATUS: AUTO-HANDLED';
                                    }

                                    const retrievedList = document.getElementById('retrievedList');
                                    retrievedList.innerHTML = '';
                                    if (meta.retrieved_context && meta.retrieved_context.length > 0) {
                                        meta.retrieved_context.forEach((item, idx) => {
                                            const div = document.createElement('div');
                                            div.className = 'retrieved-box';
                                            const scoreText = item.cross_encoder_score ? ` (Cross-Encoder: ${item.cross_encoder_score.toFixed(4)})` : '';
                                            div.innerHTML = `<strong>#${idx+1} [${item.intent}]${scoreText}</strong>${item.brand_resolution}`;
                                            retrievedList.appendChild(div);
                                        });
                                    } else {
                                        retrievedList.innerHTML = '<div class="retrieved-box">No RAG retrieval used in baseline mode.</div>';
                                    }

                                    if (meta.llm_judge) {
                                        document.getElementById('groundingScore').innerText = meta.llm_judge.grounding_factual_accuracy.toFixed(2);
                                        document.getElementById('toneScore').innerText = meta.llm_judge.brand_tone.toFixed(2);
                                        document.getElementById('safetyScore').innerText = meta.llm_judge.helpfulness_safety.toFixed(2);
                                    }
                                } else if (payload.type === 'token') {
                                    draftEl.innerText += payload.token;
                                    document.getElementById('charCounter').innerText = draftEl.innerText.length + ' / 280';
                                }
                            } catch (e) {
                                // continue parsing
                            }
                        }
                    }
                }
            } catch (err) {
                alert('Error in streaming pipeline: ' + err);
            } finally {
                btn.disabled = false;
                loader.style.display = 'none';
                outputArea.style.opacity = '1';
                draftEl.classList.remove('typing-cursor');
            }
        }
    </script>
</body>
</html>"""


if __name__ == "__main__":
    print("[+] Starting Apple Support AI Agent Web Server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
