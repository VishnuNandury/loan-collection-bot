"""
Loan Collection Agent - FastAPI Server
=======================================
Serves the WebRTC frontend and handles signaling for voice calls.
Run with: python server.py
"""

import os
import argparse
from contextlib import asynccontextmanager
from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from loguru import logger

from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection

from bot import run_bot, session_data, FLOW_NODES

load_dotenv(override=True)

# Track active peer connections
pcs_map: Dict[str, SmallWebRTCConnection] = {}


def get_ice_servers():
    """Build ICE server list from env vars. Uses RTCIceServer objects throughout."""
    from aiortc import RTCIceServer

    servers = [RTCIceServer(urls="stun:stun.l.google.com:19302")]

    turn_url = os.getenv("TURN_URL")
    turn_username = os.getenv("TURN_USERNAME")
    turn_credential = os.getenv("TURN_CREDENTIAL")
    if turn_url and turn_username and turn_credential:
        # Support multiple TURN URLs (comma-separated)
        turn_urls = [u.strip() for u in turn_url.split(",")]
        servers.append(
            RTCIceServer(
                urls=turn_urls,
                username=turn_username,
                credential=turn_credential,
            )
        )
        logger.info(f"TURN configured: {turn_urls}")
    else:
        logger.warning("TURN not configured - WebRTC may fail behind NAT/cloud")

    return servers


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Cleanup all connections on shutdown
    for pc_id, conn in pcs_map.items():
        await conn.disconnect()
    pcs_map.clear()


app = FastAPI(lifespan=lifespan)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main frontend page."""
    with open(os.path.join("static", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/offer")
async def webrtc_offer(request: Request):
    """Handle WebRTC SDP offer from the browser client."""
    body = await request.json()
    sdp = body.get("sdp")
    sdp_type = body.get("type", "offer")
    pc_id = body.get("pc_id")
    tts_type = body.get("tts_type", "deepgram")

    if not sdp:
        raise HTTPException(status_code=400, detail="Missing SDP offer")

    # Reuse existing connection (renegotiation)
    if pc_id and pc_id in pcs_map:
        logger.info(f"Renegotiating connection: {pc_id}")
        connection = pcs_map[pc_id]
        await connection.renegotiate(sdp=sdp, type=sdp_type)
        answer = connection.get_answer()
        return JSONResponse({"sdp": answer["sdp"], "pc_id": pc_id, "type": "answer"})

    # New connection
    connection = SmallWebRTCConnection(
        ice_servers=get_ice_servers()
    )

    @connection.event_handler("on_closed")
    async def on_closed(connection):
        cid = connection.pc_id
        if cid in pcs_map:
            del pcs_map[cid]
            logger.info(f"Connection closed and removed: {cid}")
        # Clean up session data
        session_data.pop(cid, None)

    await connection.initialize(sdp=sdp, type=sdp_type)
    answer = connection.get_answer()
    pc_id = connection.pc_id
    pcs_map[pc_id] = connection

    logger.info(f"New connection: {pc_id}")

    # Start the bot pipeline in the background
    task = BackgroundTask(run_bot, connection, tts_type)

    return JSONResponse(
        {"sdp": answer["sdp"], "pc_id": pc_id, "type": "answer"},
        background=task,
    )


@app.post("/api/disconnect")
async def webrtc_disconnect(request: Request):
    """Handle client disconnect."""
    body = await request.json()
    pc_id = body.get("pc_id")

    if pc_id and pc_id in pcs_map:
        connection = pcs_map.pop(pc_id)
        await connection.disconnect()
        session_data.pop(pc_id, None)
        logger.info(f"Disconnected: {pc_id}")
        return JSONResponse({"status": "disconnected"})

    return JSONResponse({"status": "not_found"}, status_code=404)


@app.get("/api/session-data/{pc_id}")
async def get_session_data(pc_id: str):
    """Return live session data for the dashboard — flow state, transcript, metrics."""
    data = session_data.get(pc_id)
    if not data:
        return JSONResponse({"current_node": "", "nodes": FLOW_NODES, "transcript": [], "metrics": {}})

    # Extract transcript from the live context messages
    transcript = []
    ctx = data.get("_context")
    if ctx:
        try:
            messages = ctx.messages if hasattr(ctx, "messages") else []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role in ("user", "assistant") and content:
                    transcript.append({"role": role, "text": content})
        except Exception:
            pass

    # Estimate token usage from context messages
    all_msgs = []
    if ctx:
        try:
            all_msgs = ctx.messages if hasattr(ctx, "messages") else []
        except Exception:
            pass
    total_chars = sum(len(m.get("content", "")) for m in all_msgs)
    user_msgs = sum(1 for m in all_msgs if m.get("role") == "user")
    assistant_msgs = sum(1 for m in all_msgs if m.get("role") == "assistant")
    system_msgs = sum(1 for m in all_msgs if m.get("role") == "system")

    return JSONResponse({
        "current_node": data.get("current_node", ""),
        "nodes": FLOW_NODES,
        "transcript": transcript,
        "tts_type": data.get("tts_type", "deepgram"),
        "metrics": {
            "est_tokens": total_chars // 3,
            "total_messages": len(all_msgs),
            "user_messages": user_msgs,
            "assistant_messages": assistant_msgs,
            "system_messages": system_msgs,
            "llm": "gemini-2.5-flash",
            "stt": "Deepgram Nova-2",
            "tts": "Edge TTS" if data.get("tts_type") == "edge" else "Deepgram Aura-2",
        },
    })


@app.get("/api/ice-servers")
async def ice_servers():
    """Return ICE server config for the browser client."""
    servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    turn_url = os.getenv("TURN_URL")
    turn_username = os.getenv("TURN_USERNAME")
    turn_credential = os.getenv("TURN_CREDENTIAL")
    if turn_url and turn_username and turn_credential:
        turn_urls = [u.strip() for u in turn_url.split(",")]
        servers.append({
            "urls": turn_urls,
            "username": turn_username,
            "credential": turn_credential,
        })
    return servers


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_connections": len(pcs_map),
        "google_api": "configured" if os.getenv("GOOGLE_API_KEY") else "missing",
        "deepgram_api": "configured" if os.getenv("DEEPGRAM_API_KEY") else "missing",
        "turn": "configured" if os.getenv("TURN_URL") else "missing",
    }


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Loan Collection Agent Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8080")), help="Port to listen on")
    args = parser.parse_args()

    is_production = os.getenv("RENDER") or os.getenv("AWS_EXECUTION_ENV")

    print(f"\n{'='*60}")
    print(f"  Loan Collection Agent - QuickFinance Ltd.")
    print(f"  Server starting at http://localhost:{args.port}")
    print(f"{'='*60}\n")

    uvicorn.run("server:app", host=args.host, port=args.port, reload=not is_production)
