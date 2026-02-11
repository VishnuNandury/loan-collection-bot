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


# @app.get("/api/session-data/{pc_id}")
# async def get_session_data(pc_id: str):
#     """Return live session data for the dashboard — flow state, transcript, metrics."""
#     data = session_data.get(pc_id)
#     if not data:
#         return JSONResponse({"current_node": "", "nodes": FLOW_NODES, "transcript": [], "metrics": {}})

#     # Transcript from frame processors (real-time capture)
#     transcript = data.get("transcript", [])

#     # Estimate token usage from context messages
#     ctx = data.get("_context")
#     all_msgs = []
#     if ctx:
#         try:
#             all_msgs = ctx.messages if hasattr(ctx, "messages") else []
#         except Exception:
#             pass
#     total_chars = 0
#     for m in all_msgs:
#         c = m.get("content", "") if isinstance(m, dict) else ""
#         if isinstance(c, str):
#             total_chars += len(c)
#     user_msgs = sum(1 for m in all_msgs if m.get("role") == "user")
#     assistant_msgs = sum(1 for m in all_msgs if m.get("role") == "assistant")
#     system_msgs = sum(1 for m in all_msgs if m.get("role") == "system")

#     return JSONResponse({
#         "current_node": data.get("current_node", ""),
#         "nodes": FLOW_NODES,
#         "transcript": transcript,
#         "tts_type": data.get("tts_type", "deepgram"),
#         "metrics": {
#             "est_tokens": total_chars // 3,
#             "total_messages": len(all_msgs),
#             "user_messages": user_msgs,
#             "assistant_messages": assistant_msgs,
#             "system_messages": system_msgs,
#             "llm": "gemini-2.5-flash",
#             "stt": "Deepgram Nova-2",
#             "tts": "Edge TTS" if data.get("tts_type") == "edge" else "Deepgram Aura-2",
#         },
#     })

# @app.get("/api/session-data/{pc_id}")
# async def get_session_data(pc_id: str):
#     """
#     Return session data for the dashboard:
#     - transcript
#     - current flow node
#     - flow pipeline structure
#     - metrics
#     """
#     session = session_data.get(pc_id)
#     if not session:
#         return {"error": "Session not found"}

#     transcript = session.get("transcript", [])
#     current_node = session.get("current_node", "")
    
#     # ✅ FIX: Build the nodes array from your flow config
#     nodes = [
#         {"id": "greeting", "label": "Greeting", "type": "start"},
#         {"id": "overdue_info", "label": "Overdue Info", "type": "process"},
#         {"id": "payment_discussion", "label": "Payment Discussion", "type": "process"},
#         {"id": "promise_to_pay", "label": "Promise to Pay", "type": "process"},
#         {"id": "partial_payment", "label": "Partial Payment", "type": "process"},
#         {"id": "cannot_pay", "label": "Cannot Pay", "type": "process"},
#         {"id": "escalation", "label": "Escalation", "type": "process"},
#         {"id": "closing", "label": "Closing", "type": "end"}
#     ]

#     # Metrics calculation
#     metrics = {}
#     try:
#         agg = session.get("context_aggregator")
#         if agg and hasattr(agg, "get_messages_for_persistent_storage"):
#             all_msgs = agg.get_messages_for_persistent_storage()
            
#             # ⚠️ Handle both dict and Pydantic Content objects
#             def get_role(msg):
#                 if isinstance(msg, dict):
#                     return msg.get("role")
#                 else:
#                     return getattr(msg, "role", None)
            
#             def get_parts(msg):
#                 if isinstance(msg, dict):
#                     return msg.get("parts", [])
#                 else:
#                     return getattr(msg, "parts", [])
            
#             user_msgs = sum(1 for m in all_msgs if get_role(m) == "user")
#             assistant_msgs = sum(1 for m in all_msgs if get_role(m) == "model")
#             system_msgs = sum(1 for m in all_msgs if get_role(m) == "system")
            
#             # Estimate tokens
#             total_text = ""
#             for m in all_msgs:
#                 parts = get_parts(m)
#                 for part in parts:
#                     if isinstance(part, dict):
#                         total_text += part.get("text", "")
#                     else:
#                         total_text += getattr(part, "text", "")
            
#             est_tokens = len(total_text.split())

#             metrics = {
#                 "llm": "Google Gemini",
#                 "stt": "Deepgram",
#                 "tts": "Edge TTS",
#                 "total_messages": len(all_msgs),
#                 "user_messages": user_msgs,
#                 "assistant_messages": assistant_msgs,
#                 "system_messages": system_msgs,
#                 "est_tokens": est_tokens
#             }
#     except Exception as e:
#         logger.error(f"Error calculating metrics: {e}")
#         metrics = {
#             "llm": "Google Gemini",
#             "stt": "Deepgram",
#             "tts": "Edge TTS",
#             "total_messages": 0,
#             "user_messages": 0,
#             "assistant_messages": 0,
#             "system_messages": 0,
#             "est_tokens": 0
#         }

#     return {
#         "transcript": transcript,
#         "current_node": current_node,
#         "nodes": nodes,
#         "metrics": metrics
#     }

@app.get("/api/session-data/{pc_id}")
async def get_session_data(pc_id: str):
    """
    Return session data for the dashboard:
    - transcript
    - current flow node
    - flow pipeline structure
    - metrics
    """
    session = session_data.get(pc_id)
    if not session:
        logger.warning(f"Session not found for pc_id: {pc_id}")
        return {"error": "Session not found"}

    transcript = session.get("transcript", [])
    current_node = session.get("current_node", "")
    
    # Build the nodes array from your flow config
    nodes = [
        {"id": "greeting", "label": "Greeting", "type": "start"},
        {"id": "overdue_info", "label": "Overdue Info", "type": "process"},
        {"id": "understand_situation", "label": "Situation", "type": "process"},
        {"id": "payment_options", "label": "Options", "type": "process"},
        {"id": "commitment", "label": "Commitment", "type": "process"},
        {"id": "promise_to_pay", "label": "PTP", "type": "process"},
        {"id": "end", "label": "Complete", "type": "end"}
    ]

    # Metrics calculation with better error handling
    metrics = {
        "llm": "Google Gemini 2.5 Flash",
        "stt": "Deepgram Nova-2",
        "tts": "Edge TTS (hi-IN-SwaraNeural)" if session.get("tts_type") == "edge" else "Deepgram Aura-2",
        "total_messages": 0,
        "user_messages": 0,
        "assistant_messages": 0,
        "system_messages": 0,
        "est_tokens": 0
    }
    
    try:
        agg = session.get("context_aggregator")
        
        if not agg:
            logger.warning(f"No context_aggregator found for session {pc_id}")
        elif not hasattr(agg, "get_messages_for_persistent_storage"):
            logger.warning(f"context_aggregator doesn't have get_messages_for_persistent_storage method")
        else:
            all_msgs = agg.get_messages_for_persistent_storage()
            logger.debug(f"Retrieved {len(all_msgs)} messages from context aggregator for {pc_id}")
            
            # Handle both dict and Pydantic Content objects
            def get_role(msg):
                if isinstance(msg, dict):
                    return msg.get("role")
                else:
                    return getattr(msg, "role", None)
            
            def get_parts(msg):
                if isinstance(msg, dict):
                    return msg.get("parts", [])
                else:
                    return getattr(msg, "parts", [])
            
            def get_text_from_part(part):
                """Extract text from various part formats."""
                if isinstance(part, dict):
                    return part.get("text", "")
                elif isinstance(part, str):
                    return part
                else:
                    # Pydantic object
                    return getattr(part, "text", "")
            
            user_msgs = 0
            assistant_msgs = 0
            system_msgs = 0
            total_text = ""
            
            for m in all_msgs:
                role = get_role(m)
                if role == "user":
                    user_msgs += 1
                elif role == "model":  # Google uses "model" instead of "assistant"
                    assistant_msgs += 1
                elif role == "system":
                    system_msgs += 1
                
                # Extract text from parts
                parts = get_parts(m)
                for part in parts:
                    text = get_text_from_part(part)
                    total_text += text + " "
            
            # Token estimation: words * 1.3 (accounts for subword tokenization)
            word_count = len(total_text.split())
            est_tokens = int(word_count * 1.3)
            
            metrics = {
                "llm": "Google Gemini 2.5 Flash",
                "stt": "Deepgram Nova-2",
                "tts": "Edge TTS (hi-IN-SwaraNeural)" if session.get("tts_type") == "edge" else "Deepgram Aura-2",
                "total_messages": len(all_msgs),
                "user_messages": user_msgs,
                "assistant_messages": assistant_msgs,
                "system_messages": system_msgs,
                "est_tokens": est_tokens
            }
            
            logger.debug(f"Metrics for {pc_id}: {metrics}")
            
    except Exception as e:
        logger.error(f"Error calculating metrics for {pc_id}: {e}", exc_info=True)

    return {
        "transcript": transcript,
        "current_node": current_node,
        "nodes": nodes,
        "metrics": metrics
    }

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
