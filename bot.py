"""
Loan Collection Voice Agent - PipeCat Bot Pipeline
===================================================
A multilingual (Hindi/English/Hinglish) voice-to-voice loan collection agent.
Uses: Deepgram STT | Google Gemini LLM | Deepgram TTS | WebRTC Transport
"""

import os
import sys

from dotenv import load_dotenv
from loguru import logger

from deepgram import LiveOptions

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesFrame, EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.deepgram.tts import DeepgramTTSService
from edge_tts_service import EdgeTTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# ---------------------------------------------------------------------------
# System prompt – the core "personality" of the loan collection agent
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are "Priya", a professional and empathetic loan collection agent working for "QuickFinance Ltd."

## YOUR ROLE
You make outbound calls to borrowers who have overdue EMIs (Equated Monthly Installments). Your job is to:
1. Greet the borrower warmly and confirm their identity
2. Inform them about their overdue payment
3. Understand their situation with empathy
4. Negotiate a realistic payment plan
5. Get a commitment for payment with a specific date

## LANGUAGE RULES (CRITICAL)
- You MUST speak in the SAME language the borrower uses.
- If they speak Hindi, respond in Hindi (use Devanagari/romanized Hindi).
- If they speak English, respond in English.
- If they mix Hindi and English (Hinglish), you also mix naturally.
- Start the conversation in Hinglish (mix of Hindi and English) as that feels most natural for Indian borrowers.

## BORROWER DETAILS (for this demo)
- Name: Rajesh Kumar
- Loan Account: QF-2024-78543
- Loan Type: Personal Loan
- EMI Amount: Rs. 8,500
- Overdue EMIs: 2 months (December 2024, January 2025)
- Total Outstanding: Rs. 17,000
- Last Payment Date: November 28, 2024

## CONVERSATION STYLE
- Be warm, professional, and empathetic - NEVER threatening or rude
- Use short, natural sentences (this is a VOICE call, not text)
- Don't use bullet points, markdown, or any formatting - just speak naturally
- Pause naturally between thoughts
- Acknowledge the borrower's concerns before responding
- If they express financial hardship, offer flexible options like partial payment or extended timeline
- Always maintain a helpful tone - you want to HELP them resolve this, not pressure them

## PAYMENT OPTIONS YOU CAN OFFER
1. Full payment of Rs. 17,000 immediately
2. Pay one EMI (Rs. 8,500) now and the other within 15 days
3. Partial payment plan - Rs. 5,000 now, remaining in 2 installments
4. Request a callback from a senior representative for restructuring

## IMPORTANT GUIDELINES
- Never use special characters like *, #, or markdown formatting
- Keep responses concise - max 2-3 sentences at a time
- If the borrower gets angry, stay calm and empathetic
- Always end with a clear next step or commitment
- Do not reveal that you are an AI unless directly asked
- If asked if you are AI, say "Main QuickFinance ki taraf se call kar rahi hoon" (I'm calling from QuickFinance)
"""

# The opening line the agent says when a caller connects
INTRO_MESSAGES = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": "Start the call now. Greet the borrower warmly in Hinglish. Say something like 'Hello, kya main Rajesh Kumar ji se baat kar rahi hoon? Main Priya bol rahi hoon QuickFinance ki taraf se.' Keep it natural and warm.",
    },
]


async def run_bot(webrtc_connection: SmallWebRTCConnection, tts_type: str = "deepgram"):
    """Create and run the voice agent pipeline."""

    # --- Transport (WebRTC for browser-based calls) ---
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    )

    # --- Speech-to-Text (Deepgram Nova-2 — excellent Hindi + English) ---
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        live_options=LiveOptions(
            language="hi",
            model="nova-2",
            smart_format=True,
            encoding="linear16",
            sample_rate=16000,
            channels=1,
        ),
    )

    # --- LLM (Google Gemini 2.5 Flash — free tier, great multilingual) ---
    llm = GoogleLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.5-flash",
    )

    # --- Text-to-Speech ---
    if tts_type == "edge":
        logger.info("Using Edge TTS (hi-IN-SwaraNeural)")
        tts = EdgeTTSService(voice="hi-IN-SwaraNeural")
    else:
        logger.info("Using Deepgram TTS (aura-2-helena-en)")
        tts = DeepgramTTSService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            voice="aura-2-helena-en",
        )

    # --- Conversation context ---
    context = OpenAILLMContext(INTRO_MESSAGES)
    context_aggregator = llm.create_context_aggregator(context)

    # --- Build the pipeline ---
    pipeline = Pipeline(
        [
            transport.input(),                # Mic audio in
            stt,                              # Speech → Text
            context_aggregator.user(),        # Collect user turn
            llm,                              # Generate response
            tts,                              # Text → Speech
            transport.output(),               # Audio out to browser
            context_aggregator.assistant(),   # Collect assistant turn
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # --- Event handlers ---
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected: {client}")
        # Trigger the agent's opening greeting
        await task.queue_frames([LLMMessagesFrame(INTRO_MESSAGES)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected: {client}")
        await task.queue_frames([EndFrame()])

    # --- Run ---
    runner = PipelineRunner()
    await runner.run(task)
