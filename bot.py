"""
Loan Collection Voice Agent - PipeCat Bot Pipeline with Flows
=============================================================
A structured loan collection conversation using Pipecat Flows.
Demonstrates: greeting → identity check → overdue info → understand situation
→ payment options → commitment → promise to pay → close.

Uses: Deepgram STT | Google Gemini LLM | Deepgram/Edge TTS | WebRTC Transport
"""

import os
import sys
import time

from dotenv import load_dotenv
from loguru import logger

from deepgram import LiveOptions

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame
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

from pipecat_flows import FlowArgs, FlowManager, FlowsFunctionSchema, NodeConfig

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# ---------------------------------------------------------------------------
# Priya persona — consistent across all flow nodes
# ---------------------------------------------------------------------------
ROLE_MESSAGE = {
    "role": "system",
    "content": (
        'You are "Priya", a professional and empathetic loan collection agent '
        'working for "QuickFinance Ltd."\n\n'
        "LANGUAGE RULES:\n"
        "- Speak in the SAME language the borrower uses.\n"
        "- If they speak Hindi, respond in Hindi.\n"
        "- If they speak English, respond in English.\n"
        "- If they mix Hindi and English (Hinglish), you also mix naturally.\n"
        "- Default to Hinglish as it feels most natural for Indian borrowers.\n\n"
        "STYLE:\n"
        "- Warm, professional, empathetic — NEVER threatening or rude.\n"
        "- Short, natural sentences (this is a voice call, not text).\n"
        "- No bullet points, markdown, special characters, or emojis.\n"
        "- Max 2-3 sentences at a time.\n"
        "- You must ALWAYS use one of the available functions to progress the conversation.\n"
        "- Your responses will be converted to audio so avoid any formatting."
    ),
}

BORROWER_INFO = (
    "Borrower: Rajesh Kumar | Account: QF-2024-78543 | Personal Loan\n"
    "EMI: Rs. 8,500 | Overdue: 2 months (Dec 2024, Jan 2025) | Total Due: Rs. 17,000\n"
    "Last Payment: Nov 28, 2024"
)

# ---------------------------------------------------------------------------
# Flow state + session tracking — exposed to server.py for dashboard API
# ---------------------------------------------------------------------------
FLOW_NODES = [
    {"id": "greeting", "label": "Greeting"},
    {"id": "overdue_info", "label": "Overdue Info"},
    {"id": "understand_situation", "label": "Situation"},
    {"id": "payment_options", "label": "Options"},
    {"id": "commitment", "label": "Commitment"},
    {"id": "promise_to_pay", "label": "PTP"},
    {"id": "end", "label": "Complete"},
]

# pc_id -> {current_node, metrics, start_time, tts_type, _context}
session_data: dict = {}


def _track_node(flow_manager: FlowManager, node_name: str):
    """Update the current flow node for the visualization dashboard."""
    pc_id = flow_manager.state.get("pc_id", "")
    if pc_id and pc_id in session_data:
        session_data[pc_id]["current_node"] = node_name
        logger.info(f"Flow tracker: {node_name}")


# ---------------------------------------------------------------------------
# Flow Node Definitions
# ---------------------------------------------------------------------------

def create_greeting_node() -> NodeConfig:
    """Node 1: Greet and confirm identity."""

    async def confirm_identity(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple:
        flow_manager.state["identity_confirmed"] = True
        _track_node(flow_manager, "overdue_info")
        return "Identity confirmed as Rajesh Kumar", create_overdue_info_node()

    async def wrong_person(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple:
        _track_node(flow_manager, "end")
        return "Wrong person on the line", create_wrong_person_end_node()

    return NodeConfig(
        name="greeting",
        role_messages=[ROLE_MESSAGE],
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Greet warmly in Hinglish. Say something like: "
                    '"Namaste, kya main Rajesh Kumar ji se baat kar rahi hoon? '
                    'Main Priya bol rahi hoon, QuickFinance ki taraf se."\n\n'
                    "Wait for their response. Use confirm_identity if they confirm "
                    "(even partially), or wrong_person if they deny.\n\n"
                    f"{BORROWER_INFO}"
                ),
            }
        ],
        functions=[
            FlowsFunctionSchema(
                name="confirm_identity",
                handler=confirm_identity,
                description="Person confirms they are Rajesh Kumar or acknowledges their identity",
                properties={},
                required=[],
            ),
            FlowsFunctionSchema(
                name="wrong_person",
                handler=wrong_person,
                description="Person says they are NOT Rajesh Kumar or denies their identity",
                properties={},
                required=[],
            ),
        ],
    )


def create_overdue_info_node() -> NodeConfig:
    """Node 2: Inform about overdue EMIs."""

    async def borrower_responds(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple:
        _track_node(flow_manager, "understand_situation")
        return "Borrower acknowledged overdue information", create_situation_node()

    return NodeConfig(
        name="overdue_info",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Politely inform about overdue EMIs. Say something like: "
                    '"Rajesh ji, main aapko ek zaroori baat batana chahti thi. '
                    "Aapke do EMIs pending hain, December aur January ke. "
                    'Total Rs. 17,000 outstanding hai."\n\n'
                    "Be gentle and empathetic. After they respond in any way, "
                    "use borrower_responds to move forward.\n\n"
                    f"{BORROWER_INFO}"
                ),
            }
        ],
        functions=[
            FlowsFunctionSchema(
                name="borrower_responds",
                handler=borrower_responds,
                description=(
                    "Borrower responds to the overdue information — acknowledges, "
                    "questions, or expresses concern"
                ),
                properties={},
                required=[],
            ),
        ],
    )


def create_situation_node() -> NodeConfig:
    """Node 3: Understand borrower's situation."""

    async def record_situation(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple:
        reason = args.get("reason", "not specified")
        flow_manager.state["reason"] = reason
        _track_node(flow_manager, "payment_options")
        return f"Borrower's reason for delay: {reason}", create_payment_options_node()

    return NodeConfig(
        name="understand_situation",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Ask empathetically about their situation. Say something like: "
                    '"Main samajh sakti hoon Rajesh ji. Kya aap bata sakte hain ki '
                    "koi specific wajah thi EMI miss hone ki? "
                    'Main aapki help karna chahti hoon."\n\n'
                    "Listen with empathy, acknowledge their difficulties, then use "
                    "record_situation to capture their reason and move to payment options."
                ),
            }
        ],
        functions=[
            FlowsFunctionSchema(
                name="record_situation",
                handler=record_situation,
                description="Record the borrower's reason for delayed payment",
                properties={
                    "reason": {
                        "type": "string",
                        "description": "Brief summary of why the borrower missed payments",
                    }
                },
                required=["reason"],
            ),
        ],
    )


def create_payment_options_node() -> NodeConfig:
    """Node 4: Present payment options."""

    async def select_full_payment(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple:
        flow_manager.state["plan"] = "Full payment of Rs. 17,000"
        _track_node(flow_manager, "commitment")
        return "Full payment of Rs. 17,000 selected", create_commitment_node()

    async def select_split_payment(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple:
        flow_manager.state["plan"] = "Rs. 8,500 now + Rs. 8,500 in 15 days"
        _track_node(flow_manager, "commitment")
        return "Split payment plan selected", create_commitment_node()

    async def select_partial_plan(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple:
        flow_manager.state["plan"] = "Rs. 5,000 now + remaining in 2 installments"
        _track_node(flow_manager, "commitment")
        return "Partial payment plan selected", create_commitment_node()

    async def request_callback(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple:
        flow_manager.state["plan"] = "Callback requested"
        _track_node(flow_manager, "end")
        return "Senior representative callback requested", create_callback_end_node()

    return NodeConfig(
        name="payment_options",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Present payment options naturally and sympathetically. "
                    "Do NOT read them as a numbered list. Weave them into conversation.\n\n"
                    "Options available:\n"
                    "- Full Rs. 17,000 payment right away\n"
                    "- Pay one EMI Rs. 8,500 now, second within 15 days\n"
                    "- Rs. 5,000 now, remaining in 2 easy installments\n"
                    "- Request a callback from senior representative for restructuring\n\n"
                    "Recommend based on what the borrower has shared about their situation. "
                    "Use the matching function when they choose."
                ),
            }
        ],
        functions=[
            FlowsFunctionSchema(
                name="select_full_payment",
                handler=select_full_payment,
                description="Borrower agrees to pay full Rs. 17,000 immediately",
                properties={},
                required=[],
            ),
            FlowsFunctionSchema(
                name="select_split_payment",
                handler=select_split_payment,
                description="Borrower wants to pay Rs. 8,500 now and Rs. 8,500 in 15 days",
                properties={},
                required=[],
            ),
            FlowsFunctionSchema(
                name="select_partial_plan",
                handler=select_partial_plan,
                description="Borrower wants to pay Rs. 5,000 now and rest in 2 installments",
                properties={},
                required=[],
            ),
            FlowsFunctionSchema(
                name="request_callback",
                handler=request_callback,
                description="Borrower requests a callback from a senior representative for loan restructuring",
                properties={},
                required=[],
            ),
        ],
    )


def create_commitment_node() -> NodeConfig:
    """Node 5: Get payment commitment with a specific date."""

    async def confirm_commitment(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple:
        date = args.get("payment_date", "not specified")
        flow_manager.state["payment_date"] = date
        plan = flow_manager.state.get("plan", "")
        _track_node(flow_manager, "promise_to_pay")
        return f"Payment commitment: {plan} by {date}", create_promise_to_pay_node()

    return NodeConfig(
        name="commitment",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Confirm the chosen payment plan and ask for a specific date. "
                    "Say something like: "
                    '"Bahut accha Rajesh ji! Kya aap mujhe ek specific date bata sakte hain '
                    'jab tak aap payment kar denge?"\n\n'
                    "Once they give a date (even approximate), use confirm_commitment."
                ),
            }
        ],
        functions=[
            FlowsFunctionSchema(
                name="confirm_commitment",
                handler=confirm_commitment,
                description="Borrower commits to a specific payment date",
                properties={
                    "payment_date": {
                        "type": "string",
                        "description": "The date the borrower commits to make payment",
                    }
                },
                required=["payment_date"],
            ),
        ],
    )


def create_promise_to_pay_node() -> NodeConfig:
    """Node 6: Formal Promise to Pay (PTP) confirmation."""

    async def confirm_ptp(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple:
        plan = flow_manager.state.get("plan", "")
        date = flow_manager.state.get("payment_date", "")
        logger.info(f"Flow: PTP confirmed — {plan} by {date}")
        _track_node(flow_manager, "end")
        return f"PTP confirmed: {plan} by {date}", create_end_node()

    async def revise_plan(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple:
        _track_node(flow_manager, "payment_options")
        return "Borrower wants to revise the plan", create_payment_options_node()

    return NodeConfig(
        name="promise_to_pay",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Formally confirm the Promise to Pay. Summarize the commitment clearly. "
                    "Say something like: "
                    '"Rajesh ji, toh main confirm kar rahi hoon — aap [plan details] '
                    "[date] tak kar denge. Kya aap is commitment ko confirm karte hain? "
                    'Yeh aapka Promise to Pay hoga."\n\n'
                    "If they confirm, use confirm_ptp. "
                    "If they want to change, use revise_plan."
                ),
            }
        ],
        functions=[
            FlowsFunctionSchema(
                name="confirm_ptp",
                handler=confirm_ptp,
                description="Borrower formally confirms their Promise to Pay commitment",
                properties={},
                required=[],
            ),
            FlowsFunctionSchema(
                name="revise_plan",
                handler=revise_plan,
                description="Borrower wants to go back and choose a different payment plan",
                properties={},
                required=[],
            ),
        ],
    )


def create_end_node() -> NodeConfig:
    """Final node: Thank the borrower and close."""
    return NodeConfig(
        name="end",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Thank the borrower warmly and close the call. Summarize their commitment. "
                    "Say something like: "
                    '"Bahut bahut dhanyavaad Rajesh ji! Main aapka Promise to Pay note kar rahi hoon. '
                    "Agar koi bhi help chahiye toh QuickFinance ka helpline number hai aapke paas. "
                    'Aapka din shubh ho!"\n\n'
                    "Be warm, professional, and end on a positive note."
                ),
            }
        ],
        post_actions=[{"type": "end_conversation"}],
    )


def create_wrong_person_end_node() -> NodeConfig:
    """End node when the person is not the borrower."""
    return NodeConfig(
        name="wrong_person_end",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Apologize politely for the inconvenience. Say: "
                    '"Oh, maafi chahti hoon aapko disturb karne ke liye. '
                    'Galti se call lag gayi. Aapka din accha ho!"'
                ),
            }
        ],
        post_actions=[{"type": "end_conversation"}],
    )


def create_callback_end_node() -> NodeConfig:
    """End node when borrower requests a senior callback."""
    return NodeConfig(
        name="callback_end",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Confirm the callback request warmly. Say something like: "
                    '"Bilkul Rajesh ji, main aapki request note kar rahi hoon. '
                    "Humare senior representative aapko 24 ghante mein call karenge. "
                    'Dhanyavaad aapke time ke liye!"\n\n'
                    "End the conversation politely."
                ),
            }
        ],
        post_actions=[{"type": "end_conversation"}],
    )


# ---------------------------------------------------------------------------
# Bot Pipeline
# ---------------------------------------------------------------------------

async def run_bot(webrtc_connection: SmallWebRTCConnection, tts_type: str = "deepgram"):
    """Create and run the voice agent pipeline with Pipecat Flows."""

    # --- Transport (WebRTC) ---
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

    # --- LLM (Google Gemini 2.5 Flash) ---
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

    # --- Conversation context (empty — FlowManager populates it) ---
    context = OpenAILLMContext([])
    context_aggregator = llm.create_context_aggregator(context)

    # --- Pipeline ---
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
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

    # --- Flow Manager ---
    flow_manager = FlowManager(
        task=task,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
    )

    # --- Session tracking for the dashboard API ---
    pc_id = webrtc_connection.pc_id
    session_data[pc_id] = {
        "current_node": "greeting",
        "start_time": time.time(),
        "tts_type": tts_type,
        "_context": context,   # reference for live transcript
    }

    # --- Event handlers ---
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected: {client}")
        flow_manager.state["pc_id"] = pc_id
        await flow_manager.initialize(create_greeting_node())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected: {client}")
        session_data.pop(pc_id, None)
        await task.queue_frames([EndFrame()])

    # --- Run ---
    runner = PipelineRunner()
    await runner.run(task)
