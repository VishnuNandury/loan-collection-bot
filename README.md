# Loan Collection Voice Agent - QuickFinance Ltd.

A multilingual (Hindi / English / Hinglish) voice-to-voice loan collection agent built with **PipeCat**.

## Tech Stack

| Component | Service | Cost |
|-----------|---------|------|
| **STT** (Speech-to-Text) | Deepgram Nova-2 | Free ($200 credit on signup) |
| **LLM** (Brain) | Google Gemini 2.5 Flash | Free (250 req/day) |
| **TTS** (Text-to-Speech) | Gemini 2.5 Flash TTS | Free (same Gemini API key) |
| **Transport** | WebRTC (browser-based) | Free |
| **VAD** | Silero (local) | Free |

## Quick Start (5 minutes)

### Step 1: Get API Keys (both free)

**Google Gemini API Key:**
1. Go to https://aistudio.google.com/apikey
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the key

**Deepgram API Key:**
1. Go to https://console.deepgram.com/signup
2. Create an account (no credit card needed)
3. You get **$200 free credit**
4. Go to **API Keys** > **Create a New API Key**
5. Copy the key

### Step 2: Setup Environment

```bash
cd Agent_pipecat

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configure API Keys

```bash
# Copy the example env file
copy .env.example .env        # Windows
# cp .env.example .env        # Linux/Mac

# Edit .env and paste your keys
```

Open `.env` in any editor and replace the placeholder values:
```
GOOGLE_API_KEY=AIzaSy...your_actual_key
DEEPGRAM_API_KEY=abc123...your_actual_key
```

### Step 4: Run

```bash
python server.py
```

Open **http://localhost:8080** in your browser and click the call button.

## How It Works

```
Browser Mic  -->  [WebRTC]  -->  [Deepgram STT]  -->  [Gemini LLM]  -->  [Gemini TTS]  -->  [WebRTC]  -->  Browser Speaker
   (voice)                        (speech to text)     (thinks)          (text to speech)                    (agent voice)
```

1. You speak into your browser microphone
2. Audio streams via WebRTC to the server
3. Deepgram converts your speech to text (supports Hindi + English)
4. Gemini processes the text and generates a contextual response
5. Gemini TTS converts the response to natural Hindi speech
6. Audio streams back to your browser

## Agent Personality

The agent plays **"Priya"** from QuickFinance Ltd., calling about overdue EMIs:
- Speaks Hindi, English, or Hinglish based on the borrower's language
- Empathetic and professional tone
- Can negotiate payment plans
- Handles objections gracefully

## Customization

### Change the borrower details
Edit `SYSTEM_PROMPT` in `bot.py` - update the name, loan amount, etc.

### Change the agent voice
In `bot.py`, change the `voice_id` parameter in `GeminiTTSService`:
- `Kore` - Female (default, works well with Hindi)
- `Charon` - Male
- `Puck` - Energetic
- `Aoede` - Female, warm
- `Fenrir` - Male, deep

See full list: 30 voices available in Gemini TTS

### Change the language
In `bot.py`, update the Deepgram `language` parameter:
- `hi` - Hindi (default, also handles English well)
- `hi-Latn` - Romanized Hindi
- `en` - English only

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "API keys not configured" | Check your `.env` file has valid keys |
| No audio from agent | Allow microphone permission in browser |
| Agent doesn't respond | Check terminal logs for errors |
| Slow responses | Normal on first call (model loading). Subsequent calls are faster |
| Port 8080 in use | Run with `python server.py --port 9090` |
