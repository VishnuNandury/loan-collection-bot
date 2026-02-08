"""
Edge TTS Service for PipeCat
=============================
Custom TTSService wrapping Microsoft Edge TTS (free, no API key needed).
Supports Hindi, English, and many other languages with natural-sounding voices.
"""

import io
from typing import AsyncGenerator

import av
import edge_tts
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

EDGE_TTS_SAMPLE_RATE = 24000


class EdgeTTSService(TTSService):
    """Microsoft Edge TTS service for PipeCat.

    Uses the edge-tts library to synthesize speech. No API key required.
    Edge TTS outputs MP3 which is decoded to raw PCM for the pipeline.

    Args:
        voice: Edge TTS voice ID (e.g. "hi-IN-SwaraNeural").
    """

    def __init__(self, *, voice: str = "hi-IN-SwaraNeural", **kwargs):
        super().__init__(sample_rate=EDGE_TTS_SAMPLE_RATE, **kwargs)
        self.set_voice(voice)

    def _decode_mp3_to_pcm(self, mp3_bytes: bytes) -> bytes:
        """Decode MP3 bytes to 16-bit mono PCM at 24 kHz."""
        buf = io.BytesIO(mp3_bytes)
        container = av.open(buf, format="mp3")
        resampler = av.AudioResampler(
            format="s16", layout="mono", rate=EDGE_TTS_SAMPLE_RATE
        )

        pcm_chunks = []
        for frame in container.decode(audio=0):
            resampled = resampler.resample(frame)
            for rf in resampled:
                pcm_chunks.append(rf.to_ndarray().tobytes())

        container.close()
        return b"".join(pcm_chunks)

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"EdgeTTSService: Generating TTS [{text}]")

        try:
            await self.start_ttfb_metrics()
            yield TTSStartedFrame()

            communicate = edge_tts.Communicate(text, voice=self._voice_id)

            # Collect all MP3 chunks then decode in one shot
            # (edge-tts streams small MP3 fragments that aren't independently decodable)
            mp3_data = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_data.extend(chunk["data"])

            if mp3_data:
                await self.stop_ttfb_metrics()
                pcm_data = self._decode_mp3_to_pcm(bytes(mp3_data))

                # Yield in chunks for smooth streaming
                chunk_size = self.chunk_size
                for i in range(0, len(pcm_data), chunk_size):
                    yield TTSAudioRawFrame(
                        audio=pcm_data[i : i + chunk_size],
                        sample_rate=EDGE_TTS_SAMPLE_RATE,
                        num_channels=1,
                    )

            yield TTSStoppedFrame()

        except Exception as e:
            logger.error(f"EdgeTTSService error: {e}")
            yield ErrorFrame(error=str(e))
