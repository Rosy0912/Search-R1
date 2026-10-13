#!/usr/bin/env python3
"""Compose an original, quiet piano-and-pad cue for the question-probe film.

No samples, downloaded audio, or model inference are used.  All notes below are
authored for this cue; numpy only evaluates the oscillator/envelope equations.
The score is nine 4/4 bars at 72 BPM, exactly 30 seconds.  The exported waveform
is deterministic, stereo, 48 kHz, and 16-bit PCM.  Its restrained RMS leaves
headroom for scientific narration or captions; it is not a loud music master.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import wave

import numpy as np


SAMPLE_RATE = 48_000
DURATION = 30.0
BPM = 72.0
BEAT = 60.0 / BPM
BAR = 4.0 * BEAT
SEED = 20261013

# Original score. Each chord is (label, bass, four gently voiced notes).
CHORDS = [
    ("D add9", "D3", ("D4", "F#4", "A4", "E5")),
    ("B minor7", "B2", ("D4", "F#4", "A4", "B4")),
    ("G major7", "G2", ("D4", "F#4", "G4", "B4")),
    ("A sus4 add9", "A2", ("D4", "E4", "A4", "B4")),
    ("F# minor7", "F#2", ("C#4", "E4", "A4", "C#5")),
    ("G major7", "G2", ("D4", "F#4", "G4", "B4")),
    ("E minor7", "E3", ("D4", "E4", "G4", "B4")),
    ("A add9", "A2", ("C#4", "E4", "A4", "B4")),
    ("D add9", "D3", ("D4", "F#4", "A4", "E5")),
]

# (bar, beat within bar, pitch, relative velocity).  The descending response and
# final D5 deliberately make the short film feel resolved rather than looping.
MELODY = [
    (0, 1.0, "D5", 0.46), (0, 2.5, "E5", 0.35),
    (1, 0.0, "F#5", 0.44), (1, 2.0, "E5", 0.36),
    (2, 0.5, "D5", 0.42), (2, 2.5, "B4", 0.34),
    (3, 1.0, "E5", 0.40), (3, 3.0, "D5", 0.30),
    (4, 0.5, "C#5", 0.39), (4, 2.0, "A4", 0.30),
    (5, 0.0, "B4", 0.38), (5, 1.5, "D5", 0.35),
    (6, 0.5, "E5", 0.41), (6, 2.5, "D5", 0.32),
    (7, 0.0, "C#5", 0.38), (7, 2.0, "B4", 0.30),
    (8, 0.0, "A4", 0.30), (8, 1.5, "D5", 0.43),
]


def frequency(note: str) -> float:
    semitones = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4,
                 "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9,
                 "A#": 10, "B": 11}
    name, octave = note[:-1], int(note[-1])
    midi = (octave + 1) * 12 + semitones[name]
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def add_voice(mix: np.ndarray, voice: np.ndarray, start: float, pan: float,
              level: float) -> None:
    offset = round(start * SAMPLE_RATE)
    length = min(len(voice), len(mix) - offset)
    if offset < 0 or length <= 0:
        return
    # Equal-power panorama; the modest +/-0.22 field remains pleasant in mono.
    angle = (pan + 1.0) * math.pi / 4.0
    mix[offset:offset + length, 0] += voice[:length] * level * math.cos(angle)
    mix[offset:offset + length, 1] += voice[:length] * level * math.sin(angle)


def piano(note: str, duration: float = 4.7) -> np.ndarray:
    """Soft additive piano: struck attack, damped overtones, two string pairs."""
    t = np.arange(round(duration * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    f = frequency(note)
    # Slight string inharmonicity and a very quiet second string make the sound
    # rounded without a sharp synth-bell attack. Higher partials die first.
    partials = (1.0, 0.28, 0.105, 0.046, 0.026, 0.014, 0.006)
    voice = np.zeros_like(t)
    fundamental_decay = 1.15 + 180.0 / f
    for index, amplitude in enumerate(partials, 1):
        partial_f = f * index * math.sqrt(1 + 0.00007 * index * index)
        decay = fundamental_decay / (index ** 0.72)
        phase = 2.0 * math.pi * partial_f * t
        pair = (np.sin(phase) + 0.16 * np.sin(phase * 1.00072 + 0.13)) / 1.16
        voice += amplitude * pair * np.exp(-t / decay)
    attack = 1.0 - np.exp(-t / 0.014)
    release = np.minimum(1.0, np.maximum(0.0, duration - t) / 0.15)
    return voice * attack * release


def pad(note: str, duration: float = 5.9) -> np.ndarray:
    """Very low, warm, slow string-like tone, with no high-frequency hiss."""
    t = np.arange(round(duration * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    f = frequency(note)
    vibrato = 0.0007 * np.sin(2.0 * math.pi * 0.23 * t)
    phase = 2.0 * math.pi * f * t + vibrato
    tone = (np.sin(phase) + 0.10 * np.sin(phase * 2.0001)
            + 0.04 * np.sin(phase * 3.0004))
    tone = 0.65 * tone + 0.35 * np.sin(phase * 0.99945 + 0.4)
    attack = np.minimum(1.0, t / 1.25)
    attack = attack * attack * (3.0 - 2.0 * attack)
    release = np.minimum(1.0, np.maximum(0.0, duration - t) / 2.2)
    release = release * release * (3.0 - 2.0 * release)
    return tone * attack * release


def compose() -> np.ndarray:
    mix = np.zeros((round(DURATION * SAMPLE_RATE), 2), dtype=np.float64)
    rng = np.random.default_rng(SEED)
    # A sparse six-note arpeggio leaves room for the original melody.
    arpeggio = ((0.0, 0), (0.5, 2), (1.0, 1), (2.0, 3), (2.5, 2), (3.0, 1))
    for bar_index, (_, bass, chord) in enumerate(CHORDS):
        start = bar_index * BAR
        add_voice(mix, piano(bass, 5.5), start, -0.10, 0.050)
        for beat, index in arpeggio:
            if bar_index == 8 and beat > 1.0:
                continue  # last chord can ring out under the ending
            velocity = 0.056 * (0.90 + 0.14 * rng.random())
            add_voice(mix, piano(chord[index]), start + beat * BEAT,
                      -0.22 + 0.44 * index / 3.0, velocity)
        for index, note in enumerate(chord[:3]):
            add_voice(mix, pad(note), start, -0.18 + index * 0.18, 0.007)

    for bar_index, beat, note, velocity in MELODY:
        add_voice(mix, piano(note, 5.3), bar_index * BAR + beat * BEAT,
                  0.07, velocity * 0.155)

    # Stereo room reflections: parallel taps, not a feedback loop. No tail can
    # accumulate unboundedly. The late taps are quieter and lightly softened.
    dry = mix.copy()
    for seconds, gain in ((0.071, 0.085), (0.137, 0.075), (0.229, 0.060),
                          (0.347, 0.048), (0.521, 0.033), (0.811, 0.021)):
        delay = round(seconds * SAMPLE_RATE)
        room = dry[:-delay, ::-1]
        soft = (room + np.roll(room, 1, axis=0) + np.roll(room, 2, axis=0)) / 3.0
        mix[delay:] += gain * soft

    # Strip numerical DC, and use smoothstep fades to avoid clicks.
    mix -= mix.mean(axis=0, keepdims=True)
    for seconds, at_end in ((0.6, False), (1.5, True)):
        size = round(seconds * SAMPLE_RATE)
        ramp = np.linspace(0.0, 1.0, size)
        ramp = ramp * ramp * (3.0 - 2.0 * ramp)
        if at_end:
            mix[-size:] *= ramp[::-1, None]
        else:
            mix[:size] *= ramp[:, None]
    # Render modestly. The final video can target -24 LUFS with a -2 dBTP cap.
    rms = float(np.sqrt(np.mean(mix ** 2)))
    mix *= 10 ** (-25.0 / 20.0) / rms
    peak = float(np.max(np.abs(mix)))
    if peak > 10 ** (-5.0 / 20.0):
        mix *= 10 ** (-5.0 / 20.0) / peak
    return mix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mix = compose()
    if not np.isfinite(mix).all():
        raise RuntimeError("Non-finite audio sample")
    pcm = np.rint(np.clip(mix, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(args.output), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm.tobytes())
    # Read the actual file back; metrics are from quantized disk samples.
    with wave.open(str(args.output), "rb") as actual:
        data = np.frombuffer(actual.readframes(actual.getnframes()), dtype="<i2")
        data = data.astype(np.float64).reshape(-1, 2) / 32768.0
        mono = data.mean(axis=1)
        power = np.abs(np.fft.rfft(mono)) ** 2
        frequency_bins = np.fft.rfftfreq(len(mono), 1.0 / SAMPLE_RATE)
        metadata = {
            "original_composition": True,
            "sample_rate": actual.getframerate(), "channels": actual.getnchannels(),
            "sample_width_bytes": actual.getsampwidth(),
            "frames": actual.getnframes(),
            "duration_seconds": actual.getnframes() / actual.getframerate(),
            "peak_dbfs": 20.0 * math.log10(float(np.max(np.abs(data)))),
            "rms_dbfs": 20.0 * math.log10(float(np.sqrt(np.mean(data ** 2)))),
            "dc_offset": data.mean(axis=0).tolist(),
            "clipped_sample_count": int(np.sum(np.abs(data) >= 0.999)),
            "stereo_correlation": float(np.corrcoef(data.T)[0, 1]),
            "energy_above_4000hz_fraction": float(power[frequency_bins > 4000].sum() / power.sum()),
            "energy_above_8000hz_fraction": float(power[frequency_bins > 8000].sum() / power.sum()),
            "first_sample": data[0].tolist(), "last_sample": data[-1].tolist(),
            "maximum_adjacent_sample_change": float(np.max(np.abs(np.diff(data, axis=0)))),
            "fade_in_seconds": 0.6, "fade_out_seconds": 1.5,
            "bpm": BPM, "bars": len(CHORDS), "seed": SEED,
            "bytes": args.output.stat().st_size,
            "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            "recommended_video_audio_filter": "loudnorm=I=-24:TP=-2:LRA=7",
            "notes": "Original soft piano arpeggio and melody, very quiet warm pad; no voice or percussion.",
        }
    if metadata["clipped_sample_count"]:
        raise RuntimeError("Clipped audio")
    if metadata["energy_above_4000hz_fraction"] > 0.001:
        raise RuntimeError("Unexpected high-frequency energy")
    if metadata["stereo_correlation"] < 0.8:
        raise RuntimeError("Unexpected stereo cancellation")
    if np.max(np.abs(data[[0, -1]])) != 0:
        raise RuntimeError("Non-zero waveform endpoint")
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
