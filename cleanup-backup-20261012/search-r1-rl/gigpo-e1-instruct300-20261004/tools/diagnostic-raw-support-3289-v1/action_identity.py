"""Exact sampled action prefixes, separate from the frozen text environment.

The Qwen byte-level vocabulary can have several token paths for the same text.
Never encode decoded text to obtain policy targets. This module does not change
search/answer parsing, sampler parameters, rewards, or a later turn's context.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numbers
import numpy as np


IDENTITY_MODE = 'sampled_action_prefix_v1'
LEGACY_MODE = 'legacy_canonical'
MAX_ACTIONS = 4
MAX_PREFIX = 4096
MAX_ACTION = 500


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _byte_decoder():
    # Qwen's ByteLevel decoder uses this reversible GPT-2 byte alphabet.
    values = list(range(ord('!'), ord('~') + 1)) + list(range(161, 173)) + list(range(174, 256))
    characters = values.copy()
    extra = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            characters.append(256 + extra)
            extra += 1
    return {chr(character): value for value, character in zip(values, characters)}


_BYTE_DECODER = _byte_decoder()


def canonical_environment_text(decoded):
    """Exact original closing priority, including search-before-answer."""
    if '</search>' in decoded:
        return decoded.split('</search>')[0] + '</search>'
    if '</answer>' in decoded:
        return decoded.split('</answer>')[0] + '</answer>'
    return decoded


def sampled_ids_with_evidence(ids, sampled_length, sampled_mask):
    """Use actual engine length/mask, never token-ID-based padding removal."""
    _require(isinstance(sampled_length, numbers.Integral) and not isinstance(sampled_length, bool),
             'Actual sampled length must be an integer')
    length = int(sampled_length)
    _require(0 <= length <= len(ids) and length <= MAX_ACTION, 'Invalid actual sampled length')
    _require(len(sampled_mask) == len(ids), 'Missing actual sampled mask coverage')
    _require(all(isinstance(value, numbers.Integral) and value in (0, 1) for value in sampled_mask),
             'Invalid actual sampled mask')
    _require(list(map(int, sampled_mask)) == [1] * length + [0] * (len(ids) - length),
             'Actual sampled mask must agree with the engine length')
    _require(all(isinstance(token, numbers.Integral) and not isinstance(token, bool) and int(token) >= 0
                 for token in ids), 'Invalid sampled token IDs')
    return [int(token) for token in ids[:length]]


@dataclass(frozen=True)
class ActionIdentity:
    actor_ids: tuple
    canonical_text: str
    sampled_length: int
    terminal_eos_excluded: bool
    selected_closing: str | None
    closing_byte_end: int | None
    closing_token_index: int | None
    closing_inside_token: bool
    sampled_ids_sha256: str

    def metadata(self):
        return dict(identity_mode=IDENTITY_MODE, sampled_length=self.sampled_length,
                    actor_length=len(self.actor_ids), terminal_eos_excluded=self.terminal_eos_excluded,
                    selected_closing=self.selected_closing, closing_byte_end=self.closing_byte_end,
                    closing_token_index=self.closing_token_index, closing_inside_token=self.closing_inside_token,
                    sampled_ids_sha256=self.sampled_ids_sha256)


def retain_sampled_action(tokenizer, ids, sampled_length, sampled_mask, canonical_text):
    """Return the shortest whole sampled-token prefix covering the old text cut.

    A boundary inside one token retains that entire atomic token. With no
    closing, retain every real sampled token except a genuine terminal EOS,
    consistent with the original skip-special textual endpoint. Interior
    special/PAD tokens remain in the actor token path, even though the frozen
    environment decoder skips them. Byte-decoder incompatibility is rejected.
    """
    sampled = sampled_ids_with_evidence(ids, sampled_length, sampled_mask)
    eos = getattr(tokenizer, 'eos_token_id', None)
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else (() if eos is None else (eos,)))
    _require(not any(token in eos_ids for token in sampled[:-1]),
             'Nonterminal EOS contradicts the supported sampler termination')
    terminal_eos = bool(sampled and sampled[-1] in eos_ids)
    supported = sampled[:-1] if terminal_eos else sampled
    decoded = tokenizer.decode(sampled, skip_special_tokens=True)
    _require(canonical_environment_text(decoded) == canonical_text,
             'Canonical environment text differs from the frozen postprocess')
    closing = ('</search>' if '</search>' in decoded else
               '</answer>' if '</answer>' in decoded else None)
    sampled_digest = hashlib.sha256(np.asarray(sampled, dtype=np.int64).tobytes()).hexdigest()
    if closing is None:
        # No byte boundary is needed. In particular, an incomplete UTF-8 final
        # token remains the original sampled token, regardless of how the
        # official decoder renders replacement characters in environment text.
        return ActionIdentity(tuple(supported), canonical_text, len(sampled), terminal_eos,
                              None, None, None, False, sampled_digest)
    special_ids = set(getattr(tokenizer, 'all_special_ids', ()))
    special_ids.update(eos_ids)
    pieces = []
    for token in supported:
        if token in special_ids:
            pieces.append(b'')
            continue
        piece = tokenizer.convert_ids_to_tokens(token)
        _require(isinstance(piece, str) and all(character in _BYTE_DECODER for character in piece),
                 'Unsupported non-ByteLevel token representation')
        pieces.append(bytes(_BYTE_DECODER[character] for character in piece))
    payload = b''.join(pieces)
    _require(payload.decode('utf-8', errors='replace') == decoded,
             'ByteLevel evidence differs from the official tokenizer decoder')
    byte_end = token_index = None
    inside = False
    length = len(supported)
    if closing is not None:
        marker = closing.encode('ascii')
        byte_end = payload.find(marker) + len(marker)
        _require(byte_end >= len(marker), 'Decoded closing has no byte-level source')
        _require(payload[:byte_end].decode('utf-8', errors='replace') == canonical_text,
                 'Frozen closing cut differs from the actual sampled bytes')
        total = 0
        for index, piece in enumerate(pieces):
            total += len(piece)
            if total >= byte_end:
                token_index = index
                length = index + 1
                inside = total > byte_end
                break
        _require(token_index is not None, 'Closing cannot be mapped to a complete sampled token')
    return ActionIdentity(tuple(supported[:length]), canonical_text, len(sampled), terminal_eos,
                          closing, byte_end, token_index, inside,
                          sampled_digest)
