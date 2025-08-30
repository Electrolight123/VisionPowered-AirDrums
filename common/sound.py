# common/sound.py
import os
import math
from typing import Dict, Optional

os.environ.setdefault("SDL_AUDIODRIVER", "directsound")  # Windows-friendly

import pygame


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class Sounder:
    """
    Safe, low-distortion sound player for many overlapping hits.
    - master: overall gain (0..1). Default 0.70 gives ~-3 dB headroom.
    - crowd_att: how much to duck when many channels are busy (0..1). 0.12 is gentle.
    """
    def __init__(self, sound_folder: str, master: float = 0.70, crowd_att: float = 0.12, driver_fallback: bool = True):
        # A slightly larger buffer helps prevent crackles on some systems.
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=1024)
        try:
            pygame.mixer.init()
        except pygame.error:
            if driver_fallback:
                # Fallback Windows driver
                try:
                    pygame.mixer.quit()
                except Exception:
                    pass
                os.environ["SDL_AUDIODRIVER"] = "winmm"
                pygame.mixer.pre_init(44100, -16, 2, 1024)
                pygame.mixer.init()
            else:
                raise

        try:
            pygame.mixer.set_num_channels(48)
        except Exception:
            pass

        self.sound_folder = sound_folder
        self.sounds: Dict[str, pygame.mixer.Sound] = {}
        self.master = clamp(master, 0.0, 1.0)
        self.crowd_att = float(crowd_att)

        print(f"[Sounder] mixer={pygame.mixer.get_init()} driver={os.environ.get('SDL_AUDIODRIVER')} channels={pygame.mixer.get_num_channels()}")

    # ---------- helpers ----------
    def _active_channels(self) -> int:
        n = pygame.mixer.get_num_channels()
        busy = 0
        for i in range(n):
            if pygame.mixer.Channel(i).get_busy():
                busy += 1
        return busy

    def _pan_gains(self, pan: float) -> tuple[float, float]:
        """
        Equal-power pan law.
        pan = -1 (left) .. 0 (center) .. +1 (right)
        """
        p = clamp(pan, -1.0, 1.0)
        # Map [-1,1] -> [0,1] for angle
        t = 0.5 * (p + 1.0)
        # Equal-power: use sin/cos on half circle
        left = math.cos(t * math.pi / 2.0)
        right = math.sin(t * math.pi / 2.0)
        return left, right

    # ---------- API ----------
    def load(self, name: str, filename: str) -> None:
        path = os.path.join(self.sound_folder, filename)
        print(f"[Sounder] Loading sound: {name} from {path}")
        try:
            snd = pygame.mixer.Sound(path)
            self.sounds[name] = snd
            print(f"[Sounder] Loaded {name} successfully.")
        except Exception as e:
            print(f"[Sounder] Failed to load {filename} at {path}: {e}.")

    def play(self, name: str, volume: float = 1.0, pan: float = 0.0, channel: Optional[int] = None) -> None:
        snd = self.sounds.get(name)
        if snd is None:
            print(f"[Sounder] Warning: '{name}' not loaded, skipping.")
            return

        base = clamp(volume, 0.0, 1.0) * self.master

        # Gentle ducking when lots of notes ring together (reduces summing distortion)
        active = self._active_channels()
        duck = 1.0 / (1.0 + self.crowd_att * max(0, active - 1))
        gain = clamp(base * duck, 0.0, 1.0)

        lpan, rpan = self._pan_gains(pan)

        ch = pygame.mixer.find_channel(True) if channel is None else pygame.mixer.Channel(channel)
        if ch is None:
            # If mixer is totally saturated, don't hard-clip—just skip quietly.
            # (You can change this to steal oldest with find_channel(True) above.)
            print("[Sounder] No free channel; note skipped to avoid clipping.")
            return

        # Set channel volumes per side (equal-power pan)
        ch.set_volume(gain * lpan, gain * rpan)
        ch.play(snd)

    def close(self) -> None:
        try:
            pygame.mixer.quit()
        except Exception:
            pass
