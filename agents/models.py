# agents/example_agent/models.py

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

UTC = timezone.utc

# These are the core data structures (models) used by both
# the AiClient and the main AiMentionApp (UI).
# Separating them ensures clean dependencies.

# =============== Models ===============

@dataclass
class EmotionState:
    valence: float = 0.0      # -1..1
    arousal: float = 0.2      # 0..1
    curiosity: float = 0.5    # 0..1
    anxiety: float = 0.1      # 0..1
    trust_to_user: float = 0.5  # 0..1

    def to_short_str(self) -> str:
        """Let's create emoji using values"""
        valence_emoji = "😊" if self.valence > 0.3 else "😐" if self.valence > -0.3 else "😞"
        arousal_emoji = "🔥" if self.arousal > 0.6 else "🌿"
        curiosity_emoji = "🤔" if self.curiosity > 0.5 else "😴"
        anxiety_emoji = "😰" if self.anxiety > 0.4 else "😌"
        trust_emoji = "💞" if self.trust_to_user > 0.6 else "🕳️"
        return f"{valence_emoji}={self.valence:.2f}/{arousal_emoji}={self.arousal:.2f}/{curiosity_emoji}={self.curiosity:.2f}/{anxiety_emoji}={self.anxiety:.2f}/{trust_emoji}={self.trust_to_user:.2f}"
        # return (
        #     f"V={self.valence:.2f}, A={self.arousal:.2f}, "
        #     f"C={self.curiosity:.2f}, Anx={self.anxiety:.2f}, "
        #     f"T={self.trust_to_user:.2f}"
        # )

    def clone(self) -> "EmotionState":
        return EmotionState(**asdict(self))

    def to_emoji(self) -> str:
        """Let's create emoji using values"""
        valence_emoji = "😊" if self.valence > 0.3 else "😐" if self.valence > -0.3 else "😞"
        arousal_emoji = "🔥" if self.arousal > 0.6 else "🌿"
        curiosity_emoji = "🤔" if self.curiosity > 0.5 else "😴"
        anxiety_emoji = "😰" if self.anxiety > 0.4 else "😌"
        trust_emoji = "💞" if self.trust_to_user > 0.6 else "🕳️"
        return f"{valence_emoji}{arousal_emoji}{curiosity_emoji}{anxiety_emoji}{trust_emoji}"
    
    # --- ADD THIS NEW METHOD ---
    def get_qualitative_description(self) -> str:
        """
        Translates the numerical emotion state into a qualitative,
        descriptive string for the LLM to understand.
        """
        desc = []

        # 1. Valence (Pleasure/Displeasure)
        if self.valence > 0.6:
            desc.append("feeling very happy and positive")
        elif self.valence > 0.2:
            desc.append("feeling pleasant and content")
        elif self.valence < -0.6:
            desc.append("feeling very unhappy and negative")
        elif self.valence < -0.2:
            desc.append("feeling a bit down and pessimistic")
        else:
            desc.append("feeling neutral")

        # 2. Arousal (Activation)
        if self.arousal > 0.7:
            desc.append("and highly energetic and active")
        elif self.arousal < 0.2:
            desc.append("and very calm and passive")
        
        # 3. Anxiety
        if self.anxiety > 0.7:
            desc.append(", but also extremely anxious and worried")
        elif self.anxiety > 0.4:
            desc.append(", with some underlying anxiety")

        # 4. Curiosity
        if self.curiosity > 0.7:
            desc.append(". You are very curious about the topic")
        elif self.curiosity < 0.3:
            desc.append(". You are somewhat bored or uninterested")

        # 5. Trust
        if self.trust_to_user < 0.3:
            desc.append(". You feel distrustful of the user")

        return ", ".join(desc) + "."

@dataclass
class ChatMessage:
    author: str               # "AI" or "User"
    text: str
    timestamp: datetime
    emotion_snapshot: EmotionState


@dataclass
class MentionThread:
    id: str
    title: str
    root_message: ChatMessage
    replies: list[ChatMessage] = field(default_factory=list)

    @property
    def root_emotion(self) -> EmotionState:
        return self.root_message.emotion_snapshot
