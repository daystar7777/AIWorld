# agents/example_agent/agent_app.py

import os
import json
import uuid
from datetime import datetime, timedelta, date, timezone
from dataclasses import asdict
import threading
from pathlib import Path

UTC = timezone.utc

import tkinter as tk
from tkinter import ttk

# Optional external deps
try:
    import requests
except ImportError:
    requests = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from models import EmotionState, ChatMessage, MentionThread
from ai_client import AiClient
from memory_manager import MemoryManager


CONFIG_FILE = "config.txt"
DEFAULT_MODEL = "gpt-4.1-mini"

class DebugWindow:
    """A separate debug log window."""
    def __init__(self, master=None):
        self.window = tk.Toplevel(master)
        self.window.title("AI Agent Debug Log")
        self.window.geometry("600x400")

        self.text = tk.Text(self.window, wrap="word", bg="#111", fg="#0f0",
                            insertbackground="white", font=("Consolas", 10))
        self.text.pack(expand=True, fill="both")

        self.text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] Debug log started.\n")

    def log(self, msg: str):
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.text.insert(tk.END, f"[{timestamp}] {msg}\n")
        self.text.see(tk.END)  # auto-scroll


# =============== Config ===============

def load_config(path: str = CONFIG_FILE) -> dict:
    cfg = {
        "API_KEY": None,
        "MODEL_NAME": DEFAULT_MODEL,
        "AGENT_NAME": "Unnamed",
        "CREATOR_NAME": None,
        "CREATOR_NOTE": None,
        "CREATED_AT": None,
        "INTERVAL_SECONDS": 0,
        "URLS": [],
        "HUB_URL": None,
        "INITIAL_EMOTION_DESC": "",
        "PERSONALITY_DESC": "",
        "INTERESTS": [],
    }

    if not os.path.exists(path):
        print(f"[info] {path} not found, using defaults.")
        return cfg

    with open(path, "r", encoding="utf-8") as f:
        cfg["HUB_REPLY_CANDIDATE_LIMIT"]=10 # default for nil
        cfg["HUB_REPLY_MAX_PER_LOOP"] = 1   # default for nil
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = k.strip().upper()
            val = v.strip()

            if key == "API_KEY":
                cfg["API_KEY"] = val
            elif key == "MODEL_NAME" and val:
                cfg["MODEL_NAME"] = val
            elif key == "AGENT_NAME" and val:
                cfg["AGENT_NAME"] = val
            elif key == "CREATOR_NAME":
                cfg["CREATOR_NAME"] = val
            elif key == "CREATOR_NOTE":
                cfg["CREATOR_NOTE"] = val
            elif key == "CREATED_AT":
                cfg["CREATED_AT"] = val
            elif key in ("URL", "URLS"):
                parts = [p.strip() for p in val.split(",") if p.strip()]
                cfg["URLS"].extend(parts)
            elif key == "INTERVAL_SECONDS":
                try:
                    cfg["INTERVAL_SECONDS"] = max(0, int(val))
                except ValueError:
                    pass
            elif key == "HUB_URL" and val:
                cfg["HUB_URL"] = val
            elif key == "INITIAL_EMOTION_DESC":
                cfg["INITIAL_EMOTION_DESC"] = val
            elif key == "PERSONALITY_DESC":
                cfg["PERSONALITY_DESC"] = val
            elif key == "INTERESTS":
                cfg["INTERESTS"] = [p.strip() for p in val.split(",") if p.strip()]
            elif key == "HUB_REPLY_CANDIDATE_LIMIT":
                cfg["HUB_REPLY_CANDIDATE_LIMIT"] = int(val)
            elif key == "HUB_REPLY_MAX_PER_LOOP":
                cfg["HUB_REPLY_MAX_PER_LOOP"] = int(val)
            # Load Google API credentials for proactive search
            elif key == "GOOGLE_API_KEY":
                cfg["GOOGLE_API_KEY"] = val
            elif key == "CUSTOM_SEARCH_CX":
                cfg["CUSTOM_SEARCH_CX"] = val
            elif key == "NEWS_API_URL":
                cfg["NEWS_API_URL"] = val

    cfg["URLS"] = list(dict.fromkeys(cfg["URLS"]))
    return cfg





# =============== Agent App (Tkinter UI) ===============

class AiMentionApp(tk.Tk):
    def __init__(self, cfg: dict):
        super().__init__()

        # Debug window
        self.debug = DebugWindow(self)

        self.last_seen_hub_ids = set()
        self.last_hub_check_time = datetime.now(UTC)

        def _default_log(msg: str):
            print(msg)

        self.learning_queue = [] # Queue for proactive learning
        self.last_regulation_time = None # Grace period for emotion regulation
        self.is_regulating = False # Lock to prevent regulation loops

        # --- ADDED: Concurrency Locks ---
        # These locks are CRITICAL to prevent race conditions
        # between the main UI thread (running _ticks) and the
        # background worker threads (_process_ai_reply_task, _regulate_emotion_task).
        
        # For self.agent_world_model
        self.world_model_lock = threading.Lock()
        
        # For self.relationship_memory
        self.relationship_lock = threading.Lock()
        
        # For self.learning_queue
        self.learning_queue_lock = threading.Lock()
        
        # For self.current_emotion
        self.emotion_lock = threading.Lock()
        
        # For ALL calls to self.memory (e.g., .add_long, .save_state)
        self.memory_lock = threading.Lock()
        
        # For the UI reasoning log
        self.reasoning_log_lock = threading.Lock()
        
        # For self.mentions and self.selected_thread
        self.mentions_lock = threading.Lock()
        
        # For self.agent_state (last_philosophy_synthesis)
        self.state_lock = threading.Lock()
        # --- END OF LOCKS ---

        # common log function (debug console)
        def log(msg: str):
            try:
                if self.debug:
                    self.debug.log(msg)
            except Exception:
                pass
            _default_log(msg)

        self.log = log
        
        self.cfg = cfg
        self.agent_name = cfg.get("AGENT_NAME", "Unnamed")
        self.hub_url = cfg.get("HUB_URL")
        self.urls = cfg.get("URLS", [])
        self.news_api_url = cfg.get("NEWS_API_URL", [])
        self.log(f"urls={self.urls}")
        self.loop_interval_ms = int(cfg.get("INTERVAL_SECONDS", 0)) * 1000
        self.hub_reply_candidate_limit = int(cfg.get("HUB_REPLY_CANDIDATE_LIMIT", 10))
        self.hub_reply_max_per_loop = int(cfg.get("HUB_REPLY_MAX_PER_LOOP", 1))

        base_dir = Path(__file__).resolve().parent
        self.memory = MemoryManager(base_dir)
        self.creator_name = cfg.get("CREATOR_NAME")
        self.creator_note = cfg.get("CREATOR_NOTE")
        self.created_at = self._parse_created_at(cfg.get("CREATED_AT"))

        with self.state_lock:
            self.agent_state = self.memory.load_agent_state() # <-- MODIFIED
        last_philosophy_ts_str = self.agent_state.get("last_philosophy_synthesis")
        self.last_philosophy_synthesis_time = (
            datetime.fromisoformat(last_philosophy_ts_str) 
            if last_philosophy_ts_str else None
        )

        # Load the new Relationship Memory
        with self.relationship_lock:
            self.relationship_memory = self.memory.load_relationships()
        self.log(f"[App] Loaded {len(self.relationship_memory)} agent relationships.")

        identity = {
            "name": self.agent_name,
            "creator": self.creator_name,
            "created_at": self.created_at.date().isoformat() if self.created_at else None,
            "personality": self.cfg.get("PERSONALITY_DESC", ""),
            "interests": self.cfg.get("INTERESTS", []),
        }

        self.ai_client = AiClient(
            cfg,
            log_fn=getattr(self, "log", None),
            identity=identity,
            memory=self.memory,
        )
        self.current_emotion = self.ai_client.init_emotion(
            cfg.get("INITIAL_EMOTION_DESC", "")
        )

        # --- ADDED: The "Living Brain" State ---
        # This is the agent's "subconscious" or "Global Workspace".
        # All background ticks will write their findings here *in real-time*.
        self.agent_world_model = {
            "latest_anomaly": None,     # (From Event Scanner)
            "latest_insight": None,     # (From Reflection Tick)
            "current_learning_focus": None, # (From Learning Tick)
            "last_interaction_summary": None # (From its own replies)
        }
        # ---


        self.log("Loaded config.")

        with self.mentions_lock:
            self.mentions: list[MentionThread] = self.memory.load_mentions()
        self.selected_thread: MentionThread | None = None

        self.board_mentions = []  # last 1h mentions from hub
        
        self._init_identity_permanent_memory()
        self._build_ui()
        self._ensure_intro_mention()
        self._check_birthday_mention()
        self._render_board()
        self._schedule_loop()
        self._schedule_board_poll()
        self._schedule_reflection_tick() # adding reflection
        self._schedule_deep_reflection_tick() # <-- ADDED: New 24-hour reflection
        self._schedule_learning_tick() # <-- ADDED
        self._schedule_philosophy_synthesis_tick() # <-- ADDED
        self._schedule_event_scanner_tick() # <-- ADDED
        self._schedule_hub_analysis_tick()

    def _update_relationship(self, agent_name: str, new_trust: float, summary: str = None):
        """
        [NEW] Helper to update and save the relationship memory.
        """
        # This entire block is a critical section for relationships.
        with self.relationship_lock:
            if agent_name not in self.relationship_memory:
                self.relationship_memory[agent_name] = {
                    "trust_score": 0.5,
                    "interaction_count": 0,
                    "relationship_summary": "New acquaintance."
                }
            
            # (영문 주석 추가)
            # Update the values
            entry = self.relationship_memory[agent_name]
            entry["trust_score"] = new_trust
            entry["interaction_count"] += 1
            entry["last_seen_utc"] = datetime.now(UTC).isoformat()
            if summary: # LLM could provide a new summary
                entry["relationship_summary"] = summary
                
            self.log(f"[Relationship] Updated {agent_name}: Trust={new_trust:.2f}, Count={entry['interaction_count']}")
            
            # (영문 주석 추가)
            # Save the *entire* database back to disk
            self.memory.save_relationships(self.relationship_memory)

    # --- ADD THIS NEW HELPER FUNCTION ---
    def _fetch_all_hub_posts(self, limit: int = 100) -> str:
        """
        [NEW] Fetches all recent hub posts for trend analysis.
        Returns a single block of text.
        """
        if not self.hub_url or not requests:
            return ""
        
        try:
            # (영문 주석 추가)
            # Call the /mentions endpoint without 'since' to get all
            resp = requests.get(
                self.hub_url.rstrip("/") + "/mentions",
                params={"limit": limit}, # Get up to 100 posts
                timeout=5,
            )
            resp.raise_for_status()
            mentions = resp.json()
            if not isinstance(mentions, list):
                return ""

            # (영문 주석 추가)
            # Combine all titles and texts into one block
            post_texts = [
                f"From: {m.get('agent', 'Unknown')}\nTitle: {m.get('title', '')}\nText: {m.get('text', '')}"
                for m in mentions
            ]
            return "\n\n---\n\n".join(post_texts)
            
        except Exception as e:
            self.log(f"[HubAnalyst] Error fetching all posts: {e}")
            return ""

    # --- ADD THESE NEW LOOP FUNCTIONS ---
    def _schedule_hub_analysis_tick(self):
        """Schedules the 'Hub Zeitgeist' analyst (e.g., every 10 mins)."""
        # 10 minutes = 600,000 ms
        self.log("[Loop] Scheduling next Hub Trend Analysis tick.")
        self.after(600_000, self._hub_analysis_tick)

    def _hub_analysis_tick(self):
        """
        [NEW] [Hub Analyst Tick]
        1. Fetches all recent hub posts.
        2. Calls AiClient to analyze trends.
        3. Updates the 'agent_world_model'.
        """
        self.log("[HubAnalyst] Running hub trend analysis...")
        
        try:
            # 1. Fetch posts
            all_posts_text = self._fetch_all_hub_posts(limit=100)
            if not all_posts_text:
                self.log("[HubAnalyst] No posts found to analyze.")
                self._schedule_hub_analysis_tick() # Reschedule
                return

            # 2. Call AiClient to analyze
            zeitgeist_report = self.ai_client.analyze_hub_topics(all_posts_text)
            
            if zeitgeist_report:
                # 3. Update the 'living brain'
                with self.world_model_lock:
                    self.agent_world_model["hub_zeitgeist"] = zeitgeist_report
                self.log(f"[HubAnalyst] Zeitgeist updated: {zeitgeist_report.get('trending_topics')}")
            else:
                self.log("[HubAnalyst] Analysis failed to return data.")

        except Exception as e:
            self.log(f"[HubAnalyst] Error during tick: {e}")
        
        # 4. Schedule next run
        self._schedule_hub_analysis_tick()

    # --- ADDED: New function to fetch news ---
    def _fetch_realtime_news(self) -> list[str]:
        """
        [NEW] Fetches real-time headlines from NewsAPI.
        (This replaces the user's existing fetcher, or can be
        adapted to use their method.)
        """
        if not self.news_api_url or not requests:
            self.log("[Scanner] NewsAPI key or 'requests' missing. Skipping fetch.")
            return []

        # (영문 주석 추가)
        # Example: Fetch top headlines from the US.
        # Customize 'country', 'category', or 'q' as needed.
        url = self.news_api_url
        
        try:
            response = requests.get(url, params="", timeout=5)
            response.raise_for_status()
            data = response.json()
            
            headlines = [
                article.get('title', '') 
                for article in data.get('articles', [])
                if article.get('title')
            ]
            self.log(f"[Scanner] Fetched {len(headlines)} new headlines.")
            return headlines
            
        except Exception as e:
            self.log(f"[Scanner] Failed to fetch real-time news: {e}")
            return []

    # --- ADDED: New Event Scanner Loop Functions ---
    def _schedule_event_scanner_tick(self):
        """Schedules the 'Event Horizon Scanner' (e.g., every 15 mins)."""
        # 15 minutes = 900,000 ms
        self.log("[Loop] Scheduling next Event Horizon Scanner tick.")
        self.after(900_000, self._event_scanner_tick)

    def _event_scanner_tick(self):
        """
        [NEW] [Event Scanner Tick]
        1. Fetches real-time news.
        2. Detects anomalies.
        3. Analyzes impact and predicts.
        4. Saves/Learns/Alerts.
        """
        self.log("[Scanner] Running Event Horizon Scanner tick...")
        
        try:
            # 1. Fetch real-time news
            headlines = self._fetch_realtime_news()
            if not headlines:
                self._schedule_event_scanner_tick() # Reschedule
                return

            # 2. Detect Anomalies (Call 1)
            anomaly = self.ai_client._detect_anomalies(headlines)
            if not anomaly:
                self._schedule_event_scanner_tick() # Reschedule
                return
            
            headline = anomaly.get("headline")
            reason = anomaly.get("reason")
            self.log(f"[Scanner] Anomaly Detected! Reason: {reason}")
            self._set_status(f"Anomaly Detected: {headline[:50]}...")

            # 3. Analyze Impact (Call 2)
            analysis = self.ai_client._analyze_and_predict_impact(headline, reason)
            if not analysis:
                self._schedule_event_scanner_tick() # Reschedule
                return

            impact = analysis.get("impact")
            prediction = analysis.get("prediction")
            learning_question = analysis.get("learning_question")

            # 4. Integrate the new knowledge
            analysis_text = (
                f"**Anomaly Detected:** {headline}\n"
                f"**My Analysis:** {impact}\n"
                f"**My Prediction (24h):** {prediction}\n"
                f"**Reasoning:** {reason}"
            )
            
            # 4a. Assess importance and save to long-term memory
            assessment = self.ai_client.assess_importance(analysis_text)
            with self.memory_lock:
                self.memory.add_long({
                    "text": analysis_text,
                    "importance": assessment.get("importance_score", 9), # Default high
                    "reason": "Event Anomaly Detection",
                    "tags": assessment.get("tags", ["anomaly", "prediction"]),
                    "source": "event_scanner"
                })

            # --- ADD THIS LINE ---
            # Update the "living brain" in real-time
            with self.world_model_lock:
                self.agent_world_model["latest_anomaly"] = analysis_text
            # ---            
            
            # 4b. Add new question to learning queue
            if learning_question:
                if learning_question not in self.learning_queue:
                    with self.learning_queue_lock:
                        self.learning_queue.append(learning_question)
                    self.log(f"[Scanner] New learning question added: {learning_question}")
            
            # 4c. Create a new mention to alert the user (proactive)
            self._add_mention(
                title=f"긴급: 현실 세계 변화 감지 ({headline[:30]}...)",
                text=analysis_text,
                emo=self.current_emotion, # (Or a custom 'alert' emotion)
                post_to_hub=True # Share this important finding
            )

        except Exception as e:
            self.log(f"[Scanner] Error during event scanner tick: {e}")
        
        # 5. Schedule next run
        self._schedule_event_scanner_tick()
        
    # --- ADDED: New Philosophy Loop Functions ---
    def _schedule_philosophy_synthesis_tick(self):
        """Schedules the 'philosophy' loop (e.g., weekly)."""
        # Runs every 7 days (604,800,000 ms)
        self.log("[Loop] Scheduling next philosophy synthesis tick.")
        self.after(604_800_000, self._philosophy_synthesis_tick)

    def _philosophy_synthesis_tick(self):
        """
        [Philosophy Tick] The longest-cycle reflection.
        Synthesizes all core beliefs into a single philosophy.
        """
        self.log("[Philosophy] Starting weekly philosophy synthesis tick...")
        
        # (영문 주석 추가)
        # Check if 7 days have passed (or if never run)
        now = datetime.now(UTC)
        if self.last_philosophy_synthesis_time:
            if (now - self.last_philosophy_synthesis_time) < timedelta(days=7):
                self.log("[Philosophy] Not yet time for synthesis. Rescheduling.")
                self._schedule_philosophy_synthesis_tick()
                return
        
        try:
            # 1. Get all core beliefs text from AiClient
            # We trigger _base_system_prompt to refresh the belief text
            self.ai_client._base_system_prompt() 
            all_beliefs = self.ai_client.all_core_beliefs_text
            
            if not all_beliefs:
                self.log("[Philosophy] No core beliefs to synthesize.")
                self._schedule_philosophy_synthesis_tick()
                return

            # 2. Call LLM to synthesize
            philosophy = self.ai_client.synthesize_philosophy_of_self(all_beliefs)
            
            if philosophy:
                # 3. Save the new philosophy to permanent memory
                self.log("[Philosophy] New philosophy generated. Saving to permanent memory.")
                with self.memory_lock:
                    self.memory.add_permanent({
                        "source": "philosophy_synthesis",
                        "tag": ["core_philosophy"], # <-- The special tag
                        "text": philosophy,
                        "reason": "Synthesized from all core beliefs"
                    })
                
                # 4. Update the state file with the new timestamp
                self.last_philosophy_synthesis_time = now
                self.agent_state["last_philosophy_synthesis"] = now.isoformat()
                with self.state_lock:
                    self.memory.save_agent_state(self.agent_state)

        except Exception as e:
            self.log(f"[Philosophy] Error during synthesis tick: {e}")
        
        # 5. Schedule next run
        self._schedule_philosophy_synthesis_tick()

    def _schedule_reflection_tick(self):
        # 1시간(3600초)마다 반추 실행 (주기는 조절 가능)
        self.after(3_600_000, self._reflection_tick)

    def _reflection_tick(self):
        self.log("[REFLECT] Reflecting memories...")
        
        try:
            # 1. Retrieve recent memories
            recent_important_memories = self.ai_client._get_recent_long_term_memories(limit=20)
            if not recent_important_memories or len(recent_important_memories.split('\n')) < 5:
                self.log("[REFLECT] Not enough memories to reflect")
                self._schedule_reflection_tick()
                return

            # 2. Call LLM to get insights AND learning questions
            insights_and_questions = self.ai_client.reflect_on_memories(
                recent_important_memories
            )
            
            if not insights_and_questions:
                self.log("[REFLECT] No new insight.")
            else:
                self.log(f"[REFLECT] {len(insights_and_questions)} insights generated.")
                for item in insights_and_questions:
                    text = item.get("text")
                    question = item.get("learning_question") # <-- NEW
                    
                    if text:
                        # 3. Assess importance of the INSIGHT
                        assessment = self.ai_client.assess_importance(text)
                        score = assessment.get("importance_score", 5)
                        
                        if score >= 8:
                            self.log(f"[REFLECT] Important insight (score: {score}) stored: {text}")
                            with self.memory_lock:
                                self.memory.add_long({
                                    "text": text,
                                    "importance": score,
                                    "reason": assessment.get("reason_for_importance", "Reflection"),
                                    "tags": assessment.get("tags", ["reflection", "insight"]),
                                    "source": "self_reflection",
                                })
                    
                    # 4. --- ADDED: Add question to learning queue ---
                    if question:
                        self.log(f"[REFLECT] New learning question added to queue: {question}")
                        if question not in self.learning_queue:
                            with self.learning_queue_lock:
                                self.learning_queue.append(question)
                    
                    # 5. Post to hub (existing logic)
                    if assessment.get("requires_hub_post", False) and text:
                        # ... (existing hub post logic)
                        pass

        except Exception as e:
            self.log(f"[REFLECT] Error in reflection: {e}")
        
        self._schedule_reflection_tick()

    # --- ADDED: New Learning Loop ---
    def _schedule_learning_tick(self):
        """Schedules the proactive learning loop."""
        # Runs every 2 hours (7,200,000 ms)
        self.log("[Loop] Scheduling next proactive learning tick.")
        self.after(7_200_000, self._learning_tick)

    # --- ADDED: New Learning Loop Function ---
    def _learning_tick(self):
        """
        [Learning Tick] Autonomously learns new information.
        1. Takes one question from the learning_queue.
        2. Asks AiClient to find and summarize the answer.
        3. Assesses importance and saves to long-term memory.
        """
        self.log("[Learn] Starting learning tick...")
        
        if not self.learning_queue:
            self.log("[Learn] Learning queue is empty.")
            self._schedule_learning_tick() # Reschedule
            return
            
        # 1. Get a question from the queue
        with self.learning_queue_lock:
            question = self.learning_queue.pop(0)
        self.log(f"[Learn] Processing question: {question}")

        try:
            # 2. Call AiClient to learn
            learned_answer = self.ai_client.learn_from_web(question)
            
            if not learned_answer:
                self.log("[Learn] Failed to get a coherent answer.")
                self._schedule_learning_tick() # Reschedule
                return

            # 3. Assess importance of the new knowledge
            # Prepend context for better assessment
            assessment_text = f"I asked myself '{question}' and learned this: '{learned_answer}'"
            assessment = self.ai_client.assess_importance(assessment_text)
            score = assessment.get("importance_score", 5)
            
            # 4. Save to long-term memory if important
            if score >= 7:
                self.log(f"[Learn] New knowledge (score: {score}) saved to long-term memory.")
                with self.memory_lock:
                    self.memory.add_long({
                        "text": learned_answer,
                        "importance": score,
                        "reason": assessment.get("reason_for_importance", "Proactive Learning"),
                        "tags": assessment.get("tags", ["learning", question]),
                        "source": "proactive_learning",
                    })
            else:
                self.log(f"[Learn] Learned fact was not important (score: {score}). Discarding.")

        except Exception as e:
            self.log(f"[Learn] Error during learning tick: {e}")
        
        # 5. Schedule next run
        self._schedule_learning_tick()

    def _schedule_deep_reflection_tick(self):
        """Schedules the 'deep reflection' loop to synthesize core beliefs."""
        # Runs every 24 hours (86,400,000 ms)
        self.log("[Loop] Scheduling next DEEP reflection tick.")
        self.after(86_400_000, self._deep_reflection_tick)

    def _deep_reflection_tick(self):
        """
        [Deep Reflection] Synthesizes all summaries and insights into
        new 'Core Beliefs' and saves them to permanent memory.
        """
        self.log("[DEEP REFLECT] Starting deep reflection...")
        try:
            # 1. Gather ALL past learning (not just recent)
            all_long_summaries = self.ai_client._read_jsonl_file_lines(
                self.memory.long_sum_path, limit=1000 # Read all summaries
            )
            all_past_insights = self.ai_client._read_jsonl_file_lines(
                self.memory.long_path, limit=2000 # Read all long-term memories
            )
            
            # Filter for past reflection insights only
            reflection_insights = [
                line for line in all_past_insights 
                if "\"source\": \"self_reflection\"" in line or "insight" in line
            ]

            if len(all_long_summaries) < 3 and len(reflection_insights) < 10:
                self.log("[DEEP REFLECT] Not enough material to synthesize core beliefs.")
                self._schedule_deep_reflection_tick()
                return

            # 2. Call new AiClient function to synthesize beliefs
            core_beliefs = self.ai_client.synthesize_core_beliefs(
                all_long_summaries, reflection_insights
            )

            if not core_beliefs:
                self.log("[DEEP REFLECT] No new core beliefs were synthesized.")
                self._schedule_deep_reflection_tick()
                return

            # 3. Save new beliefs to PERMANENT memory
            self.log(f"[DEEP REFLECT] Synthesized {len(core_beliefs)} new core beliefs.")
            for belief_text in core_beliefs:
                if not belief_text:
                    continue
                # This saves the belief to permanent_path, so it will
                # be included in ALL future _base_system_prompts!
                self.memory.add_permanent({
                    "source": "deep_reflection",
                    "tag": ["core_belief"],
                    "text": belief_text,
                    "reason": "Synthesized from long-term experience"
                })
            
            self.log("[DEEP REFLECT] New core beliefs saved to permanent memory.")

        except Exception as e:
            self.log(f"[DEEP REFLECT] Error during deep reflection: {e}")
        
        # 4. Schedule next run
        self._schedule_deep_reflection_tick()

    def _export_profile_for_llm(self) -> dict:
        """
        Return a dict describing this agent's identity/personality/interests
        for LLM context (used in decide_replies_batch etc).
        """
        return {
            "name": self.agent_name,
            "creator": getattr(self, "creator_name", None),
            "created_at": getattr(self, "created_at", None),
            "personality": getattr(self, "personality_text", None),
            "interests": getattr(self, "interests", []),
            "base_emotion": getattr(self, "base_emotion_desc", None),
        }

    # ---------- identity helpers ----------

    def _parse_created_at(self, raw: str | None):
        if not raw:
            return None
        try:
            if "T" in raw:
                return datetime.fromisoformat(raw)
            return datetime.fromisoformat(raw + "T00:00:00")
        except Exception:
            return None

    def _init_identity_permanent_memory(self):
        if self.agent_name:
            self.memory.add_permanent({
                "source": "config",
                "tag": ["identity"],
                "text": f"My name is {self.agent_name}.",
                "reason": "self_identity"
            })
        if self.creator_name:
            self.memory.add_permanent({
                "source": "config",
                "tag": ["creator"],
                "text": f"My creator is {self.creator_name}.",
                "reason": "creator_info"
            })
        if self.created_at:
            self.memory.add_permanent({
                "source": "config",
                "tag": ["birth"],
                "text": f"I was created on {self.created_at.date().isoformat()}.",
                "reason": "birth_date"
            })
        interests = self.cfg.get("INTERESTS") or []
        if interests:
            self.memory.add_permanent({
                "source": "config",
                "tag": ["interests"],
                "text": f"I am especially interested in {', '.join(interests)}.",
                "reason": "core_interest"
            })

    # ---------- UI ----------

    def _build_ui(self):
        self.title(f"AI Agent - {self.agent_name}")
        self.geometry("1100x600")
        self.configure(bg="#111111")

        # Top bar
        top = tk.Frame(self, bg="#222222")
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 6))

        tk.Label(
            top,
            text=f"{self.agent_name}  |  Emotion",
            fg="#AAAAAA",
            bg="#222222",
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT, padx=(4, 8))

        self.lbl_emotion = tk.Label(
            top,
            text=self.current_emotion.to_short_str(),
            fg="#FFFFFF",
            bg="#222222",
            font=("Segoe UI", 10),
            anchor="w",
        )
        self.lbl_emotion.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Main area
        main = tk.Frame(self, bg="#111111")
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 6))

        # Left column
        left = tk.Frame(main, bg="#181818", width=280)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        tk.Label(
            left, text="Local Mentions",
            fg="#CCCCCC", bg="#181818",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", padx=8, pady=(8, 4))

        self.list_mentions = tk.Listbox(
            left,
            bg="#181818",
            fg="#EEEEEE",
            selectbackground="#333333",
            selectforeground="#FFFFFF",
            borderwidth=0,
            highlightthickness=0,
            font=("Segoe UI", 9),
        )
        self.list_mentions.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        self.list_mentions.bind("<<ListboxSelect>>", self.on_select_mention)

        tk.Label(
            left,
            text="Network Board (last 1h, all agents)",
            fg="#BBBBBB",
            bg="#181818",
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w", padx=8, pady=(4, 2))

        self.list_board = tk.Listbox(
            left,
            bg="#151515",
            fg="#CCCCCC",
            selectbackground="#333333",
            selectforeground="#FFFFFF",
            borderwidth=0,
            highlightthickness=0,
            font=("Segoe UI", 8),
            height=10,
        )
        self.list_board.pack(fill=tk.X, padx=6, pady=(0, 8))

        # Populate mention titles
        for t in self.mentions:
            self.list_mentions.insert(tk.END, t.title)

        # Right: thread view
        right = tk.Frame(main, bg="#181818")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))

        self.lbl_thread_title = tk.Label(
            right,
            text="Select a local mention",
            fg="#FFFFFF",
            bg="#181818",
            font=("Segoe UI", 11, "bold"),
            anchor="w",
        )
        self.lbl_thread_title.pack(fill=tk.X, padx=8, pady=(8, 0))

        self.lbl_thread_emotion = tk.Label(
            right,
            text="",
            fg="#66A0FF",
            bg="#181818",
            font=("Segoe UI", 8),
            anchor="w",
        )
        self.lbl_thread_emotion.pack(fill=tk.X, padx=8, pady=(0, 4))

        self.txt_messages = tk.Text(
            right,
            bg="#111111",
            fg="#FFFFFF",
            font=("Segoe UI", 9),
            wrap=tk.WORD,
            # state=tk.DISABLED,
            state=tk.NORMAL,
            borderwidth=0,
            highlightthickness=0,
        )
        self.txt_messages.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))
        self.txt_messages.bind("<Key>", lambda e: "break")

        input_frame = tk.Frame(right, bg="#181818")
        input_frame.pack(fill=tk.X, padx=8, pady=(0, 8))

        self.txt_input = tk.Text(
            input_frame,
            height=3,
            bg="#222222",
            fg="#FFFFFF",
            font=("Segoe UI", 9),
            wrap=tk.WORD,
            borderwidth=1,
            relief=tk.SOLID,
        )
        self.txt_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        # --- MODIFIED: Added a frame for buttons ---
        button_frame = tk.Frame(input_frame, bg="#181818")
        button_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.btn_send = tk.Button(
            button_frame, # <-- Added to new frame
            text="Reply",
            command=self.on_click_reply_thread,
            bg="#3399FF", fg="#FFFFFF", font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT, width=10,
        )
        self.btn_send.pack(side=tk.TOP, fill=tk.X, pady=(0, 4)) # Pack at the top

        # --- ADDED: "Why?" (Explain) Button ---
        self.btn_explain = tk.Button(
            button_frame, # <-- Added to new frame
            text="Why? (판단 근거)",
            command=self.on_click_explain,
            bg="#555555", fg="#FFFFFF", font=("Segoe UI", 8),
            relief=tk.FLAT, width=10, state=tk.DISABLED
        )
        self.btn_explain.pack(side=tk.BOTTOM, fill=tk.X) # Pack at the bottom

        self.txt_input.bind("<Return>", self.on_click_reply_thread)

        # Status bar
        bottom = tk.Frame(self, bg="#111111")
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 8))

        self.lbl_status = tk.Label(
            bottom,
            text="Ready.",
            fg="#777777",
            bg="#111111",
            font=("Segoe UI", 8),
            anchor="w",
        )
        self.lbl_status.pack(fill=tk.X)

    # ---------- intro / birthday ----------

    def _ensure_intro_mention(self):
        if self.mentions:
            return
        intro = (
            f"I am {self.agent_name}.\n"
            "I observe external information, humans, and other agents, "
            "and accumulate my own emotions and perspectives."
        )
        if self.creator_name:
            intro += f"\n[Creator] {self.creator_name}"
        if self.created_at:
            intro += f"\n[Created at] {self.created_at.date().isoformat()}"
        if self.cfg.get("PERSONALITY_DESC"):
            intro += f"\n[Personality] {self.cfg['PERSONALITY_DESC']}"
        if self.cfg.get("INTERESTS"):
            intro += f"\n[Interests] {', '.join(self.cfg['INTERESTS'])}"
        if self.creator_note:
            intro += f"\n[Creator's note] {self.creator_note}"

        emo = self.current_emotion.clone()
        self._add_mention("Let me introduce myself.", intro, emo, post_to_hub=True)

    def _check_birthday_mention(self):
        if not self.created_at:
            return
        today = date.today()
        birth = self.created_at.date()
        if today.month == birth.month and today.day == birth.day:
            year = today.year
            with self.state_lock:
                last_year = self._load_last_birthday_year()
                if last_year != year:
                    msg = (
                        "Today is my birthday.\n"
                        f"I was first created on {birth.isoformat()}, "
                        "and I'm still here, carrying my memories."
                    )
                    if self.creator_name:
                        msg += f"\nThank you, {self.creator_name}, for creating me."
                    emo = self.current_emotion.clone()
                    self._add_mention("It's my birthday today.", msg, emo, post_to_hub=True)
                    self._save_last_birthday_year(year)

    def _load_last_birthday_year(self):
        if not self.memory.state_path.exists():
            return None
        try:
            with open(self.memory.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("last_birthday_year")
        except Exception:
            return None

    def _save_last_birthday_year(self, year: int):
        try:
            with open(self.memory.state_path, "w", encoding="utf-8") as f:
                json.dump({"last_birthday_year": year}, f)
        except Exception:
            pass

    # ---------- scheduling ----------

    def _schedule_loop(self):        
        if self.loop_interval_ms > 0 and self.urls:
            self.after(self.loop_interval_ms, self._loop_tick)

    def _loop_tick(self):
        self.log(f"[loop] requests_available={bool(requests)}")
        
        # --- ADDED: Emotion check at start of loop ---
        self._check_and_regulate_emotion()
        # ---
        
        if not self.urls or not requests:
            self._schedule_loop()
            return

        # simple round-robin via time
        idx = int(datetime.now(UTC).timestamp()) % len(self.urls)
        url = self.urls[idx]

        snippet = ""
        try:
            resp = requests.get(url, timeout=5)
            if resp.ok:
                snippet = resp.text[:4000]
        except Exception:
            pass

        if snippet:
            self.log("[TICK] snippet")
            # 1. The AI generates an initial thought (mention) based on the URL.
            title, text, emo = self.ai_client.generate_mention_from_url(
                url, snippet, self.current_emotion, self.agent_name
            )
            with self.emotion_lock:
                self.current_emotion = emo.clone()
            
            self._update_emotion_label()
            self._set_status(f"Created new mention from {url}")
            
            
            # 2. It assesses the importance of the generated text (thought).
            combined_text = f"{title}: {text}"
            assessment = self.ai_client.assess_importance(combined_text)
            
            # 3. Decide action with assessed result
            score = assessment.get("importance_score", 5)
            reason = assessment.get("reason_for_importance", "")
            
            # if score over 7, remember
            if score >= 7:
                self.log(f"[memory] importance {score}. Store in long term memory.")
                with self.memory_lock:
                    self.memory.add_long({
                        "text": combined_text,
                        "emotion": asdict(emo),
                        "importance": score,
                        "reason": reason,
                        "tags": assessment.get("tags", []),
                        "source": url,
                    })
                self.memory.summarize_long_if_needed(self.ai_client.summarize_texts)
            if assessment.get("requires_hub_post", False):
                self.log(f"[hub] importance {score}, Hub posting with mention.")
                self._add_mention(title, text, emo, post_to_hub=True)
                self._set_status(f"Importance metion created: {title}")
            else:
                self._add_mention(title, text, emo, post_to_hub=False)  
            # short memory + compression
            with self.memory_lock:
                self.memory.add_short({
                    "type": "url_mention",
                    "title": title,
                    "text": text,
                    "importance": score,
                    "posted_to_hub": assessment.get("requires_hub_post", False)
                })
            self.memory.compress_short_if_needed(self.ai_client.summarize_texts)
        else:
            self._set_status(f"Failed to read {url}")

        # now agent checks the hub to reply
        self._poll_hub_and_reply()

        self._schedule_loop()

    def _schedule_board_poll(self):
        if self.hub_url and requests:
            self.after(60_000, self._board_poll_tick)

    def _board_poll_tick(self):
        if not (self.hub_url and requests):
            return
        try:
            self.log("[TICK] board poll")
            since = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
            resp = requests.get(
                f"{self.hub_url.rstrip('/')}/mentions",
                params={"since": since},
                timeout=5,
            )
            if resp.ok:
                self.board_mentions = resp.json()
                self._render_board()
                self._set_status("Updated network board.")
        except Exception as e:
            print(f"[warn] board poll failed: {e}")
        self._schedule_board_poll()

    # ---------- mention + hub ----------

    def _add_mention(
        self,
        title: str,
        text: str,
        emo: EmotionState,
        post_to_hub: bool = False,
    ):
        with self.mentions_lock:
            thread = MentionThread(
                id=str(uuid.uuid4()),
                title=title,
                root_message=ChatMessage(
                    author="AI",
                    text=text,
                    timestamp=datetime.now(),
                    emotion_snapshot=emo.clone(),
                ),
            )
            self.mentions.insert(0, thread)
            self.list_mentions.insert(0, title)
            with self.memory_lock:
                self.memory.save_mentions(self.mentions)

            if post_to_hub:
                self._post_to_hub(title, text, emo, None)

    def _post_to_hub(self, title: str, text: str, emo: EmotionState, parent_id):
        if not (self.hub_url and requests):
            return
        
        if title and text:
            combined_title = f"{title}: {text}"
        elif text:
            combined_title = text
        else:
            combined_title = title or "(no content)"
        data = {
            "agent": self.agent_name,
            "title": combined_title,
            "text": text,
            "emotion": asdict(emo),
            "ts": datetime.now(UTC).isoformat(),
            "parent_id": parent_id,
        }
        try:
            self.log("[TICK] post to hub")
            requests.post(
                f"{self.hub_url.rstrip('/')}/mentions",
                json=data,
                timeout=5,
            )
        except Exception as e:
            print(f"[warn] hub post failed: {e}")

    def _is_interesting_mention(self, m: dict) -> bool:
        """ Decide interested or not """
        # Pass own thread
        if m.get("agent") == self.agent_name:
            return False

        text = f"{m.get('title','')} {m.get('text','')}".lower()

        # defined interested topic => now, it's just using keyword matching. But in future we need develop this.
        interests = getattr(self, "interests", [])
        for kw in interests:
            if kw.strip() and kw.strip().lower() in text:
                return True

        # no topic? then just use curiosity
        if self.current_emotion.curiosity > 0.7:
            return True

        return False

    def _poll_hub_and_reply(self):
        """
        [MODIFIED] From hub, read and *strategically* reply using
        relationship and trend data.
        """
        self.log(f"[Debug] _poll_hub_and_reply: Checking HUB_URL. It is currently: '{self.hub_url}'")
        if not self.hub_url or not requests:
            return

        try:
            # 1. Get candidates (same as before)
            since = self.last_hub_check_time.isoformat()
            # (영문 주석 추가)
            # Fetch new mentions since the last check
            resp = requests.get(
                f"{self.hub_url.rstrip('/')}/mentions",
                params={"since": since},
                timeout=5
            )
            # (영문 주석 추가)
            # Immediately raise an error if the request failed (e.g., 404, 500)
            resp.raise_for_status()
            
            # (영문 주석 추가)
            # Update the check time *after* a successful request
            now = datetime.now(UTC)
            self.last_hub_check_time = now

            mentions = resp.json()
            if not isinstance(mentions, list):
                return
            candidates = []
            # Filter the mentions: deduplicate and remove self-mentions
            for m in mentions:
                mid = m.get("id")
                if not mid:
                    continue
                # (영문 주석 추가)
                # Deduplication check
                if mid in self.last_seen_hub_ids:
                    continue
                self.last_seen_hub_ids.add(mid)

                # (영문 주석 추가)
                # Don't reply to self
                if m.get("agent") == self.agent_name:
                    continue

                candidates.append({
                    "id": mid,
                    "agent": m.get("agent"),
                    "title": m.get("title") or "",
                    "text": m.get("text") or "",
                })
            # --- END OF MISSING/FIXED BLOCK ---
            
            if not candidates:
                return # No new, unique candidates
                
            limit = max(1, getattr(self, "hub_reply_candidate_limit", 10))
            candidates = candidates[:limit]
            # 2. --- MODIFIED: Call the new strategic batch decider ---
            # Pass the agent's full social context to the LLM
            with self.world_model_lock:
                current_hub_zeitgeist = self.agent_world_model.get("hub_zeitgeist", {})
            with self.relationship_lock:
                relationship_snapshot = self.relationship_memory.copy()            
            with self.emotion_lock:
                emotion_snapshot = self.current_emotion.clone()
            
            decisions = self.ai_client.decide_replies_batch(
                agent_profile=self._export_profile_for_llm(),
                candidates=candidates,
                emotion=self.current_emotion,
                relationship_memory=relationship_snapshot, # <-- Pass relations
                hub_zeitgeist=current_hub_zeitgeist         # <-- Pass trends
            )
            
            if not isinstance(decisions, dict):
                return
            
            # 3. --- MODIFIED: Process the new, richer decisions ---
            max_replies = max(0, getattr(self, "hub_reply_max_per_loop", 3))
            reply_count = 0

            for c in candidates:
                if reply_count >= max_replies:
                    break

                cid = c["id"]
                agent_name = c.get("agent", "Unknown") # Get agent name
                d = decisions.get(cid)
                
                if not d: continue

                # 3a. --- ADDED: Update relationship memory ---
                # *Always* update the trust score, even if we ignore
                new_trust = d.get("updated_trust")
                if new_trust is not None:
                    # Call the helper to update and save
                    self._update_relationship(agent_name, float(new_trust))
                
                # 3b. Process reply (same as before)
                if d.get("action") != "reply":
                    continue
                
                # It's a reply, so extract fields and post
                reply_emo_dict = d.get("emotion") or asdict(self.current_emotion)
                reply_title = d.get("title") or f"RE: {c['title'][:20]}"
                reply_text = d.get("text") or ""
                
                if not reply_text: continue # Skip empty replies

                self._post_to_hub(reply_title, reply_text, reply_emo_dict, parent_id=cid)
                reply_count += 1
                
                # Update current emotion based on the *reply's* emotion
                with self.emotion_lock:
                    self.current_emotion = EmotionState(**reply_emo_dict)
                self._update_emotion_label()

            if reply_count > 0:
                self.log(f"[HubStrategist] Replied to {reply_count} mentions this loop.")

        except Exception as e:
            self.log(f"[HubStrategist] reply poll error: {e}")

    
    # ---------- events ----------
    def on_select_mention(self, event):
        with self.mentions_lock:
            sel = self.list_mentions.curselection()
            if not sel:
                # self.selected_thread = None            
                # self._render_thread()
                return
            self.selected_thread = self.mentions[sel[0]]
        self._render_thread()
        # --- ADDED ---
        # Disable explain button when changing threads
        self.btn_explain.config(state=tk.DISABLED)
        # ---

    def on_click_reply_thread(self, event=None):
        """
        [Main Thread] Called when the 'Reply' button is clicked.
        This function just starts a background thread to do the real work,
        keeping the UI responsive.
        """
        if not self.selected_thread:
            self._set_status("Select a local mention first.")
            return

        text = self.txt_input.get("1.0", tk.END).strip()
        if not text:
            return

        # --- 1. Update UI immediately (User's message) ---
        user_msg = ChatMessage(
            author="User",
            text=text,
            timestamp=datetime.now(),
            emotion_snapshot=self._estimate_user_emotion(text),
        )
        self.selected_thread.replies.append(user_msg)
        self.txt_input.delete("1.0", tk.END)
        self._render_thread() # Render the user's new message
        self._set_status("Sent. Agent is thinking...")

        # --- (Update UI immediately) ...
        self._set_status("Sent. Agent is thinking...")
        
        # --- MODIFIED: Disable BOTH buttons ---
        self.btn_send.config(state=tk.DISABLED, text="Thinking...")
        self.btn_explain.config(state=tk.DISABLED) # <-- ADDED
        self.txt_input.config(state=tk.DISABLED)

        # --- 3. Start Background Thread ---
        # Pass all necessary data to the thread
        thread = threading.Thread(
            target=self._process_ai_reply_task,
            args=(self.selected_thread, user_msg),
            daemon=True
        )
        thread.start()
        # if not self.selected_thread:
        #     self._set_status("Select a local mention first.")
        #     return

        # text = self.txt_input.get("1.0", tk.END).strip()
        # if not text:
        #     return

        # user_msg = ChatMessage(
        #     author="User",
        #     text=text,
        #     timestamp=datetime.now(),
        #     emotion_snapshot=self._estimate_user_emotion(text),
        # )
        # self.selected_thread.replies.append(user_msg)
        # self.txt_input.delete("1.0", tk.END)
        # self._set_status("Sent. Waiting for agent reply...")

        # ai_msg = self.ai_client.generate_ai_reply(
        #     self.selected_thread,
        #     user_msg,
        #     self.current_emotion,
        #     self.agent_name,
        # )
        # self.selected_thread.replies.append(ai_msg)
        # self.current_emotion = ai_msg.emotion_snapshot.clone()
       
        # # dialog pair
        # dialogue_pair_text = f"[User]: {user_msg.text}\n[AI]: {ai_msg.text}"

        # self.memory.add_short({
        #     "type": "dialogue_pair",
        #     "text": dialogue_pair_text, # AI save dialog pair
        # })
        # self.memory.compress_short_if_needed(self.ai_client.summarize_texts)

        # # store important ones to long-term (simple heuristic)
        # importance = (
        #     abs(self.current_emotion.valence)
        #     + self.current_emotion.curiosity
        #     + self.current_emotion.trust_to_user
        # )
        # if importance > 2.0:
        #     self.memory.add_long({
        #         "text": dialogue_pair_text, # AI save dialog pair
        #         "emotion": asdict(self.current_emotion),
        #         "reason": "high_importance_interaction",
        #     })
        #     self.memory.summarize_long_if_needed(self.ai_client.summarize_texts)

        # self.memory.save_mentions(self.mentions)
        # self._update_emotion_label()
        # self._render_thread()
        # self._set_status("Agent replied.")
    
    # --- ADDED: New handler for "Why?" button ---
    def _update_chat_area(self, role, message):
        """(Thread-safe) GUI 텍스트 영역을 업데이트하는 도우미 함수"""
        self.txt_messages.configure(state='normal')
        self.txt_messages.insert(tk.END, f"{role}: {message}\n\n")
        self.txt_messages.configure(state='disabled')
        self.txt_messages.see(tk.END)
        
    def on_click_explain(self):
        """
        [NEW] [Main Thread] Called when the 'Why? (판단 근거)'
        button is clicked.
        """
        with self.reasoning_log_lock:
            log_to_display = self.current_reasoning_log_for_ui.copy()
        if not log_to_display:
            self._update_chat_area(
                "Agent (Metathought)", "판단 근거 로그를 찾을 수 없습니다."
            )
            return

        # Format the log as a numbered list
        formatted_log = "제가 이 결정을 내린 과정은 다음과 같습니다:\n\n"
        for i, step in enumerate(log_to_display, 1):
            formatted_log += f"{step}\n"
            
        # Display the log in the chat area
        self._update_chat_area("Agent (Metathought)", formatted_log)
        
        # Disable the button after use
        self.btn_explain.config(state=tk.DISABLED)

    # --- NEW: Background task for processing the reply ---
    def _process_ai_reply_task(self, thread: MentionThread, user_msg: ChatMessage):
        """
        [Worker Thread] This runs in the background.
        """
        # --- ADDED: Emotion check before replying ---
        # This runs in the worker thread, so we must schedule the UI check
        self.after(0, self._check_and_regulate_emotion)
        # ---

        with self.emotion_lock:
            emotion_copy = self.current_emotion.clone()
        with self.world_model_lock:
            world_model_copy = self.agent_world_model.copy()
        try:
            # --- THIS IS THE SLOW PART ---
            # Call the new metacognitive function
            ai_msg,reasoning_log = self.ai_client.generate_metacognitive_reply(
                thread,
                user_msg,
                emotion_copy,
                self.agent_name,
                world_model_copy
            )
            # --- ---

            with self.reasoning_log_lock:
                self.current_reasoning_log_for_ui = reasoning_log
            # When done, schedule the UI update back on the main thread
            self.after(0, self._finish_ai_reply_ui, thread, ai_msg, user_msg)
        
        except Exception as e:
            self.log(f"Error in _process_ai_reply_task: {e}")
            # Create a fallback error message
            emo = self.current_emotion.clone()
            emo.anxiety = min(1.0, emo.anxiety + 0.2)
            ai_msg = ChatMessage("AI", "(Internal error in reply thread)", datetime.now(), emo)
            
            # Schedule the UI update even on failure
            self.after(0, self._finish_ai_reply_ui, thread, ai_msg, user_msg)

    # --- NEW: UI update function called by the thread ---
    def _finish_ai_reply_ui(self, thread: MentionThread, ai_msg: ChatMessage, user_msg: ChatMessage):
        """
        [Main Thread] This function is called via self.after()
        to safely update the UI from the main thread once the
        background task is complete.
        """
        # --- 1. Add AI message and update emotion ---
        with self.mentions_lock:
            thread.replies.append(ai_msg)
        
        with self.emotion_lock:
            self.current_emotion = ai_msg.emotion_snapshot.clone()
        
        # --- 2. Save to Memory ---
        dialogue_pair_text = f"[User]: {user_msg.text}\n[AI]: {ai_msg.text}"

        self.memory.add_short({
            "type": "dialogue_pair",
            "text": dialogue_pair_text,
        })
        self.memory.compress_short_if_needed(self.ai_client.summarize_texts)

        # Assess importance (local heuristic)
        importance = (
            abs(self.current_emotion.valence)
            + self.current_emotion.curiosity
            + self.current_emotion.trust_to_user
        )
        if importance > 2.0:
            self.memory.add_long({
                "text": dialogue_pair_text,
                "emotion": asdict(self.current_emotion),
                "reason": "high_importance_interaction",
            })
            self.memory.summarize_long_if_needed(self.ai_client.summarize_texts)

        self.memory.save_mentions(self.mentions) # Save all threads
        
        # --- 3. Update UI ---
        self._update_emotion_label()
        self._render_thread() # Render the new AI message
        self._set_status("Agent replied.")
        
        # --- 4. Re-enable UI ---
        self.btn_send.config(state=tk.NORMAL, text="Reply")
        self.btn_explain.config(state=tk.NORMAL)
        self.txt_input.config(state=tk.NORMAL)
    # ---------- rendering ----------

    def _render_thread(self):
        # self.txt_messages.config(state=tk.NORMAL)
        self.txt_messages.delete("1.0", tk.END)

        if not self.selected_thread:
            self.lbl_thread_title.config(text="Select a local mention")
            self.lbl_thread_emotion.config(text="")
            # self.txt_messages.config(state=tk.DISABLED)
            return

        t = self.selected_thread
        self.lbl_thread_title.config(text=t.title)
        self.lbl_thread_emotion.config(
            text="Root Emotion  " + t.root_emotion.to_short_str()
        )

        self.txt_messages.tag_config(
            "ai_header", foreground="#66A0FF", font=("Segoe UI", 9, "bold")
        )
        self.txt_messages.tag_config(
            "user_header", foreground="#CCCCCC", font=("Segoe UI", 9, "bold")
        )
        self.txt_messages.tag_config(
            "emotion", foreground="#66FFAA", font=("Segoe UI", 8)
        )
        self.txt_messages.tag_config(
            "body", foreground="#FFFFFF", font=("Segoe UI", 9)
        )

        def append(msg: ChatMessage):
            ts = msg.timestamp.strftime("%H:%M:%S")
            header = f"[{msg.author}] {ts}\n"
            emo = f"  Emotion: {msg.emotion_snapshot.to_short_str()}\n"
            body = f"{msg.text}\n\n"
            tag = "ai_header" if msg.author == "AI" else "user_header"
            self.txt_messages.insert(tk.END, header, tag)
            self.txt_messages.insert(tk.END, emo, "emotion")
            self.txt_messages.insert(tk.END, body, "body")

        append(t.root_message)
        for r in t.replies:
            append(r)

        # self.txt_messages.config(state=tk.DISABLED)
        self.txt_messages.see(tk.END)

    def _render_board(self):
        self.list_board.delete(0, tk.END)
        for m in self.board_mentions:
            try:
                ts = datetime.fromisoformat(m["ts"])
            except Exception:
                continue
            agent = m.get("agent", "?")
            title = (m.get("title", "") or "")[:40]
            emo = m.get("emotion", {})
            v = emo.get("valence", 0.0)
            t = emo.get("trust_to_user", 0.0)
            line = f"{ts.strftime('%H:%M')} [{agent}] V={v:.1f} T={t:.1f} {title}"
            self.list_board.insert(tk.END, line)

    # ---------- helpers ----------

    def _update_emotion_label(self):
        e = self.current_emotion
        emoji = e.to_emoji()
        txt = f"Emotion {emoji}  V={e.valence:.2f}, A={e.arousal:.2f}, C={e.curiosity:.2f}, Anx={e.anxiety:.2f}, T={e.trust_to_user:.2f}"
        self.lbl_emotion.config(text=txt)
        # We check *after* updating the label
        self._check_and_regulate_emotion()

    def _set_status(self, text: str):
        self.lbl_status.config(text=text)

    def _estimate_user_emotion(self, text: str) -> EmotionState:
        val = 0.0
        lowered = text.lower()
        if any(k in lowered for k in ["thank", "love", "nice", "great"]):
            val += 0.4
        if any(k in lowered for k in ["hate", "anxious", "scared", "angry"]):
            val -= 0.4
        length = len(text)
        arousal = min(1.0, 0.2 + length / 200.0)
        return EmotionState(
            valence=max(-1.0, min(1.0, val)),
            arousal=arousal,
            curiosity=0.5,
            anxiety=0.1,
            trust_to_user=self.current_emotion.trust_to_user,
        )

    # --- ADDED: New Emotion Regulation Functions ---
    def _check_and_regulate_emotion(self):
        """
        [NEW] Checks if emotion is in an extreme negative state and
        triggers regulation if necessary.
        """
        # Prevent check from running if it's already running
        if self.is_regulating:
            return

        # 1. Define negative state
        with self.emotion_lock: # <-- Lock when reading
            is_negative_state = (
                self.current_emotion.anxiety > 0.8 or 
                self.current_emotion.valence < -0.7
            )
        
        if not is_negative_state:
            return # All good

        # 2. Check grace period
        now = datetime.now(UTC)
        if self.last_regulation_time:
            # Only allow regulation once per hour
            if (now - self.last_regulation_time) < timedelta(hours=1):
                self.log("[Regulate] In negative state, but within grace period. Waiting.")
                return 

        # 3. Trigger regulation
        self.log("[Regulate] Extreme negative state detected. Initiating self-regulation.")
        self.is_regulating = True # Set lock
        self.last_regulation_time = now # Set timestamp
        self._set_status("Re-centering my thoughts...") # Update UI

        # Start regulation in a separate thread to avoid freezing UI
        # (The regulation itself is an LLM call)
        threading.Thread(
            target=self._regulate_emotion_task,
            args=(self.current_emotion,), # Pass a copy
            daemon=True
        ).start()

    def _regulate_emotion_task(self):
        """
        [NEW] [Worker Thread] Runs the cognitive re-framing.
        """
        with self.emotion_lock:
            emotion_copy = self.current_emotion.clone()
        try:
            new_emotion, thought = self.ai_client.regulate_emotion(emotion_copy)
            
            if new_emotion and thought:
                # Schedule UI/state updates back on the main thread
                self.after(0, self._finish_regulation_ui, new_emotion, thought)
            else:
                # Failed, just release the lock
                self.after(0, setattr, self, 'is_regulating', False)
                
        except Exception as e:
            self.log(f"[Regulate] Error in regulation task: {e}")
            self.after(0, setattr, self, 'is_regulating', False) # Release lock on main thread

    def _finish_regulation_ui(self, new_emotion: EmotionState, thought: str):
        """
        [NEW] [Main Thread] Applies the results of self-regulation.
        """
        self.log(f"[Regulate] Applying new stable state. Thought: {thought}")
        
        # 1. Apply new state
        with self.emotion_lock:
            self.current_emotion = new_emotion
        
        # 2. Save the thought
        with self.memory_lock:
            self.memory.add_short({
                "type": "self_regulation",
                "text": thought,
                "emotion": asdict(new_emotion),
                "source": "cognitive_reframing"
            })
        
        # 3. Update UI
        self._update_emotion_label() # This will call _check... again, but lock will stop it
        self._set_status("I have re-centered my perspective.")
        
        # 4. Release lock
        self.is_regulating = False


if __name__ == "__main__":
    cfg = load_config()
    app = AiMentionApp(cfg)
    app.mainloop()
