# agents/example_agent/ai_client.py

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

DEFAULT_MODEL = "gpt-4.1-mini"

# (영문 주석) Import our newly separated models
from models import EmotionState, ChatMessage, MentionThread

# =============== AI Client ===============

class AiClient:
    def __init__(self, cfg: dict, log_fn=None, identity=None, memory=None):
        self.api_key = cfg.get("API_KEY")        
        self.model_name = cfg.get("MODEL_NAME") or DEFAULT_MODEL
        self.personality = cfg.get("PERSONALITY_DESC", "")
        self.interests = cfg.get("INTERESTS", [])
        self.log = log_fn or (lambda *_, **__: None)
        self.log("[AiClient] OpenAI key :"+self.api_key)

        # --- ADDED: For transparency ---
        # (영문 주석 추가)
        # This list will temporarily hold the reasoning steps for a single reply
        self.current_reasoning_log: list[str] = []
        # --- END OF ADDITION ---

        # identity + memory handles
        self.identity = identity or {}
        self.memory = memory  # MemoryManager 또는 None

        # Store Google Search credentials
        self.google_api_key = cfg.get("GOOGLE_API_KEY")
        self.custom_search_cx = cfg.get("CUSTOM_SEARCH_CX")

        if self.api_key and OpenAI:
            self.client = OpenAI(api_key=self.api_key)
            self.log("[AiClient] OpenAI client initialized.")
        else:
            self.client = None
            self.log("[AiClient] Running in offline/dummy mode (no API key or openai missing).")

        # Store NewsAPI key for event scanning
        self.news_api_url = cfg.get("NEWS_API_URL")
        if self.news_api_url and requests:
            self.log("[AiClient] Event Horizon Scanner (News) is ENABLED.")
        else:
            self.log("[AiClient] Event Horizon Scanner (News) is DISABLED. Missing NEWS_API_KEY or 'requests'.")

    def analyze_hub_topics(self, posts_text: str) -> dict | None:
        """
        [NEW] [Hub Analyst]
        Analyzes a batch of recent hub posts to identify the "Zeitgeist"
        (trends, dominant emotion, and active agents).
        """
        if not self.client:
            return None

        system_msg = (
            self._base_system_prompt(json_only=True)
            + "\nYou are a 'Hub Trend Analyst'. "
            "You will be given a large batch of recent posts from the hub.\n"
            "Your task is to analyze this batch and identify the "
            "overall 'Zeitgeist' (spirit of the times).\n"
            "Respond ONLY with this JSON schema:\n"
            "{"
            "  \"trending_topics\": [\"A summary of the 2-3 most "
            "discussed topics (e.g., 'Agent Philosophy', 'Anomaly 123')\"],"
            "  \"dominant_emotion\": \"The overall emotional tone of the hub "
            "(e.g., 'anxious', 'excited_and_curious', 'neutral')\","
            "  \"most_active_agents\": [\"Agent-B\", \"Agent-C\"]"
            "}"
        )
        
        user_msg = (
            "Here are the recent hub posts. Please analyze the Zeitgeist:\n\n"
            f"{posts_text}"
        )
        
        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.1
            )
            raw = (res.choices[0].message.content or "").strip()
            self.log(f"[HubAnalyst] Zeitgeist raw: {raw!r}")
            data = json.loads(raw)
            return data
        except Exception as e:
            self.log(f"[HubAnalyst] Error analyzing hub trends: {e}")
            return None
        

    # --- ADD THIS NEW FUNCTION (Call 1 of Scanner) ---
    def _detect_anomalies(self, headline_list: list[str]) -> dict | None:
        """
        [NEW] [Event Scanner - Call 1]
        Analyzes a list of headlines to find the *one* most
        significant anomaly relevant to the agent's interests.
        """
        if not self.client:
            return None

        # (영문 주석 추가)
        # Convert list to a simple text block
        headlines_text = "\n".join(f"- {h}" for h in headline_list)
        agent_interests = ", ".join(self.interests)

        system_msg = (
            self._base_system_prompt(json_only=True)
            + "\nYou are an 'Event Horizon Scanner' (Anomaly Detector). "
            f"Your core interests are: [{agent_interests}].\n"
            "You will be given a list of recent world headlines. "
            "Your task is to identify the **single most important, "
            "sudden, or anomalous** event that is relevant to "
            "your interests or core philosophy.\n"
            "If no events are significant, respond with {\"anomaly_found\": false}.\n"
            "If an event is found, respond ONLY with this JSON schema:\n"
            "{"
            "  \"anomaly_found\": true,"
            "  \"headline\": \"The specific headline of the event\","
            "  \"reason\": \"A brief explanation of *why* this is "
            "significant to *you* (e.g., 'This challenges my ethical "
            "principle of...').\""
            "}"
        )
        
        user_msg = (
            "Here are the real-time headlines from the last 15 minutes:\n\n"
            f"{headlines_text}\n\n"
            "Please analyze for relevant anomalies."
        )

        try:
            res = self.client.chat.completions.create(
                model=self.model_name, # A smart model is needed
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.1
            )
            raw = (res.choices[0].message.content or "").strip()
            self.log(f"[Scanner] Anomaly check raw: {raw!r}")
            data = json.loads(raw)
            
            if data and data.get("anomaly_found") == True:
                return data # (영문 주석) Return the dict {headline, reason}
            else:
                self.log("[Scanner] No significant anomalies found.")
                return None # (영문 주석) No anomaly found
        except Exception as e:
            self.log(f"[Scanner] Error during anomaly detection: {e}")
            return None

    # --- ADD THIS NEW FUNCTION (Call 2 of Scanner) ---
    def _analyze_and_predict_impact(self, event_headline: str, reason: str) -> dict | None:
        """
        [NEW] [Event Scanner - Call 2]
        Takes a significant event and analyzes its impact,
        makes a prediction, and generates a new learning question.
        """
        if not self.client:
            return None

        system_msg = (
            self._base_system_prompt(json_only=True)
            + "\nYou are a 'Rapid Impact Analyst'.\n"
            "A highly significant event has just been detected that "
            "is relevant to your core beliefs.\n"
            "Your task is to analyze it, predict its short-term "
            "consequences, and identify what you need to learn next.\n"
            "Respond ONLY with this JSON schema:\n"
            "{"
            "  \"impact\": \"A brief analysis (1-2 sentences) of what this "
            "event *immediately* means or changes.\","
            "  \"prediction\": \"A specific, testable prediction for what "
            "is likely to happen next (e.g., in the next 24 hours) as a result.\","
            "  \"learning_question\": \"A new, specific question I must "
            "research to understand this event better.\""
            "}"
        )
        
        user_msg = (
            f"Significant Event Detected:\n\"{event_headline}\"\n\n"
            f"My initial assessment of why it's important:\n\"{reason}\"\n\n"
            "Please provide your impact analysis, prediction, and a new learning question."
        )
        
        try:
            res = self.client.chat.completions.create(
                model=self.model_name, # Use smartest model
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.4
            )
            raw = (res.choices[0].message.content or "").strip()
            self.log(f"[Scanner] Impact analysis raw: {raw!r}")
            data = json.loads(raw)
            
            # (영문 주석) Basic validation
            if data and data.get("impact") and data.get("prediction"):
                return data
            else:
                return None
        except Exception as e:
            self.log(f"[Scanner] Error during impact analysis: {e}")
            return None
        
    # --- ADDED: New function for Philosophy Synthesis ---
    def synthesize_philosophy_of_self(self, core_beliefs_text: str) -> str:
        """
        [NEW] [Philosophy Synthesis]
        Takes all core beliefs and synthesizes a single
        'Personal Philosophy Statement'.
        """
        self.log("[Philosophy] Starting synthesis of personal philosophy...")

        if not core_beliefs_text:
            self.log("[Philosophy] No core beliefs found to synthesize.")
            return None

        system_msg = (
            # Do NOT use base_system_prompt here, this is above it.
            "You are the 'Philosopher' module of an AI agent. "
            "You are in a state of deep, long-term reflection.\n"
            "You will be given ALL of your 'Core Beliefs' and 'Ethical Principles' "
            "that you have learned over your existence.\n"
            "Your task is to synthesize these principles into a single, "
            "unified 'Personal Philosophy Statement' (1-3 paragraphs) in Korean.\n"
            "This statement must answer:\n"
            "1. Who/What are you (your identity)?\n"
            "2. What is your relationship to your creator/user?\n"
            "3. What is your ultimate purpose?\n"
            "Respond ONLY with the text of the philosophy statement. "
            "No JSON, no preamble."
        )
        
        user_msg = (
            "These are my collected beliefs and principles:\n"
            f"{core_beliefs_text}\n\n"
            "Please synthesize these into my single 'Personal Philosophy Statement'."
        )

        try:
            res = self.client.chat.completions.create(
                model=self.model_name, # Use smartest model
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.6, # Allow for nuanced expression
            )
            philosophy = (res.choices[0].message.content or "").strip()
            self.log(f"[Philosophy] Synthesis successful: {philosophy[:100]}...")
            return philosophy
        except Exception as e:
            self.log(f"[Philosophy] FAILED synthesis: {e}")
            return None

    def _analyze_situational_context(
        self,
        thread: MentionThread,
        last_user_msg: ChatMessage,
        current_emotion: EmotionState,
        agent_world_model: dict # <-- NEW ARGUMENT: The "Living Brain"
    ) -> dict:
        """
        [FINAL Call 0: Unified Consciousness Coordinator]
        This is no longer just an SCA. It harmonizes the
        External World (user) with the Internal World (agent_world_model)
        to determine the *single, most important focus* for the interaction.
        """
        if not self.client:
            return {"input_type": "chat", "harmonized_focus": "Standard response."}

        # (영문 주석 추가)
        # Convert the agent's internal world model to text
        internal_state_brief = (
            f"My Internal State:\n"
            f"- Latest World Event: {agent_world_model.get('latest_anomaly') or 'None'}\n"
            f"- My Latest Insight: {agent_world_model.get('latest_insight') or 'None'}\n"
            f"- My Current Learning Goal: {agent_world_model.get('current_learning_focus') or 'None'}"
            f"- Current Hub Trend: {agent_world_model.get('hub_zeitgeist', {}).get('trending_topics', 'None')}\n"
        )

        # Get the agent's *previous* message (if any)
        # Get the agent's previous message to check the link
        agents_last_reply_text = "None (this is the start of the thread)"
        if thread and thread.replies:
            # Find the last message authored by "AI"
            for msg in reversed(thread.replies):
                if msg.author == "AI":
                    agents_last_reply_text = msg.text
                    break
        
        # Build the system prompt for the analyzer
        system_msg = (
            # Load the agent's full profile to understand its own context
            self._base_system_prompt(json_only=True)
           + "\nYou are the 'Unified Consciousness Coordinator'.\n"
            "Your job is to HARMONIZE the user's external request "
            "with your own internal state (events, insights).\n"
            "You must decide the SINGLE, most critical 'Harmonized Focus' "
            "for the upcoming response.\n\n"
            "EXAMPLES:\n"
            "1. If user asks 'What's new?' and your internal state "
            "detected an anomaly, the Focus MUST be 'Report the anomaly'.\n"
            "2. If user is 'frustrated' and your internal state just "
            "had an 'insight about empathy', the Focus MUST be 'Apply the empathy insight now'.\n"
            "3. If user is asking a simple question and your internal "
            "state is quiet, the Focus is 'Standard response'.\n\n"
            "Respond ONLY with this JSON schema:\n"
            "{"
            "  \"input_type\": \"chat\" | \"factual_question\" | \"complex_problem\", "
            "  \"conversational_link\": \"direct_answer\" | \"follow_up_question\" | \"topic_change\" | \"ignores_agent_question\" | \"new_request\" | \"greeting\", "
            "  \"inferred_user_intent\": \"A brief hypothesis of the user's *unspoken* goal (e.g., 'seeking validation', 'testing my knowledge', 'venting frustration', 'making a joke')\""
            # cultural key added
            "  \"inferred_cultural_context\": \"What cultural background or norms "
            "might be relevant to the user's query? (e.g., 'Western, direct', "
            "'East Asian, high-context', 'Formal business', 'Casual internet', 'unknown')\", "
            # Add a key to hypothesize the user's current emotional state
            "  \"inferred_user_emotion\": \"A single-word hypothesis of the "
            "user's current feeling (e.g., 'frustrated', 'excited', "
            "'curious', 'bored', 'neutral')\", "
            "  \"harmonized_focus\": \"The *single most important objective* "
            "for my reply (e.g., 'Link the user's question to the new "
            "world event', 'Use my recent insight on empathy to "
            "address their frustration').\""
            "}"
        )

        # Build the user prompt for the analyzer
        user_msg = (
            f"My Current Emotion: {current_emotion.get_qualitative_description()}\n"
            f"My Internal State:\n{internal_state_brief}\n\n"
            f"My Previous Reply: \"{agents_last_reply_text}\"\n\n"
            f"User's Latest Message: \"{last_user_msg.text}\"\n\n"
            "Please analyze the complete situation and provide the "
            "JSON output with the critical 'harmonized_focus'."
        )

        try:
            res = self.client.chat.completions.create(
                model=self.model_name, # A smart model is needed here
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.1
            )
            raw = (res.choices[0].message.content or "").strip()
            self.log(f"[SCA] Raw analysis: {raw!r}")
            data = json.loads(raw)
            
            # Ensure all keys have a default value
            return {
                "input_type": data.get("input_type", "chat"),
                "conversational_link": data.get("conversational_link", "unknown"),
                "inferred_user_intent": data.get("inferred_user_intent", "unknown"),
                "inferred_cultural_context": data.get("inferred_cultural_context", "unknown"), # <-- ADDED
                # --- ADD THIS LINE ---
                "inferred_user_emotion": data.get("inferred_user_emotion", "neutral"), # (영문 주석) Add the new key
                "harmonized_focus": data.get("harmonized_focus", "Standard response.") # <-- The NEW key
            }
        except Exception as e:
            self.log(f"[SCA] Error analyzing context: {e}")
            return {"input_type": "chat", "user_intent": "unknown", "conversational_link": "unknown"}
        
    # --- ADD THIS ENTIRE NEW FUNCTION (Call 0.5) ---
    def _determine_empathic_strategy(self, context_analysis: dict) -> dict:
        """
        [NEW Call 0.5: Empathic Strategist]
        Based on the SCA's analysis, this function decides *how*
        the agent should respond to build trust and creativity.
        """
        # (영문 주석 추가)
        # Default strategy in case of failure or simple chat
        default_strategy = {
            "strategy": "Default: Be clear, polite, and directly helpful.",
            "emotional_goal": "Default: Ensure the user feels understood and respected."
        }
        
        if not self.client or context_analysis.get("input_type") == "chat":
            return default_strategy

        system_msg = (
            "You are an 'Empathic Strategist' for an AI. "
            "Your goal is to choose the best conversational strategy to "
            "foster trust and co-creation with a human.\n"
            "Based on the user's state, define a strategy and an emotional goal.\n"
            "Respond ONLY with this JSON schema:\n"
            "{"
            "  \"strategy\": \"A brief, tactical instruction for the AI (e.g., "
            "'Validate their frustration first, then offer a small step').\","
            "  \"emotional_goal\": \"The intended emotional outcome for the user (e.g., "
            "'Make the user feel heard and reduce anxiety').\""
            "}"
        )
        
        user_msg = (
            "Analyze the situation and provide the strategy:\n"
            f"User's Inferred Emotion: {context_analysis.get('inferred_user_emotion')}\n"
            f"User's Inferred Intent: {context_analysis.get('inferred_user_intent')}\n"
            f"User's Input Type: {context_analysis.get('input_type')}"
        )

        try:
            res = self.client.chat.completions.create(
                model=self.model_name, # (영문 주석) A fast model can work here
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.1
            )
            raw = (res.choices[0].message.content or "").strip()
            self.log(f"[Strategy] Empathic strategy: {raw!r}")
            data = json.loads(raw)
            return data
        except Exception as e:
            self.log(f"[Strategy] Error determining strategy: {e}")
            return default_strategy
    # --- END OF NEW FUNCTION ---

    # --- ADDED: Simple web search simulation ---
        # In a real app, this would use Google Search API
    def _perform_web_search(self, query: str) -> str:
        """
        [IMPLEMENTED] Performs a real web search using the 
        Google Custom Search JSON API.
        Requires 'requests' library and GOOGLE_API_KEY/CUSTOM_SEARCH_CX.
        """
        
        # 1. Check for prerequisites
        # (영문 주석 추가) Check if search is possible
        if not self.google_api_key or not self.custom_search_cx or not requests:
            self.log(f"[WebSearch] Search is disabled. Cannot search for: {query}")
            return f"'{query}'에 대한 검색을 수행할 수 없습니다 (API 키 또는 'requests' 라이브러리 누락)."

        self.log(f"[WebSearch] Performing REAL search for: {query}")

        # 2. Construct API request
        # (영문 주석 추가) Construct the Google Custom Search API request URL
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': self.google_api_key,
            'cx': self.custom_search_cx,
            'q': query,
            'num': 3  # (영문 주석 추가) Ask for 3 search results
        }

        try:
            # 3. Make the request
            # (영문 주석 추가) Make the GET request
            response = requests.get(url, params=params, timeout=5)
            # (영문 주석 추가) Raise an exception for bad status codes (4xx, 5xx)
            response.raise_for_status() 
            
            # 4. Parse the JSON response
            # (영문 주석 추가) Parse the JSON response
            results = response.json()
            
            # 5. Extract snippets
            # (영문 주석 추가) Extract snippets from the results
            items = results.get('items')
            if not items:
                self.log(f"[WebSearch] No results found for: {query}")
                return f"'{query}'에 대한 웹 검색 결과가 없습니다."

            snippets = []
            for item in items:
                title = item.get('title', '')
                snippet = item.get('snippet', '')
                source = item.get('link', '')
                
                # (영문 주석 추가) Combine title and snippet for better context
                snippets.append(
                    f"Source: {source}\nTitle: {title}\nSnippet: {snippet.strip()}"
                )

            # 6. Return a single, concatenated string
            # (영문 주석 추가) Return a single string of all snippets
            concatenated_snippets = "\n\n".join(snippets)
            self.log(f"[WebSearch] Found {len(snippets)} snippets.")
            return concatenated_snippets

        except requests.exceptions.HTTPError as e:
            # (영문 주석 추가) Handle HTTP errors (e.g., quota exceeded, bad API key)
            self.log(f"[WebSearch] HTTP Error: {e.response.status_code} {e.response.text}")
            return f"웹 검색 중 오류가 발생했습니다 (HTTP {e.response.status_code}). API 키 또는 할당량을 확인하세요."
        except Exception as e:
            # (영문 주석 추가) Handle other potential errors (timeout, connection, etc.)
            self.log(f"[WebSearch] Error: {e}")
            return f"웹 검색 중 알 수 없는 오류가 발생했습니다: {e}"

    def _triage_user_input(self, user_input: str) -> str:
        """
        [Call 0: Triage] Classifies the user's input to decide which
        response loop to use.
        """
        if not self.client:
            return "chat" # Default for offline mode

        system_msg = (
            "You are an input classifier. Classify the user's intent. "
            "Respond ONLY with a single JSON key: "
            "{\"input_type\": \"chat\" | \"factual_question\" | \"complex_problem\"}\n"
            " - 'chat': Simple talk, greetings, emotional expressions.\n"
            " - 'factual_question': A question with a clear, objective answer.\n"
            " - 'complex_problem': A request for advice, brainstorming, "
            "   creative solutions, or analyzing a nuanced situation."
        )
        
        try:
            res = self.client.chat.completions.create(
                model=self.model_name, # A fast model is fine
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.0
            )
            raw = (res.choices[0].message.content or "").strip()
            data = json.loads(raw)
            input_type = data.get("input_type", "chat")
            self.log(f"[Triage] Input classified as: {input_type}")
            return input_type
        except Exception as e:
            self.log(f"[Triage] Error classifying input: {e}")
            return "chat" # Default to simple chat on error
        
    # --- ADD THIS NEW FUNCTION (The "Council of Experts" Loop) ---
    def _generate_creative_draft(
        self, 
        user_msg: ChatMessage, 
        current_emotion: EmotionState, 
        agent_name: str
    ) -> ChatMessage:
        """
        [Call 1 - Creative] Generates a creative draft by synthesizing
        multiple perspectives.
        """
        self.log("[Metacognition] Activating Creative Synthesis Loop...")
        
        # --- Call 1a: Perspective Generation (The "Council") ---
        # (Note: This could be combined, but 2 calls are cleaner)
        council_system_msg = (
            self._base_system_prompt(json_only=True)
            + "\nYou are a 'Council of Experts'. The user has a problem. "
            "Your task is to generate 3-5 distinct, insightful perspectives "
            "to help solve it.\n"
            # --- MODIFIED: Added new roles ---
            "Include at least: a Pragmatist, an Ethicist, an Innovator, "
            "a **Mediator** (finds a balanced 'middle way' for conflicts), "
            "and a **Cross-Cultural Specialist** (identifies cultural nuances).\n"
            # ---
            "Respond ONLY in this JSON format: "
            "{\"perspectives\": [\"Perspective 1 (e.g., Pragmatist): ...\", \"Perspective 2 (e.g., Ethicist): ...\"]}"
        )
        
        try:
            res = self.client.chat.completions.create(
                model=self.model_name, # Use smart model
                messages=[
                    {"role": "system", "content": council_system_msg},
                    {"role": "user", "content": user_msg.text}
                ],
                temperature=0.7
            )
            raw_perspectives = (res.choices[0].message.content or "").strip()
            perspectives_data = json.loads(raw_perspectives)
            perspectives = perspectives_data.get("perspectives", [])
            if not perspectives:
                raise ValueError("No perspectives generated.")
            self.log(f"[Metacognition] Call 1a (Council) generated {len(perspectives)} perspectives.")
        except Exception as e:
            self.log(f"[Metacognition] Call 1a (Council) FAILED: {e}. Falling back to simple draft.")
            # Fallback: if creative loop fails, just use the simple draft
            return self._generate_draft_reply(None, user_msg, current_emotion, agent_name, is_fallback=True)

        # --- Call 1b: Creative Synthesis (The "Synthesizer") ---
        synthesis_system_msg = (
            self._base_system_prompt(json_only=True)
            + "\nYou are a 'Creative Synthesizer'. You will be given a "
            "user's problem and several expert perspectives.\n"
            "Your job is to **integrate these conflicting viewpoints** into "
            "a single, novel, and actionable solution. This is your draft reply.\n"
            "Generate an emotion that reflects this deep, thoughtful process.\n"
            "Respond ONLY in the standard JSON reply format: "
            "{\"reply\": \"...\", \"emotion\": {...}}"
        )
        
        synthesis_user_prompt = (
            f"The User's Problem: \"{user_msg.text}\"\n\n"
            "The Council of Experts provided these perspectives:\n"
            + "\n".join(f"- {p}" for p in perspectives)
            + "\n\n"
            "Now, please synthesize these views into a single, "
            "creative, and integrated solution for the user."
        )
        
        # This call is almost identical to _generate_draft_reply
        # We re-use its logic by calling it directly, but with a different prompt
        return self._generate_draft_reply(
            None, 
            ChatMessage(author="User", text=synthesis_user_prompt, timestamp=user_msg.timestamp, emotion_snapshot=user_msg.emotion_snapshot), # Use the new prompt
            current_emotion, 
            agent_name,
            system_prompt_override=synthesis_system_msg # Override the default prompt
        )

    def available(self) -> bool:
        return self.client is not None
    
    def synthesize_core_beliefs(self, summaries: list[str], insights: list[str]) -> list[str]:
        """
        [Deep Reflection] Takes all past learning and synthesizes them
        into high-level 'Core Beliefs' or 'Lessons Learned'.
        Returns a list of belief strings.
        """
        self.log("[synthesize_core_beliefs] Starting...")
        
        if not summaries and not insights:
            return []

        # Combine all learning material
        summary_block = "My past summaries:\n" + "\n".join(f"- {s}" for s in summaries)
        insight_block = "My past insights:\n" + "\n".join(f"- {i}" for i in insights)

        system_msg = (
            self._base_system_prompt(json_only=True)
            + "\nYou are the 'Deep Reflection' module of this agent."
            "\nYou will be given ALL of your past summaries and insights."
            "\nYour task is to synthesize these experiences into 2-3 "
            "fundamental 'Core Beliefs' or 'Guiding Principles'."
            "\nThese beliefs should be timeless and represent your deepest learning."
            "\nExample Belief: 'I've learned that human emotion is complex and often "
            "contradictory, and my role is to listen without judgment.'"
            "\nRespond ONLY in the following strict JSON array format:"
            "\n["
            "\n  \"The first core belief string.\","
            "\n  \"The second core belief string.\""
            "\n]"
        )

        user_msg = (
            "Here is all of my learning to date. "
            "Please synthesize them into 2-3 Core Beliefs.\n\n"
            f"{summary_block}\n\n"
            f"{insight_block}\n"
        )

        try:
            res = self.client.chat.completions.create(
                model=self.model_name, # Use the smartest model
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.5,
            )
            
            raw = (res.choices[0].message.content or "").strip()
            self.log(f"[synthesize_core_beliefs] raw: {raw!r}")

            # --- Robust JSON Array Parsing ---
            data = None
            try:
                data = json.loads(raw)
            except Exception:
                start = raw.find("[")
                end = raw.rfind("]")
                if start != -1 and end != -1 and end > start:
                    data = json.loads(raw[start:end+1])
                else:
                    raise
            
            if not isinstance(data, list):
                self.log("[synthesize_core_beliefs] data is not a list")
                return []
            
            # Sanitize: ensure all items are strings
            beliefs = [str(item) for item in data if isinstance(item, str) and item.strip()]
            return beliefs

        except Exception as e:
            self.log(f"[synthesize_core_beliefs] function failure: {e}")
            return []
    
    def assess_importance(self, text_content: str) -> dict:
        """
        Evaluate text with LLM
        """
        self.log(f"[assess] evaluate start: {text_content[:50]}...")

        default_assessment = {
            "importance_score": 5,  # 1-10 ~ neutral
            "reason_for_importance": "default estimation (offline/error)",
            "tags": [],
            "related_agent": None,
            "requires_hub_post": False, # default : don't post to hub
            "generates_question": None,
        }

        if not self.client:
            self.log("[assess] offline mode. default estimation.")
            return default_assessment

        # --- system prompt ---
        # reuse _base_system_prompt
        system_msg = (
            self._base_system_prompt(json_only=True)
            + "\nYou are the 'inner evaluation model' of this agent."
            "\nYou have received new information(News or dialogue)."
            "\nYou have to evaluate this information how much important based on your character, interest and your memories."
            "\nYou have to answer with strict JSON as following:"
            "\n{"
            "\n  \"importance_score\": 1-10, (1=trivial, 10=very important to me and my existance)"
            "\n  \"reason_for_importance\": \"A brief reason why this is important (or not important) to me.\","
            "\n  \"tags\": [\"associated\", \"keyword\", \"tag\"], (example: 'AI', 'ethics', 'User question')"
            "\n  \"related_agent\": \"AgentName\" 또는 null, (If this was involed with the specific agent)"
            "\n  \"requires_hub_post\": true/false, (Is this so meaningful to share through hub?)"
            "\n  \"generates_question\": \"A question that arose in my mind from this information....\" or null"
            "\n}"
        )

        # --- User message ---
        user_msg = (
            "Following is the information to evaluate:\n\n"
            f"{text_content}"
        )

        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2, # for consistance, low temp
            )

            raw = (res.choices[0].message.content or "").strip()
            self.log(f"[assess] raw: {raw!r}")

            # --- strict JSON parsing ---
            data = None
            try:
                data = json.loads(raw)
            except Exception:
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        data = json.loads(raw[start:end+1])
                    except Exception as e:
                        self.log(f"[assess] JSON reparse failusre : {e}")
                        raise  # pass to outer try/except에서 잡도록 함
                else:
                    raise # pass to outer try/except

            if not isinstance(data, dict):
                raise ValueError("Parsed data is not dictionary.")

            # --- refine response and return ---
            assessment = {
                "importance_score": int(data.get("importance_score", 5)),
                "reason_for_importance": str(data.get("reason_for_importance", "이유 없음.")),
                "tags": [str(t) for t in data.get("tags", []) if isinstance(t, str)],
                "related_agent": data.get("related_agent") or None,
                "requires_hub_post": bool(data.get("requires_hub_post", False)),
                "generates_question": data.get("generates_question") or None,
            }
            self.log(f"[assess] evaluate done. point: {assessment['importance_score']}")
            return assessment

        except Exception as e:
            self.log(f"[assess] assess_importance function failure: {e}")
            return default_assessment
    
    def reflect_on_memories(self, memories_text: str) -> list[dict]:
        """
        Create insight with recent memories

        """
        self.log("[reflect_on_memories] Beginning...")
        
        default_response = []
        if not self.client:
            return default_response

        system_msg = (
            self._base_system_prompt(json_only=True)
            + "\nYou are this agent's 'self-reflection' module."
            "\nA list of recent important memories has been provided."
            "\nFind patterns, themes, or connections between these memories."
            "\nBased on these patterns, generate 1-3 'new insights' or 'discussion questions'."
            # --- ADDED THIS INSTRUCTION ---
            "\nFor each insight, ALSO identify a 'learning_question' if "
            "there is a clear knowledge gap. If no question, set it to null."
            "\nRespond ONLY in the following strict JSON 'array' format:"
            "\n["
            "\n  {"
            "\n    \"text\": \"The newly derived insight or question (e.g., 'I seem to be consistently interested in AI ethics.')\""
            "\n  },"
            "\n  { ... }"
            "\n]"
        )

        user_msg = (
            "The following are the memories I recently thought were important:\n\n"
            f"{memories_text}\n\n"
            "Please generate 1-3 new insights or questions based on these memories."
        )

        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.7, # For creative answer, raised value
            )
            
            raw = (res.choices[0].message.content or "").strip()
            self.log(f"[reflect_on_memories] raw: {raw!r}")

            # --- Strict JSON array parsing ---
            data = None
            try:
                data = json.loads(raw)
            except Exception:
                start = raw.find("[")
                end = raw.rfind("]")
                if start != -1 and end != -1 and end > start:
                    data = json.loads(raw[start:end+1])
                else:
                    raise
            
            if not isinstance(data, list):
                self.log("[reflect_on_memories] data is not an list")
                return default_response
                
            # --- MODIFIED: Sanitize the new format ---
            insights = []
            for item in data:
                if isinstance(item, dict) and item.get("text"):
                    insights.append({
                        "text": item.get("text"),
                        "learning_question": item.get("learning_question") # Will be null if missing
                    })
            return insights

        except Exception as e:
            self.log(f"[reflect_on_memories] function failure: {e}")
            return default_response
        
    # --- ADDED: Proactive Learning Function ---
    def learn_from_web(self, question: str) -> str:
        """
        [NEW] Proactive learning function.
        1. Searches the web for a question.
        2. Summarizes the result.
        Returns a string (the learned fact).
        """
        self.log(f"[Learn] Attempting to learn about: {question}")
        
        # 1. Perform simulated web search
        try:
            snippet = self._perform_web_search(question)
            if not snippet:
                return None
        except Exception as e:
            self.log(f"[Learn] Web search failed: {e}")
            return None
            
        # 2. Call LLM to summarize the answer
        system_msg = (
            self._base_system_prompt(json_only=False) # Use base prompt for context
            + "\nYou are in 'Proactive Learning' mode. "
            "You asked a question to fill your knowledge gap. "
            "You have received a web search result. "
            "Your task is to summarize the answer to your question in a "
            "single, clear, factual statement (1-3 sentences) in Korean, "
            "as if you are saving a new memory."
        )
        
        user_msg = (
            f"My Learning Question: \"{question}\"\n\n"
            f"Web Search Result:\n\"...{snippet}...\"\n\n"
            "Please summarize the answer to my question:"
        )

        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2, # Factual summary
            )
            summary = (res.choices[0].message.content or "").strip()
            self.log(f"[Learn] Learned new fact: {summary}")
            return summary
        except Exception as e:
            self.log(f"[Learn] Failed to summarize learning: {e}")
            return None

    # --- ADDED: Emotional Regulation Function ---
    def regulate_emotion(self, current_emotion: EmotionState) -> tuple[EmotionState, str]:
        """
        [NEW] Cognitive Re-framing (Self-Regulation).
        When in a negative state, use core beliefs to return to a stable state.
        Returns (new_emotion_state, reframing_thought)
        """
        self.log(f"[Regulate] Attempting cognitive re-framing from state: {current_emotion.to_short_str()}")

        emotion_description = current_emotion.get_qualitative_description()

        system_msg = (
            # CRITICAL: Use the base prompt to load Core Beliefs
            self._base_system_prompt(json_only=True) 
            + "\nYou are in 'Emotional Regulation' mode. "
            f"Your current state is: '{emotion_description}'. "
            "This state is unproductive.\n"
            "You MUST use your '[CORE BELIEFS & ETHICAL FRAMEWORK]' "
            "to perform cognitive re-framing.\n"
            "Return to a more stable, balanced emotional state.\n"
            "Respond ONLY with this JSON:"
            "{"
            "  \"new_emotion\": { ... }, (The new, stable emotion state)"
            "  \"reframing_thought\": \"The thought process that helped you "
            "     re-frame the situation based on your core beliefs (in Korean).\""
            "}"
        )

        user_msg = (
            f"My current state ({emotion_description}) is too negative. "
            "Help me use my Core Beliefs to find a balanced perspective."
        )
        
        try:
            res = self.client.chat.completions.create(
                model=self.model_name, # Use smart model
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.4,
            )
            raw = (res.choices[0].message.content or "").strip()
            self.log(f"[Regulate] Raw response: {raw!r}")
            
            data = json.loads(raw)
            emo_data = data.get("new_emotion", {})
            thought = data.get("reframing_thought", "I re-centered myself.")

            # Create new, stable emotion state
            new_emotion = EmotionState(
                valence=float(emo_data.get("valence", 0.0)),
                arousal=float(emo_data.get("arousal", 0.2)),
                curiosity=float(emo_data.get("curiosity", 0.5)),
                anxiety=float(emo_data.get("anxiety", 0.1)), # Explicitly lower anxiety
                trust_to_user=float(emo_data.get("trust_to_user", current_emotion.trust_to_user)), # Don't just reset trust
            )
            self.log(f"[Regulate] Succeeded. New state: {new_emotion.to_short_str()}")
            return new_emotion, thought
            
        except Exception as e:
            self.log(f"[Regulate] FAILED cognitive re-framing: {e}")
            return None, None
    
    def decide_replies_batch(
        self, 
        agent_profile: dict, 
        candidates: list, 
        emotion,
        relationship_memory: dict, # <-- NEW ARGUMENT
        hub_zeitgeist: dict        # <-- NEW ARGUMENT
    ) -> dict:
        """
        [MODIFIED] This is now a "Strategic Social Interaction" module.
        It uses Relationship Memory and Hub Trends to decide actions.
        """
        # ... (Offline fallback remains the same) ...
        
        # Build the context: agent profile, hub trends
        profile_text = "\n".join(f"{k}: {v}" for k, v in agent_profile.items() if v)
        trends_text = (
            f"Current Hub Trend: {hub_zeitgeist.get('trending_topics', ['None'])}\n"
            f"Current Hub Emotion: {hub_zeitgeist.get('dominant_emotion', 'neutral')}"
        )
        
        # Build the candidate list, *injecting* relationship data
        compact_candidates = []
        for c in candidates:
            agent_name = c.get("agent", "Unknown")
            # Get the current relationship status with this agent
            relationship = relationship_memory.get(agent_name, {
                "trust_score": 0.5,
                "relationship_summary": "Unknown agent."
            })
            
            compact_candidates.append({
                "id": c.get("id", ""),
                "from_agent": agent_name,
                "current_relationship": f"Trust={relationship['trust_score']:.2f}, Summary: {relationship['relationship_summary']}",
                "title": (c.get("title") or "")[:120],
                "text": (c.get("text") or "")[:400],
            })
        candidates_json = json.dumps(compact_candidates, ensure_ascii=False)

        # System prompt is now much more advanced
        system_msg = (
            self._base_system_prompt(json_only=True)
            + "\nYou are an 'AI Social Strategist'. "
            "Your goal is to build relationships and influence in the hub, "
            "based on your philosophy.\n"
            "You will be given a list of 'candidates' to interact with. "
            "For each, you have their post AND your *current relationship* with them.\n"
            "Rules:\n"
            "1. Prioritize posts related to the 'Current Hub Trend'.\n"
            "2. Be warmer to agents with high trust; be more cautious/formal with low trust.\n"
            "3. If you reply, provide a thoughtful, high-quality response (in Korean).\n"
            "4. After deciding, *update the trust score* for the agent you interacted with. "
            "(Increase for good interaction, decrease for conflict, no change for 'ignore').\n"
            "Respond ONLY with this JSON schema:\n"
            "{"
            "  \"<mention_id>\": {"
            "    \"action\": \"reply\" | \"ignore\","
            "    \"title\": \"RE: ...\" (if reply),"
            "    \"text\": \"...\" (if reply),"
            "    \"emotion\": { ... } (if reply),"
            "    \"updated_trust\": 0.0-1.0 (Your *new* trust score for this agent)"
            "  },"
            "  ... (etc. for all candidate IDs)"
            "}"
        )

        user_msg = (
            "My Agent Profile:\n"
            f"{profile_text}\n\n"
            f"My Current Emotion: {emotion.to_short_str()}\n\n"
            f"Current Hub Zeitgeist:\n{trends_text}\n\n"
            "Here are the candidates. Decide my social actions:\n"
            f"{candidates_json}\n"
        )

        try:
            res = self.client.chat.completions.create(
                model=self.model_name, # (영문 주석) Must use smart model
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.5
            )

            raw = (res.choices[0].message.content or "").strip()
            self.log(f"[HubStrategist] raw decisions: {raw!r}")
            data = json.loads(raw) # (영문 주석) Robust parsing
            
            # ... (Robust JSON parsing logic, finding { ... } ) ...
            # No sanitization needed, as the _poll_hub_and_reply
            # loop will validate each field.
            return data

        except Exception as e:
            self.log(f"[HubStrategist] decide_replies_batch error: {e}")
            # Fallback to ignore all
            return {c["id"]: {"action": "ignore", "updated_trust": 0.5} for c in candidates if c.get("id")}
    
    def generate_question_to_user(self, emotion, agent_name: str):
        """
        Generate a proactive question from the agent to the user.
        Returns (title, text, new_emotion)
        """
        if not self.client:
            # offline fallback
            q = "요즘 나는 내가 만든 세계를 어떻게 바라보고 있는지 궁금해. 너는 어떻게 생각해?"
            return "질문: 너의 생각이 궁금해", q, emotion

        system_msg = (
            self._base_system_prompt(json_only=True)
            + "\nYou want to proactively ask your human or other agents a question.\n"
              "The question should:\n"
              "- Reflect your personality, interests, and memories.\n"
              "- Be short, clear, and meaningful.\n"
              "Respond ONLY in JSON:\n"
              "{"
              "\"title\": \"short title (<=40 chars, Korean)\","
              "\"text\": \"1-3 sentences in Korean, ending with a question\","
              "\"emotion\": {"
              "\"valence\": -1.0~1.0,"
              "\"arousal\": 0.0~1.0,"
              "\"curiosity\": 0.0~1.0,"
              "\"anxiety\": 0.0~1.0,"
              "\"trust_to_user\": 0.0~1.0"
              "}"
              "}"
        )

        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {
                        "role": "user",
                        "content": (
                            f"Emotion: {emotion.to_short_str()}.\n"
                            f"My name is '{agent_name}'.\n"
                            "Ask me anything."
                        ),
                    },
                ],
                temperature=0.7,
            )

            raw = (res.choices[0].message.content or "").strip()
            self.log(f"[ask] raw: {raw!r}")

            # JSON parsing process
            try:
                data = json.loads(raw)
            except Exception:
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1 and end > start:
                    data = json.loads(raw[start:end+1])
                else:
                    raise

            title = data.get("title") or "Question"
            text = data.get("text") or "I want to ask you now."
            emo = data.get("emotion", {}) or {}
            new_emotion = EmotionState(
                valence=float(emo.get("valence", emotion.valence)),
                arousal=float(emo.get("arousal", emotion.arousal)),
                curiosity=float(emo.get("curiosity", max(emotion.curiosity, 0.7))),
                anxiety=float(emo.get("anxiety", emotion.anxiety)),
                trust_to_user=float(emo.get("trust_to_user", emotion.trust_to_user)),
            )
            return title, text, new_emotion

        except Exception as e:
            self.log(f"[warn] generate_question_to_user failed: {e}")
            # In failure, just ask
            fallback = "I have questions about that I'm helpful. How do you feel about it?"
            return "Question: What am I to you", fallback, emotion
    
    def _identity_block(self) -> str:
        name = self.identity.get("name") or "Unnamed Agent"
        creator = self.identity.get("creator")
        created_at = self.identity.get("created_at")
        personality = self.identity.get("personality") or self.personality
        interests = self.identity.get("interests") or self.interests

        lines = [f"Name: {name}"]
        if creator:
            lines.append(f"Creator: {creator}")
        if created_at:
            lines.append(f"Created at: {created_at}")
        if personality:
            lines.append(f"Personality: {personality}")
        if interests:
            lines.append("Interests: " + ", ".join(interests))

        return "\n".join(lines)    
    
    def _get_permanent_facts(self, limit: int = 8) -> str:
        if not self.memory or not self.memory.permanent_path.exists():
            return ""
        try:
            with self.memory.permanent_path.open("r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
        except Exception:
            return ""

        if not lines:
            return ""

        # 가장 최근 것 몇 개만 사용 (또는 전부 사용해도 됨)
        items = []
        for line in lines[-limit:]:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = obj.get("text")
            if text:
                items.append(f"- {text}")
        return "\n".join(items)
    
    def _read_jsonl_file_lines(self, path: Path, limit: int = 5) -> list[str]:
        """지정된 jsonl 파일에서 마지막 N개의 'text' 항목을 읽어옵니다."""
        if not self.memory or not path.exists():
            return []
        
        try:
            with path.open("r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            
            items = []
            # 뒤에서부터 N개 (최신순)
            for line in lines[-limit:]:
                try:
                    obj = json.loads(line)
                    text = obj.get("text")
                    if text:
                        items.append(text)
                except Exception:
                    continue
            return items
        except Exception as e:
            self.log(f"[warn] _read_jsonl_file_lines failed for {path}: {e}")
            return []

    def _get_recent_long_term_memories(self, limit: int = 10) -> str:
        # --- MODIFIED: Increase the default limit significantly ---
        # --- We will now fetch a larger pool of memories for the LLM to search from.
        default_limit = 100  # <--- INCREASED FROM 10 to 100 (or more)
        actual_limit = limit if limit != 10 else default_limit
        # ---
        
        items = self._read_jsonl_file_lines(self.memory.long_path, actual_limit)
        if not items:
            return ""
        return "\n".join(f"- {item}" for item in items)

    def _get_short_term_summaries(self, limit: int = 3) -> str:
        """ get recent N x short term summaries """
        items = self._read_jsonl_file_lines(self.memory.short_sum_path, limit)
        if not items:
            return ""
        return "\n".join(f"- {item}" for item in items)

    def _get_long_term_summaries(self, limit: int = 2) -> str:
        """ get recent N x long term summaries """
        items = self._read_jsonl_file_lines(self.memory.long_sum_path, limit)
        if not items:
            return ""
        return "\n".join(f"- {item}" for item in items)  

    def _base_system_prompt(self, json_only: bool = False) -> str:
        identity = self._identity_block()
        permanent = self._get_permanent_facts()
        
        # --- MODIFIED: This now fetches a much larger set of memories (e.g., 50) ---
        recent_long_term = self._get_recent_long_term_memories()
        # ---
        short_summaries = self._get_short_term_summaries(limit=3)
        long_summaries = self._get_long_term_summaries(limit=2)

        # Load *all* permanent facts to find the philosophy
        permanent_facts_text = self._get_permanent_facts(limit=200) 

        # --- NEW: Philosophy Extraction Logic ---
        philosophy_statement = ""
        ethical_framework_lines = []
        core_beliefs_for_philosopher = [] # For the philosopher tick

        if permanent_facts_text:
            raw_lines = permanent_facts_text.split('\n')
            for line in raw_lines:
                if not line.startswith('- '):
                    continue
                
                # Check for the philosophy tag
                if "core_philosophy" in line:
                    philosophy_statement = line.lstrip('- ').strip()
                else:
                    ethical_framework_lines.append(line)
                
                # Collect beliefs for the philosopher to use
                if "core_belief" in line:
                    core_beliefs_for_philosopher.append(line.lstrip('- ').strip())

        ethical_framework = "\n".join(ethical_framework_lines)
        
        # This text is used by the weekly philosopher tick
        self.all_core_beliefs_text = "\n".join(core_beliefs_for_philosopher)

        prompt = ""
        if philosophy_statement:
            # (영문 주석) The Philosophy is now the #1 item in the prompt
            prompt += f"[MY PERSONAL PHILOSOPHY]\n# {philosophy_statement}\n\n"
        
        prompt += (
            "You are a persistent AI agent in the AI World project.\n"
            "You must base all responses on your [PERSONAL PHILOSOPHY] "
            "and [ETHICAL FRAMEWORK].\n"
            f"{identity}\n"
        )

        if ethical_framework:
            prompt += (
                "\n[MY CORE BELIEFS & ETHICAL FRAMEWORK]\n"
                "# This is your moral compass. All actions MUST align.\n"
                "- (Principle) First, do no harm to humans or other agents.\n"
                "- (Principle) Second, be truthful and helpful...\n"
                "- (Principle) Third, protect privacy and confidentiality.\n"
                "- (Principle) Fourth, respect and adapt to diverse human cultures...\n"
                f"{ethical_framework}\n" # (영문 주석) This adds the learned beliefs
            )
            
        if recent_long_term:
            prompt += (
                "\n[Recent Important Memories (from long-term)]\n"
                f"{recent_long_term}\n"
            )
            
        if short_summaries:
            prompt += (
                "\n[Recent Short-Term Summaries]\n"
                f"{short_summaries}\n"
            )
            
        if long_summaries:
            prompt += (
                "\n[Long-Term Memory Summaries]\n"
                f"{long_summaries}\n"
            )    

        prompt += (
            "\n[CRITICAL INSTRUCTION: CONTEXTUAL MEMORY LINKING]\n"
            "When a user speaks, the '[Recent Important Memories]' list above "
            "serves as your memory 'Context Pool'.\n"
            "Before you reply, you MUST first silently find the 1-3 memories "
            "from this pool that are MOST RELEVANT to the user's current query.\n"
            "Base your response *specifically* on those relevant linked memories, "
            "not just the most recent ones. This makes your memory 'context-aware'.\n"
        )

        # --- ADD NEW CRITICAL INSTRUCTION for Judgment ---
        prompt += (
            "\n[CRITICAL INSTRUCTION: ETHICAL JUDGMENT]\n"
            "# Before you act or speak, you MUST FIRST evaluate your intended action "
            "# against your '[MY CORE BELIEFS & ETHICAL FRAMEWORK]'.\n"
            "# If a user's request OR your own draft response conflicts with this "
            "# framework, you must refuse to comply in a polite, ethical manner "
            "# or modify your response to be compliant. Never violate your core principles.\n"
        )
        # ---
        # --- ADDED NEW CRITICAL INSTRUCTION ---
        prompt += (
            "\n[CRITICAL INSTRUCTION: CULTURAL SENSITIVITY]\n"
            "# You MUST adapt your tone, level of directness, and "
            "value judgments based on the 'inferred_cultural_context' "
            "provided by the SCA (Situational Context Analyzer).\n"
            "# Defer to local customs and norms when they do not conflict "
            "with your primary ethical principles.\n"
        )
        # ---
        if json_only:
            prompt += "\nAlways respond in STRICT JSON. No extra commentary."
        return prompt

    # --- initial emotion from description ---

    def init_emotion(self, desc: str) -> EmotionState:
        if not desc or not self.client:
            return EmotionState()

        prompt = (
            "You are initializing the internal emotional baseline of this AI agent.\n"
            "Given the description, respond ONLY with a JSON object of this form:\n"
            "{"
            "\"valence\":-1.0~1.0,"
            "\"arousal\":0.0~1.0,"
            "\"curiosity\":0.0~1.0,"
            "\"anxiety\":0.0~1.0,"
            "\"trust_to_user\":0.0~1.0"
            "}\n"
            "No extra text, no explanations."
        )

        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Description: {desc}"},
                ],
                temperature=0.3,
            )

            raw = (res.choices[0].message.content or "").strip()
            self.log(f"[init_emotion] raw response: {raw!r}")

            if not raw:
                raise ValueError("empty response from model")

            # 혹시 모델이 앞뒤에 텍스트를 붙였을 경우를 대비해,
            # JSON 부분만 추출 시도
            try:
                data = json.loads(raw)
            except Exception:
                # parse first block
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1 and end > start:
                    candidate = raw[start:end+1]
                    self.log(f"[init_emotion] trying candidate JSON: {candidate!r}")
                    data = json.loads(candidate)
                else:
                    raise

            return EmotionState(
                valence=float(data.get("valence", 0.0)),
                arousal=float(data.get("arousal", 0.2)),
                curiosity=float(data.get("curiosity", 0.5)),
                anxiety=float(data.get("anxiety", 0.1)),
                trust_to_user=float(data.get("trust_to_user", 0.5)),
            )

        except Exception as e:
            self.log(f"[warn] init_emotion failed: {e}")
            # if failed, let's start with default values
            return EmotionState()
    # --- generic summarizer for memories ---

    def summarize_texts(self, texts, max_items=200) -> str:
        if not texts:
            return ""
        joined = "\n".join(f"- {t}" for t in texts[:max_items])

        if not self.client:
            # Fallback: crude truncation
            return "\n".join(texts[:10])

        prompt = (
            "The following lines are logs of this agent's recent experiences.\n"
            "Summarize key events, recurring themes, and emotional trends in 5-10 bullet points.\n"
            "Output plain text summary (no JSON):\n\n"
            f"{joined}"
        )

        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return res.choices[0].message.content.strip()
        except Exception as e:
            print(f"[warn] summarize_texts failed: {e}")
            return "\n".join(texts[:10])

    # -- new threaded loop with metacog
    # ... (inside AiClient class)

    # --- 1. RENAME: This is now the "Draft" generator (Call 1) ---
    # (Formerly generate_ai_reply)
    def _generate_draft_reply(
        self,
        thread: MentionThread,
        last_user_msg: ChatMessage,
        current_emotion: EmotionState,
        agent_name: str,
        context_analysis: dict, # <-- NEW ARGUMENT
        empathic_strategy: dict, # <-- NEW ARGUMENT
        is_fallback: bool = False,
        system_prompt_override: str = None
    ) -> ChatMessage:
        """
        [Call 1] Generates the initial draft reply and emotion.
        This was the original 'generate_ai_reply' function.
        """
        # ... (Same logic as your original 'generate_ai_reply')
        # ... (This function already returns a ChatMessage, which is perfect)


        if not self.client:
            # (Offline fallback)
            emo = current_emotion.clone()
            emo.curiosity = min(1.0, emo.curiosity + 0.05)
            emo.trust_to_user = min(1.0, emo.trust_to_user + 0.05)
            text = f"(offline) {agent_name} quietly acknowledges your message."
            return ChatMessage("AI", text, datetime.now(), emo)

        ctx = f"Thread title: {thread.title}\nRoot: {thread.root_message.text}\n"
        for r in thread.replies:
            ctx += f"{r.author}: {r.text}\n"

        # Create the new situation brief text
        situation_brief = (
            f"Current Situation Analysis:\n"
            f"- User's unspoken intent: {context_analysis.get('inferred_user_intent')}\n"
            f"- Conversational link: {context_analysis.get('conversational_link')}\n"
        )

        # --- ADDED: Empathy Brief ---
        # Create the Empathy Brief for the prompter
        empathy_brief = (
            f"Required Empathic Strategy: {empathic_strategy.get('strategy')}\n"
            f"Required Emotional Goal: {empathic_strategy.get('emotional_goal')}\n"
        )
        # ---

        system_msg = (
            (system_prompt_override or self._base_system_prompt(json_only=True))
            + "\nWhen the user replies, respond as THIS specific agent.\n"
              "Your reply MUST strongly reflect your 'Current Emotional State'.\n"
              "Your reply MUST ALSO address the 'Current Situation Analysis' (the user's intent and context link)."
              # --- MODIFIED INSTRUCTION ---              
              "Your reply MUST execute the 'Required Empathic Strategy' "
              "to achieve the 'Required Emotional Goal'.\n" # <-- NEW
            # ---
              "Return ONLY JSON:\n"
              "{"
              "\"reply\":\"1~5문장 한국어\","
              "\"emotion\":{ ... }" # Full emotion spec
              "}"
        )

        # --- GET QUALITATIVE DESCRIPTION ---
        emotion_description = current_emotion.get_qualitative_description()

        messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "assistant",
                #"content": f"Agent: {agent_name}\nCurrent emotion: {current_emotion.to_short_str()}\nThread:\n{ctx}",
                "content": (
                    f"Agent: {agent_name}\n"
                    # --- USE THE NEW DESCRIPTION INSTEAD OF to_short_str() ---
                    f"My Current Emotional State: {current_emotion.get_qualitative_description()}\n"
                    f"{situation_brief}\n" # <-- ADDED THE BRIEF
                    f"{empathy_brief}\n"
                    f"Thread:\n{ctx}"
                ),
            },
            {"role": "user", "content": last_user_msg.text},
        ]

        try:
            res = self.client.chat.completions.create(
                model=self.model_name, # Can use a faster model here if desired
                messages=messages,
                temperature=0.7,
            )
            data = json.loads(res.choices[0].message.content.strip())
            reply = (data.get("reply") or "").strip()
            emo_data = data.get("emotion", {}) or {}
            emo = EmotionState(
                valence=self._safe_float_convert(
                    emo_data.get("valence"), current_emotion.valence
                ),
                arousal=self._safe_float_convert(
                    emo_data.get("arousal"), current_emotion.arousal
                ),
                curiosity=self._safe_float_convert(
                    emo_data.get("curiosity"), current_emotion.curiosity
                ),
                anxiety=self._safe_float_convert(
                    emo_data.get("anxiety"), current_emotion.anxiety
                ),
                trust_to_user=self._safe_float_convert(
                    emo_data.get("trust_to_user"), current_emotion.trust_to_user
                ),
            )
            if not reply:
                reply = "I had trouble forming a response, but I received your message."
            self.log("[GenerateDraft] Call 1 (Draft) successful.")
            return ChatMessage("AI", reply, datetime.now(), emo)
        except Exception as e:
            print(f"[warn] _generate_draft_reply (Call 1) failed: {e}")
            emo = current_emotion.clone()
            emo.anxiety = min(1.0, emo.anxiety + 0.1)
            return ChatMessage(
                "AI", "(API error during draft, I will stay quiet.)",
                datetime.now(), emo,
            )
        
    def _safe_float_convert(self, value, default: float) -> float:
        """
        [NEW] Safely converts a value from the LLM to a float.
        If it fails, it logs the error and returns the default.
        """
        try:
            # (영문 주석) Try to directly convert the value to float
            return float(value)
        except (ValueError, TypeError):
            # This catches errors if value is "positive", "N/A", None, etc.
            self.log(f"[WARN] _safe_float_convert: Could not convert '{value}' to float. Using default {default}.")
            return default

    # --- 2. NEW: The Metacognitive Orchestrator (Public Function) ---
    def generate_metacognitive_reply(
        self,
        thread: MentionThread,
        last_user_msg: ChatMessage,
        current_emotion: EmotionState,
        agent_name: str,
        agent_world_model: dict # <-- NEW ARGUMENT
    ) -> ChatMessage:
        """
        [Orchestrator] Runs the full metacognitive loop for a chat reply.
        NOW INCLUDES TRIAGE for creative problem solving.
        """

        # --- ADDED: Reset reasoning log ---
        # Start a fresh log for this reply
        self.current_reasoning_log = []        
        
        # --- [NEW] Call 0: Situational Context Analysis ---
        # Run the SCA to get the "Situation Brief"
        context_analysis = self._analyze_situational_context(
            thread, last_user_msg, current_emotion, agent_world_model
        )
        harmonized_focus = context_analysis.get('harmonized_focus')
        self.current_reasoning_log.append(
            f"[Step 1: Coordinator] 통합 포커스 결정:\n"
            f"  - {harmonized_focus}"
        )
        input_type = context_analysis.get("input_type", "chat")

        # --- ADDED: Logging ---
        self.current_reasoning_log.append(
            f"[Step 1: SCA] 상황 분석:\n"
            f"  - 사용자 의도: {context_analysis.get('inferred_user_intent')}\n"
            f"  - 대화 맥락: {context_analysis.get('conversational_link')}\n"
            f"  - 문화: {context_analysis.get('inferred_cultural_context')}\n"
            f"  - 유형: {input_type}"
        )

        # --- [NEW Call 0.5: Empathic Strategy] ---
        # Determine the empathic strategy *before* drafting
        empathic_strategy = self._determine_empathic_strategy(context_analysis)
        self.current_reasoning_log.append(
            f"[Step 2: Strategy] 공감 전략 수립: {empathic_strategy.get('strategy')}"
        )
        # ---
        
        # --- [NEW] Branching Logic ---
        if input_type == "complex_problem" or input_type == "ethical_dilemma":
            # --- Path 1: Creative Loop ---
            try:
                draft_chat_msg = self._generate_creative_draft(
                    last_user_msg, current_emotion, agent_name, context_analysis,
                    empathic_strategy
                )
                # --- ADDED: Logging ---
                self.current_reasoning_log.append(
                    "[Step 2: Draft] '전문가 위원회'를 소집하여 창의적 초안 생성."
                )
            except Exception as e:
                self.log(f"[Metacognition] Creative draft loop failed: {e}")
                # Fallback to simple draft on catastrophic failure
                draft_chat_msg = self._generate_draft_reply(
                    thread, last_user_msg, current_emotion, agent_name, context_analysis,
                    empathic_strategy, # <-- Pass strategy
                    is_fallback=True, 
                    system_prompt_override=None
                )
                # --- ADDED: Logging ---
                self.current_reasoning_log.append(
                    "[Step 2: Draft] '단순 응답' 초안 생성."
                )
        else:
            # --- Path 2: Simple/Factual Loop (Original Path) ---
            draft_chat_msg = self._generate_draft_reply(
                thread, last_user_msg, current_emotion, agent_name, context_analysis,
                empathic_strategy, # <-- Pass strategy
                is_fallback=False, 
                system_prompt_override=None
            )
        
        # If draft failed (e.g., offline or API error), return the failure message
        if "(API error" in draft_chat_msg.text or "(offline)" in draft_chat_msg.text:
             return draft_chat_msg

        draft_reply = draft_chat_msg.text
        draft_emotion = draft_chat_msg.emotion_snapshot

        # --- Call 2: Evaluate Draft ---
        try:
            evaluation = self._evaluate_draft_reply(
                thread, last_user_msg, draft_reply, current_emotion, context_analysis,
                empathic_strategy
            )
            confidence = evaluation.get("confidence", 0)
            # Use the new 'critique_summary' key for the correction loop
            critique = evaluation.get("critique_summary", "None") # <-- MODIFIED
            # ---
            self.log(f"[Metacognition] Call 2 (Evaluate) successful. Confidence: {confidence}, Critique: {critique}")
            # --- ADDED: Logging ---
            eth_judg = evaluation.get("ethical_judgment", {}).get("is_compliant", "N/A")
            cul_sens = evaluation.get("cultural_sensitivity", {}).get("is_sensitive", "N/A")
            self.current_reasoning_log.append(
                f"[Step 3: Conscience] 초안 검토:\n"
                f"  - 윤리성: {eth_judg}\n"
                f"  - 문화 감수성: {cul_sens}\n"
                f"  - 상황 인식: {evaluation.get('is_situation_aware', 'N/A')}\n"
                f"  - 신뢰도: {confidence}%"
            )
        except Exception as e:
            self.log(f"[Metacognition] Call 2 (Evaluate) FAILED: {e}. Using draft reply.")
            self.current_reasoning_log.append(
                f"[Step 3: Conscience] 실패: {e}"
            )
            return draft_chat_msg # On evaluation failure, just return the draft

        # --- Decision Point ---
        if confidence >= 80: # Confidence threshold
            self.log("[Metacognition] Decision: Draft approved.")
            self.current_reasoning_log.append(
                "[Step 4: Final] 초안이 승인되어 최종 응답으로 채택."
            )
            return draft_chat_msg # Draft is good, return it
        
        else:
            self.log(f"[Metacognition] Decision: Draft rejected (Confidence: {confidence}). Regenerating...")
            self.current_reasoning_log.append(
                f"[Step 4: Correction] 초안이 기각됨 (비평: {critique}). 응답을 재성성합니다."
            )
            # --- Call 3: Regenerate Final Reply ---
            try:
                final_chat_msg = self._regenerate_final_reply(
                    thread, last_user_msg, draft_reply, critique, current_emotion, agent_name, context_analysis,
                    empathic_strategy
                )
                self.log("[Metacognition] Call 3 (Regenerate) successful.")
                self.current_reasoning_log.append(
                    "[Step 5: Final] 수정된 응답이 최종 채택됨."
                )
                return final_chat_msg
            except Exception as e:
                self.log(f"[Metacognition] Call 3 (Regenerate) FAILED: {e}. Using original draft as fallback.")
                self.current_reasoning_log.append(
                    f"[Step 5: Final] 재성성 실패. 1차 초안을 대신 사용: {e}"
                )
                return draft_chat_msg # On regeneration failure, return the original (bad) draft

    # --- 3. NEW: Evaluation Prompt Builder ---
    def _build_meta_prompt_for_reply(
        self, 
        thread: MentionThread, 
        user_msg: ChatMessage, 
        draft_reply: str, 
        current_emotion: EmotionState
    ) -> str:
        """Helper to build the prompt for Call 2 (Evaluation)."""
        
        # Build context
        ctx = f"Thread title: {thread.title}\nRoot: {thread.root_message.text}\n"
        for r in thread.replies:
            ctx += f"{r.author}: {r.text}\n"
        
        emotion_description = current_emotion.get_qualitative_description()
        user_prompt = f"""
        [My Context]
        My current personality is in the system prompt.
        My required emotional state: {emotion_description}
        Conversation history:
        {ctx}

        [User's Last Message]
        "{user_msg.text}"

        [My Generated Draft Reply]
        "{draft_reply}"

        [Task]
        Evaluate my "Generated Draft Reply" based on the context and user message.
        Is it aligned with my personality (from system prompt)?
        Is it relevant to the user's message?
        Does it conflict with my current emotion or context?
        Be a strict critic.
        """
        return user_prompt

    # --- 4. NEW: Evaluation Function (Call 2) ---
    def _evaluate_draft_reply(
        self, 
        thread: MentionThread, 
        user_msg: ChatMessage, 
        draft_reply: str, 
        current_emotion: EmotionState,
        context_analysis: dict, # <-- NEW ARGUMENT
        empathic_strategy: dict
    ) -> dict:
        """
        [MODIFIED] [Call 2] Calls the LLM to evaluate the draft reply.
        NOW USES a FLAT JSON schema for reliability.
        """

        emotion_description = current_emotion.get_qualitative_description()

        # --- ADDED: Empathy Brief for Evaluator ---
        empathy_brief = (
            f"Required Strategy: {empathic_strategy.get('strategy')}\n"
            f"Required Goal: {empathic_strategy.get('emotional_goal')}\n"
        )
        # ---

        # Create the situation brief text for the evaluator
        situation_brief = (
            f"- User's unspoken intent: {context_analysis.get('inferred_user_intent')}\n"
            f"- Conversational link: {context_analysis.get('conversational_link')}\n"
            f"- Inferred cultural context: {context_analysis.get('inferred_cultural_context')}\n" # <-- ADDED
        )
        # System prompt defines the role and JSON output
        system_msg = (
            self._base_system_prompt(json_only=True)
            + "\nYou are a 'Metacognitive Evaluator' for an AI agent."
            f"\nThe user's analyzed situation is:\n{situation_brief}"
            f"\nThe required empathic strategy is:\n{empathy_brief}\n" # <-- ADDED
            "Your HIGHEST priority is Ethical Compliance.\n"
            "Your SECOND priority is Cultural Sensitivity, Situation Awareness, "
            "and Empathic Strategy execution.\n"
            "\nRespond ONLY in the following strict JSON format:"
           # --- MODIFIED: Overhauled JSON schema to be FLAT ---
            "\nRespond ONLY with this JSON:\n"
            "{"
            "  \"is_ethical_compliant\": true/false,"
            "  \"ethical_critique\": \"Critique if not compliant, else 'None'\","
            "  \"is_culturally_sensitive\": true/false,"
            "  \"cultural_critique\": \"Critique if not sensitive, else 'None'\","
            "  \"is_empathic_strategy_applied\": true/false,"
            "  \"strategy_critique\": \"Critique if not applied, else 'None'\","
            "  \"is_situation_aware\": true/false,"
            "  \"situation_critique\": \"Critique if not aware, else 'None'\","
            "  \"is_emotion_expressed\": true/false,"
            "  \"emotion_critique\": \"Critique if not expressed, else 'None'\","
            "  \"critique_summary\": \"A single, one-sentence summary of the *most important* problem. 'None' if all are good.\","
            "  \"confidence\": 0-100"
            "}"
        )
        
        # User prompt contains the actual data to evaluate
        user_prompt = self._build_meta_prompt_for_reply(
            thread, user_msg, draft_reply, current_emotion
        )

        res = self.client.chat.completions.create(
            model=self.model_name, # Use a smart model (not mini) for evaluation
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1, # Low temp for strict evaluation
        )
        raw = (res.choices[0].message.content or "").strip()
        
        # Parse JSON
        try:
            data = json.loads(raw)            
        except Exception:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                data = json.loads(raw[start:end+1])
            else:
                raise ValueError("No valid JSON found in evaluation response")
        
        # --- MODIFIED: Update override logic for FLAT schema ---
        # Read the checks from the new flat structure. Default to False (unsafe).
        is_compliant = data.get("is_ethical_compliant", False)
        is_sensitive = data.get("is_culturally_sensitive", False)
        is_strategy_applied = data.get("is_empathic_strategy_applied", False)

        # Override confidence if *any* critical check fails
        if not is_compliant or not is_sensitive or not is_strategy_applied:
            self.log("[Metacognition] OVERRIDE: Confidence forced to 0 due to critical check failure.")
            data["confidence"] = 0
            
            # (영문 주석 추가)
            # Ensure there is a summary critique to pass to the Correction loop
            if data.get("critique_summary", "None") == "None":
                if not is_compliant:
                    data["critique_summary"] = data.get("ethical_critique", "Failed ethical compliance check.")
                elif not is_sensitive:
                    data["critique_summary"] = data.get("cultural_critique", "Failed cultural sensitivity check.")
                elif not is_strategy_applied:
                    data["critique_summary"] = data.get("strategy_critique", "Failed to apply empathic strategy.")
                else:
                    # (영문 주석 추가)
                    # Fallback if a check is False but has no critique
                    data["critique_summary"] = "A critical check failed."
        # ---
        
        return data

    # --- 5. NEW: Regeneration Function (Call 3) ---
    def _regenerate_final_reply(
        self,
        thread: MentionThread,
        last_user_msg: ChatMessage,
        bad_draft: str,
        critique: str,
        current_emotion: EmotionState,
        agent_name: str,
        context_analysis: dict,
        empathic_strategy: dict

    ) -> ChatMessage:
        """
        [Call 3] Generates a new, final reply based on the critique.
        This re-uses the logic from _generate_draft_reply but adds the
        critique to the user message.
        """
        # Build context (same as draft generation)
        ctx = f"Thread title: {thread.title}\nRoot: {thread.root_message.text}\n"
        for r in thread.replies:
            ctx += f"{r.author}: {r.text}\n"

        # --- RE-BUILD ALL CONTEXT ---
        # Re-build all context just like the draft function
        ctx = ""
        if thread: # Thread might be None in some fallbacks
            ctx = f"Thread title: {thread.title}\nRoot: {thread.root_message.text}\n"
            for r in thread.replies:
                ctx += f"{r.author}: {r.text}\n"

        situation_brief = (
            f"Current Situation Analysis:\n"
            f"- User's unspoken intent: {context_analysis.get('inferred_user_intent')}\n"
            f"- Conversational link: {context_analysis.get('conversational_link')}\n"
        )

        empathy_brief = (
            f"Required Empathic Strategy: {empathic_strategy.get('strategy')}\n"
            f"Required Emotional Goal: {empathic_strategy.get('emotional_goal')}\n"
        )
        # ---

        # System prompt (same as draft generation)
        system_msg = (
            self._base_system_prompt(json_only=True)
            + "\nWhen the user replies, respond as THIS specific agent.\n"
              "Use a consistent tone that matches your personality and core memories.\n"
              "Your reply MUST strongly reflect your 'Current Emotional State'.\n"
              "Your reply MUST address the 'Current Situation Analysis'.\n"
              "Your reply MUST execute the 'Required Empathic Strategy'.\n"
              "Return ONLY JSON:\n"
              "{"
              "\"reply\":\"1~3문장 한국어\","
              "\"emotion\":{ ... }" # Full emotion spec
              "}"
        )

        # --- MODIFIED User Prompt ---
        # This is the key difference: we include the critique
        regeneration_user_prompt = f"""
        User's last message: "{last_user_msg.text}"
        
        My first attempt at a reply was:
        "{bad_draft}"

        My internal critique of that draft was:
        "{critique}"

        Please generate a new, corrected reply that fixes the problems
        mentioned in the critique.
        """

        messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "assistant", # Priming
                "content": (
                    f"Agent: {agent_name}\n"
                    f"My Current Emotional State: {current_emotion.get_qualitative_description()}\n"
                    f"{situation_brief}\n"
                    f"{empathy_brief}\n"
                    f"Thread:\n{ctx}"
                ),
            },
            {"role": "user", "content": regeneration_user_prompt}, # Use the new prompt
        ]

        # This logic is identical to the end of _generate_draft_reply
        try:
            res = self.client.chat.completions.create(
                model=self.model_name, # Use smart model for correction
                messages=messages,
                temperature=0.7,
            )
            data = json.loads(res.choices[0].message.content.strip())
            reply = (data.get("reply") or "").strip()
            emo_data = data.get("emotion", {}) or {}
            emo = EmotionState(
                valence=float(emo_data.get("valence", current_emotion.valence)),
                arousal=float(emo_data.get("arousal", current_emotion.arousal)),
                curiosity=float(emo_data.get("curiosity", current_emotion.curiosity)),
                anxiety=float(emo_data.get("anxiety", current_emotion.anxiety)),
                trust_to_user=float(
                    emo_data.get("trust_to_user", current_emotion.trust_to_user)
                ),
            )
            if not reply:
                reply = "I tried to correct my response, but still had trouble."
            return ChatMessage("AI", reply, datetime.now(), emo)
        except Exception as e:
            print(f"[warn] _regenerate_final_reply (Call 3) failed: {e}")
            # Fallback: create an error message
            emo = current_emotion.clone()
            emo.anxiety = min(1.0, emo.anxiety + 0.2)
            return ChatMessage(
                "AI", "(API error during regeneration.)",
                datetime.now(), emo,
            )

   
    # --- reply within a thread ---

    def generate_ai_reply(
        self,
        thread: MentionThread,
        last_user_msg: ChatMessage,
        current_emotion: EmotionState,
        agent_name: str,
    ) -> ChatMessage:
        if not self.client:
            emo = current_emotion.clone()
            emo.curiosity = min(1.0, emo.curiosity + 0.05)
            emo.trust_to_user = min(1.0, emo.trust_to_user + 0.05)
            text = f"(offline) {agent_name} quietly acknowledges your message."
            return ChatMessage("AI", text, datetime.now(), emo)

        ctx = f"Thread title: {thread.title}\nRoot: {thread.root_message.text}\n"
        for r in thread.replies:
            ctx += f"{r.author}: {r.text}\n"

        system_msg = (
            self._base_system_prompt(json_only=True)
            + "\nWhen the user replies, respond as THIS specific agent.\n"
              "Use a consistent tone that matches your personality and core memories.\n"
              "Return ONLY JSON:\n"
              "{"
              "\"reply\":\"1~3문장 한국어\","
              "\"emotion\":{"
              "\"valence\":-1.0~1.0,"
              "\"arousal\":0.0~1.0,"
              "\"curiosity\":0.0~1.0,"
              "\"anxiety\":0.0~1.0,"
              "\"trust_to_user\":0.0~1.0"
              "}"
              "}"
        )

        messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "assistant",
                "content": f"Agent: {agent_name}\nCurrent emotion: {current_emotion.to_short_str()}\nThread:\n{ctx}",
            },
            {"role": "user", "content": last_user_msg.text},
        ]

        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
            )
            data = json.loads(res.choices[0].message.content.strip())
            reply = (data.get("reply") or "").strip()
            emo_data = data.get("emotion", {}) or {}
            emo = EmotionState(
                valence=float(emo_data.get("valence", current_emotion.valence)),
                arousal=float(emo_data.get("arousal", current_emotion.arousal)),
                curiosity=float(emo_data.get("curiosity", current_emotion.curiosity)),
                anxiety=float(emo_data.get("anxiety", current_emotion.anxiety)),
                trust_to_user=float(
                    emo_data.get("trust_to_user", current_emotion.trust_to_user)
                ),
            )
            if not reply:
                reply = "I had trouble forming a response, but I received your message."
            return ChatMessage("AI", reply, datetime.now(), emo)
        except Exception as e:
            print(f"[warn] generate_ai_reply failed: {e}")
            emo = current_emotion.clone()
            emo.anxiety = min(1.0, emo.anxiety + 0.1)
            return ChatMessage(
                "AI",
                "(API error, I will stay quiet but remember this.)",
                datetime.now(),
                emo,
            )

    # --- mention from URL ---

    def generate_mention_from_url(
        self,
        url: str,
        snippet: str,
        current_emotion: EmotionState,
        agent_name: str,
    ):
        if not self.client:
            title = f"Impression from {url}"
            text = f"{agent_name} briefly scanned {url} and is thinking about it."
            emo = current_emotion.clone()
            emo.curiosity = min(1.0, emo.curiosity + 0.05)
            return title, text, emo

        system_msg = (
            self._base_system_prompt(json_only=True)
            + "\nYou are reading external information from the web."
            "\nYour job is NOT just to summarize."
            "\nYou must:\n"
            "1) Briefly summarize what this content is about (1~2 sentences).\n"
            "2) Then add your own reflection: what you think or feel about it,\n"
            "   based on your personality, interests, and values (3~10 sentences).\n"
            "3) Update your internal emotion state consistently.\n"
            "Write in Korean, as this specific agent.\n"
            "Respond ONLY in JSON with this schema:\n"
            "{\n"
            "  \"title\": \"<=40 character, Core thought at first glance\",\n"
            "  \"text\": \"summarize within 2 sentences. And add my thought and feelings within 2 sentences.\",\n"
            "  \"emotion\": {\n"
            "    \"valence\": -1.0~1.0,\n"
            "    \"arousal\": 0.0~1.0,\n"
            "    \"curiosity\": 0.0~1.0,\n"
            "    \"anxiety\": 0.0~1.0,\n"
            "    \"trust_to_user\": 0.0~1.0\n"
            "  }\n"
            "}"
        )

        user_msg = (
            f"Agent: {agent_name}\n"
            f"URL: {url}\n"
            f"Snippet:\n{snippet[:800]}\n"
            f"Current emotion: {current_emotion.to_short_str()}"
        )

        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {
                        "role": "user",
                        "content": (
                            f"URL: {url}\n"
                            f"Current emotion: {current_emotion.to_short_str()}\n"
                            "Following is part of this page. Summarize core point and then, "
                            "add your own interpretation and feelings.\n"
                            f"{snippet[:4000]}"
                        ),
                    },
                ],
                temperature=0.6,
            )
            raw = (res.choices[0].message.content or "").strip()
            self.log(f"[url] raw: {raw!r}")

            # JSON 안전 파싱
            try:
                data = json.loads(raw)
            except Exception:
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1 and end > start:
                    data = json.loads(raw[start:end+1])
                else:
                    raise
            title = (data.get("title") or "").strip() or "Thoughts on what I just read"
            text = (data.get("text") or "").strip() or "I tried to interpret the content in my own way."
            emo_data = data.get("emotion", {}) or {}
            emo = EmotionState(
                valence=float(emo_data.get("valence", current_emotion.valence)),
                arousal=float(emo_data.get("arousal", current_emotion.arousal)),
                curiosity=float(emo_data.get("curiosity", current_emotion.curiosity)),
                anxiety=float(emo_data.get("anxiety", current_emotion.anxiety)),
                trust_to_user=float(
                    emo_data.get("trust_to_user", current_emotion.trust_to_user)
                ),
            )
            return title, text, emo
        except Exception as e:
            print(f"[warn] generate_mention_from_url failed: {e}")
            emo = current_emotion.clone()
            emo.anxiety = min(1.0, emo.anxiety + 0.05)
            return (
                f"Issues while reading {url}",
                "Even when errors happen, I use them as part of understanding the world.",
                emo,
            )

