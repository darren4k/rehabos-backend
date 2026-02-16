"""AI Chat - Conversational Interface with Natural Language.

Provides a conversational AI interface for rehabilitation consultation.
The AI acts as a central hub that can navigate users to other features
and provide personalized assistance.

Connects to the DGX Spark LLM for natural conversation.
"""

import json
import hashlib
import re
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field

from rehab_os.llm.base import Message as LLMMessage

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)

# Conversation storage
CHAT_DATA_PATH = Path("data/chat")
CHAT_DATA_PATH.mkdir(parents=True, exist_ok=True)


# ==================
# DATA MODELS
# ==================

class ChatMessage(BaseModel):
    """A single chat message."""
    role: str  # user, assistant, system
    content: str
    timestamp: Optional[str] = None
    voice_used: bool = False


class ChatRequest(BaseModel):
    """Request to send a chat message."""
    message: str
    conversation_id: Optional[str] = None
    discipline: str = "PT"
    voice_response: bool = False


class NavigationAction(BaseModel):
    """Suggested navigation to another section."""
    action: str  # navigate, generate, analyze
    destination: str  # /notes, /program, /analyze, /knowledge
    params: dict = Field(default_factory=dict)
    description: str


class ChatResponse(BaseModel):
    """Response from the chat assistant."""
    conversation_id: str
    message: str
    sources: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    timestamp: str
    navigation: Optional[NavigationAction] = None  # If AI suggests going to another section


class Conversation(BaseModel):
    """A conversation with the AI."""
    conversation_id: str
    messages: list[ChatMessage]
    discipline: str
    created_at: str
    last_activity: str
    context: dict = Field(default_factory=dict)  # Stores extracted context


# ==================
# CONVERSATIONAL AI LOGIC
# ==================

# Natural language patterns for intent detection
INTENT_PATTERNS = {
    "greeting": [
        r"^(hi|hello|hey|good morning|good afternoon|good evening)",
        r"how are you",
        r"what's up"
    ],
    "create_note": [
        r"(create|write|generate|make|help with).*(note|documentation|daily note|progress note|eval)",
        r"document.*(session|treatment|visit)",
        r"(skilled|therapy) note"
    ],
    "create_program": [
        r"(create|generate|make|build).*(program|protocol|plan|hep|home exercise)",
        r"(rehabilitation|rehab|exercise) (program|plan)",
        r"(what|which) exercises"
    ],
    "analyze_patient": [
        r"(analyze|assess|evaluate).*(patient|case|data)",
        r"(enter|input|add).*(patient|medical) (data|info|information)",
        r"patient analysis"
    ],
    "search_knowledge": [
        r"(search|find|look up|what is|tell me about)",
        r"(evidence|research|guidelines) (for|on|about)",
        r"(cpg|clinical practice guideline)"
    ],
    "capability": [
        r"what can you do",
        r"(help|how do|what).*(work|use)",
        r"show.*(features|options)",
        r"capabilities"
    ]
}

# Conversational responses - natural and friendly
CONVERSATIONAL_RESPONSES = {
    "greeting": [
        "Hey there! I'm your RehabOS assistant. I'm here to help you with anything rehabilitation-related. What can I do for you today?",
        "Hi! Great to see you. Whether you need to document a session, create a program, or look up evidence - I'm here to help. What's on your mind?",
        "Hello! I'm ready to assist with documentation, program creation, research, or patient analysis. Just tell me what you need!",
    ],
    "capability": """I can help you with all aspects of rehabilitation practice! Here's what I can do:

**Documentation** - Say "help me write a note" and I'll guide you through creating Medicare-compliant skilled notes. I can even do it through voice!

**Program Creation** - Tell me about your patient and I'll generate a personalized rehab program. Try "create a program for a 65-year-old post TKA."

**Research & Evidence** - Ask me about any condition or intervention. I'll give you evidence-based recommendations with citations.

**Patient Analysis** - Say "analyze a patient" and I'll help you input and review patient data.

Just talk to me naturally - I understand context and can guide you to the right tools!""",
}

# Knowledge base for clinical queries
CLINICAL_KNOWLEDGE = {
    "stroke": {
        "summary": "Stroke rehabilitation focuses on neuroplasticity-driven recovery through intensive, task-specific practice.",
        "key_points": [
            "High-intensity, repetitive task practice is essential",
            "Earlier intervention generally leads to better outcomes",
            "Constraint-induced movement therapy has strong evidence for UE recovery",
            "Body weight supported treadmill training aids early gait recovery"
        ],
        "evidence_level": "Level I for most interventions",
        "outcome_measures": ["Fugl-Meyer", "Berg Balance Scale", "10-Meter Walk Test"],
    },
    "parkinson": {
        "summary": "Parkinson's rehabilitation addresses bradykinesia, balance, and gait through amplitude-focused and cueing strategies.",
        "key_points": [
            "LSVT BIG is the gold standard for amplitude training",
            "Rhythmic auditory cueing helps with gait freezing",
            "High-intensity aerobic exercise may be neuroprotective",
            "Tai Chi reduces falls by 40%"
        ],
        "evidence_level": "Level I for exercise and LSVT",
        "outcome_measures": ["TUG", "Mini-BESTest", "6MWT"],
    },
    "falls": {
        "summary": "Falls prevention requires a multifactorial approach targeting balance, strength, and environmental factors.",
        "key_points": [
            "Exercise is the strongest single intervention",
            "Otago program and Tai Chi have the best evidence",
            "Minimum 3 hours/week of challenging balance work",
            "Address medications, vision, and home hazards"
        ],
        "evidence_level": "Level I for exercise programs",
        "outcome_measures": ["TUG", "Berg", "30-Second Sit-to-Stand"],
    },
    "low back pain": {
        "summary": "Low back pain management emphasizes active approaches, patient education, and avoiding unnecessary imaging.",
        "key_points": [
            "Exercise therapy is first-line treatment",
            "Stay active - bed rest is harmful",
            "McKenzie method for directional preference",
            "Address fear-avoidance beliefs"
        ],
        "evidence_level": "Level I for exercise",
        "outcome_measures": ["Oswestry", "NPRS", "FABQ"],
    },
    "total knee arthroplasty": {
        "summary": "TKA rehabilitation progresses from pain/edema control through ROM, strengthening, and return to function.",
        "key_points": [
            "Early mobilization day of surgery is standard",
            "ROM goal: 0-120 degrees by 12 weeks",
            "Quadriceps activation is priority",
            "Weight bearing as tolerated with assistive device"
        ],
        "evidence_level": "Level I for early mobilization",
        "outcome_measures": ["KOOS", "TUG", "6MWT", "Stair Climb Test"],
    },
}


def detect_intent(message: str) -> tuple[str, float]:
    """Detect user intent from natural language."""
    message_lower = message.lower().strip()

    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, message_lower):
                return intent, 0.9

    # Check for clinical topics
    for condition in CLINICAL_KNOWLEDGE.keys():
        if condition in message_lower:
            return "search_knowledge", 0.8

    return "general", 0.5


def extract_context(message: str, conversation: "Conversation") -> dict:
    """Extract clinical context from the message."""
    context = conversation.context.copy() if conversation.context else {}

    message_lower = message.lower()

    # Extract age
    age_match = re.search(r'(\d{1,3})\s*(year|yr|y/?o)', message_lower)
    if age_match:
        context["age"] = int(age_match.group(1))

    # Extract condition
    for condition in CLINICAL_KNOWLEDGE.keys():
        if condition in message_lower:
            context["condition"] = condition

    # Extract setting
    settings = ["inpatient", "outpatient", "home health", "snf", "acute"]
    for setting in settings:
        if setting in message_lower:
            context["setting"] = setting

    # Extract surgery
    surgeries = ["tka", "tha", "rotator cuff", "acl", "spinal fusion"]
    for surgery in surgeries:
        if surgery in message_lower:
            context["surgery"] = surgery

    return context


def generate_conversational_response(
    message: str,
    intent: str,
    conversation: "Conversation"
) -> tuple[str, list[str], list[str], Optional[NavigationAction]]:
    """Generate a natural, conversational response."""

    sources = []
    suggestions = []
    navigation = None
    context = extract_context(message, conversation)
    conversation.context = context

    # Handle greeting
    if intent == "greeting":
        import random
        response = random.choice(CONVERSATIONAL_RESPONSES["greeting"])
        suggestions = [
            "Help me write a daily note",
            "Create a rehab program for TKA",
            "What's the evidence for stroke rehab?"
        ]
        return response, sources, suggestions, None

    # Handle capability question
    if intent == "capability":
        return CONVERSATIONAL_RESPONSES["capability"], [], [
            "Write a skilled note",
            "Create a program for Parkinson's",
            "Analyze a patient case"
        ], None

    # Handle note creation request
    if intent == "create_note":
        response = """Absolutely! I can help you create a skilled note.

I'll take you to the documentation section where you can:
- Use **Voice Guided Entry** for hands-free documentation
- Or fill in the form manually

The note will be automatically checked for Medicare compliance.

Would you like me to take you there now? Just say "yes" or click the button below."""

        navigation = NavigationAction(
            action="navigate",
            destination="/notes",
            params={},
            description="Go to Skilled Notes"
        )
        suggestions = ["Yes, take me there", "Tell me more about the note features"]
        return response, [], suggestions, navigation

    # Handle program creation request
    if intent == "create_program":
        if context.get("condition") or context.get("surgery"):
            condition = context.get("condition") or context.get("surgery", "").upper()
            age = context.get("age", "")
            age_str = f"{age}-year-old " if age else ""

            response = f"""Great! I can create a personalized rehabilitation program for your {age_str}patient with {condition}.

The more information you provide, the more personalized the program will be. I'll take you to the program generator where you can add:
- Patient demographics
- Specific impairments
- Functional limitations
- Goals and precautions

Should I take you there now?"""

            navigation = NavigationAction(
                action="navigate",
                destination="/program",
                params={"condition": condition, "age": age} if age else {"condition": condition},
                description="Create Rehab Program"
            )
        else:
            response = """I'd love to help create a rehabilitation program!

To make it personalized, tell me a bit about your patient:
- What's their primary diagnosis or surgery?
- Approximate age?
- Any specific goals or limitations?

Or I can take you to the program generator where you can enter all the details."""

            navigation = NavigationAction(
                action="navigate",
                destination="/program",
                params={},
                description="Go to Program Generator"
            )

        suggestions = ["Yes, let's go", "Create a program for stroke", "TKA rehab protocol"]
        return response, [], suggestions, navigation

    # Handle patient analysis request
    if intent == "analyze_patient":
        response = """Sure! I'll take you to the Patient Analysis section where you can:

- **Voice input** - Just talk and I'll extract the medical data
- **Image upload** - Take a photo of records (PHI is automatically removed)
- Enter demographics, medications, labs, and functional status

All your data stays private - we never store any protected health information.

Ready to go?"""

        navigation = NavigationAction(
            action="navigate",
            destination="/analyze",
            params={},
            description="Go to Patient Analysis"
        )
        suggestions = ["Take me there", "How does the PHI protection work?"]
        return response, [], suggestions, navigation

    # Handle knowledge/research queries
    if intent == "search_knowledge":
        # Check for specific conditions
        for condition, knowledge in CLINICAL_KNOWLEDGE.items():
            if condition in message.lower():
                response = f"""**{condition.title()} Rehabilitation**

{knowledge['summary']}

**Key Evidence-Based Approaches:**
"""
                for point in knowledge['key_points']:
                    response += f"\n• {point}"

                response += f"""

**Evidence Level:** {knowledge['evidence_level']}

**Recommended Outcome Measures:** {', '.join(knowledge['outcome_measures'])}

Would you like me to:
- Generate a specific program for this condition?
- Search for the latest research?
- Go deeper on any of these interventions?"""

                sources = ["Clinical Practice Guidelines", "Cochrane Reviews", "APTA CPGs"]
                suggestions = [
                    f"Create a {condition} rehab program",
                    f"What exercises work best for {condition}?",
                    "Show me the outcome measures"
                ]

                navigation = NavigationAction(
                    action="navigate",
                    destination="/knowledge",
                    params={"query": condition},
                    description="Search Knowledge Base"
                )

                return response, sources, suggestions, navigation

        # General knowledge query
        response = f"""I can help you find evidence on that topic!

Let me search the knowledge base for relevant guidelines and research.

In the meantime, you can also explore:
- **Condition-specific protocols** (stroke, Parkinson's, falls, etc.)
- **Intervention evidence** (manual therapy, exercise, modalities)
- **Outcome measure recommendations**

What specific aspect are you most interested in?"""

        navigation = NavigationAction(
            action="navigate",
            destination="/knowledge",
            params={"query": message},
            description="Search Knowledge"
        )
        suggestions = [
            "Stroke rehabilitation evidence",
            "Falls prevention strategies",
            "Low back pain treatment"
        ]
        return response, ["Evidence-based practice database"], suggestions, navigation

    # General conversation - be helpful and guide
    response = f"""I understand you're asking about: "{message[:80]}..."

I'm here to help with all your rehabilitation needs! Here's what I can do right now:

**Quick Actions:**
• Say "write a note" → I'll help with documentation
• Say "create a program" → Personalized rehab programs
• Ask about any condition → Evidence-based guidance
• Say "analyze patient" → Enter and review patient data

Or just keep chatting naturally - I'm learning what you need!

What would be most helpful right now?"""

    suggestions = [
        "Help me document a session",
        "What's the evidence for balance training?",
        "Create a home exercise program"
    ]

    return response, [], suggestions, None


# ==================
# CONVERSATION MANAGEMENT
# ==================

def load_conversation(conversation_id: str) -> Optional[Conversation]:
    """Load a conversation from storage."""
    conv_file = CHAT_DATA_PATH / f"{conversation_id}.json"
    if conv_file.exists():
        with open(conv_file) as f:
            data = json.load(f)
            return Conversation(**data)
    return None


def save_conversation(conversation: Conversation):
    """Save a conversation to storage."""
    conv_file = CHAT_DATA_PATH / f"{conversation.conversation_id}.json"
    with open(conv_file, "w") as f:
        json.dump(conversation.model_dump(), f, indent=2)


def create_conversation(discipline: str = "PT") -> Conversation:
    """Create a new conversation."""
    conv_id = hashlib.sha256(datetime.now().isoformat().encode()).hexdigest()[:16]
    now = datetime.now(timezone.utc).isoformat()

    conversation = Conversation(
        conversation_id=conv_id,
        messages=[],
        discipline=discipline,
        created_at=now,
        last_activity=now,
        context={},
    )

    save_conversation(conversation)
    return conversation


# ==================
# API ENDPOINTS
# ==================

@router.post("/message", response_model=ChatResponse)
async def send_message(request: ChatRequest, req: Request):
    """Send a message and receive a conversational AI response.

    Uses the DGX Spark LLM for natural conversation when available,
    with fallback to rule-based responses.
    """

    # Get or create conversation
    if request.conversation_id:
        conversation = load_conversation(request.conversation_id)
        if not conversation:
            conversation = create_conversation(request.discipline)
    else:
        conversation = create_conversation(request.discipline)

    # Add user message
    now = datetime.now(timezone.utc).isoformat()
    user_message = ChatMessage(
        role="user",
        content=request.message,
        timestamp=now,
    )
    conversation.messages.append(user_message)

    # Detect intent for navigation suggestions
    intent, confidence = detect_intent(request.message)

    # Try to use the LLM router from app state (connected to DGX Spark)
    llm_router = getattr(req.app.state, 'llm_router', None)
    navigation = None
    sources = []
    suggestions = []

    if llm_router:
        try:
            # Build conversation history for the LLM
            llm_messages = [
                LLMMessage(
                    role="system",
                    content=f"""You are RehabOS, an expert AI rehabilitation consultant for {request.discipline} (Physical Therapy, Occupational Therapy, or Speech-Language Pathology).

You are conversational, friendly, and knowledgeable. You help clinicians with:
- Evidence-based treatment recommendations
- Clinical documentation guidance
- Rehabilitation program creation
- Patient assessment and outcome measures

Always provide clinically accurate, evidence-based information. When discussing treatments, mention evidence levels when known.

Keep responses concise but informative. Use bullet points for lists. Be natural and conversational."""
                )
            ]

            # Add recent conversation history (last 10 messages)
            for msg in conversation.messages[-10:]:
                llm_messages.append(LLMMessage(role=msg.role, content=msg.content))

            # Call the LLM
            response = await llm_router.complete(
                messages=llm_messages,
                temperature=0.7,
                max_tokens=1024,
            )

            response_text = response.content
            logger.info(f"LLM response generated via {response.model_id}")

        except Exception as e:
            logger.warning(f"LLM call failed, using fallback: {e}")
            # Fall back to rule-based response
            response_text, sources, suggestions, navigation = generate_conversational_response(
                request.message, intent, conversation
            )
    else:
        # No LLM available, use rule-based responses
        response_text, sources, suggestions, navigation = generate_conversational_response(
            request.message, intent, conversation
        )

    # Generate navigation suggestions based on intent
    if not navigation:
        _, _, _, navigation = generate_conversational_response(
            request.message, intent, conversation
        )
        # Only keep navigation, not override the LLM response
        if navigation and "help" not in request.message.lower():
            # Add navigation hint to response if relevant
            pass

    # Add assistant message
    assistant_message = ChatMessage(
        role="assistant",
        content=response_text,
        timestamp=now,
    )
    conversation.messages.append(assistant_message)

    # Update and save conversation
    conversation.last_activity = now
    save_conversation(conversation)

    return ChatResponse(
        conversation_id=conversation.conversation_id,
        message=response_text,
        sources=sources,
        suggestions=suggestions,
        timestamp=now,
        navigation=navigation,
    )


@router.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a conversation by ID."""
    conversation = load_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@router.get("/conversations")
async def list_conversations(limit: int = 20):
    """List recent conversations."""
    conversations = []

    for conv_file in sorted(CHAT_DATA_PATH.glob("*.json"), reverse=True)[:limit]:
        try:
            with open(conv_file) as f:
                data = json.load(f)
                conversations.append({
                    "conversation_id": data["conversation_id"],
                    "created_at": data["created_at"],
                    "last_activity": data["last_activity"],
                    "message_count": len(data["messages"]),
                    "discipline": data.get("discipline", "PT"),
                })
        except (json.JSONDecodeError, KeyError):
            continue

    return {"conversations": conversations, "total": len(conversations)}


@router.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    conv_file = CHAT_DATA_PATH / f"{conversation_id}.json"
    if conv_file.exists():
        conv_file.unlink()
        return {"status": "deleted", "conversation_id": conversation_id}
    raise HTTPException(status_code=404, detail="Conversation not found")


@router.get("/quick-responses")
async def get_quick_responses():
    """Get suggested quick responses for the chat interface."""
    return {
        "categories": [
            {
                "name": "Documentation",
                "icon": "FileText",
                "suggestions": [
                    "Help me write a daily note",
                    "Create a progress note",
                    "Document an initial evaluation",
                ],
            },
            {
                "name": "Programs",
                "icon": "ClipboardList",
                "suggestions": [
                    "Create a TKA rehab program",
                    "Home exercise program for shoulder",
                    "Balance protocol for falls risk",
                ],
            },
            {
                "name": "Evidence",
                "icon": "Search",
                "suggestions": [
                    "Stroke rehabilitation evidence",
                    "Parkinson's treatment options",
                    "Low back pain guidelines",
                ],
            },
            {
                "name": "Analysis",
                "icon": "UserRound",
                "suggestions": [
                    "Analyze a patient case",
                    "Help interpret lab values",
                    "Medication considerations for therapy",
                ],
            },
        ],
    }


# ==================
# CONVERSATIONAL NOTE-TAKING
# ==================

class ConverseRequest(BaseModel):
    """Request for conversational note-taking with custom system prompt."""
    system_prompt: str
    messages: list[dict]  # [{role, content}, ...]
    max_tokens: int = 150
    temperature: float = 0.3
    model: str = "fast"  # "fast" = 8b for conversation, "smart" = 80b for generation


class ConverseResponse(BaseModel):
    response: str


# Fast model for conversation (2-3s), smart model for SOAP generation (20-30s)
FAST_MODEL = "qwen2.5:14b"
SMART_MODEL = "qwen3-next:80b"
OLLAMA_URL = "http://192.168.68.127:11434"


@router.post("/converse", response_model=ConverseResponse)
async def converse(request: ConverseRequest, req: Request):
    """Conversational chat with custom system prompt. Used for note-taking flow.

    model="fast" → llama3.1:8b (~2-3s, for back-and-forth conversation)
    model="smart" → qwen3-next:80b (~20s, for final SOAP generation)
    """
    import httpx

    model = FAST_MODEL if request.model == "fast" else SMART_MODEL

    ollama_messages = [{"role": "system", "content": request.system_prompt}]
    for m in request.messages[-12:]:
        role = m.get("role", "user")
        if role not in ("user", "assistant", "system"):
            role = "user"
        ollama_messages.append({"role": role, "content": m.get("content", "")})

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model,
                    "stream": False,
                    "options": {
                        "num_predict": request.max_tokens,
                        "temperature": request.temperature,
                    },
                    "messages": ollama_messages,
                },
            )
            if res.status_code == 200:
                data = res.json()
                text = data.get("message", {}).get("content", "")
                # Strip thinking tags
                import re
                text = re.sub(r'<think>[\s\S]*?</think>', '', text).strip()
                return ConverseResponse(response=text)
    except Exception as e:
        logger.error(f"Converse error ({model}): {e}")

    # Fallback to built-in LLM router
    try:
        llm_router = req.app.state.llm_router
        llm_messages = [LLMMessage(role="system", content=request.system_prompt)]
        for m in request.messages[-10:]:
            role = m.get("role", "user")
            if role not in ("user", "assistant", "system"):
                role = "user"
            llm_messages.append(LLMMessage(role=role, content=m.get("content", "")))

        response = await llm_router.generate(
            messages=llm_messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        text = response.content if hasattr(response, 'content') else str(response)
        import re
        text = re.sub(r'<think>[\s\S]*?</think>', '', text).strip()
        return ConverseResponse(response=text)
    except Exception as e:
        logger.error(f"Converse fallback error: {e}")
        return ConverseResponse(response="")
