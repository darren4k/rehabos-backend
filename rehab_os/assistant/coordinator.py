"""Chat coordinator — brain of the AI clinical assistant.

Session-aware agentic assistant with command execution, navigation,
messaging, fax delivery, and multi-step workflow automation.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from rehab_os.assistant.page_context import get_page_context
from rehab_os.llm.base import Message, MessageRole
from rehab_os.llm.router import LLMRouter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session model — persists across pages for a provider
# ---------------------------------------------------------------------------
@dataclass
class AssistantSession:
    """In-memory session state for a single provider."""

    provider_id: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    active_patient_id: str | None = None
    active_patient_name: str | None = None
    active_encounter_id: str | None = None
    current_page: str = "dashboard"
    recent_patients: list[dict] = field(default_factory=list)
    messages: deque = field(default_factory=lambda: deque(maxlen=100))
    tasks_in_progress: list[dict] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_activity: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def set_patient(self, patient_id: str, patient_name: str) -> None:
        self.active_patient_id = patient_id
        self.active_patient_name = patient_name
        self.recent_patients = [p for p in self.recent_patients if p["id"] != patient_id]
        self.recent_patients.insert(0, {"id": patient_id, "name": patient_name})
        self.recent_patients = self.recent_patients[:5]
        self.last_activity = datetime.now(timezone.utc).isoformat()

    def clear_patient(self) -> None:
        self.active_patient_id = None
        self.active_patient_name = None
        self.active_encounter_id = None

    def add_message(self, role: str, content: str, msg_type: str = "text") -> None:
        self.messages.append({
            "role": role,
            "content": content,
            "type": msg_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.last_activity = datetime.now(timezone.utc).isoformat()

    def add_task(self, task_id: str, description: str, status: str = "started") -> None:
        self.tasks_in_progress.append({
            "id": task_id,
            "description": description,
            "status": status,
            "started": datetime.now(timezone.utc).isoformat(),
        })

    def complete_task(self, task_id: str) -> None:
        self.tasks_in_progress = [
            {**t, "status": "complete"} if t["id"] == task_id else t
            for t in self.tasks_in_progress
        ]

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "provider_id": self.provider_id,
            "active_patient_id": self.active_patient_id,
            "active_patient_name": self.active_patient_name,
            "active_encounter_id": self.active_encounter_id,
            "current_page": self.current_page,
            "recent_patients": self.recent_patients,
            "tasks_in_progress": self.tasks_in_progress,
            "message_count": len(self.messages),
            "created_at": self.created_at,
            "last_activity": self.last_activity,
        }


# ---------------------------------------------------------------------------
# Enhanced response
# ---------------------------------------------------------------------------
@dataclass
class AssistantResponse:
    """Structured response from the assistant."""

    type: str  # "text", "suggestion", "patient_list", "note_draft", "billing",
               # "error", "navigation", "form_action", "task_started", "task_complete"
    content: str
    data: Optional[dict[str, Any]] = None
    suggestions: Optional[list[str]] = None
    requires_approval: bool = False
    approval_action: Optional[str] = None
    navigation: Optional[str] = None
    form_action: Optional[dict[str, Any]] = None
    speak: bool = True

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "type": self.type,
            "content": self.content,
            "data": self.data,
            "suggestions": self.suggestions,
            "requires_approval": self.requires_approval,
            "approval_action": self.approval_action,
        }
        if self.navigation is not None:
            d["navigation"] = self.navigation
        if self.form_action is not None:
            d["form_action"] = self.form_action
        if not self.speak:
            d["speak"] = False
        return d


# ---------------------------------------------------------------------------
# Navigation map
# ---------------------------------------------------------------------------
NAVIGATION_MAP: dict[str, str] = {
    "dashboard": "/dashboard",
    "home": "/dashboard",
    "patients": "/patients",
    "patient list": "/patients",
    "caseload": "/patients",
    "scheduling": "/scheduling",
    "schedule": "/scheduling",
    "calendar": "/scheduling",
    "clinic mode": "/clinic-mode",
    "clinic": "/clinic-mode",
    "skilled notes": "/skilled-notes",
    "notes": "/skilled-notes",
    "documentation": "/skilled-notes",
    "intake": "/intake",
    "referral": "/intake",
    "reports": "/reports",
    "analytics": "/reports",
    "billing": "/billing",
    "settings": "/settings",
    "rehab program": "/rehab-program",
    "exercises": "/rehab-program",
    "messages": "/messages",
    "messaging": "/messages",
}


# ---------------------------------------------------------------------------
# Command definition
# ---------------------------------------------------------------------------
@dataclass
class CommandDef:
    """A regex-matched command."""

    name: str
    pattern: re.Pattern
    handler_name: str
    description: str


def _cmd(name: str, pattern: str, handler: str, description: str) -> CommandDef:
    return CommandDef(
        name=name,
        pattern=re.compile(pattern, re.IGNORECASE),
        handler_name=handler,
        description=description,
    )


# Original commands + new agentic commands
COMMANDS: list[CommandDef] = [
    # --- Communication commands ---
    _cmd("send_message", r"send\s+(?:a\s+)?message\s+to\s+(.+)", "_cmd_send_message", "Send message to care team"),
    _cmd("send_fax", r"(?:send|fax)\s+(?:a\s+)?(?:fax|note)\s+to\s+(.+)", "_cmd_send_fax", "Send fax"),
    _cmd("send_to_md", r"send\s+(?:note|summary|report)\s+to\s+(?:dr\.?|doctor)\s+(.+)", "_cmd_send_to_md", "Send to physician"),

    # --- Patient management ---
    _cmd("set_patient", r"(?:switch|select|set|focus)\s+(?:to\s+)?patient\s+(.+)", "_cmd_set_patient", "Set active patient"),
    _cmd("patient_timeline", r"(?:show|get)\s+(?:patient\s+)?timeline", "_cmd_patient_timeline", "Patient timeline"),
    _cmd("add_diagnosis", r"add\s+diagnosis\s+(.+)", "_cmd_add_diagnosis", "Add diagnosis"),
    _cmd("add_goal", r"add\s+goal\s+(.+)", "_cmd_add_goal", "Add SMART goal"),

    # --- Documentation ---
    _cmd("dictate_note", r"dictate\s+(?:a\s+)?(?:note|soap)", "_cmd_dictate_note", "Start SOAP dictation"),
    _cmd("sign_note", r"sign\s+(?:note|notes)\s+(?:for\s+)?(.+)?", "_cmd_sign_note", "Sign notes"),

    # --- Scheduling ---
    _cmd("schedule_visit", r"schedule\s+(?:a\s+)?visit\s+(?:for\s+)?(.+)?", "_cmd_schedule_visit", "Schedule visit"),
    _cmd("cancel_visit", r"cancel\s+(?:visit|appointment)\s+(.+)?", "_cmd_cancel_visit", "Cancel visit"),

    # --- Navigation ---
    _cmd("navigate", r"(?:go\s+to|open|show\s+me|navigate\s+to|take\s+me\s+to)\s+(.+)", "_cmd_navigate", "Navigate to page"),

    # --- Workflow automation ---
    _cmd("onboard_patient", r"onboard\s+(?:patient\s+)?(.+)", "_cmd_onboard_patient", "Onboard new patient"),
    _cmd("start_session", r"start\s+(?:a\s+)?session\s+(?:with|for)\s+(.+)", "_cmd_start_session", "Start patient session"),
    _cmd("complete_visit", r"(?:complete|finish|close)\s+(?:this\s+)?(?:visit|session)", "_cmd_complete_visit", "Complete visit"),
    _cmd("run_compliance_check", r"(?:run|check)\s+compliance", "_cmd_run_compliance_check", "Run compliance check"),

    # --- Query commands ---
    _cmd("who_is", r"who\s+is\s+(?:my\s+)?(?:current\s+)?patient", "_cmd_who_is", "Current patient info"),
    _cmd("what_page", r"what\s+page|where\s+am\s+i", "_cmd_what_page", "Current page"),
    _cmd("my_schedule", r"(?:my|today'?s?)\s+schedule", "_cmd_my_schedule", "Today's schedule"),

    # --- Original commands ---
    _cmd("show_patients", r"(show|list|my)\s*(patients?|caseload)", "_cmd_show_patients", "List my patients"),
    _cmd("patient_summary", r"(summarize|summary)\s*(this\s*)?(patient)?", "_cmd_patient_summary", "Summarize this patient"),
    _cmd("suggest_goals", r"suggest\s*(smart\s*)?goals?", "_cmd_suggest_goals", "Suggest SMART goals"),
    _cmd("suggest_interventions", r"suggest\s*(evidence[- ]based\s*)?interventions?", "_cmd_suggest_interventions", "Suggest interventions"),
    _cmd("improve_note", r"(improve|enhance|polish)\s*(my\s*)?(soap\s*)?(note|documentation)", "_cmd_improve_note", "Improve SOAP note"),
    _cmd("suggest_cpt", r"suggest\s*(billing\s*|cpt\s*)(codes?)?", "_cmd_suggest_cpt", "Suggest CPT codes"),
    _cmd("check_authorization", r"check\s*(insurance\s*)?auth(orization)?", "_cmd_check_authorization", "Check insurance auth"),
    _cmd("draft_care_plan", r"draft\s*(a\s*)?care\s*plan", "_cmd_draft_care_plan", "Draft a care plan"),
    _cmd("draft_progress_note", r"draft\s*(a\s*)?progress\s*note", "_cmd_draft_progress_note", "Draft progress note"),
    _cmd("draft_discharge_summary", r"draft\s*(a\s*)?discharge\s*(summary|note)", "_cmd_draft_discharge_summary", "Draft discharge summary"),
    _cmd("suggest_hep", r"suggest\s*(a\s*)?(hep|home\s*exercise)", "_cmd_suggest_hep", "Suggest HEP"),
    _cmd("red_flag_check", r"(red\s*flag|safety)\s*(check|screen)", "_cmd_red_flag_check", "Run safety screening"),
    _cmd("show_schedule", r"(show|my|today.s?)\s*schedule", "_cmd_show_schedule", "Show today's schedule"),
    _cmd("pending_notes", r"(pending|unsigned)\s*notes?", "_cmd_pending_notes", "Show pending notes"),
    _cmd("overdue_evals", r"overdue\s*(eval(uation)?s?|re-?eval)", "_cmd_overdue_evals", "Show overdue evals"),
    _cmd("patient_history", r"(patient\s*)?(encounter\s*)?history", "_cmd_patient_history", "Show encounter history"),
    _cmd("evidence_search", r"(search|find)\s*(clinical\s*)?evidence", "_cmd_evidence_search", "Search clinical evidence"),
    _cmd("mcid_check", r"(mcid|minimal\s*clinically\s*important)", "_cmd_mcid_check", "Check MCID thresholds"),
    _cmd("next_visit_plan", r"(plan|prepare)\s*(next|upcoming)\s*visit", "_cmd_next_visit_plan", "Plan next visit"),
    _cmd("documentation_tips", r"documentation\s*(tips?|improvements?|suggestions?)", "_cmd_documentation_tips", "Documentation tips"),
    _cmd("skilled_justification", r"skilled\s*(care\s*)?justification", "_cmd_skilled_justification", "Skilled care justification"),
    _cmd("eight_min_rule", r"(8[- ]?min(ute)?|billing\s*units?)\s*(rule|calc)", "_cmd_eight_min_rule", "Calculate billing units"),
    _cmd("functional_progress", r"functional\s*progress", "_cmd_functional_progress", "Functional progress summary"),
    _cmd("discharge_criteria", r"discharge\s*(criteria|readiness|ready)", "_cmd_discharge_criteria", "Check discharge criteria"),
    _cmd("peer_comparison", r"(peer|benchmark)\s*compar(ison|e)", "_cmd_peer_comparison", "Compare to benchmarks"),
]


# ---------------------------------------------------------------------------
# Session store (in-memory, keyed by provider_id)
# ---------------------------------------------------------------------------
_sessions: dict[str, AssistantSession] = {}


def get_session(provider_id: str) -> AssistantSession:
    """Get or create a session for the given provider."""
    if provider_id not in _sessions:
        _sessions[provider_id] = AssistantSession(provider_id=provider_id)
    return _sessions[provider_id]


def get_all_sessions() -> dict[str, AssistantSession]:
    """Return the session store (for testing/inspection)."""
    return _sessions


class ChatCoordinator:
    """Routes assistant messages to command handlers or LLM fallback.

    Maintains per-provider sessions that persist across page navigations.
    """

    def __init__(self, llm: LLMRouter):
        self.llm = llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process(
        self,
        provider_id: str,
        message: str,
        page_context: str = "dashboard",
        patient_id: Optional[str] = None,
        patient_name: Optional[str] = None,
        conversation_history: Optional[list[dict]] = None,
    ) -> AssistantResponse:
        """Process a user message and return a response.

        1. Load/update session.
        2. Try regex command match (new agentic commands first).
        3. Try navigation detection.
        4. Fall back to LLM with full session context.
        """
        session = get_session(provider_id)
        session.current_page = page_context

        # If patient info provided, update session
        if patient_id:
            name = patient_name or f"Patient {patient_id[:8]}"
            session.set_patient(patient_id, name)

        ctx = {
            "provider_id": provider_id,
            "page": page_context,
            "patient_id": session.active_patient_id,
            "patient_name": session.active_patient_name,
            "page_info": get_page_context(page_context),
            "history": conversation_history or list(session.messages),
            "session": session,
        }

        # Record user message in session
        session.add_message("user", message)

        # Try command match
        for cmd in COMMANDS:
            m = cmd.pattern.search(message)
            if m:
                handler = getattr(self, cmd.handler_name, None)
                if handler:
                    logger.info("Command matched: %s", cmd.name)
                    result = await handler(message, ctx, match=m)
                    session.add_message("assistant", result.content, result.type)
                    return result

        # Try implicit navigation ("go to patients", "open scheduling")
        nav_result = self._try_navigation(message)
        if nav_result:
            session.add_message("assistant", nav_result.content, nav_result.type)
            return nav_result

        # Auto-detect patient mentions and update session
        self._detect_patient_mention(message, session)

        # LLM fallback with session context
        result = await self._llm_fallback(message, ctx)
        session.add_message("assistant", result.content, result.type)
        return result

    # ------------------------------------------------------------------
    # Navigation detection
    # ------------------------------------------------------------------

    def _try_navigation(self, message: str) -> Optional[AssistantResponse]:
        """Check if the message is a navigation request."""
        msg_lower = message.lower().strip()
        for key, path in NAVIGATION_MAP.items():
            if key in msg_lower:
                return AssistantResponse(
                    type="navigation",
                    content=f"Taking you to {key.title()}.",
                    navigation=path,
                    suggestions=["what_page"],
                )
        return None

    # ------------------------------------------------------------------
    # Patient mention detection
    # ------------------------------------------------------------------

    def _detect_patient_mention(self, message: str, session: AssistantSession) -> None:
        """Try to detect patient name mentions and update session context."""
        # Check recent patients for name match
        msg_lower = message.lower()
        for patient in session.recent_patients:
            name_lower = patient["name"].lower()
            # Check if any part of the patient name appears in the message
            parts = name_lower.split()
            if any(part in msg_lower for part in parts if len(part) > 2):
                session.set_patient(patient["id"], patient["name"])
                logger.info("Auto-detected patient mention: %s", patient["name"])
                return

    # ------------------------------------------------------------------
    # LLM fallback with session context
    # ------------------------------------------------------------------

    def _build_system_prompt(self, session: AssistantSession, page: str) -> str:
        """Build context-rich system prompt."""
        page_info = get_page_context(page)
        recent_list = ", ".join(p["name"] for p in session.recent_patients) or "None"
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        return (
            "You are a clinical documentation assistant for rehabilitation professionals "
            "(PT/OT/SLP).\n\n"
            "You can:\n"
            "- Help with SOAP notes, care plans, billing codes, and evidence-based treatment\n"
            "- Navigate the app (suggest \"go to patients\" or \"open scheduling\")\n"
            "- Send messages and faxes to physicians\n"
            "- Schedule and manage visits\n"
            "- Run compliance checks\n"
            "- Automate onboarding workflows\n\n"
            "You NEVER make autonomous clinical decisions -- you suggest and the clinician "
            "approves. Be concise. Use clinical terminology. Cite evidence when available.\n\n"
            f"Current context:\n"
            f"- Page: {page_info['label']}\n"
            f"- Active patient: {session.active_patient_name or 'None selected'}\n"
            f"- Recent patients: {recent_list}\n"
            f"- Time: {now}\n"
        )

    async def _llm_fallback(self, message: str, ctx: dict) -> AssistantResponse:
        """Send message to the LLM with full session context."""
        session: AssistantSession = ctx["session"]
        system = self._build_system_prompt(session, ctx["page"])

        messages: list[Message] = [Message(role=MessageRole.SYSTEM, content=system)]

        # Inject recent conversation from session (last 10 turns)
        for turn in list(session.messages)[-10:]:
            role = MessageRole.USER if turn["role"] == "user" else MessageRole.ASSISTANT
            messages.append(Message(role=role, content=turn["content"]))

        messages.append(Message(role=MessageRole.USER, content=message))

        try:
            resp = await self.llm.complete(messages, temperature=0.4, max_tokens=2048)
            page_info = ctx["page_info"]
            return AssistantResponse(
                type="text",
                content=resp.content,
                suggestions=page_info.get("commands", [])[:3],
            )
        except Exception as e:
            logger.exception("LLM fallback failed")
            return AssistantResponse(
                type="error",
                content=f"Sorry, I could not process that request: {e}",
            )

    # ------------------------------------------------------------------
    # LLM helper for clinical generation
    # ------------------------------------------------------------------

    async def _generate(self, prompt: str, context_label: str) -> str:
        """Call the LLM with a clinical prompt and return text."""
        messages = [
            Message(role=MessageRole.SYSTEM, content=(
                "You are a clinical documentation assistant for rehabilitation professionals "
                "(PT/OT/SLP). You help with SOAP notes, care plans, billing, and evidence-based "
                "treatment. You NEVER make autonomous clinical decisions -- you suggest and the "
                "clinician approves. Be concise, use clinical terminology, cite evidence when "
                "available."
            )),
            Message(role=MessageRole.USER, content=prompt),
        ]
        resp = await self.llm.complete(messages, temperature=0.3, max_tokens=2048)
        return resp.content

    # ------------------------------------------------------------------
    # New agentic command handlers
    # ------------------------------------------------------------------

    async def _cmd_send_message(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        recipient = match.group(1).strip() if match else "unknown"
        return AssistantResponse(
            type="task_started",
            content=f"Message queued for {recipient}. What would you like to say?",
            data={"action": "send_message", "recipient": recipient, "provider_id": ctx["provider_id"]},
            form_action={"action": "open_dialog", "type": "compose_message", "prefill": {"to": recipient}},
            suggestions=["send_fax", "send_to_md"],
        )

    async def _cmd_send_fax(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        recipient = match.group(1).strip() if match else "unknown"
        session: AssistantSession = ctx["session"]
        task_id = str(uuid.uuid4())
        session.add_task(task_id, f"Fax to {recipient}")
        return AssistantResponse(
            type="task_started",
            content=f"Fax queued to {recipient}. I'll prepare a clinical summary to send.",
            data={
                "action": "send_fax",
                "task_id": task_id,
                "recipient": recipient,
                "patient_id": ctx.get("patient_id"),
                "docpilot_endpoint": "http://localhost:3847/api/v1/docpilot/fax",
            },
            requires_approval=True,
            approval_action="confirm_fax",
            suggestions=["send_to_md", "patient_summary"],
        )

    async def _cmd_send_to_md(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        doctor = match.group(1).strip() if match else "physician"
        patient_name = ctx.get("patient_name") or "current patient"
        text = await self._generate(
            f"Generate a brief clinical summary for Dr. {doctor} regarding {patient_name}. "
            "Include: diagnosis, current status, treatment plan, and reason for communication.",
            "send_to_md",
        )
        return AssistantResponse(
            type="note_draft",
            content=f"**Summary for Dr. {doctor}:**\n\n{text}",
            data={"action": "send_to_md", "doctor": doctor, "patient_id": ctx.get("patient_id")},
            requires_approval=True,
            approval_action="confirm_send_md",
            suggestions=["send_fax", "send_message"],
        )

    async def _cmd_set_patient(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        patient_query = match.group(1).strip() if match else ""
        session: AssistantSession = ctx["session"]
        # Check recent patients first
        for patient in session.recent_patients:
            if patient_query.lower() in patient["name"].lower():
                session.set_patient(patient["id"], patient["name"])
                return AssistantResponse(
                    type="text",
                    content=f"Switched to **{patient['name']}**.",
                    data={"action": "set_patient", "patient_id": patient["id"], "patient_name": patient["name"]},
                    suggestions=["patient_summary", "patient_timeline", "draft_care_plan"],
                )
        # Not in recent -- return search action for frontend
        return AssistantResponse(
            type="patient_list",
            content=f"Searching for patient: **{patient_query}**...",
            data={"action": "search_patient", "query": patient_query},
            suggestions=["show_patients"],
        )

    async def _cmd_patient_timeline(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        if not ctx.get("patient_id"):
            return AssistantResponse(
                type="error",
                content="No patient selected. Say 'set patient [name]' first.",
                suggestions=["set_patient", "show_patients"],
            )
        return AssistantResponse(
            type="text",
            content=f"Loading timeline for **{ctx.get('patient_name', 'patient')}**...",
            data={"action": "patient_timeline", "patient_id": ctx["patient_id"]},
            navigation="/patients/" + ctx["patient_id"] + "/timeline",
            suggestions=["patient_summary", "functional_progress"],
        )

    async def _cmd_add_diagnosis(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        diagnosis = match.group(1).strip() if match else ""
        if not ctx.get("patient_id"):
            return AssistantResponse(
                type="error",
                content="No patient selected. Say 'set patient [name]' first.",
                suggestions=["set_patient"],
            )
        return AssistantResponse(
            type="form_action",
            content=f"Adding diagnosis: **{diagnosis}** to {ctx.get('patient_name', 'patient')}.",
            data={"action": "add_diagnosis", "diagnosis": diagnosis, "patient_id": ctx["patient_id"]},
            requires_approval=True,
            approval_action="confirm_diagnosis",
            form_action={"action": "open_dialog", "type": "add_diagnosis", "prefill": {"diagnosis": diagnosis}},
            suggestions=["add_goal", "draft_care_plan"],
        )

    async def _cmd_add_goal(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        goal_text = match.group(1).strip() if match else ""
        if not ctx.get("patient_id"):
            return AssistantResponse(
                type="error",
                content="No patient selected. Say 'set patient [name]' first.",
                suggestions=["set_patient"],
            )
        text = await self._generate(
            f"Convert this into a SMART rehabilitation goal: {goal_text}\n"
            "Format: Goal | Measure | Target | Timeline",
            "add_goal",
        )
        return AssistantResponse(
            type="suggestion",
            content=text,
            data={"action": "add_goal", "patient_id": ctx["patient_id"], "raw_goal": goal_text},
            requires_approval=True,
            approval_action="confirm_goal",
            suggestions=["suggest_goals", "suggest_interventions"],
        )

    async def _cmd_dictate_note(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        if not ctx.get("patient_id"):
            return AssistantResponse(
                type="error",
                content="No patient selected. Say 'set patient [name]' first.",
                suggestions=["set_patient"],
            )
        return AssistantResponse(
            type="form_action",
            content=(
                "Starting SOAP dictation for **" + (ctx.get("patient_name") or "patient") + "**.\n\n"
                "Tell me the **Subjective** findings first (patient report, pain level, complaints)."
            ),
            data={"action": "dictate_note", "patient_id": ctx["patient_id"], "section": "subjective"},
            form_action={"action": "start_dictation", "type": "soap", "patient_id": ctx["patient_id"]},
            suggestions=["improve_note", "suggest_cpt"],
        )

    async def _cmd_sign_note(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        target = match.group(1).strip() if match and match.group(1) else "all unsigned"
        return AssistantResponse(
            type="task_started",
            content=f"Preparing to sign {target} notes...",
            data={"action": "sign_notes", "target": target, "provider_id": ctx["provider_id"]},
            requires_approval=True,
            approval_action="confirm_sign",
            suggestions=["pending_notes", "show_schedule"],
        )

    async def _cmd_schedule_visit(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        target = match.group(1).strip() if match and match.group(1) else ctx.get("patient_name")
        if not target and not ctx.get("patient_id"):
            return AssistantResponse(
                type="error",
                content="No patient specified. Say 'schedule visit for [patient name]'.",
                suggestions=["set_patient", "show_patients"],
            )
        return AssistantResponse(
            type="form_action",
            content=f"Opening scheduler for **{target or 'patient'}**.",
            data={"action": "schedule_visit", "patient_id": ctx.get("patient_id"), "patient_name": target},
            form_action={"action": "open_dialog", "type": "schedule_visit", "prefill": {"patient": target}},
            requires_approval=True,
            approval_action="confirm_schedule",
            navigation="/scheduling",
            suggestions=["my_schedule", "cancel_visit"],
        )

    async def _cmd_cancel_visit(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        target = match.group(1).strip() if match and match.group(1) else ""
        return AssistantResponse(
            type="task_started",
            content=f"Looking up appointment to cancel: **{target or 'next visit'}**.",
            data={"action": "cancel_visit", "query": target, "provider_id": ctx["provider_id"]},
            requires_approval=True,
            approval_action="confirm_cancel",
            suggestions=["my_schedule", "schedule_visit"],
        )

    async def _cmd_navigate(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        destination = match.group(1).strip().lower() if match else ""
        path = NAVIGATION_MAP.get(destination)
        if not path:
            # Fuzzy match
            for key, p in NAVIGATION_MAP.items():
                if destination in key or key in destination:
                    path = p
                    destination = key
                    break
        if path:
            session: AssistantSession = ctx["session"]
            session.current_page = destination.replace(" ", "_")
            return AssistantResponse(
                type="navigation",
                content=f"Taking you to **{destination.title()}**.",
                navigation=path,
                suggestions=["what_page"],
            )
        return AssistantResponse(
            type="text",
            content=f"I don't know where '{destination}' is. Try: {', '.join(NAVIGATION_MAP.keys())}",
            suggestions=["navigate"],
        )

    async def _cmd_onboard_patient(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        patient_name = match.group(1).strip() if match else "new patient"
        task_id = str(uuid.uuid4())
        session: AssistantSession = ctx["session"]
        session.add_task(task_id, f"Onboard {patient_name}")
        return AssistantResponse(
            type="task_started",
            content=(
                f"Starting onboarding for **{patient_name}**.\n\n"
                "I'll:\n"
                "1. Create the patient record\n"
                "2. Navigate to intake\n"
                "3. Prompt you for clinical details\n\n"
                "Opening new patient form..."
            ),
            data={"action": "onboard_patient", "task_id": task_id, "patient_name": patient_name},
            form_action={"action": "open_dialog", "type": "new_patient", "prefill": {"name": patient_name}},
            navigation="/intake",
            suggestions=["red_flag_check", "draft_care_plan"],
        )

    async def _cmd_start_session(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        patient_query = match.group(1).strip() if match else ""
        session: AssistantSession = ctx["session"]
        # Check recent patients
        for patient in session.recent_patients:
            if patient_query.lower() in patient["name"].lower():
                session.set_patient(patient["id"], patient["name"])
                return AssistantResponse(
                    type="navigation",
                    content=f"Starting session with **{patient['name']}**. Opening Clinic Mode.",
                    data={"action": "start_session", "patient_id": patient["id"]},
                    navigation="/clinic-mode",
                    suggestions=["dictate_note", "patient_summary", "suggest_interventions"],
                )
        return AssistantResponse(
            type="patient_list",
            content=f"Looking up **{patient_query}**...",
            data={"action": "search_patient", "query": patient_query, "then": "start_session"},
            suggestions=["show_patients"],
        )

    async def _cmd_complete_visit(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        if not ctx.get("patient_id"):
            return AssistantResponse(
                type="error",
                content="No active visit. Start a session first.",
                suggestions=["start_session", "show_patients"],
            )
        task_id = str(uuid.uuid4())
        session: AssistantSession = ctx["session"]
        session.add_task(task_id, "Complete visit")
        text = await self._generate(
            "Generate a brief SOAP note summary template for visit completion. "
            "Include placeholder sections for S, O, A, P.",
            "complete_visit",
        )
        return AssistantResponse(
            type="task_started",
            content=(
                f"Completing visit for **{ctx.get('patient_name', 'patient')}**.\n\n"
                f"**Draft SOAP:**\n{text}\n\n"
                "Review and approve, then I'll navigate to Skilled Notes for sign-off."
            ),
            data={"action": "complete_visit", "task_id": task_id, "patient_id": ctx["patient_id"]},
            requires_approval=True,
            approval_action="confirm_complete_visit",
            suggestions=["sign_note", "suggest_cpt"],
        )

    async def _cmd_run_compliance_check(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        return AssistantResponse(
            type="task_started",
            content=(
                "Running compliance check...\n\n"
                "Checking:\n"
                "- Unsigned notes\n"
                "- Expiring authorizations (next 7 days)\n"
                "- Overdue re-evaluations\n"
                "- Missing documentation\n"
                "- Billing discrepancies"
            ),
            data={"action": "compliance_check", "provider_id": ctx["provider_id"]},
            suggestions=["pending_notes", "overdue_evals", "show_schedule"],
        )

    async def _cmd_who_is(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        session: AssistantSession = ctx["session"]
        if session.active_patient_id:
            return AssistantResponse(
                type="text",
                content=(
                    f"Current patient: **{session.active_patient_name}**\n"
                    f"ID: `{session.active_patient_id}`"
                ),
                data={"patient_id": session.active_patient_id, "patient_name": session.active_patient_name},
                suggestions=["patient_summary", "patient_timeline", "set_patient"],
            )
        return AssistantResponse(
            type="text",
            content="No patient currently selected. Say 'set patient [name]' or navigate to a patient record.",
            suggestions=["set_patient", "show_patients"],
        )

    async def _cmd_what_page(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        page_info = ctx["page_info"]
        session: AssistantSession = ctx["session"]
        return AssistantResponse(
            type="text",
            content=(
                f"You're on the **{page_info['label']}** page."
                + (f"\nActive patient: **{session.active_patient_name}**" if session.active_patient_name else "")
            ),
            suggestions=page_info.get("commands", [])[:3],
            speak=False,
        )

    async def _cmd_my_schedule(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        return AssistantResponse(
            type="text",
            content="Loading today's schedule...",
            data={"action": "load_schedule", "provider_id": ctx["provider_id"]},
            suggestions=["pending_notes", "schedule_visit"],
        )

    # ------------------------------------------------------------------
    # Original command handlers (updated to accept match kwarg)
    # ------------------------------------------------------------------

    async def _cmd_show_patients(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        return AssistantResponse(
            type="patient_list",
            content="Fetching your active caseload...",
            data={"action": "list_patients", "provider_id": ctx["provider_id"]},
            suggestions=["patient_summary", "overdue_evals", "pending_notes"],
        )

    async def _cmd_patient_summary(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        if not ctx.get("patient_id"):
            return AssistantResponse(
                type="error",
                content="No patient selected. Navigate to a patient record first.",
                suggestions=["show_patients"],
            )
        text = await self._generate(
            f"Provide a concise clinical summary for patient {ctx['patient_id']}. "
            "Include diagnosis, current functional status, goals progress, and visit count.",
            "patient_summary",
        )
        return AssistantResponse(
            type="suggestion",
            content=text,
            data={"patient_id": ctx["patient_id"]},
            suggestions=["suggest_goals", "patient_history", "draft_care_plan"],
        )

    async def _cmd_suggest_goals(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        text = await self._generate(
            "Suggest 3-5 SMART goals for a rehabilitation patient. "
            "Include short-term (2 weeks) and long-term (discharge) goals. "
            "Format: Goal statement | Measure | Target | Timeline.",
            "suggest_goals",
        )
        return AssistantResponse(
            type="suggestion",
            content=text,
            requires_approval=True,
            approval_action="apply_goals",
            suggestions=["suggest_interventions", "draft_care_plan"],
        )

    async def _cmd_suggest_interventions(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        text = await self._generate(
            "Suggest 4-6 evidence-based rehabilitation interventions. "
            "Include: intervention name, parameters (sets/reps/duration), "
            "evidence level, and target impairment.",
            "suggest_interventions",
        )
        return AssistantResponse(
            type="suggestion",
            content=text,
            requires_approval=True,
            approval_action="apply_interventions",
            suggestions=["suggest_cpt", "suggest_hep"],
        )

    async def _cmd_improve_note(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        text = await self._generate(
            "Review and improve the following clinical note for medical necessity, "
            "skilled care justification, and proper documentation standards. "
            "Suggest specific wording improvements.\n\n"
            f"Context: {msg}",
            "improve_note",
        )
        return AssistantResponse(
            type="note_draft",
            content=text,
            requires_approval=True,
            approval_action="apply_note_edits",
            suggestions=["suggest_cpt", "documentation_tips"],
        )

    async def _cmd_suggest_cpt(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        text = await self._generate(
            "Suggest appropriate CPT codes for rehabilitation services. "
            "Include: code, description, typical time, and documentation requirements. "
            "Consider evaluation codes (97161-97163), therapeutic exercise (97110), "
            "therapeutic activities (97530), neuromuscular re-ed (97112), "
            "manual therapy (97140), and gait training (97116).",
            "suggest_cpt",
        )
        return AssistantResponse(
            type="billing",
            content=text,
            suggestions=["eight_min_rule", "documentation_tips"],
        )

    async def _cmd_check_authorization(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        return AssistantResponse(
            type="text",
            content="Checking authorization status...",
            data={
                "action": "check_auth",
                "patient_id": ctx.get("patient_id"),
                "provider_id": ctx["provider_id"],
            },
            suggestions=["show_schedule", "pending_notes"],
        )

    async def _cmd_draft_care_plan(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        text = await self._generate(
            "Draft a comprehensive rehabilitation care plan. Include:\n"
            "1. Problem list with ICD-10 codes\n"
            "2. Short-term goals (2-4 weeks)\n"
            "3. Long-term goals (discharge)\n"
            "4. Treatment frequency and duration\n"
            "5. Interventions with rationale\n"
            "6. Precautions and contraindications\n"
            "7. Discharge criteria",
            "draft_care_plan",
        )
        return AssistantResponse(
            type="note_draft",
            content=text,
            requires_approval=True,
            approval_action="apply_care_plan",
            suggestions=["suggest_goals", "suggest_interventions"],
        )

    async def _cmd_draft_progress_note(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        text = await self._generate(
            "Draft a SOAP progress note for a rehabilitation visit. Include:\n"
            "- S: Patient subjective report (pain, function, compliance)\n"
            "- O: Objective measures, interventions performed with parameters\n"
            "- A: Assessment of progress toward goals, clinical reasoning\n"
            "- P: Plan for next visit, HEP updates, coordination needs",
            "draft_progress_note",
        )
        return AssistantResponse(
            type="note_draft",
            content=text,
            requires_approval=True,
            approval_action="apply_progress_note",
            suggestions=["suggest_cpt", "improve_note"],
        )

    async def _cmd_draft_discharge_summary(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        text = await self._generate(
            "Draft a discharge summary. Include:\n"
            "- Reason for discharge\n"
            "- Goals met / not met with objective data\n"
            "- Functional status at discharge vs admission\n"
            "- HEP provided\n"
            "- Follow-up recommendations\n"
            "- Physician notification if needed",
            "draft_discharge_summary",
        )
        return AssistantResponse(
            type="note_draft",
            content=text,
            requires_approval=True,
            approval_action="apply_discharge_summary",
            suggestions=["functional_progress", "discharge_criteria"],
        )

    async def _cmd_suggest_hep(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        text = await self._generate(
            "Suggest a home exercise program (HEP). Include:\n"
            "- Exercise name and description\n"
            "- Parameters (sets, reps, hold time, frequency)\n"
            "- Precautions\n"
            "- Progression criteria\n"
            "Format as a patient-friendly handout.",
            "suggest_hep",
        )
        return AssistantResponse(
            type="suggestion",
            content=text,
            requires_approval=True,
            approval_action="apply_hep",
            suggestions=["suggest_interventions", "suggest_goals"],
        )

    async def _cmd_red_flag_check(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        text = await self._generate(
            "Run a red flag safety screening checklist for a rehabilitation patient. "
            "Include screens for: cauda equina, fracture, infection, malignancy, "
            "vascular compromise, cardiac red flags, neurological deterioration, "
            "and fall risk. Format as a checklist.",
            "red_flag_check",
        )
        return AssistantResponse(
            type="suggestion",
            content=text,
            requires_approval=True,
            approval_action="confirm_safety_screen",
            suggestions=["suggest_goals", "draft_care_plan"],
        )

    async def _cmd_show_schedule(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        return AssistantResponse(
            type="text",
            content="Loading today's schedule...",
            data={"action": "load_schedule", "provider_id": ctx["provider_id"]},
            suggestions=["pending_notes", "overdue_evals"],
        )

    async def _cmd_pending_notes(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        return AssistantResponse(
            type="text",
            content="Checking for unsigned and pending notes...",
            data={"action": "pending_notes", "provider_id": ctx["provider_id"]},
            suggestions=["show_schedule", "overdue_evals"],
        )

    async def _cmd_overdue_evals(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        return AssistantResponse(
            type="text",
            content="Checking for overdue evaluations and re-evaluations...",
            data={"action": "overdue_evals", "provider_id": ctx["provider_id"]},
            suggestions=["show_schedule", "pending_notes"],
        )

    async def _cmd_patient_history(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        if not ctx.get("patient_id"):
            return AssistantResponse(
                type="error",
                content="No patient selected. Navigate to a patient record first.",
                suggestions=["show_patients"],
            )
        return AssistantResponse(
            type="text",
            content="Loading encounter history...",
            data={
                "action": "patient_history",
                "patient_id": ctx["patient_id"],
            },
            suggestions=["patient_summary", "functional_progress", "mcid_check"],
        )

    async def _cmd_evidence_search(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        topic = re.sub(
            r"(search|find)\s*(clinical\s*)?evidence\s*(for|on|about)?\s*",
            "",
            msg,
            flags=re.IGNORECASE,
        ).strip() or "rehabilitation best practices"
        text = await self._generate(
            f"Search for clinical evidence on: {topic}\n"
            "Summarize the top findings with citations (author, year, journal). "
            "Rate evidence level (I-V).",
            "evidence_search",
        )
        return AssistantResponse(
            type="text",
            content=text,
            suggestions=["suggest_interventions", "suggest_goals"],
        )

    async def _cmd_mcid_check(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        text = await self._generate(
            "List common MCID (Minimal Clinically Important Difference) thresholds "
            "for rehabilitation outcome measures:\n"
            "- NPRS (pain): MCID = 2 points\n"
            "- ODI (lumbar): MCID = 6-12 points\n"
            "- LEFS (lower extremity): MCID = 9 points\n"
            "- DASH (upper extremity): MCID = 10 points\n"
            "- TUG (balance): MCID = 3.4 seconds\n"
            "- 6MWT (endurance): MCID = 50 meters\n"
            "Indicate whether the patient has met MCID based on available data.",
            "mcid_check",
        )
        return AssistantResponse(
            type="text",
            content=text,
            suggestions=["functional_progress", "discharge_criteria"],
        )

    async def _cmd_next_visit_plan(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        text = await self._generate(
            "Plan the next rehabilitation visit based on current progress. Include:\n"
            "- Reassessment priorities\n"
            "- Intervention progressions or modifications\n"
            "- Outcome measures to administer\n"
            "- Patient education topics\n"
            "- Coordination needs (MD, other disciplines)",
            "next_visit_plan",
        )
        return AssistantResponse(
            type="suggestion",
            content=text,
            requires_approval=True,
            approval_action="apply_visit_plan",
            suggestions=["suggest_interventions", "suggest_goals"],
        )

    async def _cmd_documentation_tips(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        text = await self._generate(
            "Provide documentation improvement tips for medical necessity. Cover:\n"
            "- Skilled care justification language\n"
            "- Functional limitation documentation\n"
            "- Objective measure requirements\n"
            "- Prior level of function (PLOF)\n"
            "- Complexity indicators for eval codes\n"
            "- Common audit triggers to avoid",
            "documentation_tips",
        )
        return AssistantResponse(
            type="text",
            content=text,
            suggestions=["improve_note", "skilled_justification"],
        )

    async def _cmd_skilled_justification(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        text = await self._generate(
            "Generate a skilled care justification statement. Include:\n"
            "- Why skilled services are required (vs maintenance)\n"
            "- Complexity of condition requiring professional judgment\n"
            "- Objective evidence of improvement potential\n"
            "- Safety considerations requiring skilled oversight\n"
            "- Specific skilled interventions being provided",
            "skilled_justification",
        )
        return AssistantResponse(
            type="note_draft",
            content=text,
            requires_approval=True,
            approval_action="apply_justification",
            suggestions=["suggest_cpt", "documentation_tips"],
        )

    async def _cmd_eight_min_rule(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        content = (
            "**8-Minute Rule Calculator**\n\n"
            "| Total Minutes | Billable Units |\n"
            "|:---:|:---:|\n"
            "| 8-22 min | 1 unit |\n"
            "| 23-37 min | 2 units |\n"
            "| 38-52 min | 3 units |\n"
            "| 53-67 min | 4 units |\n"
            "| 68-82 min | 5 units |\n\n"
            "**Rules:**\n"
            "- Each unit = 15 minutes\n"
            "- Minimum 8 minutes of a service to bill 1 unit\n"
            "- Remaining minutes: bill the unit for whichever service has the most minutes\n"
            "- Timed codes only (97110, 97112, 97116, 97140, 97530, 97535, 97542, 97750, 97761, 97763)\n"
            "- Untimed codes (97010, 97014, 97024, 97026) = 1 unit regardless of time\n\n"
            "Provide your service times and I can calculate units."
        )
        return AssistantResponse(
            type="billing",
            content=content,
            suggestions=["suggest_cpt", "documentation_tips"],
        )

    async def _cmd_functional_progress(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        if not ctx.get("patient_id"):
            return AssistantResponse(
                type="error",
                content="No patient selected. Navigate to a patient record first.",
                suggestions=["show_patients"],
            )
        text = await self._generate(
            "Summarize functional progress over the episode of care. Include:\n"
            "- Admission vs current functional status\n"
            "- Objective outcome measure changes\n"
            "- Goals met and percentage progress on remaining goals\n"
            "- Trend (improving, plateau, declining)\n"
            "- Recommended action based on progress",
            "functional_progress",
        )
        return AssistantResponse(
            type="text",
            content=text,
            data={"patient_id": ctx["patient_id"]},
            suggestions=["mcid_check", "discharge_criteria", "next_visit_plan"],
        )

    async def _cmd_discharge_criteria(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        text = await self._generate(
            "Evaluate discharge readiness. Check:\n"
            "- Goals met (short-term and long-term)\n"
            "- Functional plateau (3+ visits without measurable change)\n"
            "- Patient independence with HEP\n"
            "- Authorization remaining\n"
            "- Safety for independent function\n"
            "Recommend: continue, discharge, or transition to maintenance.",
            "discharge_criteria",
        )
        return AssistantResponse(
            type="suggestion",
            content=text,
            requires_approval=True,
            approval_action="initiate_discharge",
            suggestions=["draft_discharge_summary", "functional_progress"],
        )

    async def _cmd_peer_comparison(self, msg: str, ctx: dict, match: re.Match | None = None) -> AssistantResponse:
        text = await self._generate(
            "Compare patient outcomes to national rehabilitation benchmarks:\n"
            "- Average visits per episode by diagnosis\n"
            "- Expected functional improvement rates\n"
            "- MCID achievement rates\n"
            "- Typical episode duration\n"
            "Use FOTO, IRF-PAI, or published norms as reference.",
            "peer_comparison",
        )
        return AssistantResponse(
            type="text",
            content=text,
            suggestions=["functional_progress", "mcid_check"],
        )
