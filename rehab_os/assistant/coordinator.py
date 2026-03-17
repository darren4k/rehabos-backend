"""Chat coordinator — brain of the AI clinical assistant."""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

from rehab_os.assistant.page_context import get_page_context
from rehab_os.llm.base import Message, MessageRole
from rehab_os.llm.router import LLMRouter

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a clinical documentation assistant for rehabilitation professionals "
    "(PT/OT/SLP). You help with SOAP notes, care plans, billing, and evidence-based "
    "treatment. You NEVER make autonomous clinical decisions — you suggest and the "
    "clinician approves. Be concise, use clinical terminology, cite evidence when "
    "available."
)


@dataclass
class AssistantResponse:
    """Structured response from the assistant."""

    type: str  # "text", "suggestion", "patient_list", "note_draft", "billing", "error"
    content: str  # Markdown-formatted response
    data: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None
    requires_approval: bool = False
    approval_action: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "content": self.content,
            "data": self.data,
            "suggestions": self.suggestions,
            "requires_approval": self.requires_approval,
            "approval_action": self.approval_action,
        }


# ---------------------------------------------------------------------------
# Command definition
# ---------------------------------------------------------------------------
@dataclass
class CommandDef:
    """A regex-matched command."""

    name: str
    pattern: re.Pattern
    handler_name: str  # method name on ChatCoordinator
    description: str


def _cmd(name: str, pattern: str, handler: str, description: str) -> CommandDef:
    return CommandDef(
        name=name,
        pattern=re.compile(pattern, re.IGNORECASE),
        handler_name=handler,
        description=description,
    )


COMMANDS: list[CommandDef] = [
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


class ChatCoordinator:
    """Routes assistant messages to command handlers or LLM fallback."""

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
        conversation_history: Optional[list[dict]] = None,
    ) -> AssistantResponse:
        """Process a user message and return a response.

        1. Try regex command match.
        2. If matched, run the handler.
        3. Otherwise fall back to LLM with context injection.
        """
        ctx = {
            "provider_id": provider_id,
            "page": page_context,
            "patient_id": patient_id,
            "page_info": get_page_context(page_context),
            "history": conversation_history or [],
        }

        # Try command match
        for cmd in COMMANDS:
            if cmd.pattern.search(message):
                handler = getattr(self, cmd.handler_name, None)
                if handler:
                    logger.info("Command matched: %s", cmd.name)
                    return await handler(message, ctx)

        # LLM fallback
        return await self._llm_fallback(message, ctx)

    # ------------------------------------------------------------------
    # LLM fallback
    # ------------------------------------------------------------------

    async def _llm_fallback(self, message: str, ctx: dict) -> AssistantResponse:
        """Send message to the LLM with full context."""
        page_info = ctx["page_info"]
        system = SYSTEM_PROMPT + (
            f"\n\nThe clinician is currently on the '{page_info['label']}' page."
        )
        if ctx.get("patient_id"):
            system += f"\nActive patient ID: {ctx['patient_id']}."

        messages: list[Message] = [Message(role=MessageRole.SYSTEM, content=system)]

        # Inject recent conversation history (last 6 turns)
        for turn in (ctx.get("history") or [])[-6:]:
            role = MessageRole.USER if turn["role"] == "user" else MessageRole.ASSISTANT
            messages.append(Message(role=role, content=turn["content"]))

        messages.append(Message(role=MessageRole.USER, content=message))

        try:
            resp = await self.llm.complete(messages, temperature=0.4, max_tokens=2048)
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
            Message(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
            Message(role=MessageRole.USER, content=prompt),
        ]
        resp = await self.llm.complete(messages, temperature=0.3, max_tokens=2048)
        return resp.content

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    async def _cmd_show_patients(self, msg: str, ctx: dict) -> AssistantResponse:
        return AssistantResponse(
            type="patient_list",
            content="Fetching your active caseload...",
            data={"action": "list_patients", "provider_id": ctx["provider_id"]},
            suggestions=["patient_summary", "overdue_evals", "pending_notes"],
        )

    async def _cmd_patient_summary(self, msg: str, ctx: dict) -> AssistantResponse:
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

    async def _cmd_suggest_goals(self, msg: str, ctx: dict) -> AssistantResponse:
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

    async def _cmd_suggest_interventions(self, msg: str, ctx: dict) -> AssistantResponse:
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

    async def _cmd_improve_note(self, msg: str, ctx: dict) -> AssistantResponse:
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

    async def _cmd_suggest_cpt(self, msg: str, ctx: dict) -> AssistantResponse:
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

    async def _cmd_check_authorization(self, msg: str, ctx: dict) -> AssistantResponse:
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

    async def _cmd_draft_care_plan(self, msg: str, ctx: dict) -> AssistantResponse:
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

    async def _cmd_draft_progress_note(self, msg: str, ctx: dict) -> AssistantResponse:
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

    async def _cmd_draft_discharge_summary(self, msg: str, ctx: dict) -> AssistantResponse:
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

    async def _cmd_suggest_hep(self, msg: str, ctx: dict) -> AssistantResponse:
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

    async def _cmd_red_flag_check(self, msg: str, ctx: dict) -> AssistantResponse:
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

    async def _cmd_show_schedule(self, msg: str, ctx: dict) -> AssistantResponse:
        return AssistantResponse(
            type="text",
            content="Loading today's schedule...",
            data={"action": "load_schedule", "provider_id": ctx["provider_id"]},
            suggestions=["pending_notes", "overdue_evals"],
        )

    async def _cmd_pending_notes(self, msg: str, ctx: dict) -> AssistantResponse:
        return AssistantResponse(
            type="text",
            content="Checking for unsigned and pending notes...",
            data={"action": "pending_notes", "provider_id": ctx["provider_id"]},
            suggestions=["show_schedule", "overdue_evals"],
        )

    async def _cmd_overdue_evals(self, msg: str, ctx: dict) -> AssistantResponse:
        return AssistantResponse(
            type="text",
            content="Checking for overdue evaluations and re-evaluations...",
            data={"action": "overdue_evals", "provider_id": ctx["provider_id"]},
            suggestions=["show_schedule", "pending_notes"],
        )

    async def _cmd_patient_history(self, msg: str, ctx: dict) -> AssistantResponse:
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

    async def _cmd_evidence_search(self, msg: str, ctx: dict) -> AssistantResponse:
        # Extract the search topic from the message after the command trigger
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

    async def _cmd_mcid_check(self, msg: str, ctx: dict) -> AssistantResponse:
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

    async def _cmd_next_visit_plan(self, msg: str, ctx: dict) -> AssistantResponse:
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

    async def _cmd_documentation_tips(self, msg: str, ctx: dict) -> AssistantResponse:
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

    async def _cmd_skilled_justification(self, msg: str, ctx: dict) -> AssistantResponse:
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

    async def _cmd_eight_min_rule(self, msg: str, ctx: dict) -> AssistantResponse:
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

    async def _cmd_functional_progress(self, msg: str, ctx: dict) -> AssistantResponse:
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

    async def _cmd_discharge_criteria(self, msg: str, ctx: dict) -> AssistantResponse:
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

    async def _cmd_peer_comparison(self, msg: str, ctx: dict) -> AssistantResponse:
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
