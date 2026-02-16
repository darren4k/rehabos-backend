"""Encounter Brain — the intelligent orchestrator for clinical documentation.

Replaces the rigid conversation script with dynamic, context-aware guidance.
Builds a custom prompt each turn based on:
  - What's been collected so far
  - Patient history (from memU / Patient-Core)
  - Medicare requirements for the note type
  - What the therapist just said
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime, timezone
from typing import Any, Optional

from rehab_os.encounter.state import (
    AssessmentData,
    EncounterPhase,
    EncounterState,
    FunctionalMobilityEntry,
    InterventionEntry,
    MMTEntry,
    ObjectiveData,
    PlanData,
    ROMEntry,
    StandardizedTestEntry,
    SubjectiveData,
    VitalsData,
)

logger = logging.getLogger(__name__)


class EncounterBrain:
    """Orchestrates a clinical documentation encounter.

    Each call to ``process_turn`` receives the current encounter state and
    a new therapist utterance, and returns the updated state plus an
    intelligent response.
    """

    def __init__(self, llm_router: Any, session_memory: Any = None):
        self.llm_router = llm_router
        self.session_memory = session_memory

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    async def process_turn(
        self,
        state: EncounterState,
        utterance: str,
    ) -> tuple[EncounterState, str, list[str]]:
        """Process a therapist utterance and return updated state + response.

        Returns:
            (updated_state, response_text, suggestions)
        """
        # 1. Record the utterance
        state.transcript.append(
            {
                "role": "user",
                "content": utterance,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        state.turn_count += 1

        # 2. Check for instant responses (greetings, simple chat — no LLM needed)
        instant = self._check_instant_response(utterance, state)
        if instant:
            state.transcript.append(
                {"role": "assistant", "content": instant[1], "timestamp": datetime.now(timezone.utc).isoformat()}
            )
            return instant

        # 3. Check for meta-commands (generate, skip, next patient)
        meta = self._check_meta_commands(utterance, state)
        if meta:
            return meta

        # 4. Extract structured data from the utterance
        state = self._extract_data(utterance, state)

        # 5. Update phase based on what's collected
        state = self._update_phase(state)

        # 6. Load patient history on first relevant turn
        if state.patient_id and not state.history.last_encounters:
            state = await self._load_patient_history(state)

        # 7. Check if we can respond without LLM (data confirmed + next question is obvious)
        fast = self._try_fast_response(utterance, state)
        if fast:
            response, suggestions = fast
            state.transcript.append(
                {"role": "assistant", "content": response, "timestamp": datetime.now(timezone.utc).isoformat()}
            )
            return state, response, suggestions

        # 8. Build dynamic prompt and get LLM response (slow path — only when needed)
        response, suggestions = await self._generate_response(state, utterance)

        # 9. Record assistant response
        state.transcript.append(
            {
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        return state, response, suggestions

    # ──────────────────────────────────────────────────────────────────────
    # Meta-commands
    # ──────────────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────────
    # Instant responses (no LLM, <50ms)
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _check_instant_response(
        utterance: str, state: EncounterState
    ) -> Optional[tuple[EncounterState, str, list[str]]]:
        """Handle greetings and simple chat instantly — no LLM needed."""
        lower = utterance.lower().strip()

        # Greetings
        if re.match(r"^(hi|hello|hey|good morning|good afternoon|good evening|howdy|what'?s up|yo)[\s!.?]*$", lower):
            return (
                state,
                "Hey! Ready when you are. Who are we documenting for?",
                ["Start a daily note", "Start an eval"],
            )

        # How are you / pleasantries
        if re.match(r"^(how are you|how'?s it going|what'?s up|how do you do)[\s!.?]*$", lower):
            return (
                state,
                "Doing great, thanks! Ready to help with documentation. Who's the next patient?",
                ["Start a daily note", "Start an eval"],
            )

        # Thank you
        if re.match(r"^(thanks?|thank you|thx|ty|appreciate it)[\s!.?]*$", lower):
            return (
                state,
                "You're welcome! Need anything else?",
                [],
            )

        # Help / what can you do
        if re.match(r"^(help|what can you do|how does this work|commands?)[\s!.?]*$", lower):
            return (
                state,
                "I help you document patient encounters through conversation. Just tell me the patient name and note type, then walk me through the visit — I'll organize everything into a SOAP note. You can also say 'skip' to jump sections or 'generate' when you're ready.",
                ["Start a daily note"],
            )

        return None

    def _try_fast_response(
        self, utterance: str, state: EncounterState
    ) -> Optional[tuple[str, list[str]]]:
        """Try to respond without LLM when the next step is obvious.

        Returns (response, suggestions) or None if LLM is needed.
        """
        # After data extraction, check if we can give an immediate confirmation + question
        lower = utterance.lower()

        # Did we extract vitals this turn?
        if state.objective.vitals and re.search(r"(?:bp|blood pressure|pulse|heart rate)", lower):
            v = state.objective.vitals
            parts = []
            if v.bp: parts.append(f"BP {v.bp}")
            if v.pulse: parts.append(f"HR {v.pulse}")
            if v.spo2: parts.append(f"SpO2 {v.spo2}%")
            confirm = f"Got it — {', '.join(parts)}."
            # What's next?
            if not state.objective.interventions:
                return f"{confirm} What interventions were performed?", self._get_suggestions(state)
            if not state.objective.tolerance:
                return f"{confirm} How did the patient tolerate treatment?", self._get_suggestions(state)
            return f"{confirm} Anything else to add?", self._get_suggestions(state)

        # Did we extract pain this turn?
        if state.subjective.pain_level is not None and re.search(r"(?:pain|complain|c/o|hurts)", lower):
            loc = f" ({state.subjective.pain_location})" if state.subjective.pain_location else ""
            confirm = f"Got it — pain {state.subjective.pain_level}/10{loc}."

            # Compare with history
            if state.history.last_encounters:
                last_pain = state.history.last_encounters[0].get("pain_level")
                if last_pain is not None:
                    if state.subjective.pain_level < last_pain:
                        confirm += f" Down from {last_pain}/10 last visit — nice improvement!"
                    elif state.subjective.pain_level > last_pain:
                        confirm += f" Up from {last_pain}/10 last visit — let's keep an eye on that."

            if not state.objective.vitals:
                return f"{confirm} Any vitals today?", self._get_suggestions(state)
            if not state.objective.interventions:
                return f"{confirm} What interventions were performed?", self._get_suggestions(state)
            return f"{confirm} What else?", self._get_suggestions(state)

        # Did we extract interventions + tolerance together?
        if state.objective.interventions and state.objective.tolerance and re.search(r"(?:exercise|gait|balance|manual|tolerat)", lower):
            names = [i.name for i in state.objective.interventions]
            confirm = f"Got it — {', '.join(names)}. Tolerated well."
            if not state.plan.next_visit and not state.plan.frequency:
                return f"{confirm} What's the plan? Continue current POC?", self._get_suggestions(state)
            if not state.missing_critical():
                return f"{confirm} I have everything I need. Say 'generate' when ready, or add anything else.", self._get_suggestions(state)
            return f"{confirm} Anything else?", self._get_suggestions(state)

        # Did we extract plan this turn?
        if (state.plan.next_visit or state.plan.frequency) and re.search(r"(?:continue|plan|poc|frequency|times?\s*(?:per|a)\s*week)", lower):
            plan_parts = []
            if state.plan.next_visit: plan_parts.append(state.plan.next_visit)
            if state.plan.frequency: plan_parts.append(state.plan.frequency)
            confirm = f"Got it — {', '.join(plan_parts)}."
            if not state.missing_critical():
                return f"{confirm} All set! Say 'generate' when you're ready for the note.", ["Generate note"]
            missing = state.missing_critical()
            return f"{confirm} Still need: {', '.join(missing)}.", self._get_suggestions(state)

        # Setup phase — extracted note type
        if state.note_type and state.phase == EncounterPhase.SUBJECTIVE and not state.subjective.chief_complaint:
            name = f" for {state.patient_name}" if state.patient_name else ""
            nt = (state.note_type or "").replace("_", " ")
            return f"Starting {nt}{name}. What's the chief complaint?", self._get_suggestions(state)

        # Nothing obvious — let LLM handle it
        return None

    def _check_meta_commands(
        self, utterance: str, state: EncounterState
    ) -> Optional[tuple[EncounterState, str, list[str]]]:
        lower = utterance.lower().strip()

        # Generate note
        if re.search(
            r"\b(generate|write it|finalize|that'?s it|that'?s all|i'?m done|go ahead|write the note)\b",
            lower,
        ):
            if state.turn_count > 2:
                state.phase = EncounterPhase.REVIEW
                missing = state.missing_critical()
                if missing:
                    return (
                        state,
                        f"Almost ready. Still missing: {', '.join(missing)}. Want to add those or generate anyway?",
                        ["Generate anyway", f"Add {missing[0]}"],
                    )
                return state, "Generating your note now.", ["generate"]

        # Skip / move to section
        if re.search(r"\b(skip|move on|next section|proceed to|let'?s go to)\b", lower):
            target = self._detect_section_target(lower)
            if target:
                state.phase = target
                phase_name = target.value.replace("_", " ")
                return state, f"Moving to {phase_name}. What do you have?", []

        # Reset / next patient
        if re.search(r"\b(next patient|new patient|start over|clear|reset)\b", lower):
            new_state = EncounterState(encounter_id=state.encounter_id)
            return new_state, "Ready for the next patient. Who are we seeing?", []

        return None

    @staticmethod
    def _detect_section_target(text: str) -> Optional[EncounterPhase]:
        text_l = text.lower()
        if "subjective" in text_l:
            return EncounterPhase.SUBJECTIVE
        if "objective" in text_l:
            return EncounterPhase.OBJECTIVE
        if "assessment" in text_l:
            return EncounterPhase.ASSESSMENT
        if "plan" in text_l:
            return EncounterPhase.PLAN
        return None

    # ──────────────────────────────────────────────────────────────────────
    # Data extraction (regex-based, fast, no LLM needed)
    # ──────────────────────────────────────────────────────────────────────

    def _extract_data(self, utterance: str, state: EncounterState) -> EncounterState:
        """Parse structured clinical data from natural speech."""
        lower = utterance.lower()

        # ── Setup: note type & date ──
        self._extract_setup(lower, utterance, state)

        # ── Pain ──
        self._extract_pain(lower, state)

        # ── Chief complaint ──
        self._extract_chief_complaint(lower, state)

        # ── Vitals ──
        self._extract_vitals(lower, state)

        # ── ROM ──
        self._extract_rom(utterance, state)

        # ── MMT ──
        self._extract_mmt(lower, state)

        # ── Standardized tests ──
        self._extract_tests(lower, state)

        # ── Interventions ──
        self._extract_interventions(lower, state)

        # ── Tolerance ──
        self._extract_tolerance(lower, state)

        # ── HEP compliance ──
        self._extract_hep(lower, state)

        # ── Assessment / progress ──
        self._extract_assessment(lower, state)

        # ── Plan ──
        self._extract_plan(lower, state)

        return state

    # ── Individual extractors ─────────────────────────────────────────────

    @staticmethod
    def _extract_setup(lower: str, raw: str, state: EncounterState) -> None:
        # Patient name — "note for Maria Santos", "documenting Maria Santos"
        if not state.patient_name:
            name_match = re.search(
                r"(?:note\s+for|documenting|seeing|patient\s+(?:is)?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
                raw,
            )
            if name_match:
                state.patient_name = name_match.group(1).strip()

        if state.phase != EncounterPhase.SETUP and state.note_type:
            return

        nt_map = {
            "daily": "daily_note",
            "progress": "progress_note",
            "eval": "initial_evaluation",
            "evaluation": "initial_evaluation",
            "discharge": "discharge",
            "recert": "recertification",
        }
        for key, val in nt_map.items():
            if re.search(rf"\b{key}\b", lower):
                state.note_type = val
                break

        date_match = re.search(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})", raw)
        if date_match:
            m, d, y = date_match.groups()
            yr = int(y) if len(y) == 4 else 2000 + int(y)
            state.date_of_service = f"{yr}-{int(m):02d}-{int(d):02d}"
        elif "today" in lower and not state.date_of_service:
            state.date_of_service = date.today().isoformat()

    @staticmethod
    def _extract_pain(lower: str, state: EncounterState) -> None:
        # "pain 3 out of 10", "3/10 pain", "pain level is 3 over 10"
        patterns = [
            r"pain\s*(?:level\s*)?(?:is\s*|of\s*)?(\d{1,2})\s*(?:out of|over|\/)\s*(?:10|ten)",
            r"(\d{1,2})\s*(?:out of|over|\/)\s*(?:10|ten)\s*(?:pain)?",
        ]
        for pat in patterns:
            m = re.search(pat, lower)
            if m:
                level = int(m.group(1))
                if 0 <= level <= 10:
                    state.subjective.pain_level = level
                    break

        # No pain
        if re.search(r"\b(no pain|pain.?free|denies pain|zero\s*(?:out of|\/)\s*10)\b", lower):
            state.subjective.pain_level = 0
            if not state.subjective.chief_complaint:
                state.subjective.chief_complaint = "No pain reported"

        # Pain location
        loc_match = re.search(
            r"(?:pain|complain\w*|hurt\w*|c/o)\s+(?:in\s+|at\s+|of\s+)?(?:the\s+)?"
            r"(?:(right|left|bilateral|r|l)\s+)?"
            r"(knee|shoulder|hip|ankle|back|neck|wrist|elbow|foot|hand|lower back|lumbar|cervical)",
            lower,
        )
        if loc_match:
            side_raw, body = loc_match.groups()
            parts = []
            if side_raw:
                s = side_raw.strip()
                parts.append({"r": "right", "l": "left"}.get(s, s))
            parts.append(body)
            state.subjective.pain_location = " ".join(parts)

    @staticmethod
    def _extract_chief_complaint(lower: str, state: EncounterState) -> None:
        if state.phase not in (EncounterPhase.SETUP, EncounterPhase.SUBJECTIVE):
            return
        m = re.search(
            r"(?:complain\w*|c/o|chief complaint|presents? with|reports?)\s+(?:of\s+)?(.+?)(?:\.|,|$)",
            lower,
        )
        if m:
            state.subjective.chief_complaint = m.group(1).strip()
        elif state.subjective.pain_location and not state.subjective.chief_complaint:
            state.subjective.chief_complaint = f"{state.subjective.pain_location} pain"

    @staticmethod
    def _extract_vitals(lower: str, state: EncounterState) -> None:
        def _ensure_vitals() -> VitalsData:
            if state.objective.vitals is None:
                state.objective.vitals = VitalsData()
            return state.objective.vitals

        # Blood pressure
        bp = re.search(r"(?:bp|blood pressure)\s*(?:is|of|was)?\s*(\d{2,3})\s*(?:over|\/)\s*(\d{2,3})", lower)
        if bp:
            _ensure_vitals().bp = f"{bp.group(1)}/{bp.group(2)}"

        # Pulse / HR
        hr = re.search(r"(?:pulse\s*(?:rate)?|heart\s*rate|hr)\s*(?:is|of|was)?\s*(\d{2,3})", lower)
        if hr:
            _ensure_vitals().pulse = int(hr.group(1))

        # SpO2
        spo2 = re.search(r"(?:sp\s*o\s*2|o2\s*sat|oxygen\s*sat)\s*(?:is|of|was)?\s*(\d{2,3})\s*%?", lower)
        if spo2:
            _ensure_vitals().spo2 = int(spo2.group(1))

        # Respiratory rate
        rr = re.search(r"(?:respiratory rate|resp rate|rr)\s*(?:is|of|was)?\s*(\d{1,2})", lower)
        if rr:
            _ensure_vitals().respiratory_rate = int(rr.group(1))

    @staticmethod
    def _extract_rom(raw: str, state: EncounterState) -> None:
        # "right knee flexion 95 degrees"
        pattern = re.compile(
            r"(?:rom\s+)?(?:(right|left|r|l|bilateral)\s+)?"
            r"(\w+(?:\s+\w+)?)\s+"
            r"(flexion|extension|abduction|adduction|(?:internal|external)\s*rotation)\s+"
            r"(\d+)\s*(?:degrees|deg|°)?",
            re.IGNORECASE,
        )
        for m in pattern.finditer(raw):
            side_raw, joint, motion, value = m.groups()
            side = {"r": "right", "l": "left"}.get((side_raw or "").lower().strip(), side_raw or "bilateral")
            state.objective.rom.append(
                ROMEntry(joint=joint.strip(), motion=motion.strip(), value=int(value), side=side.lower())
            )

        # Simpler: "flexion 95 degrees" (uses pain_location as joint)
        simple = re.compile(
            r"\b(flexion|extension|abduction|adduction)\s+(?:is\s+)?(\d+)\s*(?:degrees|deg|°)?",
            re.IGNORECASE,
        )
        for m in simple.finditer(raw):
            motion, value = m.groups()
            # Only if the full pattern didn't already catch it
            if not any(r.value == int(value) and r.motion.lower() == motion.lower() for r in state.objective.rom):
                state.objective.rom.append(
                    ROMEntry(
                        joint=state.subjective.pain_location or "unknown",
                        motion=motion.strip(),
                        value=int(value),
                    )
                )

    @staticmethod
    def _extract_mmt(lower: str, state: EncounterState) -> None:
        for m in re.finditer(
            r"(?:mmt\s+)?(\w+(?:\s+\w+)?)\s+(\d[+\-]?)\s*(?:out of|\/)\s*5\s*(?:(right|left|bilateral|r|l))?",
            lower,
        ):
            muscle, grade, side_raw = m.groups()
            side = {"r": "right", "l": "left"}.get((side_raw or "").strip(), side_raw or "bilateral")
            state.objective.mmt.append(MMTEntry(muscle_group=muscle.strip(), grade=f"{grade}/5", side=side))

    @staticmethod
    def _extract_tests(lower: str, state: EncounterState) -> None:
        test_patterns = {
            "Berg Balance Scale": r"berg\s*(?:balance)?\s*(?:score)?\s*(?:is|of|was)?\s*(\d+)",
            "TUG": r"(?:tug|timed up and go)\s*(?:is|of|was)?\s*(\d+\.?\d*)\s*(?:seconds?|sec)?",
            "Tinetti": r"tinetti\s*(?:score)?\s*(?:is|of|was)?\s*(\d+)",
            "6MWT": r"(?:6\s*(?:minute)?\s*walk\s*test|6mwt)\s*(?:is|of|was)?\s*(\d+)\s*(?:feet|ft|meters?|m)?",
            "30-Second Sit-to-Stand": r"(?:30\s*(?:second)?\s*sit\s*to\s*stand|30s?sts)\s*(?:is|of|was)?\s*(\d+)",
        }
        for test_name, pattern in test_patterns.items():
            m = re.search(pattern, lower)
            if m:
                # Avoid duplicates
                if not any(t.name == test_name for t in state.objective.standardized_tests):
                    state.objective.standardized_tests.append(
                        StandardizedTestEntry(name=test_name, score=m.group(1))
                    )

    @staticmethod
    def _extract_interventions(lower: str, state: EncounterState) -> None:
        keyword_map = {
            "therapeutic exercise": r"\b(?:ther(?:apeutic)?\s*ex(?:ercise)?|strengthening|stretching)\b",
            "gait training": r"\b(?:gait\s*training|ambul(?:ation|ated)|walk(?:ing|ed))\b",
            "balance training": r"\b(?:balance\s*training|balance\s*activities)\b",
            "manual therapy": r"\b(?:manual\s*therapy|soft\s*tissue|mobilization|joint\s*mob)\b",
            "neuromuscular re-education": r"\b(?:neuro(?:muscular)?\s*re-?ed(?:ucation)?|motor\s*control)\b",
            "transfer training": r"\b(?:transfer\s*training|sit\s*to\s*stand|bed\s*mobility)\b",
            "patient education": r"\b(?:patient\s*education|educated\s*(?:patient|pt)|HEP\s*instruction)\b",
            "modalities": r"\b(?:ultrasound|e-?stim|electrical\s*stim|hot\s*pack|cold\s*pack|ice|iontophoresis|TENS)\b",
        }
        for name, pattern in keyword_map.items():
            if re.search(pattern, lower):
                if not any(i.name == name for i in state.objective.interventions):
                    state.objective.interventions.append(InterventionEntry(name=name))

    @staticmethod
    def _extract_tolerance(lower: str, state: EncounterState) -> None:
        m = re.search(r"(?:patient|pt)\s+(?:tolerated?|handled)\s+(.+?)(?:\.|$)", lower)
        if m:
            state.objective.tolerance = m.group(1).strip()
        elif re.search(r"\b(tolerated?\s+well|no adverse|without complications)\b", lower):
            state.objective.tolerance = "Patient tolerated treatment well without adverse reaction"

    @staticmethod
    def _extract_hep(lower: str, state: EncounterState) -> None:
        m = re.search(
            r"(?:hep|home\s*exercise|home\s*program)\s*(?:compliance|adherence)?\s*(?:is|was)?\s*(.+?)(?:\.|$)",
            lower,
        )
        if m:
            state.subjective.hep_compliance = m.group(1).strip()
        elif re.search(r"\b(compliant|doing (?:the|his|her) exercises|following\s*(?:the\s*)?program)\b", lower):
            state.subjective.hep_compliance = "Compliant with HEP"
        elif re.search(r"\b(non-?compliant|not doing|hasn't been doing|skipping)\b", lower):
            state.subjective.hep_compliance = "Non-compliant with HEP"

    @staticmethod
    def _extract_assessment(lower: str, state: EncounterState) -> None:
        m = re.search(r"(?:progress|improving|plateau|regress|declined?|stable)\b.*", lower)
        if m and state.phase in (EncounterPhase.ASSESSMENT, EncounterPhase.OBJECTIVE):
            state.assessment.progress = m.group(0).strip()

    @staticmethod
    def _extract_plan(lower: str, state: EncounterState) -> None:
        if re.search(r"continue\s+(?:current\s+)?(?:poc|plan|treatment|therapy)", lower):
            state.plan.next_visit = "Continue current plan of care"
        freq = re.search(r"(\d+)\s*(?:times?|x)\s*(?:per|\/)\s*(?:week|wk)", lower)
        if freq:
            state.plan.frequency = freq.group(0)
        dc = re.search(r"discharge\s+(?:in|within)\s+(\d+)\s*(?:weeks?|visits?)", lower)
        if dc:
            state.plan.discharge_timeline = dc.group(0)

    # ──────────────────────────────────────────────────────────────────────
    # Phase management
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _update_phase(state: EncounterState) -> EncounterState:
        """Auto-advance phase based on what's been collected."""
        if state.phase == EncounterPhase.SETUP:
            # Need at least note type to move on
            if state.note_type:
                state.phase = EncounterPhase.SUBJECTIVE
        # Don't auto-advance other phases — let the Brain's prompt handle it
        return state

    # ──────────────────────────────────────────────────────────────────────
    # Patient history
    # ──────────────────────────────────────────────────────────────────────

    async def _load_patient_history(self, state: EncounterState) -> EncounterState:
        if not self.session_memory or not state.patient_id:
            return state
        try:
            from rehab_os.encounter.memory import load_patient_context

            state.history = await load_patient_context(
                patient_id=state.patient_id,
                session_memory=self.session_memory,
            )
        except Exception as e:
            logger.warning("Failed to load patient history: %s", e)
        return state

    # ──────────────────────────────────────────────────────────────────────
    # LLM response generation
    # ──────────────────────────────────────────────────────────────────────

    async def _generate_response(
        self, state: EncounterState, utterance: str
    ) -> tuple[str, list[str]]:
        prompt = self._build_prompt(state)

        # Build conversation messages for the LLM
        ollama_messages = [{"role": "system", "content": prompt}]
        for turn in state.transcript[-12:]:
            role = "user" if turn["role"] == "user" else "assistant"
            ollama_messages.append({"role": role, "content": turn["content"]})

        # Use fast model directly via Ollama for snappy conversation
        text = await self._call_fast_llm(ollama_messages)

        if not text:
            text = self._fallback_response(state)

        suggestions = self._get_suggestions(state)
        return text, suggestions

    @staticmethod
    async def _call_fast_llm(messages: list[dict], max_tokens: int = 200) -> str:
        """Call the fast conversation model directly via Ollama HTTP API.

        Uses qwen2.5:14b for ~2-4s response time instead of the 80b model
        which takes 20-30s. Falls back gracefully on any error.
        """
        import httpx

        OLLAMA_URL = "http://192.168.68.127:11434"
        FAST_MODEL = "qwen2.5:14b"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": FAST_MODEL,
                        "stream": False,
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": 0.3,
                        },
                        "messages": messages,
                    },
                )
                if res.status_code == 200:
                    data = res.json()
                    text = data.get("message", {}).get("content", "")
                    # Strip thinking tags
                    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
                    return text
        except Exception as e:
            logger.warning("Fast LLM call failed: %s", e)

        return ""

    # ──────────────────────────────────────────────────────────────────────
    # Dynamic prompt
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(state: EncounterState) -> str:
        # Patient context
        patient_ctx = ""
        if state.patient_name:
            patient_ctx = f"PATIENT: {state.patient_name}"
            if state.history.diagnosis:
                patient_ctx += f" ({', '.join(state.history.diagnosis)})"
            patient_ctx += "\n"

        # History
        history_ctx = ""
        if state.history.last_encounters:
            last = state.history.last_encounters[0]
            history_ctx = f"LAST VISIT ({last.get('date', 'unknown')}): "
            parts = []
            if last.get("pain_level") is not None:
                parts.append(f"Pain {last['pain_level']}/10")
            if last.get("summary"):
                parts.append(last["summary"][:100])
            history_ctx += ", ".join(parts) + "\n"

        if state.history.active_goals:
            history_ctx += "ACTIVE GOALS:\n"
            for g in state.history.active_goals[:3]:
                history_ctx += f"  - {g.get('area', '?')}: current {g.get('current', '?')} → target {g.get('target', '?')}\n"

        # Collected summary
        collected = state.summary_for_prompt()

        # Missing
        critical_missing = state.missing_critical()
        recommended_missing = state.missing_recommended()

        # Smart next-step suggestions
        next_steps: list[str] = []
        if not state.subjective.chief_complaint:
            next_steps.append("Ask about chief complaint and how the patient is feeling")
        elif not state.objective.vitals:
            next_steps.append("Ask about vital signs")
        elif not state.objective.interventions:
            next_steps.append("Ask what interventions were performed")
        elif not state.objective.tolerance:
            next_steps.append("Ask about patient tolerance/response")
        elif not state.plan.next_visit and not state.plan.frequency:
            next_steps.append("Ask about the plan — continue POC? Any changes?")
        else:
            next_steps.append(
                "All key sections have data. Ask if there's anything to add, or offer to generate the note."
            )

        # Pain comparison
        comparison = ""
        if state.subjective.pain_level is not None and state.history.last_encounters:
            last_pain = state.history.last_encounters[0].get("pain_level")
            if last_pain is not None:
                if state.subjective.pain_level < last_pain:
                    comparison = f"\n💡 Pain IMPROVED: {last_pain}/10 → {state.subjective.pain_level}/10"
                elif state.subjective.pain_level > last_pain:
                    comparison = f"\n⚠ Pain WORSENED: {last_pain}/10 → {state.subjective.pain_level}/10"

        return f"""You are an intelligent clinical documentation assistant helping a rehabilitation therapist document a patient encounter. Be warm, efficient, and clinically knowledgeable.

{patient_ctx}{history_ctx}
NOTE TYPE: {state.note_type or '[not yet specified]'}
DATE: {state.date_of_service or 'today'}

COLLECTED SO FAR:
{collected}{comparison}

STILL NEEDED (critical): {', '.join(critical_missing) if critical_missing else 'All critical items collected ✓'}
STILL NEEDED (recommended): {', '.join(recommended_missing) if recommended_missing else 'All recommended items collected ✓'}

WHAT TO DO NEXT:
{chr(10).join(f'- {s}' for s in next_steps)}

RULES:
1. Be BRIEF — 1-2 sentences max + 1 question. No paragraphs.
2. CONFIRM data you received ("Got it — BP 125/66, HR 62.")
3. Reference patient HISTORY when relevant ("Pain down from 5 to 3, nice improvement!")
4. NEVER invent or assume clinical data not stated by the therapist.
5. Accept data for ANY section regardless of current phase — slot it in.
6. If they say "skip" or "move on" → advance to next section.
7. If they say "done" / "that's it" / "generate" → offer to generate the note.
8. Suggest relevant measurements based on diagnosis and history.
9. Keep your clinical personality — not generic or robotic."""

    # ──────────────────────────────────────────────────────────────────────
    # Fallback (no LLM)
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _fallback_response(state: EncounterState) -> str:
        if not state.patient_name and not state.note_type:
            return "Who are we documenting for today? And what type of note?"
        if not state.note_type:
            return f"What type of note — daily, progress, eval, or discharge?"
        if not state.subjective.chief_complaint:
            return "What's the chief complaint and pain level?"
        if not state.objective.vitals:
            return "Any vitals today — BP, pulse, O2 sat?"
        if not state.objective.interventions:
            return "What interventions did you perform?"
        if not state.objective.tolerance:
            return "How did the patient tolerate treatment?"
        if not state.plan.next_visit:
            return "What's the plan? Continue current POC, any changes?"
        return "I have the key info. Say 'generate' when ready, or add anything else."

    # ──────────────────────────────────────────────────────────────────────
    # Suggestions
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_suggestions(state: EncounterState) -> list[str]:
        suggestions: list[str] = []
        if state.objective.interventions and not state.objective.tolerance:
            suggestions.append("Patient tolerated well")
        if state.subjective.chief_complaint and not state.objective.rom:
            suggestions.append("Add ROM measurements")
        if state.objective.interventions and (state.plan.next_visit or state.plan.frequency):
            suggestions.append("Generate note")
        if not state.plan.next_visit:
            suggestions.append("Continue current POC")
        return suggestions[:3]
