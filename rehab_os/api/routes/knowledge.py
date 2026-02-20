"""Knowledge and evidence retrieval API for therapists.

Therapists ask clinical questions (including patient scenarios) and get
evidence-based treatment approaches with citations.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from rehab_os.api.dependencies import get_current_user
from rehab_os.core.models import Provider

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


class KnowledgeQuery(BaseModel):
    """A clinical knowledge query from a therapist."""

    question: str = Field(..., description="Clinical question or patient scenario")
    discipline: str = Field("PT", description="PT, OT, or SLP")


class Citation(BaseModel):
    """A reference citation."""

    title: str
    source: str
    year: Optional[int] = None
    authors: Optional[str] = None
    url: Optional[str] = None
    evidence_level: Optional[str] = None


class KnowledgeResponse(BaseModel):
    """Evidence-based response to a clinical query."""

    question: str
    summary: str = Field(..., description="Brief summary of recommendations")
    detailed_answer: str = Field(..., description="Full evidence-based response")
    key_recommendations: list[str] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)
    outcome_measures: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    clinical_pearls: list[str] = Field(default_factory=list)
    related_topics: list[str] = Field(default_factory=list)


# Knowledge base with clinical scenarios and evidence
KNOWLEDGE_BASE = {
    "parkinsons": {
        "keywords": ["parkinson", "pd", "bradykinesia", "rigidity", "tremor", "festinating"],
        "summary": "Exercise is disease-modifying in Parkinson's. High-intensity, amplitude-based training (LSVT BIG), balance/gait training, and aerobic exercise are first-line interventions.",
        "detailed_answer": """## Evidence-Based Approach for Parkinson's Disease Rehabilitation

### Core Principles
- **Exercise is neuroprotective** - Evidence shows high-intensity exercise may slow disease progression
- **Amplitude-based training** - Focus on BIG movements to counteract bradykinesia
- **External cueing** - Visual, auditory, and tactile cues improve movement initiation
- **Task-specific practice** - Train functional activities directly

### Recommended Interventions

**1. LSVT BIG (Level I Evidence)**
- Amplitude-focused movement training
- 16 sessions over 4 weeks (4x/week)
- Emphasizes "think big" for all movements
- Proven effective for bradykinesia and functional mobility

**2. Balance & Falls Prevention**
- Progressive balance training (static → dynamic → reactive)
- Perturbation-based training for fall prevention
- Tai Chi (Level I evidence for falls reduction)
- Dual-task training (cognitive + motor)

**3. Gait Training**
- Cueing strategies: rhythmic auditory (metronome), visual (floor lines)
- Treadmill training with/without body weight support
- Nordic walking for outdoor mobility
- Freezing of gait strategies: laser cues, attentional strategies

**4. Transfers & Functional Mobility**
- Sit-to-stand training with attention to movement initiation
- Segmented movement strategies (break complex movements into steps)
- Environmental modifications for home safety
- Bed mobility training

**5. Aerobic Exercise**
- High-intensity (70-85% max HR) shows greatest benefit
- Cycling, treadmill, or dance-based programs
- 30-40 min, 3x/week minimum
- May have neuroprotective effects

### For Older Adults with Falls
- Screen for orthostatic hypotension (common in PD)
- Assess freezing of gait triggers
- Home safety evaluation essential
- Consider assistive device (wheeled walker with hand brakes)
- Medication timing affects function - coordinate with neurologist

### Functional Training Priorities
1. **Toilet transfers**: Forward weight shift, use of rails, movement initiation cues
2. **Couch/chair transfers**: Higher seat height, firm surface, armrest push-off technique
3. **Floor transfers**: Practice getting up from floor (important for post-fall recovery)
4. **Turning**: Step-through turns vs. pivot turns to reduce freezing""",
        "key_recommendations": [
            "High-intensity exercise (LSVT BIG or equivalent) 4x/week",
            "Balance training with perturbations and dual-task challenges",
            "Cueing strategies for gait and freezing of gait",
            "Sit-to-stand and transfer training with movement initiation focus",
            "Floor-to-stand practice for fall recovery",
            "Aerobic exercise at 70-85% max HR, 3x/week",
            "Home safety evaluation and modifications",
        ],
        "contraindications": [
            "Avoid exercise during 'off' medication periods if severely affected",
            "Monitor for orthostatic hypotension with position changes",
            "Caution with high-speed treadmill if freezing of gait present",
        ],
        "outcome_measures": [
            "Timed Up and Go (TUG) - <13.5 sec = lower fall risk",
            "Mini-BESTest - balance assessment specific to PD",
            "6-Minute Walk Test - endurance",
            "5x Sit-to-Stand - functional strength",
            "Freezing of Gait Questionnaire",
            "PDQ-39 - Parkinson's quality of life",
            "UPDRS Part III - motor examination",
        ],
        "citations": [
            Citation(
                title="Physical Therapy for Parkinson Disease: A Practice Guideline",
                source="Neurology - American Academy of Neurology",
                year=2022,
                evidence_level="CPG",
                url="https://www.neurology.org/doi/10.1212/WNL.0000000000013321",
            ),
            Citation(
                title="LSVT BIG randomized controlled trial",
                source="Movement Disorders",
                year=2015,
                authors="Ebersbach G, et al.",
                evidence_level="Level I",
            ),
            Citation(
                title="Tai Chi and Falls Prevention in Parkinson Disease",
                source="New England Journal of Medicine",
                year=2012,
                authors="Li F, et al.",
                evidence_level="Level I",
            ),
            Citation(
                title="High-intensity exercise in Parkinson disease",
                source="JAMA Neurology",
                year=2018,
                authors="Schenkman M, et al.",
                evidence_level="Level I",
            ),
        ],
        "clinical_pearls": [
            "Time therapy sessions during 'on' medication periods for best performance",
            "Freezing often occurs at doorways, turns, and when initiating movement",
            "Counting or marching rhythm can overcome freezing episodes",
            "Dual-task interference is common - train it, don't just avoid it",
            "Depression and apathy affect participation - screen and address",
            "Caregiver training is essential for carryover",
        ],
    },
    "falls": {
        "keywords": ["fall", "falling", "balance", "unsteady", "dizzy"],
        "summary": "Multimodal exercise programs including balance, strength, and functional training reduce falls by 23-40%. Home hazard modification and medication review are essential adjuncts.",
        "detailed_answer": """## Evidence-Based Falls Prevention

### Risk Factor Assessment
- Previous falls (strongest predictor)
- Gait and balance impairment
- Polypharmacy (≥4 medications)
- Visual impairment
- Cognitive impairment
- Environmental hazards
- Orthostatic hypotension
- Fear of falling

### Recommended Interventions

**1. Exercise Programs (Level I Evidence)**
- Balance training: progressive, challenging, 3+ hours/week
- Strength training: lower extremity focus
- Tai Chi: 40% reduction in falls (Level I)
- Otago Exercise Program: proven in community-dwelling older adults

**2. Home Modification**
- Remove throw rugs, secure cords
- Adequate lighting (especially stairs, bathroom)
- Grab bars at toilet and shower
- Non-slip mats
- Bed/chair height optimization

**3. Medication Review**
- Reduce psychotropics if possible
- Address orthostatic hypotension
- Review sedating medications

**4. Vision Correction**
- Updated prescription
- Single-lens glasses for outdoor walking
- Cataract surgery if indicated

### Fall Recovery Training
- Practice floor-to-stand transfers
- Teach multiple strategies (furniture crawl, backward scoot)
- Build confidence through repetition""",
        "key_recommendations": [
            "Multimodal exercise program 3+ hours/week",
            "Progressive balance training (must be challenging)",
            "Lower extremity strengthening",
            "Home safety evaluation and modifications",
            "Medication review with physician",
            "Vision assessment and correction",
            "Floor transfer training (getting up after fall)",
        ],
        "outcome_measures": [
            "Timed Up and Go (TUG)",
            "Berg Balance Scale",
            "30-Second Sit-to-Stand",
            "4-Stage Balance Test",
            "Falls Efficacy Scale (FES-I)",
            "ABC Scale (Activities-specific Balance Confidence)",
        ],
        "citations": [
            Citation(
                title="Exercise for preventing falls in older people",
                source="Cochrane Database of Systematic Reviews",
                year=2019,
                authors="Sherrington C, et al.",
                evidence_level="Level I",
            ),
            Citation(
                title="CDC STEADI Falls Prevention Toolkit",
                source="Centers for Disease Control and Prevention",
                year=2023,
                evidence_level="CPG",
            ),
        ],
        "clinical_pearls": [
            "Fear of falling is itself a fall risk factor - address it directly",
            "Balance training must be challenging to be effective",
            "Near-falls are as important as actual falls in risk assessment",
        ],
    },
    "low back pain": {
        "keywords": ["low back", "lbp", "lumbar", "back pain"],
        "summary": "Exercise therapy is first-line treatment. Patient education, activity modification, and manual therapy as adjunct. Avoid imaging for non-specific LBP <6 weeks.",
        "detailed_answer": """## Evidence-Based Low Back Pain Management

### Classification
- Non-specific LBP (90% of cases)
- Radiculopathy with neurological signs
- Serious pathology (<1% - red flags)

### First-Line Interventions

**1. Patient Education (Level I)**
- Reassurance about favorable prognosis
- Stay active, avoid bed rest
- Address fear-avoidance beliefs
- Self-management strategies

**2. Exercise Therapy (Level I)**
- Motor control exercises for chronic LBP
- McKenzie method if directional preference
- General exercise equally effective as specific
- Graded activity approach

**3. Manual Therapy (Level II)**
- Best as adjunct to exercise
- Thrust or non-thrust mobilization
- Short-term pain relief

### What Doesn't Work
- Bed rest (harmful)
- Routine imaging (not helpful for non-specific LBP)
- Passive modalities alone (TENS, ultrasound)
- Opioids for chronic LBP""",
        "key_recommendations": [
            "Exercise therapy as first-line treatment",
            "Patient education on self-management",
            "Stay active - avoid bed rest",
            "Manual therapy as adjunct to exercise",
            "Address psychosocial factors (yellow flags)",
            "No imaging for non-specific LBP <6 weeks",
        ],
        "outcome_measures": [
            "Oswestry Disability Index (ODI)",
            "Roland-Morris Disability Questionnaire",
            "Numeric Pain Rating Scale (NPRS)",
            "Fear-Avoidance Beliefs Questionnaire (FABQ)",
        ],
        "citations": [
            Citation(
                title="Clinical Practice Guidelines for Physical Therapy in Low Back Pain",
                source="JOSPT",
                year=2021,
                evidence_level="CPG",
            ),
            Citation(
                title="Exercise therapy for chronic low back pain",
                source="Cochrane Database",
                year=2021,
                authors="Hayden JA, et al.",
                evidence_level="Level I",
            ),
        ],
        "clinical_pearls": [
            "Fear-avoidance is stronger predictor than pain intensity",
            "Match exercise to patient preference for adherence",
            "Early return to activity improves outcomes",
        ],
    },
    "stroke": {
        "keywords": ["stroke", "cva", "hemiplegia", "hemiparesis"],
        "summary": "Early mobilization, high-intensity task-specific practice, and repetition drive neuroplasticity. CIMT for upper extremity, task-specific gait training for mobility.",
        "detailed_answer": """## Evidence-Based Stroke Rehabilitation

### Principles of Neuroplasticity
- Use it or lose it / Use it and improve it
- Specificity - train what you want to improve
- Repetition matters - hundreds of repetitions needed
- Intensity matters - more therapy = better outcomes
- Time matters - but recovery continues beyond 6 months

### Upper Extremity

**Constraint-Induced Movement Therapy (Level I)**
- For patients with some active wrist/finger extension
- Constrains unaffected arm 90% of waking hours
- Massed practice with affected arm (3-6 hours/day)
- Modified CIMT protocols available

**Task-Specific Training**
- High repetition of meaningful tasks
- Shaping and progression
- Real-world object manipulation

### Gait and Mobility

**Task-Specific Gait Training (Level I)**
- Overground walking preferred when possible
- Treadmill training with/without BWS
- Focus on speed, endurance, and symmetry

**Balance**
- Weight-shifting, reaching
- Perturbation training
- Dual-task progression

### Intensity Recommendations
- Minimum 3 hours/day of therapy during inpatient rehab
- Higher intensity = better outcomes (dose-response relationship)
- Continue community exercise post-discharge""",
        "key_recommendations": [
            "Early mobilization within 24-48 hours",
            "High-repetition, task-specific practice",
            "CIMT for appropriate upper extremity candidates",
            "Task-specific gait training for ambulation",
            "Minimum 3 hours/day therapy in acute rehab",
            "Continue structured exercise post-discharge",
        ],
        "outcome_measures": [
            "Fugl-Meyer Assessment",
            "Action Research Arm Test (ARAT)",
            "10-Meter Walk Test",
            "6-Minute Walk Test",
            "Berg Balance Scale",
            "Functional Independence Measure (FIM)",
        ],
        "citations": [
            Citation(
                title="Guidelines for Adult Stroke Rehabilitation",
                source="AHA/ASA",
                year=2016,
                evidence_level="CPG",
            ),
            Citation(
                title="AVERT Trial - Very Early Mobilization",
                source="Lancet",
                year=2015,
                evidence_level="Level I",
            ),
        ],
        "clinical_pearls": [
            "Learned non-use develops within days - prevent it early",
            "Screen for depression - affects 1/3 of patients",
            "Aphasia doesn't mean cognitive impairment",
        ],
    },
    "rotator cuff": {
        "keywords": ["rotator cuff", "shoulder", "supraspinatus", "impingement"],
        "summary": "Exercise therapy is effective for most rotator cuff tendinopathy and partial tears. Progressive loading, scapular control, and patient education are key.",
        "detailed_answer": """## Evidence-Based Rotator Cuff Management

### Conservative First
- 70-80% of patients improve with PT alone
- Even full-thickness tears may respond to conservative care
- Surgery for acute traumatic tears or failed conservative management

### Exercise Progression

**Phase 1: Pain Reduction**
- Relative rest (not immobilization)
- Pain-free ROM
- Isometrics in neutral

**Phase 2: Strengthening**
- Isotonic rotator cuff (ER/IR)
- Scapular stabilization
- Closed chain exercises

**Phase 3: Functional**
- Eccentric loading
- Sport/work-specific training
- Plyometrics if indicated

### Key Principles
- Load management is critical
- Address scapular dyskinesis
- Gradual return to overhead activities""",
        "key_recommendations": [
            "Progressive exercise therapy as first-line",
            "Rotator cuff and scapular strengthening",
            "Load management and activity modification",
            "Eccentric exercises for tendinopathy",
            "Surgery for failed conservative care or acute trauma",
        ],
        "outcome_measures": [
            "DASH or QuickDASH",
            "Penn Shoulder Score",
            "Shoulder ROM (goniometry)",
            "Strength testing (dynamometry)",
        ],
        "citations": [
            Citation(
                title="Exercise therapy for rotator cuff tendinopathy",
                source="BJSM",
                year=2022,
                evidence_level="Level I",
            ),
        ],
        "clinical_pearls": [
            "Painful arc 60-120° suggests subacromial involvement",
            "Lag signs indicate full-thickness tears",
            "Night pain is common, doesn't mandate surgery",
        ],
    },
    "vestibular": {
        "keywords": ["vestibular", "vertigo", "bppv", "dizzy", "dizziness"],
        "summary": "BPPV is treated with repositioning maneuvers (Epley, BBQ roll). Vestibular hypofunction requires habituation, gaze stabilization, and balance training.",
        "detailed_answer": """## Evidence-Based Vestibular Rehabilitation

### BPPV (Benign Paroxysmal Positional Vertigo)

**Posterior Canal (most common)**
- Epley maneuver (canalith repositioning)
- 80%+ success rate in 1-3 treatments
- Semont maneuver as alternative

**Horizontal Canal**
- BBQ roll / Lempert maneuver
- May need multiple treatments

### Vestibular Hypofunction

**Gaze Stabilization (Level I)**
- VORx1: fixate on target, move head
- VORx2: target and head move opposite
- Progress speed, complexity, visual background

**Habituation**
- Repeated exposure to provoking movements
- Reduces motion sensitivity
- Brandt-Daroff exercises

**Balance Training**
- Progressive challenges to balance systems
- Reduce visual/somatosensory input
- Dual-task activities""",
        "key_recommendations": [
            "Identify type: BPPV vs. vestibular hypofunction",
            "Epley maneuver for posterior canal BPPV",
            "Gaze stabilization exercises for hypofunction",
            "Habituation for motion sensitivity",
            "Progressive balance training",
        ],
        "outcome_measures": [
            "Dix-Hallpike Test (BPPV diagnosis)",
            "Dizziness Handicap Inventory (DHI)",
            "Dynamic Visual Acuity Test",
            "Balance Error Scoring System (BESS)",
        ],
        "citations": [
            Citation(
                title="Vestibular Rehabilitation for Unilateral Peripheral Vestibular Dysfunction",
                source="Cochrane Database",
                year=2015,
                evidence_level="Level I",
            ),
            Citation(
                title="BPPV Clinical Practice Guideline",
                source="JOSPT",
                year=2022,
                evidence_level="CPG",
            ),
        ],
        "clinical_pearls": [
            "BPPV: brief episodes (<1 min), position-triggered",
            "Vestibular hypofunction: constant symptoms, worse with head movement",
            "Central causes need different management - know red flags",
        ],
    },
    "dysphagia": {
        "keywords": ["dysphagia", "swallow", "swallowing", "aspiration"],
        "summary": "Dysphagia management includes diet modification, compensatory strategies, and rehabilitative exercises. Instrumental assessment (VFSS/FEES) guides treatment.",
        "detailed_answer": """## Evidence-Based Dysphagia Management

### Assessment
- Clinical swallow evaluation
- Instrumental: VFSS (videofluoroscopy) or FEES
- Identify aspiration risk and physiological deficits

### Compensatory Strategies

**Postural Techniques**
- Chin tuck: reduces airway entrance
- Head turn: directs bolus to stronger side
- Head tilt: uses gravity

**Diet Modifications**
- IDDSI framework for texture levels
- Thickened liquids if thin liquid aspiration
- Modified food textures

### Rehabilitative Exercises

**Lingual Strengthening**
- Iowa Oral Performance Instrument (IOPI)
- Tongue press exercises

**Effortful Swallow**
- Increases base of tongue retraction
- Improves pharyngeal clearance

**Mendelsohn Maneuver**
- Prolongs laryngeal elevation
- Increases UES opening

**Shaker Exercise**
- Strengthens suprahyoid muscles
- Improves UES opening""",
        "key_recommendations": [
            "Instrumental assessment to identify aspiration",
            "Match strategies to physiological deficit",
            "Diet modification based on objective findings",
            "Rehabilitative exercises for long-term improvement",
            "Oral hygiene to reduce aspiration pneumonia risk",
        ],
        "outcome_measures": [
            "Penetration-Aspiration Scale (PAS)",
            "Functional Oral Intake Scale (FOIS)",
            "EAT-10 screening tool",
            "Mann Assessment of Swallowing Ability (MASA)",
        ],
        "citations": [
            Citation(
                title="ASHA Practice Guidelines for Dysphagia",
                source="ASHA",
                year=2020,
                evidence_level="CPG",
            ),
        ],
        "clinical_pearls": [
            "Silent aspiration is common - instrumental assessment essential",
            "Oral hygiene is critical for preventing aspiration pneumonia",
            "Fatigue affects swallowing - assess at end of meals too",
        ],
    },
}


def find_matching_topics(query: str) -> list[str]:
    """Find topics that match the query based on keywords."""
    query_lower = query.lower()
    matches = []

    for topic, data in KNOWLEDGE_BASE.items():
        for keyword in data["keywords"]:
            if keyword in query_lower:
                matches.append(topic)
                break

    return matches


@router.post("/ask", response_model=KnowledgeResponse)
async def ask_clinical_question(query: KnowledgeQuery, current_user: Provider = Depends(get_current_user)):
    """Ask a clinical question and get evidence-based information.

    You can ask:
    - Direct questions: "What's the evidence for LSVT BIG?"
    - Clinical scenarios: "76 y/o male with Parkinson's, difficulty walking, falls..."
    - Treatment approaches: "How do I treat rotator cuff tendinopathy?"

    Returns synthesized evidence with citations.
    """
    matches = find_matching_topics(query.question)

    if not matches:
        # No direct match - return helpful response
        return KnowledgeResponse(
            question=query.question,
            summary="No specific evidence found for this query in the current knowledge base.",
            detailed_answer=f"""I don't have specific evidence-based information for this query yet.

**Suggestions:**
1. Try rephrasing with common clinical terms
2. Search PubMed or clinical practice guidelines directly
3. Check discipline-specific resources (APTA, AOTA, ASHA)

**Available topics in my knowledge base:**
{', '.join(KNOWLEDGE_BASE.keys())}

*Note: This is a demo version. Full implementation would search vector stores and PubMed.*""",
            key_recommendations=["Search clinical practice guidelines", "Consult PubMed for current research"],
            related_topics=list(KNOWLEDGE_BASE.keys()),
        )

    # Combine knowledge from all matching topics
    primary_topic = matches[0]
    knowledge = KNOWLEDGE_BASE[primary_topic]

    # Build comprehensive response
    detailed = knowledge["detailed_answer"]

    # If multiple topics match, add relevant info from others
    if len(matches) > 1:
        detailed += "\n\n---\n\n## Related Evidence\n\n"
        for topic in matches[1:]:
            other = KNOWLEDGE_BASE[topic]
            detailed += f"### {topic.title()}\n{other['summary']}\n\n"

    all_citations = []
    all_pearls = []
    all_recommendations = []
    all_outcome_measures = []
    all_contraindications = []

    for topic in matches:
        data = KNOWLEDGE_BASE[topic]
        all_citations.extend(data.get("citations", []))
        all_pearls.extend(data.get("clinical_pearls", []))
        all_recommendations.extend(data.get("key_recommendations", []))
        all_outcome_measures.extend(data.get("outcome_measures", []))
        all_contraindications.extend(data.get("contraindications", []))

    return KnowledgeResponse(
        question=query.question,
        summary=knowledge["summary"],
        detailed_answer=detailed,
        key_recommendations=all_recommendations,
        contraindications=all_contraindications,
        outcome_measures=all_outcome_measures,
        citations=all_citations,
        clinical_pearls=all_pearls,
        related_topics=[t for t in KNOWLEDGE_BASE.keys() if t not in matches],
    )


@router.get("/topics")
async def list_available_topics(discipline: str = Query("PT", description="PT, OT, or SLP"), current_user: Provider = Depends(get_current_user)):
    """List available knowledge topics by discipline."""
    topics_by_discipline = {
        "PT": [
            "Parkinson's Disease",
            "Falls Prevention",
            "Low Back Pain",
            "Stroke Rehabilitation",
            "Rotator Cuff",
            "Vestibular Rehabilitation",
            "Total Joint Replacement",
            "ACL Reconstruction",
        ],
        "OT": [
            "Stroke - Upper Extremity",
            "Parkinson's Disease - ADLs",
            "Hand Therapy",
            "Cognitive Rehabilitation",
            "Falls Prevention - Home Mods",
        ],
        "SLP": [
            "Dysphagia",
            "Aphasia",
            "Voice Disorders",
            "Cognitive-Communication",
            "Parkinson's Disease - LSVT LOUD",
        ],
    }

    return {
        "discipline": discipline,
        "topics": topics_by_discipline.get(discipline, topics_by_discipline["PT"]),
        "total_in_knowledge_base": len(KNOWLEDGE_BASE),
    }


@router.get("/quick/{topic}")
async def quick_lookup(
    topic: str,
    aspect: str = Query("overview", description="overview, exercises, red_flags, outcome_measures"),
    current_user: Provider = Depends(get_current_user),
):
    """Quick lookup for common clinical questions."""
    topic_lower = topic.lower().replace("-", " ").replace("_", " ")

    # Find matching topic
    matched = None
    for key, data in KNOWLEDGE_BASE.items():
        if key in topic_lower or any(kw in topic_lower for kw in data["keywords"]):
            matched = key
            break

    if not matched:
        raise HTTPException(status_code=404, detail=f"Topic not found: {topic}")

    knowledge = KNOWLEDGE_BASE[matched]

    if aspect == "overview":
        return {"topic": matched, "aspect": aspect, "content": knowledge["summary"]}
    elif aspect == "outcome_measures":
        return {"topic": matched, "aspect": aspect, "content": knowledge.get("outcome_measures", [])}
    elif aspect == "red_flags" or aspect == "contraindications":
        return {"topic": matched, "aspect": aspect, "content": knowledge.get("contraindications", ["See detailed guidelines"])}
    else:
        return {"topic": matched, "aspect": aspect, "content": knowledge.get("key_recommendations", [])}


@router.get("/guidelines")
async def list_guidelines(discipline: str = Query("PT"), current_user: Provider = Depends(get_current_user)):
    """List available clinical practice guidelines."""
    guidelines = {
        "PT": [
            {"title": "Parkinson Disease Physical Therapy Guideline", "organization": "AAN/AHA", "year": 2022},
            {"title": "Low Back Pain CPG", "organization": "APTA - JOSPT", "year": 2021},
            {"title": "Falls Prevention in Older Adults", "organization": "CDC STEADI", "year": 2023},
            {"title": "Neck Pain CPG", "organization": "APTA - JOSPT", "year": 2017},
            {"title": "BPPV Clinical Practice Guideline", "organization": "APTA - JOSPT", "year": 2022},
            {"title": "Stroke Rehabilitation Guidelines", "organization": "AHA/ASA", "year": 2016},
        ],
        "OT": [
            {"title": "Stroke Rehabilitation Practice Guidelines", "organization": "AOTA", "year": 2015},
            {"title": "TBI Practice Guidelines", "organization": "AOTA", "year": 2016},
        ],
        "SLP": [
            {"title": "Dysphagia Practice Guidelines", "organization": "ASHA", "year": 2020},
            {"title": "Aphasia Practice Guidelines", "organization": "ASHA", "year": 2019},
        ],
    }

    return {"discipline": discipline, "guidelines": guidelines.get(discipline, [])}
