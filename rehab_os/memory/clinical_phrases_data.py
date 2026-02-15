"""
Clinical phrases data for memU seeding.
Mirror of the frontend clinical-phrases.ts but structured for backend use.
"""

MEDICARE_GUIDELINES = [
    """Medicare Coverage Criteria for Rehabilitation Therapy (CMS Benefits Policy Manual Ch.15 §220.2):
Services must be: (1) Reasonable and necessary for treatment of the patient's illness or injury under §1862(a)(1)(A),
(2) Require the skills of a qualified therapist — cannot be safely/effectively carried out by nonskilled personnel,
(3) Rehabilitative in nature — aimed at improving, restoring, or maintaining function.
The deciding factors are ALWAYS whether services are considered reasonable, effective treatments
for the patient's condition and require the skills of a therapist, or whether they can be safely
and effectively carried out by nonskilled personnel.""",

    """Maintenance Programs (§220.2D): Skilled therapy services that do not meet criteria for
rehabilitative therapy may be covered as maintenance therapy. Goals: maintain functional status
or prevent/slow further deterioration. Coverage available when:
(1) Establishment/design of maintenance programs — covered when specialized skill, knowledge
and judgment of a qualified therapist are required.
(2) Instruction — covered when skilled services needed to instruct patient/caregiver.
(3) Periodic reevaluation — covered when skilled reassessment is needed.
(4) Delivery — covered ONLY when individualized assessment demonstrates:
    (a) procedures are of such complexity/sophistication that therapist skills are required, OR
    (b) patient's special medical complications require therapist skills even for otherwise routine procedures.
Unlike rehabilitative therapy, maintenance program coverage does NOT depend on potential for improvement.""",

    """Documentation Requirements for Skilled Care Justification:
Each note should demonstrate: Complexity (why this patient needs skilled assessment),
Skilled interventions (what the therapist did that requires training),
Clinical reasoning (modifications/progressions based on patient response),
Functional relevance (how interventions relate to functional goals),
Safety considerations (why unskilled personnel cannot safely perform these),
Progress or justification (evidence of improvement OR medical necessity for maintenance).
Common denial reasons: lacks skilled language, no functional goals, no clinical decision-making evidence,
progress not documented, maintenance needs not justified, generic templates without patient-specific details.""",
]

SKILLED_INTERVENTION_PHRASES = {
    "97110": {
        "Aerobic Capacity": [
            "Monitoring and modification of exercise intensity based on physiologic response",
            "Progression of exercise prescription",
            "Adaptation of exercise program given patient and/or condition specific needs",
            "Calculation of appropriate target heart rate at an intensity of 70-80% of max HR",
            "Patient/caregiver education including benefits of exercise, effects of aerobic training, disease processes and impact on function",
            "Education on energy conservation techniques, self-monitoring of exercise intensity",
        ],
        "Muscle Length / Flexibility": [
            "Static stretching, dynamic stretching, instruction in proper breathing techniques",
            "Facilitation of proper alignment, application of orthotic post stretch",
            "Contract relax, hold relax, contract-relax agonist contract",
            "Assessment of patient tolerance, assessment of abnormal joint articulation",
            "Correlation of individual's pain report with onset of tissue resistance",
            "Analysis of end feels, measurement and analysis of joint mobility",
            "Analysis of progress related to ROM improvements and resultant impact on functional outcomes",
        ],
        "Strength": [
            "Facilitation of slow and controlled movement pattern without substitution",
            "Adaptation of exercise program based on patient and/or condition specific needs",
            "Analysis of impact of intervention on functional outcomes",
            "Development and progression of exercise prescription",
            "Assessment of physiologic tolerance and monitoring/modification of exercise intensity based upon response",
            "With use of cuff weights, elastic bands, free weights, isotonic exercise machines, pulleys, body weight and gravity",
            "Utilizing AAROM, AROM, open chain and closed chain activity at resistance required to achieve momentary muscle fatigue",
            "Patient education including quality of movement, BORG/RPE scale, disease process considerations",
        ],
    },
    "97116": {
        "Gait Training": [
            "Facilitation of weight shift during mobility",
            "Facilitation of center of mass over BOS",
            "Training in correct foot placement during gait",
            "Facilitation of symmetrical stride length and stance",
            "Training in correct sequencing of gait with assistive device",
            "Training strategies to safely maneuver around obstacles",
            "Facilitation of knee stability during weight acceptance",
            "Facilitation of gait stability with dual task challenge and increased attentional demands",
        ],
        "Pre-Gait Training": [
            "Facilitate weight shifting in standing",
            "Improve ability to maintain unsupported standing and single limb stance",
            "Improve ability to initiate a step",
            "Facilitate appropriate base of support and trunk control in preparation for upright mobility",
        ],
    },
    "97530": {
        "Balance & Postural Control": [
            "Facilitation of body position awareness and correction in space",
            "Activities to facilitate postural control and reaching outside base of support",
            "Dynamic balance activities during sitting and standing",
            "Facilitation of adjustment of center of mass over base of support during functional activities",
            "Weight shifting to improve safety with unsupported sit/stand",
        ],
        "Bed Mobility": [
            "Training in log rolling, facilitation of bridging",
            "Training in repositioning to facilitate pressure relief",
            "Training in scooting techniques to facilitate repositioning in bed",
            "Training in sit to supine and supine to sit",
        ],
        "Transfers": [
            "Facilitation of adjustment of center of mass over base of support during transfers",
            "Facilitation of controlled standing to sitting and pivotal rotation during transfers",
            "Training in safe performance for sit to stand, stand pivot, sliding board transfers",
            "Training in toilet, tub/shower, car, and floor transfers",
            "Patient/caregiver training in cueing techniques to promote transfer independence and reduce fall risk",
        ],
    },
    "97112": {
        "Neuro Re-education": [
            "Joint approximation techniques, bilateral integration techniques",
            "NDT and PNF techniques",
            "Facilitate neuromuscular functional synergy patterns to improve mobility",
            "Facilitation of body awareness in space, crossing midline, isolated movement",
            "Graded tactile cues to facilitate motor planning and initiation",
            "Proprioceptive-kinesthetic cueing to facilitate motor control/patterned movements",
        ],
        "Skilled Positioning": [
            "Assessment of patient response to positioning adjustments",
            "Bed/chair positioning for contracture management and postural alignment",
            "Compensations for hypertonicity and hypotonicity",
            "Edema reduction, control and management techniques",
            "Tone inhibition and management techniques",
            "Training with emphasis on proximal stability/distal control",
        ],
        "Vestibular": [
            "Vestibular accommodation and re-training",
            "Gaze stabilization techniques with head turns",
            "Habituation exercises",
            "Training in compensatory strategies during transitional movements",
        ],
        "Balance — Sitting": [
            "Dynamic sitting balance training on unstable surfaces",
            "Facilitation of anticipatory postural adjustments",
            "Facilitation of balance and righting reactions",
            "Techniques to challenge equilibrium and limits of stability",
        ],
        "Balance — Standing": [
            "Dynamic standing balance training on unstable surfaces",
            "Single leg stance activities, tandem walking, braiding exercises",
            "Training in ankle, hip, stepping strategies",
            "Balance recovery strategies during mobility",
        ],
    },
    "97140": {
        "Soft Tissue Mobilization": [
            "Soft tissue mobilization to reduce muscle guarding and improve tissue extensibility",
            "Myofascial release techniques to address tissue restrictions",
            "Skilled assessment of tissue quality and mobility",
        ],
        "Joint Mobilization": [
            "Joint mobilization grades I-IV to improve joint mobility",
            "Assessment of joint play and accessory motion",
            "Analysis of end feels and correlation with ROM limitations",
        ],
    },
}

PROGRESS_INDICATORS = [
    "Patient compliant with adaptations",
    "Actively participated with skilled interventions",
    "Compliant with trained techniques",
    "Decreased need for demonstration of tasks",
    "Demonstrated insight regarding functional deficits/condition",
    "Decreased need for instruction",
    "Decreased need for supervision",
    "Exhibits improvement with structured tasks",
    "Generalization of trained skills exhibited",
    "Decreased need for type and amount of cues",
    "Increased activity participation / fewer functional breaks",
    "Exhibited carryover of skills to untrained tasks",
    "Increased challenge with tasks",
    "Observably less pain",
    "Exhibited good self-monitoring skills",
    "Participated in tasks of increased complexity",
    "Recalls trained strategies",
    "Reports decreased pain",
    "Safely adapting to change and self-correcting without cues",
]

BILLING_JUSTIFICATION = """Skilled Care Justification Language Patterns:
Use language that demonstrates clinical reasoning:
- "Facilitation of..." (implies skilled hands-on guidance)
- "Assessment of..." (implies clinical judgment)
- "Modification of... based on..." (implies skilled decision-making)
- "Adaptation of... given patient-specific..." (implies individualized care)
- "Analysis of... and correlation with..." (implies clinical interpretation)
- "Monitoring and modification based on physiologic response" (implies ongoing assessment)
- "Due to complexity of patient presentation" (justifies skilled care)
- "Beyond the scope of a home program" (distinguishes from unskilled care)
- "Requires ongoing skilled assessment, modification of treatment parameters, and clinical decision-making"
- "Patient's special medical complications require the skills of a qualified therapist"
"""
