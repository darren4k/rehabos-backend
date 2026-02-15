# memU Namespace Convention

All three services share the same memU/pgvector instance on DGX (192.168.68.127:5432/memu).
Patient data is isolated by namespace prefix in the user_id field.

| Service | Prefix | Example user_id | Categories |
|---|---|---|---|
| magda_health | (none) | `patient_123` | enrollment, prom_history, engagement, preferences, triggers, provider_interactions, milestones, journey |
| rehab-os | `rehab:` | `rehab:patient_123` | consultation_history, treatment_outcomes, patient_preferences, clinical_observations, functional_progress, discharge_planning |
| docpilot | `docpilot:` | `docpilot:patient_123` | encounter_notes, billing_history, treatment_provided, voice_transcriptions |

## Cross-Service Queries
To build a complete patient view, query all three namespaces:
- `patient_id` → magda engagement/enrollment data
- `rehab:{patient_id}` → clinical reasoning and outcomes
- `docpilot:{patient_id}` → documentation and billing history
