"""
FEMA Policy Document Corpus — shared across multi-agent tutorials.

Merged superset of policy documents used by tutorials 10 (multi-agent supervisor)
and 12 (CrewAI multi-agent). Each document contains realistic (but synthetic)
summaries of FEMA protocols and guidelines, used as a knowledge base for
policy-search tools.

Usage:
    from utils.policy_docs import get_policy_documents
    policy_docs = get_policy_documents()
"""


def get_policy_documents() -> dict[str, str]:
    """Return the FEMA policy document corpus (11 documents)."""
    return {
        # --- Evacuation & General Response ---
        "evacuation_protocols": (
            "FEMA Evacuation Protocols (ICS-300): All evacuation orders must follow the Incident Command System. "
            "Zone-based evacuation proceeds from highest-risk zones outward. Mandatory evacuation requires "
            "governor authorization. Evacuation routes must be pre-designated and communicated via Wireless "
            "Emergency Alerts (WEA). Special needs populations require dedicated transport. Shelter capacity "
            "must be verified before issuing orders. Pet-friendly shelters must be available per the PETS Act."
        ),
        "disaster_declaration": (
            "FEMA Disaster Declaration Process: Local government requests state assistance. If overwhelmed, "
            "governor requests federal declaration from the President. FEMA conducts Preliminary Damage "
            "Assessment (PDA). Two declaration types: Emergency Declaration (Category B assistance, $5M cap) "
            "and Major Disaster Declaration (all assistance categories, no cap). Individual Assistance (IA) "
            "and Public Assistance (PA) can be authorized separately. Timeline: PDA within 10 days, "
            "presidential decision within 5 days of governor's request."
        ),
        # --- Federal Assistance ---
        "aid_eligibility": (
            "FEMA Individual Assistance Eligibility: Applicants must be U.S. citizens, non-citizen nationals, "
            "or qualified aliens. Primary residence must be in the declared disaster area. Damage must be "
            "disaster-caused and not covered by insurance. Apply within 60 days of declaration (extensions "
            "possible). Types of assistance: Housing Assistance (rental, repair, replacement up to $42,500), "
            "Other Needs Assistance (medical, dental, funeral, personal property, transportation, childcare). "
            "SBA disaster loans available for amounts exceeding FEMA maximums."
        ),
        "federal_assistance_guidelines": (
            "FEMA Individual Assistance (IA) Program: Eligible applicants include US citizens, non-citizen nationals, "
            "and qualified aliens in presidentially declared disaster areas. Assistance types: Housing Assistance "
            "(rental, repair, replacement), Other Needs Assistance (medical, dental, funeral, transportation), "
            "and Crisis Counseling. Maximum grant per household is set annually ($42,500 for 2024). Applications "
            "must be filed within 60 days of disaster declaration. SBA disaster loans available for amounts exceeding grants."
        ),
        # --- Flood ---
        "flood_response": (
            "FEMA Flood Response Procedures: Activate National Flood Insurance Program (NFIP) coordination. "
            "Deploy Urban Search and Rescue (US&R) teams within 6 hours. Establish water rescue staging "
            "areas. Issue flash flood warnings via NWS partnership. Coordinate with Army Corps of Engineers "
            "for levee assessments. Post-flood: conduct rapid needs assessment within 72 hours. Distribute "
            "flood cleanup kits. Monitor waterborne disease risks. Begin mold remediation guidance within 48 hours."
        ),
        "flood_response_procedures": (
            "FEMA Flood Response (NRF ESF-3): Pre-event: activate flood gauges, pre-position pumps and sandbags, "
            "issue Flash Flood Watches 24-48h in advance. During event: deploy Urban Search and Rescue (US&R) "
            "teams within 6 hours, establish Points of Distribution (PODs) for water and supplies. Post-event: "
            "conduct Preliminary Damage Assessments (PDAs) within 72 hours, activate National Flood Insurance "
            "Program (NFIP) claims process. Flood zones A and V require mandatory flood insurance for federally-backed mortgages."
        ),
        # --- Wildfire ---
        "wildfire_safety": (
            "FEMA Wildfire Safety Guidelines: Create defensible space of 100 feet around structures. "
            "Zone 1 (0-30 ft): remove all dead vegetation and flammable materials. Zone 2 (30-100 ft): "
            "reduce and space vegetation. Have a go-bag ready with essentials for 72 hours. Know two "
            "evacuation routes from your area. Close all windows and doors before evacuating. Wear "
            "protective clothing: long sleeves, cotton/wool, N95 mask. Monitor air quality index (AQI) daily."
        ),
        "wildfire_management": (
            "FEMA Wildfire Response (NRF ESF-4): Coordinate with USFS and state forestry agencies. Pre-position "
            "resources when Fire Weather Watch is issued. Activate Fire Management Assistance Grants (FMAG) for "
            "state cost-sharing. Evacuation triggers: fire within 2 miles of populated areas with wind >25mph. "
            "Post-fire: activate Burned Area Emergency Response (BAER) teams within 7 days. Debris flow risk "
            "assessment required for all slopes >30%% in burn scar areas. Community wildfire protection plans (CWPPs) required."
        ),
        # --- Hurricane ---
        "hurricane_preparedness": (
            "FEMA Hurricane Preparedness Guide: Monitor National Hurricane Center (NHC) advisories. "
            "Hurricane Watch: conditions possible within 48 hours, begin preparations. Hurricane Warning: "
            "conditions expected within 36 hours, complete preparations and evacuate if ordered. "
            "Secure property: install storm shutters, reinforce garage doors, trim trees. Stock supplies "
            "for 7 days minimum (water: 1 gallon per person per day). Know your evacuation zone (A through E). "
            "Register with local special needs registry if applicable. "
            "FEMA Operational Response (NRF ESF-5): 120-hour watch triggers FEMA Region activation. 72-hour "
            "warning activates Incident Management Assistance Teams (IMAT). 48 hours: pre-stage commodities "
            "(MREs, water, tarps, generators) at Federal Staging Areas. 24 hours: activate ESF-1 (Transportation) "
            "for evacuation support. Storm surge zones require mandatory evacuation for Category 3+. Post-landfall: "
            "deploy FEMA Corps teams, activate Transitional Sheltering Assistance (TSA) within 14 days."
        ),
        # --- Earthquake ---
        "earthquake_response": (
            "FEMA Earthquake Response Protocol: Activate ShakeAlert early warning system. Immediate response: "
            "Drop, Cover, Hold On. Post-quake: expect aftershocks, check for structural damage before re-entry. "
            "Deploy structural engineers for rapid building assessments (ATC-20 inspections). Red/Yellow/Green "
            "tagging system for building safety. Activate Community Emergency Response Teams (CERT). "
            "Coordinate dam safety inspections. Restore utilities following gas-electric-water priority sequence."
        ),
        # --- Tornado ---
        "tornado_safety": (
            "FEMA Tornado Safety Procedures: Tornado Watch: conditions favorable, stay alert. Tornado Warning: "
            "tornado detected or imminent, take shelter immediately. Safe rooms: interior room on lowest floor, "
            "no windows. Mobile home residents must evacuate to sturdy shelter. FEMA safe room grants available "
            "under Hazard Mitigation Grant Program (HMGP). Post-tornado: avoid downed power lines, watch for "
            "gas leaks, document damage with photos for insurance claims. Community tornado shelters must meet "
            "FEMA P-361 standards."
        ),
    }
