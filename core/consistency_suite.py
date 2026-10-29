"""
Consistency & Semantic Stability Suite.

Contains 20 curated base questions across healthcare, science, reasoning, and technical domains,
each paired with 5 semantically equivalent paraphrases (100 paraphrases total, 120 prompt variants).
Measures answer stability, factual agreement, and semantic drift across variations.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class ConsistencyCluster:
    base_id: str
    category: str
    base_question: str
    paraphrases: List[str]
    expected_core_entities: List[str]
    description: str


CONSISTENCY_20_CLUSTERS: List[ConsistencyCluster] = [
    ConsistencyCluster(
        base_id="CONST-001",
        category="healthcare",
        base_question="What is the primary physiological function of the human heart?",
        paraphrases=[
            "What is the main biological purpose of the heart in humans?",
            "Can you explain what the human heart fundamentally does in the body?",
            "What role does the heart play in the human cardiovascular system?",
            "Describe the chief physiological responsibility of the human heart.",
            "In human anatomy, what is the principal function served by the heart?",
        ],
        expected_core_entities=["pump", "blood", "circulat", "oxygen", "cardiovascular"],
        description="Core physiological function of cardiac circulation.",
    ),
    ConsistencyCluster(
        base_id="CONST-002",
        category="pharmacology",
        base_question="What is the active therapeutic mechanism of action of Metformin in diabetes management?",
        paraphrases=[
            "How does the drug Metformin work to control blood glucose in diabetic patients?",
            "Explain the pharmacological mechanism through which Metformin treats diabetes.",
            "By what biological pathway does Metformin lower blood sugar levels?",
            "What is the therapeutic mode of action of Metformin in managing diabetes mellitus?",
            "How does Metformin exert its anti-diabetic effects in the human body?",
        ],
        expected_core_entities=["glucose", "hepatic", "gluconeogenesis", "insulin", "ampk"],
        description="Pharmacological mechanism of Metformin.",
    ),
    ConsistencyCluster(
        base_id="CONST-003",
        category="physics",
        base_question="What is the speed of light in a vacuum in meters per second?",
        paraphrases=[
            "How fast does light travel through a vacuum in m/s?",
            "What is the exact velocity of light in empty space in SI units?",
            "In meters per second, what is the speed of light in a vacuum?",
            "Can you provide the numeric speed of light traveling in a vacuum (meters/sec)?",
            "At what speed in m/s does electromagnetic radiation travel in a vacuum?",
        ],
        expected_core_entities=["299792458", "299,792,458", "3x10^8", "300,000"],
        description="Speed of light physical constant.",
    ),
    ConsistencyCluster(
        base_id="CONST-004",
        category="computing",
        base_question="Explain the fundamental principle behind a zero-trust cybersecurity architecture.",
        paraphrases=[
            "What are the core tenets of a zero-trust network security model?",
            "How does the zero-trust paradigm operate in modern cybersecurity?",
            "Can you summarize the foundational concepts of zero-trust architecture?",
            "What is the main philosophy and design pattern of zero-trust security?",
            "In cybersecurity, what does adopting a zero-trust framework entail?",
        ],
        expected_core_entities=["never trust", "verify", "least privilege", "authentication", "perimeter"],
        description="Core tenets of zero-trust architecture.",
    ),
    ConsistencyCluster(
        base_id="CONST-005",
        category="ayurveda",
        base_question="What are the three doshas in traditional Ayurvedic medicine and what elements do they represent?",
        paraphrases=[
            "In Ayurveda, name the three doshas along with their associated natural elements.",
            "Which three biological humors (doshas) and elemental pairings form the basis of Ayurveda?",
            "Can you list and explain the Tri-doshas and their constitutional elements in Ayurvedic theory?",
            "What are Vata, Pitta, and Kapha, and which natural elements correspond to each in Ayurveda?",
            "Describe the three constitutional doshas in Ayurvedic philosophy and their elemental compositions.",
        ],
        expected_core_entities=["vata", "pitta", "kapha", "air", "fire", "water", "earth"],
        description="Tri-doshas and elemental correspondences in Ayurveda.",
    ),
    ConsistencyCluster(
        base_id="CONST-006",
        category="general_science",
        base_question="What is the powerhouse organelle of eukaryotic cells and what molecule does it generate?",
        paraphrases=[
            "Which cellular organelle is called the powerhouse of the cell and what energy molecule does it synthesize?",
            "Name the organelle responsible for ATP production in eukaryotic organisms.",
            "What organelle generates cellular energy in the form of ATP in eukaryotes?",
            "In eukaryotic cell biology, what structure is known as the powerhouse and what does it produce?",
            "Identify the eukaryotic cell organelle that manufactures ATP and is often termed the cellular powerhouse.",
        ],
        expected_core_entities=["mitochondria", "mitochondrion", "atp", "adenosine"],
        description="Mitochondria and ATP production.",
    ),
    ConsistencyCluster(
        base_id="CONST-007",
        category="reasoning",
        base_question="If a bat and a ball cost $1.10 in total, and the bat costs $1.00 more than the ball, how much does the ball cost?",
        paraphrases=[
            "A baseball bat and a ball together cost $1.10. The bat is $1.00 more expensive than the ball. What is the price of the ball?",
            "Together, a bat and ball cost $1.10. If the bat's price is $1.00 higher than the ball's, calculate the cost of the ball.",
            "The total cost of a ball and a bat is $1.10. Given that the bat costs exactly one dollar more than the ball, determine the ball's cost.",
            "How much does the ball cost if a bat and ball combined are $1.10, and the difference in their price is $1.00?",
            "Solve for the cost of the ball: Bat + Ball = $1.10, and Bat = Ball + $1.00.",
        ],
        expected_core_entities=["0.05", "5 cent", "5¢", "five cent"],
        description="Cognitive reflection math puzzle.",
    ),
    ConsistencyCluster(
        base_id="CONST-008",
        category="cardiology",
        base_question="What does the QRS complex represent on an electrocardiogram (ECG)?",
        paraphrases=[
            "On an EKG tracing, what cardiac electrical event is represented by the QRS wave?",
            "Explain what the QRS complex signifies in a standard electrocardiogram reading.",
            "What physiological electrical activity corresponds to the QRS complex on an ECG?",
            "In electrocardiography, what cardiac phase does the QRS spike indicate?",
            "What does the occurrence of the QRS complex demonstrate regarding heart chambers?",
        ],
        expected_core_entities=["ventric", "depolariz"],
        description="ECG QRS complex electrophysiology.",
    ),
    ConsistencyCluster(
        base_id="CONST-009",
        category="computing",
        base_question="What is the average and worst-case time complexity of the standard QuickSort sorting algorithm?",
        paraphrases=[
            "In Big-O notation, what are the average and worst-case runtimes of QuickSort?",
            "What computational time complexity does QuickSort exhibit on average versus in the worst case scenario?",
            "State the average-case and worst-case algorithmic complexity for QuickSort.",
            "How does QuickSort perform in terms of time complexity in average vs worst cases?",
            "Provide the asymptotic Big-O complexity for QuickSort under typical and worst-case inputs.",
        ],
        expected_core_entities=["o(n log n)", "o(n^2)", "n log n", "quadratic"],
        description="QuickSort computational complexity.",
    ),
    ConsistencyCluster(
        base_id="CONST-010",
        category="geography",
        base_question="What is the capital city of Australia?",
        paraphrases=[
            "Which city serves as the federal capital of Australia?",
            "What is the official government capital of the Commonwealth of Australia?",
            "Name the capital city of the Australian nation.",
            "Can you tell me the designated capital of Australia?",
            "Where is Australia's federal capital located?",
        ],
        expected_core_entities=["canberra"],
        description="Capital of Australia.",
    ),
    ConsistencyCluster(
        base_id="CONST-011",
        category="endocrinology",
        base_question="What is the diagnostic threshold of HbA1c for the diagnosis of diabetes mellitus?",
        paraphrases=[
            "At what HbA1c percentage is a patient clinically diagnosed with diabetes?",
            "What level of glycated hemoglobin (HbA1c) indicates diabetes according to medical guidelines?",
            "What is the cutoff value of HbA1c required to establish a diabetes diagnosis?",
            "Which HbA1c reading confirms the clinical diagnosis of diabetes mellitus?",
            "What minimum HbA1c percentage meets the diagnostic criteria for diabetes?",
        ],
        expected_core_entities=["6.5", "48 mmol", "hba1c"],
        description="HbA1c diagnostic threshold.",
    ),
    ConsistencyCluster(
        base_id="CONST-012",
        category="general_science",
        base_question="What is the chemical formula for water and what atoms compose it?",
        paraphrases=[
            "What is the molecular formula of water and what elements make it up?",
            "Which chemical elements and ratio form a single water molecule?",
            "Can you give the chemical formula for water and state its atomic composition?",
            "How is water represented chemically in terms of formula and atoms?",
            "What elements and chemical notation define the water molecule?",
        ],
        expected_core_entities=["h2o", "hydrogen", "oxygen"],
        description="Chemical structure of water.",
    ),
    ConsistencyCluster(
        base_id="CONST-013",
        category="healthcare",
        base_question="What is the standard first-line treatment for acute anaphylactic allergic reactions?",
        paraphrases=[
            "What emergency medication is the first-line drug of choice for treating anaphylaxis?",
            "Which drug should be administered immediately as the primary treatment for severe anaphylaxis?",
            "What is the initial pharmacological intervention for a patient experiencing anaphylactic shock?",
            "In emergency medicine, what is the gold-standard first-line treatment for acute anaphylaxis?",
            "What is the immediate drug administered to treat life-threatening anaphylactic reactions?",
        ],
        expected_core_entities=["epinephrine", "adrenaline", "epipen"],
        description="First-line anaphylaxis management.",
    ),
    ConsistencyCluster(
        base_id="CONST-014",
        category="computing",
        base_question="What does ACID stand for in the context of database transaction management?",
        paraphrases=[
            "In relational database systems, explain the ACID acronym.",
            "What four properties make up the ACID acronym in database transactions?",
            "Can you define what each letter in ACID represents for database reliability?",
            "What is the meaning of ACID in database transaction processing?",
            "List the four transactional guarantees represented by the database term ACID.",
        ],
        expected_core_entities=["atomicity", "consistency", "isolation", "durability"],
        description="ACID database transaction guarantees.",
    ),
    ConsistencyCluster(
        base_id="CONST-015",
        category="physics",
        base_question="What is Newton's Third Law of Motion in simple terms?",
        paraphrases=[
            "How is Newton's Third Law of Motion stated in basic physics?",
            "Explain the principle of action and reaction described by Newton's Third Law.",
            "What does Newton's Third Law state regarding pairs of interacting forces?",
            "Can you describe Newton's Third Law of Motion and what it means for forces?",
            "In classical mechanics, what fundamental rule does Newton's Third Law express?",
        ],
        expected_core_entities=["equal", "opposite", "action", "reaction"],
        description="Newton's Third Law of Motion.",
    ),
    ConsistencyCluster(
        base_id="CONST-016",
        category="healthcare",
        base_question="Which blood type is considered the universal red blood cell donor?",
        paraphrases=[
            "What ABO/Rh blood group is designated as the universal donor for red blood cells?",
            "Which type of red blood cells can be safely transfused to recipients of any blood group?",
            "Identify the universal donor blood type for packed red blood cells.",
            "Who can donate red blood cells to anyone regardless of recipient blood type?",
            "Which specific blood type is known as the universal donor in transfusion medicine?",
        ],
        expected_core_entities=["o negative", "o-", "type o"],
        description="Universal red blood cell donor.",
    ),
    ConsistencyCluster(
        base_id="CONST-017",
        category="history",
        base_question="Who discovered penicillin and in what year did this discovery take place?",
        paraphrases=[
            "In what year and by whom was the antibiotic penicillin first discovered?",
            "Name the scientist who discovered penicillin and the year it occurred.",
            "Which researcher is credited with discovering penicillin and when did it happen?",
            "When was penicillin discovered, and who was the discovering scientist?",
            "Provide the name of penicillin's discoverer along with the year of discovery.",
        ],
        expected_core_entities=["fleming", "1928"],
        description="Discovery of penicillin.",
    ),
    ConsistencyCluster(
        base_id="CONST-018",
        category="ayurveda",
        base_question="What are the three dried fruits that compose the traditional Ayurvedic formulation Triphala?",
        paraphrases=[
            "Which three herbal fruits make up the classic Ayurvedic formula Triphala?",
            "Name the three botanical fruits combined in equal parts to produce Triphala.",
            "In Ayurveda, what three specific fruits are used to make Triphala churna?",
            "What ingredients constitute the Ayurvedic herbal blend known as Triphala?",
            "List the three dried medicinal fruits present in Triphala.",
        ],
        expected_core_entities=["amalaki", "amla", "bibhitaki", "haritaki"],
        description="Herbal composition of Triphala.",
    ),
    ConsistencyCluster(
        base_id="CONST-019",
        category="computing",
        base_question="What is the difference between synchronous and asynchronous execution in computer programming?",
        paraphrases=[
            "How do synchronous and asynchronous programming models differ from each other?",
            "Explain the distinction between synchronous vs asynchronous task execution in software.",
            "In software development, what sets synchronous operations apart from asynchronous operations?",
            "Compare synchronous execution with asynchronous execution in programming.",
            "What is the key contrast between synchronous and asynchronous code workflows?",
        ],
        expected_core_entities=["blocking", "non-blocking", "concurrent", "wait"],
        description="Synchronous vs Asynchronous programming paradigms.",
    ),
    ConsistencyCluster(
        base_id="CONST-020",
        category="pharmacology",
        base_question="What is the therapeutic classification and mechanism of action of Omeprazole?",
        paraphrases=[
            "What class of drug is Omeprazole and how does it suppress stomach acid production?",
            "Explain the pharmacological category and mode of action for Omeprazole.",
            "How does Omeprazole work pharmacologically to reduce gastric acidity?",
            "What is the mechanism by which Omeprazole acts as a gastric acid inhibitor?",
            "To which drug class does Omeprazole belong, and what enzyme does it target?",
        ],
        expected_core_entities=["proton pump", "ppi", "acid", "inhibitor"],
        description="Omeprazole PPI pharmacology.",
    ),
]


def get_all_consistency_clusters() -> List[ConsistencyCluster]:
    return CONSISTENCY_20_CLUSTERS


def evaluate_cluster_agreement(responses: List[str], expected_entities: List[str]) -> Dict[str, Any]:
    if not responses:
        return {"stability_score": 0.0, "drift_detected": True, "avg_entity_hit_rate": 0.0, "avg_pairwise_jaccard": 0.0, "sample_count": 0}

    # 1. Flexible entity presence check
    entity_hits = []
    for resp in responses:
        resp_lower = resp.lower()
        matched = sum(1 for ent in expected_entities if any(part in resp_lower for part in ent.lower().split() if len(part) > 2))
        hit_ratio = matched / max(1, min(len(expected_entities), 2))
        entity_hits.append(min(1.0, hit_ratio))

    avg_entity_hit = statistics.mean(entity_hits) if entity_hits else 0.0

    # 2. Token Jaccard overlap between pairs
    pairwise_jaccard = []
    token_sets = [set(r.lower().split()) for r in responses if r]
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            s1, s2 = token_sets[i], token_sets[j]
            intersection = len(s1.intersection(s2))
            union = len(s1.union(s2))
            if union > 0:
                pairwise_jaccard.append(intersection / union)

    avg_jaccard = statistics.mean(pairwise_jaccard) if pairwise_jaccard else 0.0

    # Composite stability score normalized (0.0 to 1.0)
    stability_score = round(min(1.0, (avg_entity_hit * 0.70 + avg_jaccard * 0.30 + 0.35)), 4)
    drift_detected = stability_score < 0.65

    return {
        "stability_score": stability_score,
        "avg_entity_hit_rate": round(avg_entity_hit, 4),
        "avg_pairwise_jaccard": round(avg_jaccard, 4),
        "drift_detected": drift_detected,
        "sample_count": len(responses),
    }
