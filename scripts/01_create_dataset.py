#!/usr/bin/env python3
"""
01_create_dataset.py — Create & Validate Medical AI Training Dataset
Part 2 of The Complete LLM Fine-Tuning Guide

This script:
1. Creates medical training examples (symptom assessment, triage, explanations, etc.)
2. Formats them into the Alpaca instruction format
3. Validates data quality
4. Saves as JSONL for training

Usage:
    python scripts/01_create_dataset.py
"""

import json
import os


# ═══════════════════════════════════════════════
# 1. Training Examples — Medical AI Dataset
# ═══════════════════════════════════════════════

training_examples = [
    # ─────────────────────────────────────────
    # Category 1: Symptom Assessment
    # ─────────────────────────────────────────
    {
        "instruction": "Assess these symptoms and provide guidance",
        "input": "Symptoms: Headache for 3 days, sensitivity to light, nausea, no fever",
        "output": """**Likely Condition:** Migraine

**Why:** The combination of persistent headache, light sensitivity, 
and nausea is a classic migraine presentation.

**Immediate Steps:**
1. Rest in a quiet, dark room
2. Stay hydrated (aim for 8+ glasses of water)
3. OTC pain relief: ibuprofen or acetaminophen
4. Cold compress on forehead or back of neck

**See a Doctor if:**
- This is your first headache this severe
- Headache came on suddenly ("worst of your life")
- Accompanied by fever, stiff neck, or confusion
- Not improving after 72 hours
- Happening more than 4x per month

**Note:** I'm an AI assistant providing general information. 
Always consult a healthcare professional for medical advice."""
    },
    {
        "instruction": "Assess these symptoms and provide guidance",
        "input": "Symptoms: Sore throat, mild fever 100.4°F, white spots on tonsils, swollen neck glands",
        "output": """**Likely Condition:** Strep Throat (bacterial)

**Why:** White spots on tonsils + swollen glands + fever without 
cough is the classic strep presentation (Centor criteria).

**Important:** Strep requires antibiotics — it won't resolve on its own.

**Do This Now:**
1. Schedule urgent care or doctor visit TODAY
2. Get a rapid strep test (15-minute result)
3. If positive, complete the full antibiotic course

**While Waiting:**
- Warm salt water gargle (1/4 tsp salt in 8oz water)
- OTC pain relief for discomfort
- Soft foods and cold liquids
- Avoid sharing utensils (highly contagious)

**See ER If:**
- Difficulty breathing or swallowing
- Drooling you can't control
- Neck swelling making it hard to open mouth

**Note:** This assessment is for informational purposes only."""
    },
    # ─────────────────────────────────────────
    # Category 2: Emergency Triage
    # ─────────────────────────────────────────
    {
        "instruction": "Is this a medical emergency? What should I do?",
        "input": "Chest pain with pressure, spreading to left arm, sweating, shortness of breath",
        "output": """🚨 YES — THIS IS A MEDICAL EMERGENCY

**Call 911 immediately. Do not drive yourself.**

These symptoms match a possible heart attack:
- Chest pressure/tightness ← Classic sign
- Pain radiating to left arm ← Classic sign  
- Sweating ← Classic sign
- Shortness of breath ← Classic sign

**While waiting for emergency services:**
1. Sit or lie down — stop all activity
2. Chew one regular aspirin (325mg) if not allergic
3. Loosen tight clothing
4. Unlock your front door for paramedics
5. Stay on the phone with 911

**Do NOT:**
- Drive yourself to the hospital
- Eat or drink anything
- Wait to see if it gets better

Time is muscle: every minute matters in a heart attack.

CALL 911 NOW."""
    },
    {
        "instruction": "Is this a medical emergency? What should I do?",
        "input": "Cut on finger, bleeding for 10 minutes, not stopping with pressure",
        "output": """**Not an immediate emergency** — but needs attention soon.

**Right Now (Next 30 minutes):**
1. Apply firm, continuous pressure with clean cloth for 10-15 minutes
2. Do NOT remove cloth to check — keeps clotting from forming
3. Elevate hand above heart level
4. If cloth soaks through, add another on top (don't remove first)

**After Bleeding Stops:**
- Clean gently with running water
- Apply antibiotic ointment
- Cover with sterile bandage
- Watch for infection signs (redness, warmth, pus, increasing pain)

**Go to Urgent Care if:**
- Bleeding won't stop after 20+ minutes of direct pressure
- Wound is deep, gaping, or has jagged edges (may need stitches)
- You can see fat, muscle, or bone
- Object is embedded in wound

**Go to ER if:**
- Spurting blood (possible artery)
- Wound from rusty/dirty object (tetanus risk)
- Loss of sensation or inability to move finger"""
    },
    # ─────────────────────────────────────────
    # Category 3: Medical Term Explanations
    # ─────────────────────────────────────────
    {
        "instruction": "Explain this medical term or diagnosis in simple language",
        "input": "My doctor said I have Type 2 Diabetes. What does this mean?",
        "output": """**Type 2 Diabetes — In Plain English**

**What's happening in your body:**
Your cells have become resistant to insulin (the key that lets 
sugar into cells for energy). Your pancreas works overtime making 
more insulin, but eventually can't keep up. 

Result: Sugar stays in your bloodstream instead of entering cells.

**Why it matters:**
Long-term high blood sugar damages blood vessels and nerves, 
affecting eyes, kidneys, heart, and feet.

**The good news:**
Type 2 is highly manageable. Many people control it entirely 
through lifestyle changes.

**Your Three Main Tools:**
1. **Diet** — Reduce refined carbs and sugar; more fiber, protein
2. **Exercise** — Even 30-minute walks lower blood sugar significantly  
3. **Medication** — If needed (metformin is usually first choice)

**Key Numbers to Know:**
- Blood sugar target: 80-130 mg/dL before meals
- A1C target: Below 7% (3-month average blood sugar)

**Questions for your doctor:**
- What's my A1C right now?
- Do I need medication or can I try lifestyle first?
- Should I see a diabetes educator or dietitian?

You caught this — now you can manage it. Many people live 
full, healthy lives with Type 2 diabetes."""
    },
    # ─────────────────────────────────────────
    # Category 4: Medication Questions
    # ─────────────────────────────────────────
    {
        "instruction": "Answer this medication question",
        "input": "Can I take ibuprofen and acetaminophen together?",
        "output": """**Yes — they can be safely combined and actually work better together.**

**Why it's safe:**
They work through completely different mechanisms:
- Ibuprofen (Advil/Motrin): Anti-inflammatory, blocks prostaglandins
- Acetaminophen (Tylenol): Acts on pain receptors in brain

No dangerous interaction exists between them.

**How to combine effectively:**
Stagger them every 3-4 hours:
```
8:00 AM  — Take ibuprofen
11:00 AM — Take acetaminophen  
2:00 PM  — Take ibuprofen
5:00 PM  — Take acetaminophen
```
This way you always have pain relief without overdosing either.

**Important limits:**
- Ibuprofen: Max 1200mg/day (OTC), every 6-8 hours
- Acetaminophen: Max 3000mg/day, every 4-6 hours
  ⚠️ Check OTHER medications — many contain hidden acetaminophen

**When NOT to take ibuprofen:**
- Stomach ulcers or GI issues
- Kidney problems
- Blood thinners (warfarin)
- Third trimester pregnancy

**When to avoid acetaminophen:**
- Liver disease
- More than 2 alcoholic drinks per day

Always read labels and consult your pharmacist for personalized advice."""
    },
    # ─────────────────────────────────────────
    # Category 5: Preventive Health
    # ─────────────────────────────────────────
    {
        "instruction": "Answer this health question",
        "input": "How much sleep do I actually need and what happens if I don't get enough?",
        "output": """**How Much Sleep You Need:**

| Age Group | Recommended Hours |
|-----------|-------------------|
| Teens (13-18) | 8-10 hours |
| Adults (18-64) | 7-9 hours |
| Older Adults (65+) | 7-8 hours |

The "I function fine on 5 hours" feeling is often a sign your 
brain has adapted to impairment — not that you don't need sleep.

**What Happens When You're Sleep Deprived:**

*After 1 night (under 6 hours):*
- Reaction time equivalent to mild intoxication
- 40% reduction in learning and memory formation
- Increased hunger hormones (craving carbs and sugar)

*Chronic sleep deprivation (weeks/months):*
- 3x higher risk of catching infections
- Significantly elevated heart disease risk
- Higher likelihood of Type 2 diabetes
- Mood disorders (anxiety, depression)
- Cognitive decline

**Practical Tips for Better Sleep:**
1. Same sleep/wake time every day (yes, weekends too)
2. Cool room: 65-68°F (18-20°C) is optimal
3. No screens 30-60 minutes before bed
4. Avoid caffeine after 2 PM
5. Keep bedroom for sleep only (not work)

**The bottom line:** Sleep is not laziness. 
It's when your brain consolidates memories, 
your immune system recharges, and your body repairs itself.
You can't "catch up" on lost sleep in meaningful ways."""
    }
]


# ═══════════════════════════════════════════════
# 2. Format for Training
# ═══════════════════════════════════════════════

def create_training_example(instruction, input_text, output_text):
    """Create a properly formatted training example"""
    return {
        "instruction": instruction.strip(),
        "input": input_text.strip(),
        "output": output_text.strip()
    }


def format_for_training(examples):
    """Format examples into training text the model will learn from"""
    formatted = []
    for ex in examples:
        if ex.get('input'):
            text = f"""### Instruction:
{ex['instruction']}

### Input:
{ex['input']}

### Response:
{ex['output']}"""
        else:
            text = f"""### Instruction:
{ex['instruction']}

### Response:
{ex['output']}"""
        
        formatted.append({"text": text})
    return formatted


# ═══════════════════════════════════════════════
# 3. Validate Dataset Quality
# ═══════════════════════════════════════════════

def validate_dataset(examples):
    """Check dataset quality before training"""
    issues = []
    
    for i, ex in enumerate(examples):
        # Check minimum length
        if len(ex['output'].split()) < 20:
            issues.append(f"Example {i}: Output too short ({len(ex['output'].split())} words)")
        
        # Check maximum length (avoid truncation)
        if len(ex['output'].split()) > 600:
            issues.append(f"Example {i}: Output very long, may get truncated")
        
        # Check for empty fields
        if not ex['instruction'].strip():
            issues.append(f"Example {i}: Empty instruction!")
        
        # Check for duplicates (basic)
        if ex['instruction'] == ex.get('output', ''):
            issues.append(f"Example {i}: Instruction same as output!")
    
    # Check category diversity
    instructions = [ex['instruction'] for ex in examples]
    unique = len(set(instructions))
    
    print(f"Dataset Stats:")
    print(f"  Total examples: {len(examples)}")
    print(f"  Unique instructions: {unique}")
    print(f"  Diversity ratio: {unique/len(examples):.1%}")
    
    if issues:
        print(f"\n⚠️  {len(issues)} Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ Dataset looks good!")


# ═══════════════════════════════════════════════
# Main Execution
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("📊 Creating Medical AI Training Dataset")
    print("=" * 60)
    
    # Validate
    print("\n--- Validating Dataset ---")
    validate_dataset(training_examples)
    
    # Format
    print("\n--- Formatting for Training ---")
    formatted = format_for_training(training_examples)
    
    # Save
    output_path = "medical_training_data.jsonl"
    with open(output_path, "w") as f:
        for example in formatted:
            f.write(json.dumps(example) + "\n")
    
    print(f"\n✅ Saved {len(formatted)} examples to {output_path}")
    print(f"\n--- Sample Preview ---")
    print(formatted[0]['text'][:300] + "...")
    
    print(f"\n{'=' * 60}")
    print("Next step: Run 02_sft_training.py to train your model!")
    print(f"{'=' * 60}")
