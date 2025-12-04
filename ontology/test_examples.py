#!/usr/bin/env python3
"""
Test examples for the agriculture reasoner
"""

from reasoner import AgricultureReasoner

def run_examples():
    """Run example diagnoses"""
    reasoner = AgricultureReasoner()
    
    examples = [
        ("Maize", ["YellowStreaks", "StuntedGrowth"], "Should diagnose: Maize Streak Virus"),
        ("Tomato", ["WhitePowderyGrowth"], "Should diagnose: Powdery Mildew"),
        ("Rice", ["LeafBlight"], "Should diagnose: Rice Blast"),
        ("Tomato", ["LeafCurling", "Honeydew"], "Should diagnose: Aphids"),
        ("Potato", ["ChewedLeaves"], "Should diagnose: Colorado Potato Beetle"),
        ("Wheat", ["LeafRust"], "Should diagnose: Wheat Rust"),
    ]
    
    print("Running test examples...")
    print("="*60)
    
    for crop, symptoms, expected in examples:
        print(f"\nTest: {expected}")
        print(f"Crop: {crop}, Symptoms: {symptoms}")
        diagnoses = reasoner.diagnose(crop, symptoms)
        
        if diagnoses:
            for d in diagnoses:
                print(f"  ✓ Diagnosis: {d['diagnosis']}")
                print(f"    Treatments: {', '.join(d['treatments'][:2])}")
        else:
            print(f"  ✗ No diagnosis found")
    
    print("\n" + "="*60)
    print("All tests completed!")

if __name__ == "__main__":
    run_examples()