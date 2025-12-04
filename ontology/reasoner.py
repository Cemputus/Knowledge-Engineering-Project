#!/usr/bin/env python3
"""
Agriculture Ontology Reasoner
Simplified version that will definitely work
"""

import os
import sys
import uuid
import argparse

class AgricultureReasoner:
    def __init__(self):
        """Initialize the reasoner"""
        print("Initializing Agriculture Reasoner...")
        
        # Define the knowledge base
        self.knowledge_base = self._create_knowledge_base()
        print("✓ Knowledge base loaded with expert rules")
    
    def _create_knowledge_base(self):
        """Create the knowledge base with expert rules"""
        kb = {
            # Crop definitions
            'crops': {
                'Maize': {'type': 'Crop', 'label': 'Maize'},
                'Rice': {'type': 'Crop', 'label': 'Rice'},
                'Wheat': {'type': 'Crop', 'label': 'Wheat'},
                'Tomato': {'type': 'Crop', 'label': 'Tomato'},
                'Potato': {'type': 'Crop', 'label': 'Potato'},
                'Soybean': {'type': 'Crop', 'label': 'Soybean'}
            },
            
            # Symptoms
            'symptoms': {
                'YellowStreaks': {'label': 'Yellow streaks on leaves'},
                'StuntedGrowth': {'label': 'Stunted growth'},
                'LeafBlight': {'label': 'Leaf blight lesions'},
                'WhitePowderyGrowth': {'label': 'White powdery growth'},
                'LeafRust': {'label': 'Orange-brown pustules'},
                'Wilting': {'label': 'Wilting of plants'},
                'YellowingLeaves': {'label': 'Yellowing of leaves'},
                'LeafCurling': {'label': 'Leaf curling'},
                'LeafMosaic': {'label': 'Mosaic pattern on leaves'},
                'LeafSpots': {'label': 'Circular leaf spots'},
                'FruitRot': {'label': 'Fruit rot'},
                'Honeydew': {'label': 'Honeydew secretion'},
                'LeafMinerDamage': {'label': 'Leaf miner damage'},
                'ChewedLeaves': {'label': 'Chewed leaves'},
                'StemBoring': {'label': 'Stem boring damage'}
            },
            
            # Expert rules (diagnosis rules)
            'rules': [
                # Maize rules
                {
                    'crop': 'Maize',
                    'symptoms': ['YellowStreaks', 'StuntedGrowth'],
                    'diagnosis': 'Maize Streak Virus',
                    'type': 'disease',
                    'treatments': ['Use MSV-resistant maize varieties', 'Practice field sanitation']
                },
                {
                    'crop': 'Maize',
                    'symptoms': ['ChewedLeaves'],
                    'diagnosis': 'Fall Armyworm',
                    'type': 'pest',
                    'treatments': ['Apply Imidacloprid insecticide', 'Apply neem oil']
                },
                {
                    'crop': 'Maize',
                    'symptoms': ['StemBoring', 'Wilting'],
                    'diagnosis': 'Stem Borer',
                    'type': 'pest',
                    'treatments': ['Apply Imidacloprid insecticide', 'Practice field sanitation']
                },
                
                # Rice rules
                {
                    'crop': 'Rice',
                    'symptoms': ['LeafBlight'],
                    'diagnosis': 'Rice Blast',
                    'type': 'disease',
                    'treatments': ['Apply Tebuconazole fungicide', 'Use blast-resistant rice varieties']
                },
                {
                    'crop': 'Rice',
                    'symptoms': ['LeafSpots'],
                    'diagnosis': 'Bacterial Blight',
                    'type': 'disease',
                    'treatments': ['Apply copper-based fungicide', 'Use resistant varieties']
                },
                {
                    'crop': 'Rice',
                    'symptoms': ['YellowingLeaves'],
                    'diagnosis': 'Brown Plant Hopper',
                    'type': 'pest',
                    'treatments': ['Apply Imidacloprid', 'Use resistant varieties']
                },
                
                # Wheat rules
                {
                    'crop': 'Wheat',
                    'symptoms': ['LeafRust'],
                    'diagnosis': 'Wheat Rust',
                    'type': 'disease',
                    'treatments': ['Apply Tebuconazole fungicide', 'Use rust-resistant wheat varieties']
                },
                
                # Tomato rules
                {
                    'crop': 'Tomato',
                    'symptoms': ['LeafSpots', 'FruitRot'],
                    'diagnosis': 'Late Blight',
                    'type': 'disease',
                    'treatments': ['Apply copper-based fungicide', 'Apply Chlorothalonil']
                },
                {
                    'crop': 'Tomato',
                    'symptoms': ['WhitePowderyGrowth'],
                    'diagnosis': 'Powdery Mildew',
                    'type': 'disease',
                    'treatments': ['Apply sulfur fungicide', 'Apply neem oil']
                },
                {
                    'crop': 'Tomato',
                    'symptoms': ['Wilting', 'YellowingLeaves'],
                    'diagnosis': 'Fusarium Wilt',
                    'type': 'disease',
                    'treatments': ['Practice crop rotation', 'Apply Trichoderma biological control']
                },
                {
                    'crop': 'Tomato',
                    'symptoms': ['LeafMosaic'],
                    'diagnosis': 'Tomato Mosaic Virus',
                    'type': 'disease',
                    'treatments': ['Use virus-free seeds', 'Practice field sanitation']
                },
                {
                    'crop': 'Tomato',
                    'symptoms': ['LeafCurling', 'Honeydew'],
                    'diagnosis': 'Aphids',
                    'type': 'pest',
                    'treatments': ['Apply Imidacloprid insecticide', 'Release ladybugs']
                },
                {
                    'crop': 'Tomato',
                    'symptoms': ['YellowingLeaves', 'Honeydew'],
                    'diagnosis': 'Whiteflies',
                    'type': 'pest',
                    'treatments': ['Apply Imidacloprid insecticide', 'Apply neem oil']
                },
                {
                    'crop': 'Tomato',
                    'symptoms': ['LeafMinerDamage'],
                    'diagnosis': 'Leaf Miner',
                    'type': 'pest',
                    'treatments': ['Apply neem oil', 'Practice field sanitation']
                },
                
                # Potato rules
                {
                    'crop': 'Potato',
                    'symptoms': ['LeafSpots'],
                    'diagnosis': 'Early Blight',
                    'type': 'disease',
                    'treatments': ['Apply Chlorothalonil', 'Practice crop rotation']
                },
                {
                    'crop': 'Potato',
                    'symptoms': ['ChewedLeaves'],
                    'diagnosis': 'Colorado Potato Beetle',
                    'type': 'pest',
                    'treatments': ['Apply Imidacloprid', 'Handpick beetles']
                },
                
                # Soybean rules
                {
                    'crop': 'Soybean',
                    'symptoms': ['YellowStreaks', 'StuntedGrowth'],
                    'diagnosis': 'Soybean Mosaic Virus',
                    'type': 'disease',
                    'treatments': ['Use SMV-resistant soybean varieties', 'Practice field sanitation', 'Control aphid vectors']
                },
                {
                    'crop': 'Soybean',
                    'symptoms': ['LeafRust'],
                    'diagnosis': 'Soybean Rust',
                    'type': 'disease',
                    'treatments': ['Apply Tebuconazole fungicide', 'Use resistant varieties']
                }
            ],
            
            # Additional treatments for specific cases
            'treatment_db': {
                'Soybean Mosaic Virus': ['Use SMV-resistant soybean varieties', 'Practice field sanitation', 'Control aphid vectors', 'Remove infected plants'],
                'Maize Streak Virus': ['Use MSV-resistant maize varieties', 'Practice field sanitation', 'Control leafhopper vectors'],
                'Rice Blast': ['Apply Tebuconazole fungicide', 'Use blast-resistant rice varieties', 'Avoid excessive nitrogen'],
                'Wheat Rust': ['Apply Tebuconazole fungicide', 'Use rust-resistant wheat varieties', 'Early planting'],
                'Late Blight': ['Apply copper-based fungicide', 'Apply Chlorothalonil', 'Improve air circulation'],
                'Powdery Mildew': ['Apply sulfur fungicide', 'Apply neem oil', 'Reduce humidity'],
                'Fusarium Wilt': ['Practice crop rotation', 'Apply Trichoderma biological control', 'Use resistant varieties'],
                'Fall Armyworm': ['Apply Imidacloprid insecticide', 'Apply neem oil', 'Use pheromone traps'],
                'Aphids': ['Apply Imidacloprid insecticide', 'Release ladybugs', 'Use insecticidal soap'],
                'Whiteflies': ['Apply Imidacloprid insecticide', 'Apply neem oil', 'Use yellow sticky traps']
            }
        }
        
        return kb
    
    def get_general_recommendations(self, crop_name):
        """Get general recommendations for any crop"""
        general_recommendations = [
            "Monitor plants closely for symptom development and progression",
            "Practice good field sanitation - remove and destroy infected plant debris",
            "Maintain proper spacing between plants for adequate air circulation",
            "Water plants at the base to avoid wetting foliage",
            "Consider soil testing to check for nutrient deficiencies",
            "Use crop rotation to prevent buildup of pests and diseases",
            "Apply organic mulch to maintain soil moisture and temperature",
            "Consult with local agricultural extension service for region-specific advice",
            "Keep records of symptoms, treatments, and outcomes for future reference",
            "Ensure proper drainage to prevent waterlogging"
        ]
        
        crop_specific = {
            'Maize': [
                "Plant maize at recommended spacing (75cm x 25cm)",
                "Control weeds that can harbor pests and diseases",
                "Apply balanced fertilizer based on soil test results"
            ],
            'Rice': [
                "Maintain proper water level in paddy fields",
                "Use certified disease-free seeds",
                "Practice proper field leveling for uniform water distribution"
            ],
            'Wheat': [
                "Plant at optimal time to avoid disease-prone periods",
                "Use certified seed varieties",
                "Monitor for early signs of rust and apply preventive fungicides"
            ],
            'Tomato': [
                "Stake or cage plants for better air circulation",
                "Avoid overhead watering to prevent fungal diseases",
                "Remove lower leaves that touch the ground"
            ],
            'Potato': [
                "Plant certified seed potatoes",
                "Practice hilling to protect tubers",
                "Harvest when vines are dead to prevent disease spread"
            ],
            'Soybean': [
                "Plant at recommended depth (2-3 cm)",
                "Use proper inoculation with rhizobia bacteria",
                "Monitor for aphids and other sucking pests"
            ]
        }
        
        recommendations = general_recommendations.copy()
        if crop_name in crop_specific:
            recommendations.extend(crop_specific[crop_name])
        
        return recommendations
    
    def diagnose(self, crop_name, symptoms_list):
        """Diagnose based on crop and symptoms"""
        print(f"\nDiagnosing {crop_name}...")
        print(f"Observed symptoms: {', '.join(symptoms_list)}")
        
        diagnoses = []
        
        # Check all rules
        for rule in self.knowledge_base['rules']:
            if rule['crop'] == crop_name:
                # Check if all symptoms in rule are present
                if all(symptom in symptoms_list for symptom in rule['symptoms']):
                    diagnoses.append({
                        'diagnosis': rule['diagnosis'],
                        'type': rule['type'],
                        'treatments': rule['treatments'],
                        'matched_symptoms': rule['symptoms']
                    })
        
        return diagnoses
    
    def get_crop_symptoms(self, crop_name):
        """Get all symptoms relevant to a specific crop based on rules"""
        relevant_symptoms = set()
        for rule in self.knowledge_base['rules']:
            if rule['crop'] == crop_name:
                relevant_symptoms.update(rule['symptoms'])
        return sorted(list(relevant_symptoms))
    
    def list_available_crops(self):
        """List available crops as numbered list"""
        crops = list(self.knowledge_base['crops'].keys())
        print("\nAvailable crops:")
        for i, crop_name in enumerate(crops, 1):
            print(f"  {i}. {crop_name}")
        return crops
    
    def list_crop_symptoms(self, crop_name):
        """List symptoms relevant to a specific crop as numbered list with descriptions"""
        symptoms = self.get_crop_symptoms(crop_name)
        if not symptoms:
            print(f"\nNo symptoms found for {crop_name}.")
            return []
        
        print(f"\nSymptoms for {crop_name}:")
        for i, symptom_name in enumerate(symptoms, 1):
            symptom_label = self.knowledge_base['symptoms'][symptom_name]['label']
            print(f"  {i}. {symptom_name}: {symptom_label}")
        return symptoms
    
    def interactive_diagnosis(self):
        """Interactive diagnosis mode with numbered selections"""
        print("\n" + "="*60)
        print("AGRICULTURE DIAGNOSIS SYSTEM")
        print("="*60)
        
        # Step 1: Select crop by number
        crops = self.list_available_crops()
        while True:
            try:
                crop_choice = input("\nSelect crop (enter number): ").strip()
                crop_index = int(crop_choice) - 1
                if 0 <= crop_index < len(crops):
                    crop_name = crops[crop_index]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(crops)}")
            except ValueError:
                print("Please enter a valid number")
        
        # Step 2: Show symptoms for selected crop and get selections
        symptoms = self.list_crop_symptoms(crop_name)
        
        if not symptoms:
            print(f"\nNo symptoms available for {crop_name}. Cannot proceed with diagnosis.")
            return
        
        while True:
            symptoms_input = input(f"\nSelect symptoms (enter numbers separated by commas, e.g., 1,3,5): ").strip()
            
            # Parse number selections
            try:
                selected_indices = [int(s.strip()) - 1 for s in symptoms_input.split(",") if s.strip()]
                valid_indices = [idx for idx in selected_indices if 0 <= idx < len(symptoms)]
                
                if not valid_indices:
                    print(f"Please enter valid numbers between 1 and {len(symptoms)}")
                    continue
                
                # Convert indices to symptom names
                valid_symptoms = [symptoms[idx] for idx in valid_indices]
                break
            except ValueError:
                print("Please enter numbers separated by commas (e.g., 1,3,5)")
        
        if not valid_symptoms:
            print("No valid symptoms selected.")
            return
        
        # Perform diagnosis
        diagnoses = self.diagnose(crop_name, valid_symptoms)
        
        if diagnoses:
            print("\n" + "="*60)
            print("SPECIFIC DIAGNOSIS RESULTS")
            print("="*60)
            
            for i, diagnosis in enumerate(diagnoses, 1):
                print(f"\n{i}. Diagnosis: {diagnosis['diagnosis']} ({diagnosis['type'].title()})")
                print(f"   Matched symptoms: {', '.join(diagnosis['matched_symptoms'])}")
                print(f"   Recommended treatments:")
                for treatment in diagnosis['treatments']:
                    print(f"     • {treatment}")
                
                # Show additional treatments if available
                if diagnosis['diagnosis'] in self.knowledge_base['treatment_db']:
                    additional = self.knowledge_base['treatment_db'][diagnosis['diagnosis']]
                    extra_treatments = [t for t in additional if t not in diagnosis['treatments']]
                    if extra_treatments:
                        print(f"   Additional recommendations:")
                        for treatment in extra_treatments[:2]:  # Show max 2 additional
                            print(f"     • {treatment}")
            
            # Always show general recommendations after specific diagnosis
            print("\n" + "="*60)
            print("GENERAL RECOMMENDATIONS")
            print("="*60)
            general_recs = self.get_general_recommendations(crop_name)
            for i, rec in enumerate(general_recs[:8], 1):  # Show top 8 general recommendations
                print(f"  {i}. {rec}")
        else:
            print("\nNo specific diagnosis found for the given symptoms.")
            print("\n" + "="*60)
            print("GENERAL RECOMMENDATIONS")
            print("="*60)
            general_recs = self.get_general_recommendations(crop_name)
            for i, rec in enumerate(general_recs, 1):
                print(f"  {i}. {rec}")
    
    def command_line_diagnosis(self, crop_name, symptoms_input):
        """Command-line diagnosis mode"""
        symptoms_list = [s.strip() for s in symptoms_input.split(",") if s.strip()]
        
        # Validate crop
        if crop_name not in self.knowledge_base['crops']:
            print(f"Error: Crop '{crop_name}' not recognized.")
            self.list_available_crops()
            return
        
        # Validate symptoms
        valid_symptoms = []
        invalid_symptoms = []
        for symptom in symptoms_list:
            if symptom in self.knowledge_base['symptoms']:
                valid_symptoms.append(symptom)
            else:
                invalid_symptoms.append(symptom)
        
        if invalid_symptoms:
            print(f"Warning: The following symptoms were not recognized and will be ignored: {', '.join(invalid_symptoms)}")
        
        if not valid_symptoms:
            print("Error: No valid symptoms provided.")
            self.list_available_symptoms()
            return
        
        # Perform diagnosis
        diagnoses = self.diagnose(crop_name, valid_symptoms)
        
        if diagnoses:
            print("\n" + "="*60)
            print("SPECIFIC DIAGNOSIS RESULTS")
            print("="*60)
            
            for i, diagnosis in enumerate(diagnoses, 1):
                print(f"\n{i}. Diagnosis: {diagnosis['diagnosis']} ({diagnosis['type'].title()})")
                print(f"   Matched symptoms: {', '.join(diagnosis['matched_symptoms'])}")
                print(f"   Recommended treatments:")
                for treatment in diagnosis['treatments']:
                    print(f"     • {treatment}")
            
            # Always show general recommendations after specific diagnosis
            print("\n" + "="*60)
            print("GENERAL RECOMMENDATIONS")
            print("="*60)
            general_recs = self.get_general_recommendations(crop_name)
            for i, rec in enumerate(general_recs[:8], 1):  # Show top 8 general recommendations
                print(f"  {i}. {rec}")
        else:
            print("\nNo specific diagnosis found.")
            print("\n" + "="*60)
            print("GENERAL RECOMMENDATIONS")
            print("="*60)
            general_recs = self.get_general_recommendations(crop_name)
            for i, rec in enumerate(general_recs, 1):
                print(f"  {i}. {rec}")

def main():
    parser = argparse.ArgumentParser(description="Agriculture Diagnosis Expert System")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--crop", "-c", type=str,
                       help="Crop name for diagnosis")
    parser.add_argument("--symptoms", "-s", type=str,
                       help="Comma-separated list of symptoms")
    
    args = parser.parse_args()
    
    reasoner = AgricultureReasoner()
    
    if args.interactive:
        reasoner.interactive_diagnosis()
    elif args.crop and args.symptoms:
        reasoner.command_line_diagnosis(args.crop, args.symptoms)
    else:
        # Show usage examples
        print("\nAgriculture Diagnosis Expert System")
        print("="*50)
        print("\nUsage options:")
        print("  1. Interactive mode: python reasoner.py --interactive")
        print("  2. Command-line: python reasoner.py --crop Maize --symptoms \"YellowStreaks,StuntedGrowth\"")
        
        print("\nExample diagnoses:")
        print("  - Maize + YellowStreaks + StuntedGrowth → Maize Streak Virus")
        print("  - Tomato + WhitePowderyGrowth → Powdery Mildew")
        print("  - Rice + LeafBlight → Rice Blast")
        
        print("\nTo see all available crops and symptoms, use interactive mode.")

if __name__ == "__main__":
    main()