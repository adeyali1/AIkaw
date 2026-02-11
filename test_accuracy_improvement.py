
import json
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from app_whisperx import analyze_reading_whisperx, TEXT_1, calculate_stats

FILES = [
    {"path": "d:\\cap anti\\have.mp3", "name": "Mistakes_Have", "text": TEXT_1},
    {"path": "d:\\cap anti\\v10.mp3.mp3", "name": "Mistakes_V10", "text": TEXT_1},
    {"path": "d:\\cap anti\\correct.mp3", "name": "Correct_Reading", "text": TEXT_1}
]

def run_tests():
    results = {}
    
    print("----------------------------------------------------------------")
    print(" 🧪 RUNNING ACCURACY IMPROVEMENT TESTS")
    print("----------------------------------------------------------------")
    
    for file_info in FILES:
        path = file_info["path"]
        name = file_info["name"]
        text = TEXT_1  # Assuming TEXT_1 for all
        
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            continue
            
        print(f"\n🎧 Analyzing: {name}")
        
        try:
            analysis, transcript = analyze_reading_whisperx(path, text)
            stats = calculate_stats(analysis)
            
            # Extract hesitation details
            hesitations = [w for w in analysis if w.get('hesitation', False)]
            self_corrections = [w for w in analysis if w.get('status') == 'SELF_CORRECTION']
            
            file_result = {
                "stats": stats,
                "hesitation_count": len(hesitations),
                "self_correction_count": len(self_corrections),
                "hesitation_details": [
                    {
                        "word": w['word'],
                        "duration": w['end'] - w['start'],
                        "info": w.get('info', ''),
                        "status": w['status']
                    } for w in hesitations
                ],
                "self_correction_details": [
                    {
                        "word": w['word'],
                        "duration": w['end'] - w['start'],
                        "info": w.get('info', ''),
                        "read_as": w.get('read_as', '')
                    } for w in self_corrections
                ]
            }
            results[name] = file_result
            
            print(f"   ✅ Accuracy: {stats['accuracy']:.1f}%")
            print(f"   ⚠️ Hesitations: {len(hesitations)}")
            print(f"   🔄 Self-Corrections: {len(self_corrections)}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    # Save detailed report
    with open("accuracy_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n----------------------------------------------------------------")
    print("📄 Report saved to accuracy_report.json")
    print("----------------------------------------------------------------")

if __name__ == "__main__":
    run_tests()
