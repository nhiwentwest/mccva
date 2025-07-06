#!/usr/bin/env python3
import requests
import json

def test_prediction(features, expected, name):
    try:
        response = requests.post("http://localhost:5000/predict/enhanced", 
                               json={"features": features}, timeout=5)
        result = response.json()
        svm_pred = result["model_contributions"]["svm"]["prediction"]
        final_pred = result["makespan"]
        correct = (svm_pred == expected)
        print(f"{name:15} | SVM: {svm_pred:6} | Final: {final_pred:6} | Expected: {expected:6} | {'✅' if correct else '❌'}")
        return correct
    except Exception as e:
        print(f"{name:15} | ERROR: {e}")
        return False

print("\n🧪 ACCURACY TEST RESULTS")
print("=" * 80)
print(f"{'Server Type':15} | {'SVM Pred':8} | {'Final':8} | {'Expected':8} | {'Status'}")
print("-" * 80)

tests = [
    ([2, 4, 50, 500, 1], "small", "Web Server"),
    ([4, 8, 100, 1000, 3], "medium", "Database"), 
    ([12, 32, 500, 5000, 5], "large", "ML Training"),
    ([1, 2, 20, 100, 1], "small", "API Gateway"),
    ([6, 16, 200, 2000, 4], "medium", "Cache Server"),
    ([8, 24, 300, 3000, 4], "large", "Compute Node")
]

correct = 0
total = len(tests)

for features, expected, name in tests:
    if test_prediction(features, expected, name):
        correct += 1

accuracy = (correct / total) * 100
print("-" * 80)
print(f"\n📊 FINAL ACCURACY: {correct}/{total} = {accuracy:.1f}%")

if accuracy >= 80:
    print("🎉 EXCELLENT! Model is working correctly!")
elif accuracy >= 60:
    print("✅ GOOD! Significant improvement achieved!") 
else:
    print("⚠️  Still needs improvement...")
