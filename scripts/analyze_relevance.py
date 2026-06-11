import json

def main():
    with open('data/train/scd_training_final.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    train_data = data.get("train", data) if isinstance(data, dict) else data
    keywords = ["sickle", "hbss", "hbsc", "hydroxyurea", "vaso-occlusive", "dactylitis", "hemoglobin s", "scd", "acute chest", "priapism", "sequestration"]

    suspects = []
    for x in train_data:
        text_in = x.get("input", "").lower()
        text_out = x.get("output", "").lower()
        
        # If output does not contain any of the keywords
        out_has_kw = any(kw in text_out for kw in keywords)
        
        # And input contains "sickle" only in the options list (heuristic: after '? {')
        # Or just simply, output doesn't have the keyword, let's see them
        if not out_has_kw and "medical_meadow_medqa" in x.get("source", ""):
            suspects.append(x)

    print(f"Total records remaining: {len(train_data)}")
    print(f"Found {len(suspects)} suspect records where output does not contain any sickle cell keywords (mostly medqa distractors).")
    for s in suspects[:3]:
        print("---")
        print("IN:", s.get("input")[:150], "...")
        print("OUT:", s.get("output"))

if __name__ == "__main__":
    main()
