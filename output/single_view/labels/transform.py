import os
import json

labels_dir = "output/single_view/labels"
output_json = "output/single_view/labels/labels.json"

all_labels = {}

for fname in os.listdir(labels_dir):
    if fname.endswith('.txt'):
        frame_id = os.path.splitext(fname)[0]
        with open(os.path.join(labels_dir, fname), 'r') as f:
            lines = [line.strip().split() for line in f.readlines()]
            all_labels[frame_id] = lines

with open(output_json, "w") as f:
    json.dump(all_labels, f, indent=2)

print(f"All label info saved to {output_json}")