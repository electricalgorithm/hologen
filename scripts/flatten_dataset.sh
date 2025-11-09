#!/usr/bin/env bash
# Reorganize inline holography dataset into subdirectories.
# Expected structure after running:
# object/, hologram/, reconstruction/, npz/

set -euo pipefail

DATASET_DIR="../inline-holography-dataset"
cd "$DATASET_DIR" || { echo "Directory not found: $DATASET_DIR"; exit 1; }

# Create target subdirectories if they don't exist
mkdir -p object hologram reconstruction npz

# Iterate through all files in the directory
for file in sample_*; do
    # Skip directories if any
    [[ -d "$file" ]] && continue

    # Extract base sample ID (e.g., sample_00001)
    base=$(echo "$file" | sed -E 's/^(sample_[0-9]+).*$/\1/')

    case "$file" in
        *_object.png)
            cp "$file" "object/${base}.png"
            ;;
        *_hologram.png)
            cp "$file" "hologram/${base}.png"
            ;;
        *_reconstruction.png)
            cp "$file" "reconstruction/${base}.png"
            ;;
        *.npz)
            cp "$file" "npz/${base}.npz"
            ;;
        *)
            echo "Skipping unknown file: $file"
            ;;
    esac
done

# Count the number of files in each directory
count_object=$(find object -type f | wc -l)
count_hologram=$(find hologram -type f | wc -l)
count_reconstruction=$(find reconstruction -type f | wc -l)
count_npz=$(find npz -type f | wc -l)

echo "📊 File counts:"
echo "  object:          $count_object"
echo "  hologram:        $count_hologram"
echo "  reconstruction:  $count_reconstruction"
echo "  npz:             $count_npz"

# Validation: ensure all counts are equal
if [[ $count_object -eq $count_hologram && \
      $count_hologram -eq $count_reconstruction && \
      $count_reconstruction -eq $count_npz ]]; then
    echo "🎯 Validation passed — all directories contain the same number of samples."
else
    echo "❌ Validation failed — counts differ between directories!"
    echo "Please check for missing or mismatched files."
    exit 1
fi

echo "✅ Dataset reorganized successfully!"
echo "Subdirectories created under: $DATASET_DIR"
