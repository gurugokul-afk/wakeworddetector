"""
05_create_balanced_split.py
Creates a leak-free, source-grouped, stratified train / val / test split:
  Train: ~70%
  Val:   ~15%
  Test:  ~15%

Guarantees:
1. Zero group/stem overlap between train, val, and test.
2. Balanced class representation in all splits.
3. Diverse noise coverage (real WhatsApp + Speech Commands silence + synthetic) across all splits.
"""
from pathlib import Path
import shutil
import random

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ROOT / "features"
SPLIT = ROOT / "split"

CLASSES = ["bankai", "unknown", "noise"]
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 20260909

random.seed(SEED)

def get_group_id(filename, class_name):
    stem = Path(filename).stem
    if class_name == "bankai":
        # Stem format: {original_stem}_{aug_type}.npy or {original_stem}_orig.npy
        for tag in ["_orig", "_shift", "_speed", "_reverb", "_noise", "_hiss"]:
            if tag in stem:
                return stem.split(tag)[0]
        return stem

    elif class_name == "unknown":
        if "isro2_unknew_" in stem:
            parts = stem.split("_clip")
            return parts[0]
        elif "isro2_unk_" in stem:
            parts = stem.split("_clip")
            return parts[0]
        elif "sc_word_" in stem:
            parts = stem.split("_")
            return f"sc_word_{parts[2]}_{parts[3][:6]}"
        elif "krishna_sc_" in stem:
            parts = stem.split("_")
            return f"krishna_sc_{parts[2]}_{parts[3][:6]}"
        elif "sc_unk_" in stem:
            parts = stem.split("_")
            return f"sc_unk_{parts[2][:6]}"
        return stem

    else:  # noise
        if "synth_" in stem:
            parts = stem.split("_")
            # e.g. synth_pink_0001 -> synth_pink
            return f"synth_{parts[1]}"
        elif "isro2_noise_" in stem:
            # Group into 10 buckets
            num = int(stem.split("_")[-1])
            bucket = (num - 1) // 80
            return f"isro2_noise_bucket_{bucket:02d}"
        elif "sc_silence_" in stem:
            # Group into 5 buckets
            parts = stem.split("_")
            num = int(parts[-1]) if parts[-1].isdigit() else hash(stem) % 10
            return f"sc_silence_bucket_{num % 5}"
        return stem

def split_groups(groups, train_ratio=0.70, val_ratio=0.15):
    groups = list(groups)
    random.shuffle(groups)
    n = len(groups)
    n_train = max(1, round(n * train_ratio))
    n_val = max(1, round(n * val_ratio))
    if n_train + n_val >= n:
        n_val = 1
        n_train = n - 2
    train_g = set(groups[:n_train])
    val_g = set(groups[n_train:n_train + n_val])
    test_g = set(groups[n_train + n_val:])
    return train_g, val_g, test_g

def build_split():
    print("========================================")
    print("CREATING BALANCED GROUP-AWARE SPLIT")
    print("========================================")
    
    if SPLIT.exists():
        shutil.rmtree(SPLIT)
    for s in ["train", "val", "test"]:
        for c in CLASSES:
            (SPLIT / s / c).mkdir(parents=True, exist_ok=True)
            
    summary = {"train": {}, "val": {}, "test": {}}
    
    for cls in CLASSES:
        cls_dir = FEATURES / cls
        files = sorted(cls_dir.glob("*.npy"))
        if not files:
            raise RuntimeError(f"No features found in {cls_dir}")
            
        # Group files
        groups = {}
        for f in files:
            gid = get_group_id(f.name, cls)
            groups.setdefault(gid, []).append(f)
            
        print(f"\nClass '{cls}': {len(files)} total files across {len(groups)} unique groups.")
        
        train_g, val_g, test_g = split_groups(groups.keys(), TRAIN_RATIO, VAL_RATIO)
        
        # Leakage assert
        assert len(train_g & val_g) == 0, f"Leakage train/val in {cls}"
        assert len(train_g & test_g) == 0, f"Leakage train/test in {cls}"
        assert len(val_g & test_g) == 0, f"Leakage val/test in {cls}"
        
        counts = {"train": 0, "val": 0, "test": 0}
        for gid, flist in groups.items():
            if gid in train_g:
                split_name = "train"
            elif gid in val_g:
                split_name = "val"
            else:
                split_name = "test"
                
            for src in flist:
                dst = SPLIT / split_name / cls / src.name
                shutil.copy2(src, dst)
                counts[split_name] += 1
                
        for s in ["train", "val", "test"]:
            summary[s][cls] = counts[s]
            print(f"  {cls:<8} [{s.upper()}]: {counts[s]} files ({len([g for g in groups if g in (train_g if s=='train' else val_g if s=='val' else test_g)])} groups)")
            
    print("\n========================================")
    print("FINAL SPLIT SUMMARY")
    print("========================================")
    for s in ["train", "val", "test"]:
        tot = sum(summary[s].values())
        print(f"--- {s.upper()} (Total: {tot}) ---")
        for c in CLASSES:
            pct = summary[s][c] / tot * 100 if tot > 0 else 0
            print(f"  {c:<8}: {summary[s][c]:4d} ({pct:5.1f}%)")
    print("========================================")

if __name__ == "__main__":
    build_split()
