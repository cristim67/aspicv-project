#!/usr/bin/env python3
"""
Script pentru compararea a două fișiere CSV și afișarea diferențelor.
"""

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def load_csv(file_path: str) -> Dict[str, str]:
    """Încarcă un fișier CSV și returnează un dicționar {id: label}."""
    data = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data[row['id']] = row['label']
    except FileNotFoundError:
        print(f"Eroare: Fișierul {file_path} nu a fost găsit.")
        sys.exit(1)
    except KeyError as e:
        print(f"Eroare: Coloana {e} lipsește din fișierul {file_path}.")
        sys.exit(1)
    return data


def compare_csvs(file1_path: str, file2_path: str) -> Tuple[List[Tuple[str, str, str]], int, int]:
    """
    Compară două fișiere CSV și returnează diferențele.
    
    Returns:
        Tuple[List[Tuple[id, label1, label2]], num_differences, total_rows]
    """
    data1 = load_csv(file1_path)
    data2 = load_csv(file2_path)
    
    differences = []
    all_ids = set(data1.keys()) | set(data2.keys())
    
    # ID-uri care există doar în primul fișier
    only_in_file1 = set(data1.keys()) - set(data2.keys())
    # ID-uri care există doar în al doilea fișier
    only_in_file2 = set(data2.keys()) - set(data1.keys())
    # ID-uri comune
    common_ids = set(data1.keys()) & set(data2.keys())
    
    # Găsește diferențele pentru ID-urile comune
    for id_val in sorted(common_ids):
        if data1[id_val] != data2[id_val]:
            differences.append((id_val, data1[id_val], data2[id_val]))
    
    # Adaugă ID-urile care există doar într-unul dintre fișiere
    for id_val in sorted(only_in_file1):
        differences.append((id_val, data1[id_val], "N/A (lipsește din fișierul 2)"))
    
    for id_val in sorted(only_in_file2):
        differences.append((id_val, "N/A (lipsește din fișierul 1)", data2[id_val]))
    
    return differences, len(differences), len(all_ids)


def print_differences(differences: List[Tuple[str, str, str]], 
                     file1_name: str, 
                     file2_name: str,
                     num_differences: int,
                     total_rows: int):
    """Afișează diferențele într-un format citibil."""
    print("=" * 80)
    print(f"COMPARAȚIE CSV: {file1_name} vs {file2_name}")
    print("=" * 80)
    print(f"\nTotal rânduri: {total_rows}")
    print(f"Diferențe găsite: {num_differences}")
    print(f"Potriviri: {total_rows - num_differences}")
    print("\n" + "-" * 80)
    
    if not differences:
        print("✓ Nu există diferențe între cele două fișiere!")
        return
    
    print(f"\n{'ID':<10} {'Fișier 1 (' + file1_name + ')':<30} {'Fișier 2 (' + file2_name + ')':<30}")
    print("-" * 80)
    
    for id_val, label1, label2 in differences:
        print(f"{id_val:<10} {label1:<30} {label2:<30}")
    
    print("\n" + "=" * 80)


def main():
    """Funcția principală."""
    # Setează fișierele implicite sau folosește argumentele din linia de comandă
    if len(sys.argv) == 3:
        file1 = sys.argv[1]
        file2 = sys.argv[2]
    elif len(sys.argv) == 1:
        # Folosește fișierele implicite din directorul curent
        file1 = "submission.csv"
        file2 = "submission2.csv"
    else:
        print("Utilizare: python diff-csv.py [file1.csv] [file2.csv]")
        print("Sau: python diff-csv.py  (folosește submission.csv și submission2.csv)")
        sys.exit(1)
    
    # Verifică dacă fișierele există
    if not Path(file1).exists():
        print(f"Eroare: {file1} nu există.")
        sys.exit(1)
    if not Path(file2).exists():
        print(f"Eroare: {file2} nu există.")
        sys.exit(1)
    
    # Compară fișierele
    differences, num_differences, total_rows = compare_csvs(file1, file2)
    
    # Afișează rezultatele
    print_differences(differences, file1, file2, num_differences, total_rows)
    
    # Statistici suplimentare
    if differences:
        print("\nStatistici diferențe:")
        label_changes = {}
        for _, label1, label2 in differences:
            if label1 != "N/A (lipsește din fișierul 1)" and label2 != "N/A (lipsește din fișierul 2)":
                change_key = f"{label1} → {label2}"
                label_changes[change_key] = label_changes.get(change_key, 0) + 1
        
        if label_changes:
            print("\nSchimbări de etichetă:")
            for change, count in sorted(label_changes.items(), key=lambda x: x[1], reverse=True):
                print(f"  {change}: {count} cazuri")


if __name__ == "__main__":
    main()

