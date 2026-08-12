#!/usr/bin/env python3
"""
InvokeAI Auto-Sorter & Board-Organizer
Sucht unkategorisierte/falsch einsortierte Bilder in InvokeAI, analysiert LoRAs
sowie Prompts und weist den Bildern automatisch das richtige Board zu.
"""

import argparse
import json
import re
import sqlite3
import uuid
import platform
from pathlib import Path
from tqdm import tqdm
from collections import Counter

# Plattformabhängigen Standard-Datenbankpfad definieren
if platform.system() == "Windows":
    DEFAULT_DB_PATH = Path(r"")  # Passe diesen Windows-Pfad bei Bedarf an deine Ordnerstruktur an
else:
    DEFAULT_DB_PATH = Path("") # Passe diesen Linux-Pfad bei Bedarf an deine Ordnerstruktur an
    
# --- BOARD-REGELN ---
RULES = {
    "Portrait": ["medium shot portrait", "bokeh", "festival stage"],
    "Photorealism": ["realism_lora", "cinematic_photo", "raw_photo"],
    "food": ["Fruit Pancakes", "sweet", "red strawberry sauce", "baked", "delicious", "steaming", "dish", "served", "maple sirup", "breakfast"],
}
# --- BOARD-REGELN ---
QUALITY_STOPWORDS = {
    "masterpiece", "best quality", "high quality", "ultra detailed", "detailed",
    "8k", "4k", "photorealistic", "realistic", "hyperrealistic", "trending on artstation",
    "octane render", "unreal engine", "sharp focus", "soft lighting", "cinematic",
    "dramatic lighting", "intricate details", "masterpiece quality", "raw photo",
    "absurdres", "volumetric lighting", "studio lighting", "a photo of", "a picture of",
    "a painting of", "illustration", "digital painting", "concept art"
}

# Name des Boards, in das Bilder landen, die zu keinem anderen passen
FALLBACK_BOARD_NAME = "general"

def normalize_lora_name(lora_name: str) -> str:
    if not lora_name:
        return ""
    return re.sub(r'[-_]v?\d+([\._]\d+)?(_final)?$', '', lora_name, flags=re.IGNORECASE).strip()

def extract_all_strings_from_json(data) -> list[str]:
    """Durchsucht das gesamte JSON rekursiv nach allen Strings (Modellnamen, LoRAs etc.)."""
    found_strings = []
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, str):
                found_strings.append(v)
            else:
                found_strings.extend(extract_all_strings_from_json(v))
    elif isinstance(data, list):
        for item in data:
            found_strings.extend(extract_all_strings_from_json(item))
    return found_strings

def extract_clean_prompt_words(prompt: str) -> list[str]:
    if not prompt:
        return []
    
    text = prompt.lower()
    for word in QUALITY_STOPWORDS:
        text = text.replace(word, "")
        
    text = re.sub(r'[\(\)\[\]\{\}:0-9\.]', ' ', text)
    return [w.strip() for w in re.split(r'[,;\s]+', text) if len(w.strip()) > 2]

def clean_board_name_from_tag(raw_name: str) -> str:
    """Komplett kompromisslose Bereinigung für LoRA-Dateinamen."""
    if not raw_name:
        return ""

    cleaned = raw_name

    # 1. Dateiendung am Ende hart abschneiden (.safetensors, .ckpt etc.)
    cleaned = re.sub(r'\.(safetensors|ckpt|pt|json)$', '', cleaned, flags=re.IGNORECASE)
    
    # 2. Alle eckigen Klammern samt Inhalt entfernen ([PEOPLEZIT], [PERSON], etc.)
    cleaned = re.sub(r'\[.*?\]', '', cleaned)
    
    # 3. Versionsnummern am Ende entfernen (_v2, -v2 etc.)
    cleaned = re.sub(r'[\s_\-]*v\d+([._]\d+)?$', '', cleaned, flags=re.IGNORECASE)
    
    # 4. JETZT ALLE PUNKTE WEG (wandelt "first.second" in "firstsecond")
    cleaned = cleaned.replace('.', '')
    
    # 5. Unterstriche und Bindestriche zu Leerzeichen
    cleaned = cleaned.replace('_', ' ').replace('-', ' ')
    
    # 6. Aufräumen von Leerzeichen
    cleaned = ' '.join(cleaned.split()).strip()
    
    # 7. Umlaute ersetzen und für den finalen Vergleich komplett alphanumerisch machen
    cleaned = cleaned.lower()
    cleaned = cleaned.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    cleaned = re.sub(r'[^a-z0-9]', '', cleaned)  

    blacklist = {"lora", "loras", "safetensors", "checkpoint", "model", "models", "embedding", "embeddings"}
    if cleaned in blacklist:
        return ""
        
    return cleaned

def determine_board(metadata_raw, existing_boards_lower):
    if not metadata_raw:
        return None, None

    try:
        meta = json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
    except json.JSONDecodeError:
        return None, None

    all_strings = extract_all_strings_from_json(meta)
    all_text = " ".join(all_strings)
    
    # --- 1. PRIORITÄT: LoRA / Tag-Matching (Nur in echten technischen Strings, KEIN Prompt!) ---
    for text in all_strings:
        text_lower = text.lower()
        
        # Überspringe den reinen Prompt-Text in dieser Stufe, um Konflikte zu vermeiden!
        if "prompt" in text_lower or len(text) > 150:
            continue
            
        # Prüfe nur Strings, die verdächtig nach LoRA, Datei oder Tag aussehen
        if any(marker in text_lower for marker in ['.safetensors', '.ckpt', '.pt', 'lora', 'peoplezit', 'person', 'embedding']):
            cleaned_text = clean_board_name_from_tag(text)
            if cleaned_text:
                for board_name, board_id in existing_boards_lower.items():
                    board_clean = clean_board_name_from_tag(board_name)
                    if board_clean and (board_clean == cleaned_text or board_clean in cleaned_text or cleaned_text in board_clean):
                        if len(board_clean) >= 3:
                            return board_name, f"LoRA/Tag Match ({text} -> {board_name})"

    # --- 2. ERST DANACH KOMMT DER PROMPT! ---
    prompt = meta.get('prompt', '') or meta.get('positive_prompt', '') or all_text
    if prompt:
        first_part = prompt.split(',')[0].strip()
        cleaned_keyword = clean_board_name_from_tag(first_part)
        
        for board_name, board_id in existing_boards_lower.items():
            board_clean = clean_board_name_from_tag(board_name)
            if board_clean and cleaned_keyword and (board_clean == cleaned_keyword or board_clean in cleaned_keyword or cleaned_keyword in board_clean):
                if len(board_clean) >= 3:
                    return board_name, f"Prompt-Keyword Match ({first_part} -> {board_name})"

    # -------------------------------------------------------------
    # 3. PRIORITÄT: Feste RULES
    # -------------------------------------------------------------
    for board_name, keywords in RULES.items():
        for kw in keywords:
            if kw.lower() in all_text.lower():
                return board_name, f"Rule Match ({kw})"

    # -------------------------------------------------------------
    # 4. PRIORITÄT: Fallback
    # -------------------------------------------------------------
    return None, None

def get_or_create_board(cursor, board_name, existing_boards, existing_boards_lower, dry_run, allow_creation):
    normalized = board_name.lower()
    if normalized in existing_boards_lower:
        return existing_boards_lower[normalized]
    
    # Wenn das Board nicht existiert und wir eins erstellen dürfen:
    if allow_creation:
        if not dry_run:
            new_id = str(uuid.uuid4())
            cursor.execute("INSERT INTO boards (board_id, board_name) VALUES (?, ?)", (new_id, board_name))
            existing_boards_lower[normalized] = new_id
            existing_boards[board_name] = new_id
            return new_id
        else:
            # DRY RUN: Gib den Namen (oder einen Marker) zurück, damit die Statistik weiß, dass es neu ist!
            # Wir geben hier temporär den board_name zurück, damit der Counter ihn greifen kann.
            existing_boards_lower[normalized] = f"DRY_RUN_{board_name}"
            return f"DRY_RUN_{board_name}"
            
    return None

def auto_sort_boards(db_path: Path, dry_run=True, allow_creation=False):
    """Hauptfunktion: Liest DB, berechnet Ziel-Boards und aktualisiert board_images."""
    print(f"📂 Verwende Datenbank: {db_path}")
    if not db_path.exists():
        print(f"❌ Datenbank nicht gefunden unter: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Bestehende Boards aus DB laden und Case-Insensitive-Dictionary aufbauen
    cursor.execute("SELECT board_id, board_name FROM boards")
    db_boards = cursor.fetchall()
    
    existing_boards = {name: b_id for b_id, name in db_boards}
    existing_boards_lower = {name.lower(): b_id for b_id, name in db_boards}

    # 2. ALLE Bilder abfragen (auch ohne Filter, um den gesamten Bestand zu prüfen)
    cursor.execute("""
        SELECT i.image_name, bi.board_id, i.metadata 
        FROM images i
        LEFT JOIN board_images bi ON i.image_name = bi.image_name
    """)
    rows = cursor.fetchall()

    print(f"🔍 Analysiere alle {len(rows)} Bilder in der Datenbank...")
    if dry_run:
        print("⚠️ [DRY RUN MODUS] Es werden keine Änderungen an der Datenbank vorgenommen!\n")

    updated_count = 0
    skipped_count = 0
    
    # Statistik-Zähler für die Boards
    board_summary = Counter()

    # Wir bauen uns zusätzlich ein Mapping von board_id -> board_name auf
    id_to_board_name = {b_id: name for name, b_id in existing_boards.items()}

    for img_name, current_board_id, metadata_raw in tqdm(
        rows, desc="Prüfe & Sortiere alle Bilder", unit="Bild"
    ):
        # 1. Ziel-Board über unsere angepasste determine_board Funktion ermitteln
        target_board_name, reason = determine_board(metadata_raw, existing_boards_lower)
        """
        # --- ERWEITERTER DEBUG-BLOCK FÜR LORA-SUCHE ---
        if "examplehash" in img_name:  
            print(f"\n🔍 [DEBUG LORA-SUCHE] Bild: {img_name}")
            print(f"   - Aktuelles Board: {id_to_board_name.get(current_board_id, 'Kein Board')}")
            
            try:
                meta_dict = json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
                all_found_strings = extract_all_strings_from_json(meta_dict)
                
                # Wir filtern gezielt nach LoRA-Endungen in allen gefundenen Strings:
                lora_candidates = [s for s in all_found_strings if any(ext in s.lower() for ext in ['.safetensors', '.ckpt', '.pt', 'lora'])]
                print(f"   - Gefundene LoRA-Verdächtige in Metadaten: {lora_candidates}")
                
            except Exception as e:
                print(f"   - Fehler beim Parsen: {e}")
        """
        # -------------------------------

        current_board_name = id_to_board_name.get(current_board_id, None)

        # Wenn WIRKLICH kein spezifisches Match gefunden wurde:
        if not target_board_name:
            # Fall A: Das Bild war vorher schon in einem Board -> In Ruhe lassen
            if current_board_name:
                skipped_count += 1
                continue
            else:
                # Fall B: Das Bild war in gar keinem Board -> ab ins Fallback-Board!
                target_board_name = FALLBACK_BOARD_NAME
                reason = "Fallback (Unassigned Image)"

        # 2. Ziel-Board-ID ermitteln / erstellen
        is_lora_match = "LoRA-Filename Match" in reason or "Fallback" in reason
        target_board_id = get_or_create_board(
            cursor, target_board_name, existing_boards, 
            existing_boards_lower, dry_run, allow_creation=is_lora_match
        )

        # 3. Abgleich: Weicht das aktuelle Board vom ermittelten Soll-Board ab?
        if target_board_id and (current_board_name is None or current_board_name.lower() != target_board_name.lower()):
            board_summary[target_board_name] += 1

            if not dry_run:
                cursor.execute("DELETE FROM board_images WHERE image_name = ?", (img_name,))
                cursor.execute(
                    "INSERT INTO board_images (board_id, image_name) VALUES (?, ?)",
                    (target_board_id, img_name),
                )
            updated_count += 1
        else:
            skipped_count += 1

    # --- KOMPAKTE ZUSAMMENFASSUNG AM ENDE ---
    print("\n" + "="*50)
    if dry_run:
        print("🔍 [DRY RUN ZUSAMMENFASSUNG]")
    else:
        print("✅ [ERFOLGREICH ANGEWENDET]")
    print("="*50)

    if board_summary:
        for board, count in board_summary.most_common():
            # Prüfen, ob das Board vor dem Lauf schon da war oder neu erstellt worden wäre
            is_new = board.lower() not in [b.lower() for b in existing_boards.keys()]
            
            status_label = " 🆕 [NEU]" if is_new else ""
            print(f"📁 {count} Bilder ➔ Board: {board}{status_label}")
    else:
        print("ℹ️ Keine Bilder mussten verschoben werden.")

    print("="*50)
    print(f"Gesamt verschoben: {updated_count} Bilder")
    print(f"Übersprungen / Unverändert: {skipped_count} Bilder")
    # DEBUG-HILFE: Lass dir für ein bestimmtes Bild anzeigen, was das Skript denkt
    """
    if "xyz" in (metadata_raw or "").lower():
        print(f"DEBUG Check für Bild {img_name}: Ermitteltes Ziel = {target_board_name} (Grund: {reason})")
    """
    if not dry_run:
        conn.commit()
        
    conn.close()


def main():
    parser = argparse.ArgumentParser(
        prog="invoke_auto_sorter",
        description="InvokeAI Auto-Sorter: Sortiert Bilder anhand von LoRAs und Prompts automatisch in die richtigen Boards.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Pfad zur invokeai.db (Standard: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Führt die Änderungen tatsächlich in der DB aus (ohne diesen Flag läuft nur ein Dry-Run).",
    )
    
    parser.add_argument(
        "--allow-creation",
        dest="allow_creation",
        action="store_true",
        help="Erlaubt dem Skript, automatisch neue Boards in der Datenbank anzulegen.",
    )
    parser.add_argument(
        "--no-creation",
        dest="allow_creation",
        action="store_false",
        help="Verbietet das automatische Erstellen neuer Boards.",
    )
    parser.set_defaults(allow_creation=False)  # Standardmäßig auf True oder False setzen (je nach Wunsch)

    args = parser.parse_args()
    dry_run = not args.apply
    
    # Übergebe allow_creation an deine Hauptfunktion (achte darauf, dass auto_sort_boards den Parameter auch annimmt)
    auto_sort_boards(args.db, dry_run=dry_run, allow_creation=args.allow_creation)

if __name__ == "__main__":
    main()
