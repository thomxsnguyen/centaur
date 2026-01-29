#!/usr/bin/env python3
"""
Generate transcripts for contextual bandit experiment from participant data.

This script parses CSV files containing participant data and generates
formatted transcripts showing the trial-by-trial decisions and rewards
in a contextual bandit task with three actions: Red, White, and Black.
"""

import csv
import os
import argparse
from pathlib import Path
from typing import List, Dict, Optional


def map_choice_to_action(choice: str) -> str:
    """
    Map the choice code to action name.
    
    Args:
        choice: Choice code from CSV (e.g., 'red_pirate', 'white_pirate', 'black_pirate')
        
    Returns:
        Action name: 'Red', 'White', or 'Black'
    """
    choice_lower = choice.lower()
    if 'red' in choice_lower:
        return 'Red'
    elif 'white' in choice_lower:
        return 'White'
    elif 'black' in choice_lower:
        return 'Black'
    else:
        return 'Unknown'


def parse_participant_data(csv_path: str) -> List[Dict]:
    """
    Parse participant CSV file and extract trial data.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        List of trial dictionaries containing context, choice, and reward
    """
    trials = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Only process rows that are actual trials (pirate_*)
            trial_type = row.get('TrialType', '')

            # Accept any trial_type that begins with pirate_ (avoid overly brittle checks)
            if trial_type.startswith('pirate_'):
                context = row.get('context', '')
                choice_raw = row.get('choice', '')
                reward = row.get('reward', '')

                # Skip if essential data is missing (use explicit empty-string checks)
                if context == '' or choice_raw == '' or reward == '':
                    continue

                # Map choice to action
                action = map_choice_to_action(choice_raw)

                trials.append({
                    'trial_num': len(trials) + 1,
                    'context': context,
                    'choice': action,
                    'reward': reward,
                    'trial_type': trial_type,
                })
    
    return trials


def generate_transcript(trials: List[Dict], participant_id: str) -> str:
    """
    Generate formatted transcript from trial data.
    
    Args:
        trials: List of trial dictionaries
        participant_id: Participant identifier
        
    Returns:
        Formatted transcript string
    """
    lines = []
    lines.append(f"PARTICIPANT: {participant_id}")
    lines.append("=" * 60)
    lines.append("TASK: Contextual bandit experiment.")
    lines.append("ACTIONS: Red, White, Black.")
    lines.append("GOAL: Earn as many rewards as possible.")
    lines.append("=" * 60)
    lines.append("")
    
    for trial in trials:
        lines.append(f"TRIAL {trial['trial_num']}")
        lines.append(f"CONTEXT: {trial['context']}")
        lines.append("AVAILABLE_ACTIONS: Red, White, Black")
        lines.append(f"CHOOSE_ACTION: {trial['choice']}")
        lines.append(f"REWARD: {trial['reward']}")
        lines.append("")
    
    return "\n".join(lines)


def clean_trials(trials: List[Dict], strategy: str = "trim", allow_numeric_context: bool = False) -> List[Dict]:
    """Clean a list of trial dicts.

    strategy:
      - 'trim': cut episode before first Unknown choice
      - 'drop': drop rows where CHOOSE_ACTION is Unknown
    """
    # find Unknown indices
    unknown_idxs = [i for i, t in enumerate(trials) if t.get('choice', '') == 'Unknown']
    if unknown_idxs:
        if strategy == 'trim':
            trials = trials[:unknown_idxs[0]]
        elif strategy == 'drop':
            trials = [t for t in trials if t.get('choice', '') != 'Unknown']

    def invalid_context(c):
        if not c:
            return True
        cu = c.strip().upper()
        if cu in {'NA', 'NONE'}:
            return True
        if (not allow_numeric_context) and cu.isdigit():
            return True
        return False

    # remove invalid contexts
    trials = [t for t in trials if not invalid_context(t.get('context', ''))]

    # Validate choice against canonical actions
    VALID_ACTIONS = {'Red', 'White', 'Black'}
    cleaned = [t for t in trials if t.get('choice') in VALID_ACTIONS]

    # Reindex trial numbers
    for i, t in enumerate(cleaned, start=1):
        t['trial_num'] = i

    return cleaned


def process_all_participants(data_dir: str, output_dir: str, clean: bool = False, strategy: str = 'trim', allow_numeric_context: bool = False) -> None:
    """
    Process all participant CSV files and generate transcripts.
    
    Args:
        data_dir: Directory containing participant CSV files
        output_dir: Directory to save transcript files
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Find all participant CSV files (exclude transformed versions)
    csv_files = [f for f in data_path.glob('participant_*.csv') 
                 if 'transformed' not in f.name and 'copy' not in f.name]
    
    print(f"Found {len(csv_files)} participant files to process")
    
    for csv_file in sorted(csv_files):
        try:
            # Extract participant ID from filename
            filename = csv_file.stem  # e.g., 'participant_1132'
            participant_id = filename.replace('participant_', '')

            print(f"Processing {participant_id}...", end=' ')

            # Parse trials
            trials = parse_participant_data(str(csv_file))

            if not trials:
                print(f"No trials found, skipping")
                continue

            # Optionally clean trials before generating transcript
            if clean:
                print(f"  trials before cleaning: {len(trials)}")
                cleaned = clean_trials(trials, strategy=strategy, allow_numeric_context=allow_numeric_context)
                # debug info: first Unknown index
                unknown_idxs = [i for i, t in enumerate(trials) if t.get('choice') == 'Unknown']
                if unknown_idxs:
                    print(f"  first Unknown at trial index: {unknown_idxs[0]+1}")
                # sample unique contexts
                contexts = list({t.get('context','') for t in trials})[:10]
                print(f"  sample contexts: {contexts}")
                print(f"  trials after cleaning: {len(cleaned)}")
                # Use cleaned trials for transcript generation (do not write CSVs)
                trials_to_use = cleaned
            else:
                trials_to_use = trials

            # Generate transcript
            transcript = generate_transcript(trials_to_use, participant_id)

            # Save to file: if cleaning is enabled, write only the cleaned transcript
            output_path.mkdir(parents=True, exist_ok=True)
            if clean:
                # Write cleaned transcript into the main output folder (overwrite original)
                with open(output_path / f"transcript_{participant_id}.txt", 'w', encoding='utf-8') as f:
                    f.write(transcript)
            else:
                # No cleaning requested: write the unmodified transcript
                with open(output_path / f"transcript_{participant_id}.txt", 'w', encoding='utf-8') as f:
                    f.write(transcript)

            print(f"✓ Generated transcript with {len(trials_to_use)} trials")

        except Exception as e:
            print(f"✗ Error: {e}")


def main():
    """Main entry point."""
    # Get script directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    data_dir = project_root / 'data'
    output_dir = project_root / 'transcripts'
    
    print("Contextual Bandit Transcript Generator")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    print()
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    # Parse CLI args so users can toggle cleaning and strategy
    parser = argparse.ArgumentParser(description="Generate transcripts for contextual bandit experiment")
    parser.add_argument('--data-dir', type=str, default=str(data_dir), help='Path to data directory')
    parser.add_argument('--output-dir', type=str, default=str(output_dir), help='Path to output transcripts directory')
    parser.add_argument('--clean', action=argparse.BooleanOptionalAction, default=True, help='Enable cleaning (default: true)')
    parser.add_argument('--strategy', choices=['trim', 'drop'], default='trim', help='Cleaning strategy to apply')
    parser.add_argument('--allow-numeric-context', action='store_true', default=False, help='Allow numeric-only contexts during cleaning')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print(f"Cleaning: {args.clean}, Strategy: {args.strategy}, Allow numeric context: {args.allow_numeric_context}")

    process_all_participants(str(data_dir), str(output_dir), clean=args.clean, strategy=args.strategy, allow_numeric_context=args.allow_numeric_context)
    
    print()
    print("=" * 60)
    print("Processing complete")
    print(f"Transcripts saved to: {output_dir}")


if __name__ == '__main__':
    main()
