#!/usr/bin/env python3
"""
Generate transcripts for contextual bandit experiment from participant data.

This script parses CSV files containing participant data and generates
formatted transcripts showing the trial-by-trial decisions and rewards
in a contextual bandit task with three actions: Red, White, and Black.
"""

import csv
import os
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
            # Only process rows that are actual trials (pirate_X)
            trial_type = row.get('TrialType', '')
            
            if trial_type.startswith('pirate_') and not trial_type.startswith('practice_'):
                context = row.get('context', '')
                choice_raw = row.get('choice', '')
                reward = row.get('reward', '')
                
                # Skip if essential data is missing
                if not context or not choice_raw or not reward:
                    continue
                
                # Map choice to action
                action = map_choice_to_action(choice_raw)
                
                trials.append({
                    'trial_num': len(trials) + 1,
                    'context': context,
                    'choice': action,
                    'reward': reward
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


def process_all_participants(data_dir: str, output_dir: str) -> None:
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
            
            # Generate transcript
            transcript = generate_transcript(trials, participant_id)
            
            # Save to file
            output_file = output_path / f"transcript_{participant_id}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(transcript)
            
            print(f"✓ Generated transcript with {len(trials)} trials")
            
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
    
    process_all_participants(str(data_dir), str(output_dir))
    
    print()
    print("=" * 60)
    print("Processing complete")
    print(f"Transcripts saved to: {output_dir}")


if __name__ == '__main__':
    main()
