"""
Create sample annotation file from existing data
Helper script to convert existing data structure to annotation format
"""
import json
from pathlib import Path
import argparse


def create_annotations_from_directory(data_dir: str, output_file: str, 
                                     label_mapping: dict = None):
    """
    Create annotation file from directory structure
    
    Args:
        data_dir: Directory with videos organized by label
        output_file: Output JSON file path
        label_mapping: Optional mapping from directory name to label
    """
    data_path = Path(data_dir)
    annotations = []
    
    # If label_mapping not provided, use directory names as labels
    if label_mapping is None:
        label_mapping = {}
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    for video_file in data_path.rglob('*'):
        if video_file.suffix.lower() in video_extensions:
            # Get label from directory structure
            # Assume structure: data_dir/label/video.mp4
            relative_path = video_file.relative_to(data_path)
            parts = relative_path.parts
            
            if len(parts) >= 2:
                label = parts[0]  # Parent directory name
            else:
                label = video_file.stem  # Use filename without extension
            
            # Apply mapping if provided
            label = label_mapping.get(label, label)
            
            annotation = {
                'video_path': str(relative_path),
                'sign_label': label.upper(),
                'gloss': label.upper(),
                'signer_id': 'unknown',
                'start_frame': 0,
                'end_frame': -1  # Will be determined during processing
            }
            
            annotations.append(annotation)
    
    # Save annotations
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Created {len(annotations)} annotations in {output_file}")
    print(f"Unique labels: {len(set(a['sign_label'] for a in annotations))}")
    
    return annotations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create annotation file from directory')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory with videos')
    parser.add_argument('--output-file', type=str, required=True,
                       help='Output JSON annotation file')
    parser.add_argument('--label-mapping', type=str, default=None,
                       help='JSON file with label mappings (optional)')
    
    args = parser.parse_args()
    
    label_mapping = None
    if args.label_mapping:
        with open(args.label_mapping, 'r') as f:
            label_mapping = json.load(f)
    
    create_annotations_from_directory(
        args.data_dir,
        args.output_file,
        label_mapping
    )


