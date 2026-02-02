import os
import glob
import networkx as nx
from PIL import Image
import math
from pathlib import Path

def get_image_size(path):
    with Image.open(path) as img:
        return img.size

def get_all_complete_files(complete_dir):
    """Index all png files in complete_dir by filename"""
    index = {}
    for root, dirs, files in os.walk(complete_dir):
        for f in files:
            if f.lower().endswith('.png'):
                if f not in index:
                    index[f] = []
                index[f].append(os.path.join(root, f))
    return index

def regenerate(resized_dir, complete_dir):
    complete_index = get_all_complete_files(complete_dir)
    resized_files = glob.glob(os.path.join(resized_dir, "*.png"))
    
    print(f"Found {len(resized_files)} resized images.")
    processed_count = 0
    
    for resized_path in resized_files:
        filename = os.path.basename(resized_path)
        
        if filename not in complete_index:
            print(f"[Warning] Source image for {filename} not found in complete dataset.")
            continue
            
        candidates = complete_index[filename]
        
        # Determine best match based on aspect ratio
        w_new, h_new = get_image_size(resized_path)
        aspect_new = w_new / h_new
        
        best_match = None
        min_diff = 1.0
        
        for cand in candidates:
            w_orig, h_orig = get_image_size(cand)
            aspect_orig = w_orig / h_orig
            
            diff = abs(aspect_new - aspect_orig)
            if diff < 0.05: # 5% tolerance
                if diff < min_diff:
                    min_diff = diff
                    best_match = cand
                    
        if not best_match:
            print(f"[Warning] No matching aspect ratio found for {filename}. Candidates: {candidates}")
            continue
            
        # Found source
        source_img_path = best_match
        source_base = os.path.splitext(source_img_path)[0]
        # graphml usually shares base name
        source_graphml = source_base + ".graphml"
        
        if not os.path.exists(source_graphml):
            print(f"[Warning] Annotation file missing for {source_img_path}")
            continue
            
        # Calc scale
        w_orig, h_orig = get_image_size(source_img_path)
        scale_x = w_new / w_orig
        scale_y = h_new / h_orig
        
        # Read and Modify GraphML
        try:
            graph = nx.read_graphml(source_graphml)
            
            # Modify nodes (Objects)
            for node_id, node_attrs in graph.nodes(data=True):
                bbox_keys = ['xmin', 'ymin', 'xmax', 'ymax']
                if all(key in node_attrs for key in bbox_keys):
                    # Coordinates might be strings in graphml
                    try:
                        xmin = float(node_attrs['xmin'])
                        ymin = float(node_attrs['ymin'])
                        xmax = float(node_attrs['xmax'])
                        ymax = float(node_attrs['ymax'])
                        
                        # Scale
                        new_xmin = xmin * scale_x
                        new_ymin = ymin * scale_y
                        new_xmax = xmax * scale_x
                        new_ymax = ymax * scale_y
                        
                        # Update attrs
                        node_attrs['xmin'] = str(new_xmin)
                        node_attrs['ymin'] = str(new_ymin)
                        node_attrs['xmax'] = str(new_xmax)
                        node_attrs['ymax'] = str(new_ymax)
                    except ValueError:
                        pass
            
            # Edges (Relations) preserve attributes automatically
            
            # Write to resized_dir
            dest_graphml = os.path.join(resized_dir, filename.replace('.png', '.graphml').replace('.PNG', '.graphml'))
            nx.write_graphml(graph, dest_graphml)
            processed_count += 1
            
        except Exception as e:
            print(f"[Error] Failed to process {filename}: {e}")
            
    print(f"Successfully regenerated {processed_count} annotation files.")

if __name__ == "__main__":
    RESIZED_DIR = "/mnt/ShareDB_6TB/baitianyou/RelTR-main/data/pid_resized"
    COMPLETE_DIR = "/mnt/ShareDB_6TB/baitianyou/RelTR-main/data/complete"
    regenerate(RESIZED_DIR, COMPLETE_DIR)
