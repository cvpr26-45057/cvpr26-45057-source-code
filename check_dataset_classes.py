import os
import networkx as nx
from collections import Counter

def check_classes(data_dir):
    graphml_files = [os.path.join(root, f) 
                     for root, dirs, files in os.walk(data_dir) 
                     for f in files if f.endswith('.graphml')]
    
    if not graphml_files:
        print(f"No graphml files found in {data_dir}")
        return

    obj_labels = Counter()
    rel_predicates = Counter()

    print(f"Scanning {len(graphml_files)} annotation files...")

    for f in graphml_files:
        try:
            graph = nx.read_graphml(f)
            
            # Count object labels
            for _, node_attrs in graph.nodes(data=True):
                label = node_attrs.get('label')
                if label:
                    obj_labels[label] += 1
            
            # Count relation predicates
            for _, _, edge_attrs in graph.edges(data=True):
                # Try multiple keys for predicate
                pred = edge_attrs.get('edge_label') or                        edge_attrs.get('interaction') or                        edge_attrs.get('label')
                if pred:
                    rel_predicates[pred] += 1
                    
        except Exception as e:
            print(f"Error reading {f}: {e}")

    print("\n=== Object Classes ===")
    for label, count in obj_labels.most_common():
        print(f"{label}: {count}")
        
    print("\n=== Relation Classes ===")
    for label, count in rel_predicates.most_common():
        print(f"{label}: {count}")

if __name__ == "__main__":
    check_classes("/mnt/ShareDB_6TB/baitianyou/RelTR-main/data/pid_resized")
