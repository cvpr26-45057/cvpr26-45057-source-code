import os
import random
import networkx as nx
from PIL import Image, ImageDraw, ImageFont

def extract_annotations_from_graphml(graphml_path):
    """从 GraphML 文件中提取标注信息"""
    try:
        graph = nx.read_graphml(graphml_path)
        
        # 提取节点（对象）信息
        objects = []
        for node_id, node_attrs in graph.nodes(data=True):
            # 检查是否有边界框信息
            bbox_keys = ['xmin', 'ymin', 'xmax', 'ymax']
            if all(key in node_attrs for key in bbox_keys):
                try:
                    bbox = [float(node_attrs[key]) for key in bbox_keys]
                    
                    obj_info = {
                        'id': node_id,
                        'bbox': bbox,  # [xmin, ymin, xmax, ymax]
                        'label': node_attrs.get('label', 'unknown'),
                    }
                    objects.append(obj_info)
                except (ValueError, TypeError):
                    continue
        
        # 提取边（关系）信息
        relations = []
        for src, dst, edge_attrs in graph.edges(data=True):
            # Try 'edge_label' first as per pid_datasets.py, then others if needed
            predicate = edge_attrs.get('edge_label')
            if not predicate:
                 predicate = edge_attrs.get('interaction')
            if not predicate:
                 predicate = edge_attrs.get('label', 'unknown')
            
            rel_info = {
                'subject_id': src,
                'object_id': dst,
                'predicate': predicate
            }

            relations.append(rel_info)
            
        return objects, relations

    except Exception as e:
        print(f"Error parsing dataframe {graphml_path}: {e}")
        return [], []

def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def visualize_pid_data(data_dir, output_dir, num_samples=10):
    images = [f for f in os.listdir(data_dir) if f.endswith('.png') or f.endswith('.jpg')]
    # Filter to ensure graphml exists
    images = [img for img in images if os.path.exists(os.path.join(data_dir, os.path.splitext(img)[0] + '.graphml'))]
    
    if not images:
        print("No paired image/graphml files found.")
        return

    sampled_images = random.sample(images, min(len(images), num_samples))
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for img_file in sampled_images:
        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(data_dir, img_file)
        graphml_path = os.path.join(data_dir, base_name + '.graphml')
        
        objects, relations = extract_annotations_from_graphml(graphml_path)
        
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue

        draw = ImageDraw.Draw(img)
        w, h = img.size
        
        obj_map = {} 
        
        # Draw Objects
        for obj in objects:
            bbox = obj['bbox']
            xmin, ymin, xmax, ymax = bbox
            
            # Sanity check and cleanup coordinates
            # Ensure order
            x0, x1 = sorted([xmin, xmax])
            y0, y1 = sorted([ymin, ymax])
            
            # Clip to image boundaries
            x0 = max(0, min(x0, w))
            y0 = max(0, min(y0, h))
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            
            # If box is too small or invalid after clip, skip
            if x1 - x0 < 1 or y1 - y0 < 1:
                print(f"Skipping invalid bbox in {img_file} ID {obj['id']}: orig={bbox} clipped={x0,y0,x1,y1}")
                continue

            # Update obj bbox with sanitized values for relation drawing
            obj['bbox'] = [x0, y0, x1, y1]
            obj_map[obj['id']] = obj

            color = get_random_color()
            
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            
            label = obj['label']
            # Draw label background
            try:
                text_bbox = draw.textbbox((x0, y0), label, font=font)
                draw.rectangle(text_bbox, fill=color)
            except AttributeError:
                # Older Pillow
                text_w, text_h = draw.textsize(label, font=font)
                draw.rectangle([x0, y0, x0 + text_w, y0 + text_h], fill=color)

            draw.text((x0, y0), label, fill="white", font=font)
            

        # Draw Relations (Lines and text mid-point)
        for rel in relations:
            subj = obj_map.get(rel['subject_id'])
            obj = obj_map.get(rel['object_id'])
            if subj and obj:
                pred = rel['predicate']
                
                # Center points
                sx, sy = (subj['bbox'][0] + subj['bbox'][2])/2, (subj['bbox'][1] + subj['bbox'][3])/2
                ox, oy = (obj['bbox'][0] + obj['bbox'][2])/2, (obj['bbox'][1] + obj['bbox'][3])/2
                
                # Draw line
                line_color = (255, 255, 0) # Yellow
                draw.line([(sx, sy), (ox, oy)], fill=line_color, width=2)
                
                # Draw text at midpoint
                mid_x, mid_y = (sx + ox)/2, (sy + oy)/2
                
                try:
                    text_bbox = draw.textbbox((mid_x, mid_y), pred, font=font)
                    text_w = text_bbox[2] - text_bbox[0]
                    text_h = text_bbox[3] - text_bbox[1]
                except AttributeError:
                    text_w, text_h = draw.textsize(pred, font=font)
                
                draw.rectangle([mid_x - text_w/2 - 2, mid_y - text_h/2 - 2, mid_x + text_w/2 + 2, mid_y + text_h/2 + 2], fill="black")
                draw.text((mid_x - text_w/2, mid_y - text_h/2), pred, fill="yellow", font=font)

        out_path = os.path.join(output_dir, f"viz_{base_name}.png")
        img.save(out_path)
        print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    data_dir = "/mnt/ShareDB_6TB/baitianyou/RelTR-main/data/pid_resized"
    output_dir = "/mnt/ShareDB_6TB/baitianyou/RelTR-main/viz_pid_resized"
    visualize_pid_data(data_dir, output_dir)
