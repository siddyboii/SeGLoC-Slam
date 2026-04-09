#!/usr/bin/env python3
"""Inference script for YOLO-World ONNX model with embedded classes"""

import onnxruntime as ort
import cv2
import numpy as np
import argparse
from pathlib import Path

def preprocess_image(image_path, input_size=(640, 640)):
    """Load and preprocess image for YOLO-World"""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    original_height, original_width = image.shape[:2]
    
    # Resize image
    image_resized = cv2.resize(image, input_size)
    
    # Normalize to [0, 1]
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb.astype(np.float32) / 255.0
    
    # Convert to CHW format (channels first)
    image_chw = np.transpose(image_rgb, (2, 0, 1))
    
    # Add batch dimension
    image_batch = np.expand_dims(image_chw, 0)
    
    return image_batch, image, (original_width, original_height), (input_size[1], input_size[0])

def postprocess_detections(outputs, conf_threshold=0.25, iou_threshold=0.65):
    """
    Post-process ONNX model outputs
    
    outputs[0]: scores - Shape: (1, 8400, num_classes)
    outputs[1]: boxes - Shape: (1, 8400, 4) - [cx, cy, w, h]
    """
    scores = outputs[0]  # Shape: (1, 8400, 3) - scores for 3 classes
    boxes = outputs[1]   # Shape: (1, 8400, 4) - [cx, cy, w, h]
    
    detections = []
    
    # For each detection (anchor point)
    for box_idx in range(boxes.shape[1]):
        score_vector = scores[0, box_idx]  # Scores for this anchor
        
        # Get max score and class
        class_id = np.argmax(score_vector)
        confidence = float(score_vector[class_id])
        
        # Filter by confidence threshold
        if confidence >= conf_threshold:
            # Get box coordinates
            cx, cy, w, h = boxes[0, box_idx]  # Center x, center y, width, height
            
            # Convert to x1, y1, x2, y2 format
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            
            detections.append({
                'box': [x1, y1, x2, y2],
                'class_id': int(class_id),
                'confidence': confidence
            })
    
    return detections

def draw_detections(image, detections, class_names, original_size, model_input_size):
    """Draw bounding boxes and labels on image"""
    
    original_width, original_height = original_size
    model_width, model_height = model_input_size
    
    # Scale factor for drawing on original image
    scale_x = original_width / model_width
    scale_y = original_height / model_height
    
    # Colors for each class
    colors = {
        0: (0, 255, 0),      # Green for helmet
        1: (255, 0, 0),      # Blue for head
        2: (0, 255, 255),    # Yellow for sunglasses
    }
    
    image_with_boxes = image.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['box']
        
        # Scale to original image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)
        
        class_id = det['class_id']
        confidence = det['confidence']
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        
        color = colors.get(class_id, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(image_with_boxes, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), color, 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Background for text
        cv2.rectangle(image_with_boxes, 
                     (x1_scaled, y1_scaled - label_size[1] - 4),
                     (x1_scaled + label_size[0], y1_scaled),
                     color, -1)
        
        # Text
        cv2.putText(image_with_boxes, label,
                   (x1_scaled, y1_scaled - 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return image_with_boxes

def infer_onnx(onnx_path, image_path, class_names=None, conf_threshold=0.25, input_size=(640, 640), use_gpu=True):
    """Run inference on image"""
    
    if class_names is None:
        class_names = ['helmet', 'head', 'sunglasses']
    
    print(f"\n{'='*60}")
    print(f"YOLO-World ONNX Inference")
    print(f"{'='*60}")
    print(f"Model: {Path(onnx_path).name}")
    print(f"Image: {Path(image_path).name}")
    print(f"Classes: {class_names}")
    print(f"Confidence Threshold: {conf_threshold}")
    print(f"Device: {'GPU (if available)' if use_gpu else 'CPU'}")
    
    # Create ONNX session
    print("\n✓ Loading ONNX model...")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Use GPU if available, fallback to CPU
    if use_gpu:
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    
    session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
    
    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    
    # Preprocess image
    print("✓ Preprocessing image...")
    image_batch, original_image, orig_size, model_size = preprocess_image(image_path, input_size)
    
    # Run inference
    print("✓ Running inference...")
    outputs = session.run(output_names, {input_name: image_batch})
    
    # Post-process
    print("✓ Post-processing detections...")
    detections = postprocess_detections(outputs, conf_threshold=conf_threshold)
    
    print(f"\n{'='*60}")
    print(f"Detection Results")
    print(f"{'='*60}")
    print(f"Found {len(detections)} objects:")
    
    for i, det in enumerate(detections, 1):
        class_id = det['class_id']
        confidence = det['confidence']
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        box = det['box']
        print(f"  {i}. {class_name:12s} - Confidence: {confidence:.3f} - Box: [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")
    
    # Draw detections
    print("\n✓ Drawing bounding boxes...")
    image_annotated = draw_detections(original_image, detections, class_names, orig_size, model_size)
    
    # Save result
    output_path = Path(image_path).stem + '_detected.jpg'
    cv2.imwrite(output_path, image_annotated)
    print(f"✓ Saved result to: {output_path}")
    
    print(f"{'='*60}\n")
    
    return detections, image_annotated

def main():
    parser = argparse.ArgumentParser(description='YOLO-World ONNX Inference with Embedded Classes')
    parser.add_argument('onnx_model', help='Path to ONNX model')
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('--conf-threshold', type=float, default=0.25, help='Confidence threshold (default: 0.25)')
    parser.add_argument('--input-size', type=int, nargs=2, default=[640, 640], help='Model input size (default: 640 640)')
    parser.add_argument('--classes', type=str, nargs='+', default=['helmet', 'head', 'sunglasses'], 
                        help='Class names (default: helmet head sunglasses)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference (default: use GPU if available)')
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.onnx_model).exists():
        print(f"ERROR: ONNX model not found: {args.onnx_model}")
        return 1
    
    if not Path(args.image).exists():
        print(f"ERROR: Image not found: {args.image}")
        return 1
    
    # Run inference
    try:
        detections, annotated_image = infer_onnx(
            args.onnx_model,
            args.image,
            class_names=args.classes,
            conf_threshold=args.conf_threshold,
            input_size=tuple(args.input_size),
            use_gpu=not args.cpu
        )
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())