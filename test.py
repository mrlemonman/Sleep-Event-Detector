import tensorflow as tf
import numpy as np
import sounddevice as sd
import time
from datetime import datetime

def test_tflite_model_simple(model_path="sleep_detector.tflite"):
    """
    Simple test: Record 5 seconds, analyze with TFLite, show results
    """
    print("Sleep Event Detector Test")
    print("=" * 40)
    
    # Load TFLite model
    try:
        interpreter = tf.lite.Interpreter(model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Model loaded: {model_path}")
        print(f"Expected input: {input_details[0]['shape'][0]} samples")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have the sleep_detector.tflite file")
        return
    
    # Define labels and thresholds
    yamnet_labels = ["snore", "cough", "fart", "speech", "laughter", "music", "silence"]
    sneeze_labels = ["sneeze", "sniff", "neither"]
    
    # Thresholds
    yamnet_thresholds = [0.2, 0.2, 0.2, 0.4, 0.2, 0.3, 0.5]  # snore, cough, fart, speech, laughter, music, silence
    sneeze_threshold = 0.3
    
    # Recording parameters
    duration = 5  # seconds
    sample_rate = 16000
    samples_needed = duration * sample_rate
    
    print(f"\nRecording {duration} seconds...")
    print("Make some sounds: cough, sniff, talk, laugh, or stay quiet")
    print("Recording starts in 3 seconds...")
    
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    print("Recording NOW!")
    
    # Record audio
    recording = sd.rec(samples_needed, samplerate=sample_rate, channels=1, dtype='float32')
    
    # Show countdown
    for i in range(duration, 0, -1):
        print(f"  {i} seconds left...", end='\r')
        time.sleep(1)
    
    sd.wait()  # Wait for recording to finish
    print("\nRecording finished!")
    
    # Process audio
    waveform = recording.flatten()
    audio_level = np.sqrt(np.mean(waveform**2))
    
    print(f"Audio level (RMS): {audio_level:.4f}")
    
    if audio_level < 0.001:
        print("Warning: Very low audio level. Check microphone settings.")
    
    # Run TFLite inference
    try:
        interpreter.set_tensor(input_details[0]['index'], waveform)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print("\nModel Analysis Results:")
        print("-" * 30)
        
        # Check if model produced any output
        if output.sum() == 0:
            print("Model output: All zeros (model may have conversion issues)")
            return
        
        # Analyze YAMNet results (indices 0-6)
        yamnet_scores = output[:7]
        detected_yamnet = []
        
        print("YAMNet Detection:")
        for i, (label, score, threshold) in enumerate(zip(yamnet_labels, yamnet_scores, yamnet_thresholds)):
            status = "DETECTED" if score > threshold else "below threshold"
            print(f"  {label:12}: {score:.4f} ({status})")
            
            if score > threshold:
                detected_yamnet.append((label, score))
        
        # Analyze sneeze/sniff results (indices 7-9)
        sneeze_probs = output[7:10]
        
        print(f"\nSneeze/Sniff Classification:")
        for label, prob in zip(sneeze_labels, sneeze_probs):
            marker = " <- ABOVE THRESHOLD" if (label != "neither" and prob > sneeze_threshold) else ""
            print(f"  {label:12}: {prob:.4f}{marker}")
        
        # Final detection summary
        print(f"\n" + "=" * 40)
        print("DETECTION SUMMARY")
        print("=" * 40)
        
        all_detections = []
        
        # Add YAMNet detections
        for label, score in detected_yamnet:
            all_detections.append(f"{label.upper()} (confidence: {score:.3f})")
        
        # Add sneeze/sniff detections (can be multiple)
        for label, prob in zip(sneeze_labels, sneeze_probs):
            if label != "neither" and prob > sneeze_threshold:
                all_detections.append(f"{label.upper()} (confidence: {prob:.3f})")
        
        if all_detections:
            print("Detected events:")
            for detection in all_detections:
                print(f"  • {detection}")
        else:
            print("No significant events detected")
            if audio_level > 0.01:
                print("(Audio was detected but below confidence thresholds)")
            else:
                print("(Very low audio level - try speaking louder)")
        
        print(f"\nTest completed at {datetime.now().strftime('%H:%M:%S')}")
        
    except Exception as e:
        print(f"Error during inference: {e}")

def interactive_test():
    """Run multiple tests interactively"""
    
    print("Interactive Sleep Event Detector")
    print("================================")
    
    while True:
        print(f"\nOptions:")
        print("1. Run test")
        print("2. Exit")
        
        choice = input("\nChoose (1-2): ").strip()
        
        if choice == "1":
            test_tflite_model_simple()
        elif choice == "2":
            print("Goodbye!")
            break
        else:
            print("Invalid choice")
        
        print("\n" + "-" * 50)

if __name__ == "__main__":
    # Check if model file exists
    import os
    
    model_file = "_sleep_detector.tflite"
    
    if not os.path.exists(model_file):
        print(f"Model file '{model_file}' not found!")
        print("Please run the _tflite_model.py script first to create the model.")
        
        # Try alternative model names
        alternatives = ["sleep_event_detector.tflite", "minimal_detector.tflite"]
        for alt in alternatives:
            if os.path.exists(alt):
                print(f"Found alternative model: {alt}")
                model_file = alt
                break
        else:
            print("No TFLite model found.")
            exit(1)
    
    # Run interactive test
    interactive_test()
