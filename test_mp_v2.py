try:
    from mediapipe.python.solutions import face_mesh
    print("Success: from mediapipe.python.solutions import face_mesh")
except ImportError as e:
    print(f"Failed 1: {e}")

try:
    import mediapipe.python.solutions.face_mesh
    print("Success: import mediapipe.python.solutions.face_mesh")
except ImportError as e:
    print(f"Failed 2: {e}")

try:
    from mediapipe.tasks.python import vision
    print("Success: from mediapipe.tasks.python import vision")
except ImportError as e:
    print(f"Failed 3: {e}")
