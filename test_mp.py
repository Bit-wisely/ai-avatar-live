import mediapipe as mp
print(dir(mp))
try:
    print(mp.solutions)
except Exception as e:
    print(f"Error: {e}")
