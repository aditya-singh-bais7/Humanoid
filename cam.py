# find_camera_index.py
import cv2

def find_cameras():
    """Tests camera indices and displays the feed for each one found."""
    print("Searching for available cameras...")
    for i in range(5):  # Test indices 0 through 4
        # The 'cv2.CAP_DSHOW' flag can help on Windows
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"✅ Camera found at index: {i}")
            
            # Read a frame to show in the window
            ret, frame = cap.read()
            if ret:
                cv2.imshow(f"Camera Test (Index {i}) - Press any key to continue", frame)
                cv2.waitKey(0) # Wait for you to press a key

            cap.release()
    
    cv2.destroyAllWindows()
    print("\nSearch complete.")


if __name__ == "__main__":
    find_cameras()