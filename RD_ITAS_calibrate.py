"""
RD-ITAS Calibration Helper
══════════════════════════
Run this BEFORE RD_ITAS_main.py to get the pixel coordinates
you need to fill into Section 2 of the main script.

HOW TO USE:
  1. python RD_ITAS_calibrate.py
  2. A frame from your video opens.
  3. Click any point — its (x, y) pixel coordinate prints in the terminal.
  4. Use 'D' / 'A' keys to step forward/backward through frames.
  5. Copy the coordinates you need into RD_ITAS_main.py Section 2.
  6. Press Q to quit.

WHAT TO MARK:
  APPROACH_LINE  → click left end and right end of the stop line
  INTERSECTION_BOX → click all 4 corners of the intersection box
  EXIT_LINE_LEFT   → click top and bottom of left exit edge
  EXIT_LINE_STRAIGHT → click left and right of bottom exit edge
  EXIT_LINE_RIGHT  → click top and bottom of right exit edge
  LANE boundary    → click the lane dividing line (note x coordinate)
  OCCLUSION_ZONE   → click 4 corners of the tree/obstruction
  PED_WAITING_ZONE → click 4 corners of the pedestrian waiting area
  PED_CROSSING_LINE → click left and right end of the crossing line
  CALIBRATION BAR  → click two ends of a known real-world distance
"""

import cv2
import numpy as np

VIDEO_PATH =  r"LNDC TRAFFIC LIGHTS\DJI_20260730150924_0167_D.MP4" #"LANCERS_IN_WEST_EAST_FLOW.MOV"   the name of the video here - check the extension closely

clicked_points = []                           # setting up the ROI (Region Of Interest) through clicking points polygon zones
frame_ref      = [None]

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        n = len(clicked_points)
        print(f"  Point {n:3d}: ({x:4d}, {y:4d})")

        # Draw dot on frame
        cv2.circle(frame_ref[0], (x, y), 5, (0, 0, 255), -1)
        cv2.putText(frame_ref[0], f"{n}:({x},{y})",
                    (x+6, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        cv2.imshow("RD-ITAS Calibrate", frame_ref[0])

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {VIDEO_PATH}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    frame_idx    = [0]

    def read_frame(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frm = cap.read()
        return frm if ret else None

    WINDOW = "RD-ITAS Calibrate"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)
    cv2.setMouseCallback(WINDOW, mouse_callback)

    print("═"*60)
    print("  RD-ITAS Calibration Tool")
    print("  Left-click → print pixel coordinates")
    print("  D → next frame   A → previous frame")
    print("  F → jump forward 30 frames (1s at 30fps)")
    print("  C → clear all clicked points")
    print("  S → print summary of all points so far")
    print("  Q → quit")
    print("═"*60) 

    frm = read_frame(0)
    if frm is None:
        print("ERROR: Could not read first frame.")
        return

    frame_ref[0] = frm.copy()
    display = cv2.resize(frame_ref[0], (1280, 720))
    cv2.imshow(WINDOW, display)

    while True:
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('d'):
            frame_idx[0] = min(frame_idx[0] + 1, total_frames - 1)
            frm = read_frame(frame_idx[0])
            if frm is not None:
                frame_ref[0] = frm.copy()
                # Redraw existing points
                for i, (px, py) in enumerate(clicked_points):
                    cv2.circle(frame_ref[0], (px, py), 5, (0,0,255), -1)
                    cv2.putText(frame_ref[0], f"{i+1}:({px},{py})",
                                (px+6, py-6), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (0,0,255), 1)
                display = cv2.resize(frame_ref[0], (1280, 720))
                cv2.imshow(WINDOW, display)
                print(f"  Frame {frame_idx[0]} / {total_frames}  "
                      f"({frame_idx[0]/fps:.1f}s)")

        elif key == ord('a'):
            frame_idx[0] = max(frame_idx[0] - 1, 0)
            frm = read_frame(frame_idx[0])
            if frm is not None:
                frame_ref[0] = frm.copy()
                for i, (px, py) in enumerate(clicked_points):
                    cv2.circle(frame_ref[0], (px, py), 5, (0,0,255), -1)
                    cv2.putText(frame_ref[0], f"{i+1}:({px},{py})",
                                (px+6, py-6), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (0,0,255), 1)
                display = cv2.resize(frame_ref[0], (1280, 720))
                cv2.imshow(WINDOW, display)
                print(f"  Frame {frame_idx[0]} / {total_frames}  "
                      f"({frame_idx[0]/fps:.1f}s)")

        elif key == ord('f'):
            frame_idx[0] = min(frame_idx[0] + 30, total_frames - 1)
            frm = read_frame(frame_idx[0])
            if frm is not None:
                frame_ref[0] = frm.copy()
                display = cv2.resize(frame_ref[0], (1280, 720))
                cv2.imshow(WINDOW, display)
                print(f"  Jumped to frame {frame_idx[0]} ({frame_idx[0]/fps:.1f}s)")

        elif key == ord('c'):
            clicked_points.clear()
            frm = read_frame(frame_idx[0])
            if frm is not None:
                frame_ref[0] = frm.copy()
                display = cv2.resize(frame_ref[0], (1280, 720))
                cv2.imshow(WINDOW, display)
            print("  Points cleared.")

        elif key == ord('s'):
            print("\n  ── CLICKED POINTS SUMMARY ──")
            for i, (px, py) in enumerate(clicked_points):
                print(f"    {i+1:3d}: ({px:4d}, {py:4d})")
            if len(clicked_points) >= 2:
                # Pixel distance between last two points
                p1 = clicked_points[-2]
                p2 = clicked_points[-1]
                dpx = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                print(f"\n  Last 2 points pixel distance: {dpx:.1f} px")
                print(f"  If this = 3.5 m lane width:")
                print(f"  → PIXELS_PER_METRE = {dpx:.1f} / 3.5 = {dpx/3.5:.1f}")
            print()

    cap.release()
    cv2.destroyAllWindows()

    print("\n  ── FINAL COORDINATES ──")
    for i, (px, py) in enumerate(clicked_points):
        print(f"  {i+1:3d}: ({px}, {py})")
    print("\n  Copy these into Section 2 of RD_ITAS_main.py")

if __name__ == "__main__":
    main()
