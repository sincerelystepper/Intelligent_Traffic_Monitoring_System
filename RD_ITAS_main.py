"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   ROADS DIRECTORATE — KINGDOM OF LESOTHO                                     ║
║   Intersection Traffic Analysis System  (RD-ITAS v1.0)                       ║
║                                                                              ║
║   Author  : Kopano Maketekete (BSc Eng — Electrical Intern)                  ║
║   Ref     : RD/EED/STUDY/2026/07                                             ║
║   Target  : LNDC | QUEEN 2 | NEDBANK Intersections, Maseru                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

PIPELINE OVERVIEW
─────────────────
INPUT  : Overhead/elevated video from cherry-picker or cantilever camera.

VEHICLE PIPELINE:
  1. Detect & classify  —  car | taxi | truck | bus | motorcycle
  2. Track with ByteTrack (persistent IDs, handles occlusion)
  3. APPROACH LINE  — count every vehicle before stop line (primary count)
  4. Track through INTERSECTION BOX
  5. EXIT LINES     — classify turn (Left / Straight / Right) + count again
  6. REDUNDANCY CHECK: approach_total == left + straight + right
  7. Output: volume matrix [class × movement] → vph + PCU/hr per lane

PEDESTRIAN PIPELINE:
  1. Detect all pedestrians (COCO class 0)
  2. Filter to CROSSING INTENT ZONE  (those approaching the kerb)
  3. Track across PEDESTRIAN CROSSING LINE
  4. Measure walking speed  (m/s)  from calibrated pixel displacement
  5. Measure crossing time  (s)  per individual
  6. Output: flow (ped/hr), mean speed, mean crossing time, min green needed

OUTPUTS:
  ├── Live annotated display window  (resizable, pauseable)
  ├── Annotated output video  (.mp4)
  ├── Event log CSV  (every vehicle + pedestrian crossing event)
  └── Terminal summary  (Webster vᵢ, sᵢ, yᵢ inputs ready)

HOW TO USE:
  1. Fill in Section 1 (file paths, recording duration, intersection name)
  2. Run the calibration helper first:  python RD_ITAS_calibrate.py
  3. Paste calibration coordinates into Section 2
  4. Run:  python RD_ITAS_main.py
  5. Press P to pause/resume, Q to quit early
"""

import cv2
import numpy as np
import csv
import os
from ultralytics import YOLO
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# Fill these in before each session. Everything else adapts automatically.
# ══════════════════════════════════════════════════════════════════════════════

# ── Files ──────────────────────────────────────────────────────────────────
VIDEO_PATH   = r"LNDC TRAFFIC LIGHTS\DJI_20260730150924_0167_D.MP4"     # the video filename
MODEL_PATH   = "yolo11n.pt"                        # or yolo11s.pt for accuracy
SESSION_TAG  = "LNDC_AM_PEAK"                      # used in output filenames

_ts          = datetime.now().strftime("%Y%m%d_%H%M")  # today's date, right at the point of running of the footage
#OUTPUT_VIDEO = f"RD_ITAS_{SESSION_TAG}_{_ts}.mp4"      # video output naming convention with f-string format, commented out for now for space limitation (its saves on every run)
OUTPUT_CSV   = f"RD_ITAS_{SESSION_TAG}_{_ts}.csv"      # CSV output format similar to the video format with necessary parameters

# ── Intersection Identity ───────────────────────────────────────────────────
INTERSECTION_NAME = "LNDC — Kingsway Approach (East → West)"
INTERSECTION_REF  = "LRD004"
APPROACH_DIR      = "Eastbound vehicles approaching stop line"

# ── Recording Duration ──────────────────────────────────────────────────────
# Set to the actual length of your recording in hours.
# 15 min = 0.25,  30 min = 0.50,  1 hr = 1.0
RECORDING_DURATION_HRS = 0.25

# ── Pixel-to-Metre Calibration ─────────────────────────────────────────────
# Step 1: From your overhead video, identify a known real-world distance
#         visible in the frame  (e.g. a lane width = 3.5 m, or a kerb span).
# Step 2: Run RD_ITAS_calibrate.py — click two points at each end of that
#         distance and note the pixel count it gives you.
# Step 3: Fill in below.
KNOWN_REAL_DISTANCE_M  = 3.5    # metres  — e.g. one lane width
KNOWN_PIXEL_DISTANCE   = 120    # pixels  — from calibration tool
PIXELS_PER_METRE       = KNOWN_PIXEL_DISTANCE / KNOWN_REAL_DISTANCE_M  # ratio of PIXEL to METRES....we should improve this by using calibration matrices and employing CV perspective matrix from that real world coordinates, to use in speeds estimations per vehicle 

# lets work on the accuracy of this conversion. Possibly set up the video
# such that it does not warp the lines of the video, like the stop line of
# vehicles, which will allow us to make that accurate conversion. 

# ── Signal Phase Clock (from ST950 commissioning file LRD004) ──────────────
# Used to tag each crossing event with the current signal phase.
CYCLE_LENGTH_S         = 70     # seconds — AM Peak plan
# Elapsed seconds INTO the cycle when you pressed record.
# If unknown, run a few seconds of video and note what phase you see.
VIDEO_START_OFFSET_S   = 0

# Phase table from ST950 commissioning file — (name, cycle_start_s, green_end_s)
PHASE_SCHEDULE = [
    ("Kingsway EB/WB Green",  0,  23), # name, cycle_start_s, green_end_s
    ("Intergreen A→C",       23,  28), # clearance
    ("Pioneer NB/SB Green",  28,  48),
    ("Intergreen C→E",       48,  53),
    ("Peds + RT Filter",     53,  60),
    ("Intergreen to AR",     60,  65),
    ("All Red",              65,  70),
]

# ── Detection Parameters ────────────────────────────────────────────────────
CONF_THRESHOLD  = 0.35
IOU_THRESHOLD   = 0.50

# ── Vehicle Classes (COCO) + PCU Values ────────────────────────────────────
# Color format: BGR
VEHICLE_CLASSES = {
    2: {"name": "car",        "pcu": 1.0, "color": (0,   220,   0)},
    3: {"name": "motorcycle", "pcu": 0.5, "color": (0,   220, 220)},
    5: {"name": "bus",        "pcu": 3.0, "color": (220,   0,   0)},
    7: {"name": "truck",      "pcu": 2.5, "color": (0,   128, 255)},
}
# Taxi detection: YOLO calls Quantum taxis "car" or "truck".
# If bounding box area exceeds threshold → reclassify as taxi.
TAXI_AREA_THRESHOLD_PX = 7000   # pixels² — tune from your overhead footage --- we will need to test this as per live footage, to ensure consistency
TAXI_INFO = {"name": "taxi", "pcu": 1.5, "color": (0, 200, 255)}

PEDESTRIAN_CLASS = 0
PED_COLOR        = (200,  50, 200)
PED_WAIT_COLOR   = (255,   0, 255)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ZONE DEFINITIONS
# Calibrate these coordinates using RD_ITAS_calibrate.py
# ══════════════════════════════════════════════════════════════════════════════
#
# OVERHEAD ZONE DIAGRAM (one approach):
#
#   ↑ UPSTREAM (vehicles coming toward camera)
#   │
#   │  [LANE 2]      [LANE 1]
#   │     │               │
#   ╠═══════════════════════════╣  ← APPROACH_LINE   (count all entering)
#   │     ↓               ↓
#   │ ┌─────────────────────────┐
#   │ │   INTERSECTION BOX      │
#   │ │   (track turns here)    │
#   │ └─────────────────────────┘
#   │    ↙           ↓         ↘
#   │ EXIT_LEFT  EXIT_STRAIGHT  EXIT_RIGHT
#   │
#   │  ┌──────────┐      ══════════════  ← PED_CROSSING_LINE
#   │  │PED WAIT  │
#   │  │  ZONE    │
#   │  └──────────┘
#   ↓ DOWNSTREAM

# ── Approach Line ──────────────────────────────────────────────────────────
# Draw ACROSS both incoming lanes, just before the stop line ( lets make it a little further from the stop line for lane allocation).
# Format: [(x1,y1), (x2,y2)]
APPROACH_LINE = [(1800, 978), (2118, 972)] # a little upper coordinate line to allocate vehicle approach to lanes timely (allows heavy processing on further frames)

# ── Lane x-ranges ──────────────────────────────────────────────────────────
# Vehicles counted at approach are assigned to a lane by their centroid x.
# We must dimarcate this LANE_DEFINITION from the APPROACH_LINE since our lane  definitions must map to the physical width on that specific line.
CENTER_X = 1953

LANE_DEFINITIONS = {
    "Lane 1": (599, CENTER_X),   # (x_min, x_max) for lane 1, any vehicle whose center point cx is between pixel 607 and pixel 2016 is declared to be in Lane 1
    "Lane 2": (CENTER_X, 2016),   # (x_min, x_max) for lane 2, any vehicle whosen center point cx is between pixel 2016 and pixel 2265 is declared to be in Lane 2
}

# ── Intersection Box ───────────────────────────────────────────────────────
# Polygon covering the full intersection box visible from overhead.
# Vehicles are tracked inside here before exit classification.
INTERSECTION_BOX = np.array([
    [1320, 1203],          # [540, 530]  # top-left
    [2544, 1212],           # [860, 530]  # top-right
    [2931, 1863],         # [860, 710]  # bottom-right
    [990, 1866],         # [540, 710]  # bottom-left
], dtype=np.int32)

# ── Exit Lines (one per turn direction) ────────────────────────────────────
# Place LEFT  line on the left edge of the intersection box
# Place STRAIGHT line on the bottom edge of the intersection box
# Place RIGHT line on the right edge of the intersection box
EXIT_LINES = {
    "Left":     [(2673, 1233), (2862, 1365)],   # [(540, 580), (540, 700)],
    "Straight": [(1920, 1833), (2934, 1872)],   # [(600, 710), (800, 710)],
    "Right":    [(969, 1413), (795, 1572)],   # [(860, 580), (860, 700)]
}
EXIT_COLORS = {
    "Left": (255, 220, 0), "Straight": (0, 255, 120), "Right": (255, 140, 0)
}

# ── Occlusion Zones ────────────────────────────────────────────────────────
# Add one polygon per obstruction (tree, pole, structure).
# Counting events are suppressed inside these zones.
# Tracking continues (ByteTrack reacquires on exit).
OCCLUSION_ZONES = [
    # Mokorotlong to intersection approach
    np.array([
        [990, 1860], [1917, 1806], [1953, 2145], [795, 2148]
    ], dtype=np.int32),
    # Add more zones here if needed:
    # np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], dtype=np.int32),

    # ECOL to Intersection approach
    np.array([
        [5, 413], [1089, 1284], [942, 1410], [30, 1395]
    ], dtype=np.int32),

    # Intersection to after pedestrian waiting polygon region of interest Pioneer Right Turn approach
        np.array([
        [1296, 1191], [1353, 987], [1800, 927], [1827, 1191]
    ], dtype=np.int32),
    
    # after ROI pedestrian polygon zone to Nedbank approach
        np.array([
        [1800, 918], [1203, 936], [1479, 477], [1743, 555]
    ], dtype=np.int32),

    # Pioneer to Right Turn Approach 
            np.array([
        [2937, 1842], [2628, 1407], [3654, 1440], [3651, 1653]
    ], dtype=np.int32),

]

# ── Pedestrian Zones ───────────────────────────────────────────────────────
# Waiting zone: polygon at the pedestrian kerb where people wait to cross.
PED_WAITING_ZONE = np.array([
    [1260, 1041],     # [200, 500]  # top-left
    [1335, 1032],     # [380, 500]  # top-right
    [1257, 1230],     # [380, 590]  # bottom-right
    [1125, 1230],     # [200, 590]  # bottom-left
], dtype=np.int32)

# Crossing line: line that a pedestrian crosses to complete their crossing.
PED_CROSSING_LINE = [(1353, 1032), (1275, 1218)]    # [(200, 590), (550, 590)]

# ── Outgoing Lane Exclusion ────────────────────────────────────────────────
# Polygon covering outgoing lanes — vehicles here are already past the
# intersection and must NOT be counted as new approach vehicles.
OUTGOING_LANE_ZONE = np.array([
    [804, 2142],     # [0,   600],
    [1935, 2142],     # [540, 600],
    [1737, 546],     # [540, 720],
    [1464, 636],     # [0,   720],
], dtype=np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VehicleRecord:
    """Full lifecycle record for one tracked vehicle."""
    track_id:         int
    class_id:         int
    class_name:       str
    pcu:              float                             # PCU value initialization 
    color:            tuple
    lane:             Optional[str]   = None
    approach_time:    Optional[float] = None
    turn_movement:    Optional[str]   = None
    exit_time:        Optional[float] = None
    in_intersection:  bool            = False
    counted_approach: bool            = False
    counted_exit:     bool            = False
    centroid_history: list            = field(default_factory=list) # centroid history for the vehicle, to track its movement through the intersection

@dataclass
class PedestrianRecord:
    """Full lifecycle record for one tracked pedestrian."""
    track_id:          int
    first_seen:        float
    in_waiting_zone:   bool            = False
    intent_confirmed:  bool            = False   # entered waiting zone → intends to cross
    crossing_start:    Optional[float] = None
    crossing_end:      Optional[float] = None
    crossing_time_s:   Optional[float] = None
    walking_speed_ms:  Optional[float] = None
    counted:           bool            = False
    centroid_history:  list            = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def in_any_occlusion(x, y) -> bool:
    """True if (x,y) is inside any defined occlusion zone."""
    for zone in OCCLUSION_ZONES:
        if cv2.pointPolygonTest(zone, (float(x), float(y)), False) >= 0:
            return True
    return False

def in_polygon(x, y, poly) -> bool:
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

def line_crossed(p_prev, p_curr, ls, le) -> bool:
    """Cross-product line segment intersection test."""
    def xp(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    d1 = xp(ls, le, p_prev); d2 = xp(ls, le, p_curr)
    d3 = xp(p_prev, p_curr, ls); d4 = xp(p_prev, p_curr, le)
    return (((d1>0 and d2<0) or (d1<0 and d2>0)) and
            ((d3>0 and d4<0) or (d3<0 and d4>0)))

def assign_lane(cx: int) -> str:
    for name, (lo, hi) in LANE_DEFINITIONS.items():
        if lo <= cx <= hi:
            return name
    return "Unassigned"

def classify_vehicle(cls_id: int, box) -> dict:
    """Classify vehicle type, catching taxis by bounding box area."""
    x1, y1, x2, y2 = box
    area = (x2-x1) * (y2-y1)
    if cls_id == 2 and area > TAXI_AREA_THRESHOLD_PX:
        return {**TAXI_INFO, "class_id": 98}   # 98 = internal taxi code
    return {**VEHICLE_CLASSES.get(cls_id, VEHICLE_CLASSES[2]),
            "class_id": cls_id}

def current_phase(ts: float) -> str:
    cyc_pos = (ts + VIDEO_START_OFFSET_S) % CYCLE_LENGTH_S
    for name, start, end in PHASE_SCHEDULE:
        if start <= cyc_pos < end:
            return name
    return "Unknown"

def pixel_dist(p1, p2) -> float:
    return float(np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2))

def pixels_to_metres(px: float) -> float:
    return px / PIXELS_PER_METRE


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — GLOBAL STATE
# ══════════════════════════════════════════════════════════════════════════════

vehicles:    dict[int, VehicleRecord]    = {}
pedestrians: dict[int, PedestrianRecord] = {}

# Approach counts  [class_name][lane] = count
approach_counts = defaultdict(lambda: defaultdict(int))
approach_pcu    = defaultdict(lambda: defaultdict(float))

# Turn movement counts  [class_name][movement] = count
turn_counts = defaultdict(lambda: defaultdict(int))
turn_pcu    = defaultdict(lambda: defaultdict(float))

# Pedestrian accumulators
ped_count          = 0
ped_crossing_times = []
ped_speeds         = []

# Headway buffer for saturation flow
headways          = []
last_approach_ts  = None

# CSV event buffer
csv_events = []

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — VEHICLE PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def process_vehicle(tid: int, cls_id: int, box, ts: float): # we take the track id, class id, bounding box and timestamp as input to the function
    global last_approach_ts

    x1, y1, x2, y2 = box                        # extract the bounding box coordinates
    cx, cy = int((x1+x2)/2), int((y1+y2)/2)     # compute the centroid of the bounding box

    vinfo = classify_vehicle(cls_id, box)
    cls_name = vinfo["name"]
    pcu      = vinfo["pcu"]
    color    = vinfo["color"]

    in_occ  = in_any_occlusion(cx, cy)
    in_box  = in_polygon(cx, cy, INTERSECTION_BOX)      # lets divert and think of the best way to process this logic for only approaching vehicles while ignoring the outgoing traffic and other traffics other than the approaching
    in_out  = in_polygon(cx, cy, OUTGOING_LANE_ZONE)

    # Create record on first sight
    if tid not in vehicles:
        vehicles[tid] = VehicleRecord(
            track_id=tid, class_id=cls_id,
            class_name=cls_name, pcu=pcu, color=color
        )

    rec = vehicles[tid]
    rec.centroid_history.append((cx, cy, ts))
    if in_box:
        rec.in_intersection = True

    # Need ≥ 2 history points to check crossings
    if len(rec.centroid_history) < 2 or in_occ or in_out:
        return color, in_occ

    p_prev = rec.centroid_history[-2][:2]
    p_curr = (cx, cy)

    # ── APPROACH LINE crossing ──────────────────────────────────────────────
    if not rec.counted_approach:
        if line_crossed(p_prev, p_curr, APPROACH_LINE[0], APPROACH_LINE[1]):
            lane = assign_lane(cx)
            rec.counted_approach = True
            rec.approach_time    = ts
            rec.lane             = lane

            approach_counts[cls_name][lane] += 1
            approach_pcu[cls_name][lane]    += pcu

            if last_approach_ts is not None:
                hw = ts - last_approach_ts
                if 1.0 < hw < 9.0:
                    headways.append(hw)
            last_approach_ts = ts

            phase = current_phase(ts)
            print(f"  ▶ APPROACH  {ts:7.1f}s | ID {tid:4d} | "
                  f"{cls_name:<9} | {lane} | PCU {pcu:.1f} | {phase}")

            csv_events.append({                                         # csv events log
                "event":    "approach",
                "time_s":   f"{ts:.2f}",
                "track_id": tid,
                "class":    cls_name,
                "pcu":      pcu,
                "lane":     lane,
                "movement": "",
                "phase":    phase,
                "speed_ms": "",
            })

    # ── EXIT LINE crossings (turn classification) ───────────────────────────
    if rec.in_intersection and rec.counted_approach and not rec.counted_exit:
        for movement, (ls, le) in EXIT_LINES.items():
            if line_crossed(p_prev, p_curr, ls, le):
                rec.counted_exit  = True
                rec.turn_movement = movement
                rec.exit_time     = ts

                turn_counts[cls_name][movement] += 1
                turn_pcu[cls_name][movement]    += pcu

                color_m = EXIT_COLORS[movement]
                print(f"  ↳ EXIT      {ts:7.1f}s | ID {tid:4d} | "
                      f"{cls_name:<9} | {movement:<9} | PCU {pcu:.1f}")

                csv_events.append({
                    "event":    "exit_turn",
                    "time_s":   f"{ts:.2f}",
                    "track_id": tid,
                    "class":    cls_name,
                    "pcu":      pcu,
                    "lane":     rec.lane or "",
                    "movement": movement,
                    "phase":    "",
                    "speed_ms": "",
                })
                break

    return color, in_occ


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — PEDESTRIAN PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def process_pedestrian(tid: int, box, ts: float):
    global ped_count

    x1, y1, x2, y2 = box
    cx, cy = int((x1+x2)/2), int((y1+y2)/2)

    if tid not in pedestrians:
        pedestrians[tid] = PedestrianRecord(track_id=tid, first_seen=ts)

    rec = pedestrians[tid]
    rec.centroid_history.append((cx, cy, ts))

    in_wait = in_polygon(cx, cy, PED_WAITING_ZONE)

    # Confirm crossing intent: pedestrian entered waiting zone
    if in_wait and not rec.intent_confirmed:
        rec.in_waiting_zone  = True
        rec.intent_confirmed = True
        rec.crossing_start   = ts
        print(f"  🚶 PED WAIT  {ts:7.1f}s | ID {tid:4d} | "
              f"Intent confirmed — entered waiting zone")

    # Walking speed from recent centroid displacement
    if len(rec.centroid_history) >= 3:
        ph = rec.centroid_history
        # Use 3-frame window for smoother speed
        dt  = ph[-1][2] - ph[-3][2]
        dpx = pixel_dist(ph[-1][:2], ph[-3][:2])
        if dt > 0:
            spd = pixels_to_metres(dpx / dt)
            if 0.15 < spd < 3.5:   # valid pedestrian speed range
                ped_speeds.append(spd)

    # Crossing line event
    if rec.intent_confirmed and not rec.counted and len(rec.centroid_history) >= 2:
        p_prev = rec.centroid_history[-2][:2]
        p_curr = (cx, cy)
        if line_crossed(p_prev, p_curr,
                        PED_CROSSING_LINE[0], PED_CROSSING_LINE[1]):
            rec.counted    = True
            rec.crossing_end = ts
            ped_count += 1

            if rec.crossing_start:
                ct = ts - rec.crossing_start
                rec.crossing_time_s = ct
                ped_crossing_times.append(ct)

            spd = np.mean(ped_speeds[-15:]) if len(ped_speeds) >= 3 else None
            rec.walking_speed_ms = spd
            phase = current_phase(ts)

            spd_str = f"{spd:.2f} m/s" if spd else "N/A"
            ct_str  = f"{rec.crossing_time_s:.1f}s" if rec.crossing_time_s else "N/A"
            print(f"  -> PED CROSS {ts:7.1f}s | ID {tid:4d} | "
                  f"#{ped_count} | Speed:{spd_str} | Time:{ct_str} | {phase}")

            csv_events.append({
                "event":    "pedestrian_cross",
                "time_s":   f"{ts:.2f}",
                "track_id": tid,
                "class":    "pedestrian",
                "pcu":      0,
                "lane":     "",
                "movement": "crossing",
                "phase":    phase,
                "speed_ms": f"{spd:.3f}" if spd else "",
            })

    return in_wait


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — HUD OVERLAY
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — HUD OVERLAY (SCALED FOR HIGH-RES / 4K FOOTAGE)
# ══════════════════════════════════════════════════════════════════════════════

def draw_zones(frame):
    """Draw all analysis zones with semi-transparent fills and bold labels."""
    ov = frame.copy()

    cv2.fillPoly(ov, [INTERSECTION_BOX],  (180,  80,   0))
    cv2.fillPoly(ov, [PED_WAITING_ZONE],  (160,   0, 160))
    for zone in OCCLUSION_ZONES:
        cv2.fillPoly(ov, [zone], (0, 0, 180))

    cv2.addWeighted(ov, 0.22, frame, 0.78, 0, frame)

    # Thickened Zone Borders
    cv2.polylines(frame, [INTERSECTION_BOX], True, (220, 120, 0), 3)
    cv2.polylines(frame, [PED_WAITING_ZONE], True, (220,  50, 220), 3)
    for zone in OCCLUSION_ZONES:
        cv2.polylines(frame, [zone], True, (50, 50, 220), 3)

    # Scaled Zone Labels
    cv2.putText(frame, "INTERSECTION",
                (INTERSECTION_BOX[0][0]+4, INTERSECTION_BOX[0][1]+30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.90, (220,120,0), 2, cv2.LINE_AA)
    cv2.putText(frame, "PED WAIT",
                (PED_WAITING_ZONE[0][0]+2, PED_WAITING_ZONE[0][1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.80, (220,50,220), 2, cv2.LINE_AA)
    for zone in OCCLUSION_ZONES:
        cv2.putText(frame, "OCCL",
                    (zone[0][0]+2, zone[0][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80,80,255), 2, cv2.LINE_AA)

    # Approach line
    cv2.line(frame, APPROACH_LINE[0], APPROACH_LINE[1], (0,0,255), 5)
    cv2.putText(frame, "APPROACH LINE",
                (APPROACH_LINE[0][0], APPROACH_LINE[0][1]-15),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,80,255), 3, cv2.LINE_AA)

    # Exit lines
    for mv, (ls, le) in EXIT_LINES.items():
        col = EXIT_COLORS[mv]
        cv2.line(frame, ls, le, col, 4)
        mid = ((ls[0]+le[0])//2, (ls[1]+le[1])//2)
        cv2.putText(frame, mv, (mid[0]-25, mid[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2, cv2.LINE_AA)

    # Pedestrian crossing line
    cv2.line(frame, PED_CROSSING_LINE[0], PED_CROSSING_LINE[1], (200,50,200), 4)
    cv2.putText(frame, "PED LINE",
                (PED_CROSSING_LINE[0][0], PED_CROSSING_LINE[0][1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200,50,200), 2, cv2.LINE_AA)


def draw_hud(frame, ts: float, approach_total: int):
    """Draw live stats panel — expanded and scaled for 4K visibility."""
    phase = current_phase(ts)

    # Expanded Background Panel (Width: 650px, Height: 370px)
    cv2.rectangle(frame, (0, 0), (650, 370), (15, 15, 15), -1)
    cv2.rectangle(frame, (0, 0), (650, 370), (70, 70, 70),  2)

    # Signal Phase Bar (Height: 10px)
    phase_color = (0, 200, 0) if "Green" in phase else \
                  (0, 200, 220) if "Red" in phase else (0, 180, 255)
    cv2.rectangle(frame, (0, 0), (650, 10), phase_color, -1)

    L_tot = sum(turn_counts[c].get("Left",0)     for c in turn_counts)
    S_tot = sum(turn_counts[c].get("Straight",0) for c in turn_counts)
    R_tot = sum(turn_counts[c].get("Right",0)    for c in turn_counts)
    l1pcu = sum(approach_pcu[c].get("Lane 1",0)  for c in approach_pcu)
    l2pcu = sum(approach_pcu[c].get("Lane 2",0)  for c in approach_pcu)

    # Text structure: (string, (x, y), fontScale, color, thickness)
    lines = [
        (f"RD-ITAS | {INTERSECTION_REF}",                  (15, 42),  0.85, (180,180,180), 2),
        (f"{INTERSECTION_NAME[:38]}",                      (15, 78),  0.75, (140,140,140), 2),
        (f"Phase: {phase[:32]}",                           (15, 120), 0.90, phase_color,     2),
        (f"Time: {ts:6.1f}s",                              (15, 158), 0.85, (160,160,160), 2),
        (f"APPROACH : {approach_total:4d} vehicles",       (15, 205), 1.10, (0, 240, 100),   3),
        (f"L:{L_tot:<4d}  STRT:{S_tot:<4d}  R:{R_tot}",    (15, 250), 1.00, (255, 210, 0),   2),
        (f"Peds crossed : {ped_count}",                    (15, 292), 1.00, (210,  80, 210),  2),
        (f"L1 PCU:{l1pcu:.1f}   L2 PCU:{l2pcu:.1f}",     (15, 335), 0.90, (80, 210, 255),  2),
    ]

    for text, pos, sc, col, th in lines:
        # Outer dark stroke for maximum readability over light video background
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th + 2, cv2.LINE_AA)
        # Inner colored text
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — RESULTS SUMMARY + CSV EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_summary():
    D = RECORDING_DURATION_HRS
    print("\n" + "═"*72)
    print(f"  ROADS DIRECTORATE — INTERSECTION TRAFFIC ANALYSIS SYSTEM")
    print(f"  {INTERSECTION_NAME}")
    print(f"  Ref: {INTERSECTION_REF}  |  Recording: {D*60:.0f} min")
    print("═"*72)

    # ── Approach Volume Matrix ─────────────────────────────────────────────
    print("\n  APPROACH VOLUME  (vehicles counted before stop line)")
    print(f"  {'Class':<12}  {'Lane 1':>8}  {'Lane 2':>8}  {'Total':>8}  {'vph':>8}  {'PCU/hr':>8}")
    print("  " + "─"*62)

    all_classes = sorted(set(approach_counts.keys()) | set(turn_counts.keys()))
    total_veh = 0
    for cls in all_classes:
        l1 = approach_counts[cls].get("Lane 1", 0)
        l2 = approach_counts[cls].get("Lane 2", 0)
        t  = l1 + l2
        p1 = approach_pcu[cls].get("Lane 1", 0.0)
        p2 = approach_pcu[cls].get("Lane 2", 0.0)
        total_pcu = (p1+p2)/D
        total_veh += t
        print(f"  {cls:<12}  {l1:>8}  {l2:>8}  {t:>8}  {t/D:>8.0f}  {total_pcu:>8.0f}")

    l1pcu_hr = sum(approach_pcu[c].get("Lane 1",0) for c in approach_pcu) / D
    l2pcu_hr = sum(approach_pcu[c].get("Lane 2",0) for c in approach_pcu) / D
    print(f"\n  Total vehicles : {total_veh}  →  {total_veh/D:.0f} vph")
    print(f"  Lane 1 PCU/hr : {l1pcu_hr:.0f}")
    print(f"  Lane 2 PCU/hr : {l2pcu_hr:.0f}")

    # ── Turning Movement Count Matrix ──────────────────────────────────────
    print("\n  TURNING MOVEMENT COUNT MATRIX  (exit classification)")
    print(f"  {'Class':<12}  {'Left':>8}  {'Straight':>10}  {'Right':>8}  {'Total':>8}")
    print("  " + "─"*54)
    exit_total = 0
    for cls in all_classes:
        L = turn_counts[cls].get("Left", 0)
        S = turn_counts[cls].get("Straight", 0)
        R = turn_counts[cls].get("Right", 0)
        T = L + S + R
        exit_total += T
        print(f"  {cls:<12}  {L:>8}  {S:>10}  {R:>8}  {T:>8}")

    print(f"\n  REDUNDANCY: Approach={total_veh}  Exits={exit_total}  "
          f"Δ={total_veh-exit_total}  "
          f"({'✓ consistent' if abs(total_veh-exit_total)<=2 else '⚠ check occlusion'})")

    # ── Pedestrian Analysis ────────────────────────────────────────────────
    print("\n  PEDESTRIAN CROSSING ANALYSIS")
    print(f"  Total crossings     : {ped_count}")
    if ped_crossing_times:
        mean_ct = np.mean(ped_crossing_times)
        max_ct  = np.max(ped_crossing_times)
        print(f"  Mean crossing time  : {mean_ct:.1f} s")
        print(f"  Max crossing time   : {max_ct:.1f} s")
        print(f"  Ped flow rate       : {ped_count/D:.0f} ped/hr")
    if ped_speeds:
        ms = np.mean(ped_speeds)
        print(f"  Mean walking speed  : {ms:.2f} m/s")
        print(f"  HCM standard (able) : 1.20 m/s")
        print(f"  HCM access design   : 0.90 m/s (elderly / visually impaired)")

        # Minimum pedestrian green time
        # From geometric survey: crossing_distance_m (fill from §2 of working framework)
        crossing_distance_m = 15.0   # PLACEHOLDER — fill from field measurement
        min_green = crossing_distance_m / 0.90   # use access design speed
        print(f"\n  MINIMUM PEDESTRIAN GREEN TIME (access design):")
        print(f"  Crossing distance     : {crossing_distance_m:.1f} m  "
              f"(update from geometric survey)")
        print(f"  At 0.90 m/s design    : {min_green:.1f} s minimum solid green")
        print(f"  Current ST950 Phase E : 7.0 s  (AM/PM peak)")
        print(f"  Current ST950 Phase F : 7.0 s  (AM/PM peak)")
        deficiency = min_green - 7.0
        if deficiency > 0:
            print(f"  ⚠ DEFICIENCY          : {deficiency:.1f} s underprovisioned")
        else:
            print(f"  ✓ Current provision adequate")

    # ── Saturation Flow ────────────────────────────────────────────────────
    print("\n  SATURATION FLOW ESTIMATE  (from approach headways)")
    if len(headways) >= 5:
        mh = np.mean(headways)
        sf = 3600 / mh
        print(f"  Observations   : {len(headways)}")
        print(f"  Mean headway   : {mh:.2f} s")
        print(f"  Observed sᵢ    : {sf:.0f} PCU/hr/lane")
        print(f"  HCM default    : 1900 PCU/hr/lane")
    else:
        sf = 1900
        print(f"  Insufficient observations — using HCM default: 1900 PCU/hr/lane")

    # ── Webster Inputs ─────────────────────────────────────────────────────
    vi = max(l1pcu_hr, l2pcu_hr)
    yi = vi / sf
    print("\n  WEBSTER EQUATION INPUTS  (this approach)")
    print(f"  vᵢ  (critical lane PCU/hr)  : {vi:.0f}")
    print(f"  sᵢ  (saturation flow)       : {sf:.0f}")
    print(f"  yᵢ  = vᵢ / sᵢ              : {yi:.4f}")
    print(f"\n  Enter yᵢ into Module 4 of the working framework (§4.2)")
    print("═"*72)


def save_csv():
    if not csv_events:
        print("  No events recorded — CSV not saved.")
        return
    with open(OUTPUT_CSV, "w", encoding='utf-8', newline="") as f:  # added the utf-8 encoding for the saved CSV for rendering harmony
        writer = csv.DictWriter(f, fieldnames=[
            "event","time_s","track_id","class",
            "pcu","lane","movement","phase","speed_ms"
        ])
        writer.writeheader()
        writer.writerows(csv_events)
    print(f"\n   CSV saved -> {OUTPUT_CSV}")
#    print(f"   Video saved -> {OUTPUT_VIDEO}")
# the above also saved for the purpose of space and RAM as well

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9.1 — CALIBRATION: PIXELS TO METERS
# ══════════════════════════════════════════════════════════════════════════════
# Define 4 points on the video frame that form a rectangle in the real world.
# e.g, lets say: A stretch of a known lane of 3.5m wide and 15m long.
# [Top-Left, Top-Right, Bottom-Right, Bottom-Left] in PIXELS



# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run():
    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Video file not found — {VIDEO_PATH}")
        print("Make sure the .MOV file is in the same folder as this script.")
        return

    model = YOLO(MODEL_PATH)
    cap   = cv2.VideoCapture(VIDEO_PATH)                # capture object to read the frames picture by picture

    W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps  = cap.get(cv2.CAP_PROP_FPS)
    nfr  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("═"*72)
    print(f"  RD-ITAS v1.0  |  {INTERSECTION_NAME}")
    print(f"  Video: {W}×{H} @ {fps:.1f} fps  |  {nfr} frames  |  "
          f"~{nfr/fps/60:.1f} min")
    print(f"  Controls:  P = pause/resume    Q = quit early")
    print("═"*72)

# COMMENTED OUT FOR NOW TO ENSURE WE SAVE SPACE, AND RAM. WE WRITE THE VIDEO FRAMES HERE
#    out = cv2.VideoWriter(
#        OUTPUT_VIDEO,
#        cv2.VideoWriter_fourcc(*"mp4v"),
#        fps, (W, H)
#    )

    WINDOW = f"RD-ITAS | {INTERSECTION_REF}"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)

    frame_num      = 0
    approach_total = 0
    paused         = False

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            ts = frame_num / fps

            # ── Inference ────────────────────────────────────────────────
            # we must optimize the inference call (biggest memory hog), to simulate the edge node processing behavior
            # we must therefore pass memory-limiting arguments to YOLO to force it to downscale internally, while preserving 
            # original coordinate scale for the zones, while using half-precision
            results = model.track(
                frame,
                persist=True,
                classes=[PEDESTRIAN_CLASS, 2, 3, 5, 7],
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                tracker="bytetrack.yaml",
                verbose=False,
            )

            # ── Draw zones ────────────────────────────────────────────────
            draw_zones(frame)

            # ── Process detections ────────────────────────────────────────
            if results[0].boxes.id is not None:
                boxes   = results[0].boxes.xyxy.cpu().numpy()
                ids     = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)

                for box, tid, cls in zip(boxes, ids, classes):
                    x1, y1, x2, y2 = box
                    cx = int((x1+x2)/2)
                    cy = int((y1+y2)/2)

                    if cls == PEDESTRIAN_CLASS:
                        in_wait = process_pedestrian(tid, box, ts)
                        col = PED_WAIT_COLOR if in_wait else PED_COLOR
                        cv2.rectangle(frame,
                                      (int(x1),int(y1)),(int(x2),int(y2)), col, 1)
                        cv2.putText(frame, f"P{tid}",
                                    (int(x1), int(y1)-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)
                    else:
                        col, in_occ = process_vehicle(tid, cls, box, ts)
                        if in_occ:
                            col = (80, 80, 255)

                        rec = vehicles.get(tid)
                        lbl = rec.class_name if rec else "?"
                        mv  = f"→{rec.turn_movement}" if (rec and rec.turn_movement) else ""

                        thickness = 1 if in_occ else 2
                        cv2.rectangle(frame,
                                      (int(x1),int(y1)),(int(x2),int(y2)), col, thickness)
                        cv2.putText(frame, f"{lbl} {mv} [{tid}]",
                                    (int(x1), int(y1)-7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 2)
                        cv2.circle(frame, (cx, cy), 3, col, -1)

            # ── HUD ───────────────────────────────────────────────────────
            approach_total = sum(
                sum(approach_counts[c][l] for l in approach_counts[c])
                for c in approach_counts
            )
            draw_hud(frame, ts, approach_total)

#            out.write(frame)
# COMMENTED OUT THE ABOVE TO SAVE ON SAVING SPACE AND RAM 

        # ── Display ───────────────────────────────────────────────────────
        display = cv2.resize(frame, (1280, 720))
        if paused:
            cv2.putText(display, "  ▌▌ PAUSED — press P to resume",
                        (350, 360), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 200, 255), 2)
        cv2.imshow(WINDOW, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n  [Q pressed — stopping early]")
            break
        elif key == ord('p'):
            paused = not paused
            print(f"  [{'PAUSED' if paused else 'RESUMED'} at {frame_num/fps:.1f}s]")

    cap.release()
#    out.release()
# ABOVE ALSO COMMENTED OUT TO SAVE ON STORAGE FOR SAVING PURPOSES
    cv2.destroyAllWindows()

    print_summary()
    save_csv()


if __name__ == "__main__":
    run()
