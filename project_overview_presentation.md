````carousel
# Aerial Traffic Analytics
**Project Overview & Pipeline Description**
*Drone-based video processing, vehicle tracking, and geospatial analysis.*

---

**Objective**
To process high-resolution drone footage of traffic, accurately detect and track various vehicle classes and pedestrians, and map their trajectories into real-world GPS coordinates for analytics.

<!-- slide -->
# Project Architecture
**Core Technologies**

1. **Object Detection**: YOLOv8 (Custom trained model `epoch40.pt`)
2. **Object Tracking**: ByteTrack (Multi-object trajectory tracking)
3. **Geospatial Mapping**: Homography matrix transformations
4. **Data Exports**: GeoJSON, KML, and CSV

*The system operates frame-by-frame, extracting bounding boxes, associating them with track IDs, and transforming pixel coordinates to world coordinates.*

<!-- slide -->
# Key Features & Pipeline

### 1. Robust Detection & Tracking
- **Multi-Class Detection**: Identifies Cars, HMVs (Heavy Motor Vehicles), Motorcycles, Rickshaws, and Pedestrians.
- **Trajectory History**: Retains full pixel and world coordinate trails for each vehicle.

### 2. Camera Calibration & Homography
- Uses predefined calibration JSONs (e.g., `calibration_cam1.json`) to map 2D pixel coordinates to real-world dimensions and GPS coordinates (Latitude/Longitude).
- Computes instantaneous real-world speeds (km/h) for moving objects.

<!-- slide -->
# Recent Enhancements: Noise Reduction

### Pedestrian Spatial Filtering
**The Problem**: Highway lane markings and high-speed motion caused the low-confidence object detector to generate false-positive pedestrian tracks in the middle of active traffic lanes.

**The Solution**:
- Implemented a configurable spatial boundary (`PEDESTRIAN_MIN_Y = 900`).
- The pipeline now explicitly filters out low-confidence pedestrian detections that occur in the active traffic lanes, isolating valid detections to the road shoulder/construction zone.
- **Result**: Reduced false positive tracks by over **60%**, eliminating noise while perfectly preserving true positive pedestrians.

<!-- slide -->
# Analytics & Outputs

The pipeline automatically generates several export formats for downstream geographic information systems (GIS):

- **GeoJSON**: `drone_3_trajectories.geojson` (For web mapping tools)
- **KML**: `drone_3_trajectories.kml` (For Google Earth integration)
- **CSV**: `drone_3_gps_trails.csv` (Raw tabular data for data science analytics)
- **Visuals**: Annotated `.mp4` video with bounding boxes, confidence scores, and color-coded trailing paths.
````
