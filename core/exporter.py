"""
core/exporter.py
Exports vehicle trajectory data in three formats:

  1. Static map image  — PNG satellite/OSM tile with coloured polylines per track
  2. GeoJSON           — drop into Google Maps, QGIS, Mapbox, etc.
  3. KML               — open in Google Earth or Google My Maps

All functions accept the tracker.summary() list and the stem (output filename base).

Usage (called from main.py):
    from core.exporter import export_all
    export_all(tracker.summary(), output_dir, stem)
"""
from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Color palette — one distinct color per vehicle class
# Folium / matplotlib use hex colors; KML uses aabbggrr hex
# ---------------------------------------------------------------------------

CLASS_COLORS_HEX: dict[str, str] = {
    "car":        "#00FF00",   # green
    "motorcycle": "#FFFF00",   # yellow
    "rikshaw":    "#00FFFF",   # cyan
    "HMV":        "#FF0000",   # red
    "pedestrian": "#FF00FF",   # magenta
    "unknown":    "#FFFFFF",   # white
}

# KML uses AABBGGRR (alpha, blue, green, red) — opposite of RGB
CLASS_COLORS_KML: dict[str, str] = {
    "car":        "ff00ff00",
    "motorcycle": "ff00ffff",
    "rikshaw":    "ffffff00",
    "HMV":        "ff0000ff",
    "pedestrian": "ffff00ff",
    "unknown":    "ffffffff",
}

# Matplotlib colors for static map
CLASS_COLORS_MPL: dict[str, str] = {
    "car":        "lime",
    "motorcycle": "yellow",
    "rikshaw":    "cyan",
    "HMV":        "red",
    "pedestrian": "magenta",
    "unknown":    "white",
}


def _get_color_hex(class_name: str) -> str:
    return CLASS_COLORS_HEX.get(class_name, CLASS_COLORS_HEX["unknown"])


def _get_color_kml(class_name: str) -> str:
    return CLASS_COLORS_KML.get(class_name, CLASS_COLORS_KML["unknown"])


def _get_color_mpl(class_name: str) -> str:
    return CLASS_COLORS_MPL.get(class_name, CLASS_COLORS_MPL["unknown"])


# ---------------------------------------------------------------------------
# Helper — parse GPS trail from summary row
# ---------------------------------------------------------------------------

def _parse_gps_trail(row: dict) -> list[tuple[float, float]]:
    """
    Returns list of (lat, lon) from a summary row's gps_trail field.
    Handles both the native list (if coming directly from tracker)
    and the stringified version (if read back from CSV).
    """
    trail = row.get("gps_trail")
    if trail is None:
        return []
    if isinstance(trail, list):
        return [(float(p[0]), float(p[1])) for p in trail]
    if isinstance(trail, str) and trail.strip().startswith("["):
        try:
            parsed = json.loads(trail.replace("(", "[").replace(")", "]"))
            return [(float(p[0]), float(p[1])) for p in parsed]
        except Exception:
            return []
    return []


# ---------------------------------------------------------------------------
# 1. Static map image (matplotlib + contextily)
# ---------------------------------------------------------------------------

def export_static_map(
    summary: list[dict],
    output_dir: Path,
    stem: str,
    dpi: int = 150,
) -> Path:
    """
    Saves a PNG of all vehicle trajectories plotted on an OSM satellite basemap.
    Uses contextily for tile fetching and matplotlib for rendering.

    Returns the output path.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")           # headless — no display needed
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import contextily as ctx
        from pyproj import Transformer
    except ImportError as e:
        raise ImportError(
            f"Missing dependency for static map: {e}\n"
            "Install with: pip install matplotlib contextily pyproj"
        ) from e

    # Collect all (lat, lon) points to set map extent
    all_lats, all_lons = [], []
    track_data = []  # (class_name, [(lat,lon), ...])

    for row in summary:
        trail = _parse_gps_trail(row)
        if len(trail) < 2:
            continue
        cls = row.get("class_name", "unknown")
        track_data.append((row["track_id"], cls, trail))
        lats = [p[0] for p in trail]
        lons = [p[1] for p in trail]
        all_lats.extend(lats)
        all_lons.extend(lons)

    if not all_lats:
        print("[Exporter] No GPS trails to plot — skipping static map.")
        return None

    # --- Project to Web Mercator (EPSG:3857) for contextily ---
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    plotted_classes = set()

    for track_id, cls, trail in track_data:
        lats = [p[0] for p in trail]
        lons = [p[1] for p in trail]
        xs, ys = transformer.transform(lons, lats)

        color = _get_color_mpl(cls)
        ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.8, zorder=3)

        # Start dot
        ax.scatter(xs[0], ys[0], color=color, s=18, zorder=4, alpha=0.9)
        # End arrow-head dot
        ax.scatter(xs[-1], ys[-1], color=color, s=30, marker="^", zorder=4, alpha=0.9)

        # Track ID label at midpoint
        mid = len(xs) // 2
        ax.annotate(
            f"ID{track_id}",
            (xs[mid], ys[mid]),
            color="white",
            fontsize=5,
            ha="center",
            va="bottom",
            zorder=5,
        )
        plotted_classes.add(cls)

    # Fit extent with padding
    all_xs, all_ys = transformer.transform(all_lons, all_lats)
    pad_x = (max(all_xs) - min(all_xs)) * 0.15 + 10
    pad_y = (max(all_ys) - min(all_ys)) * 0.15 + 10
    ax.set_xlim(min(all_xs) - pad_x, max(all_xs) + pad_x)
    ax.set_ylim(min(all_ys) - pad_y, max(all_ys) + pad_y)

    # Add OSM basemap tiles
    try:
        ctx.add_basemap(
            ax,
            crs="EPSG:3857",
            source=ctx.providers.OpenStreetMap.Mapnik,
            alpha=0.6,
            zoom="auto",
        )
    except Exception as e:
        print(f"[Exporter] Warning: could not fetch map tiles ({e}). Plotting without basemap.")

    # Legend
    legend_handles = [
        mpatches.Patch(color=_get_color_mpl(cls), label=cls)
        for cls in sorted(plotted_classes)
    ]
    legend_handles += [
        plt.Line2D([0], [0], marker="o",    color="w", markerfacecolor="gray",
                   markersize=6, label="Start"),
        plt.Line2D([0], [0], marker="^",    color="w", markerfacecolor="gray",
                   markersize=6, label="End"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=7,
        facecolor="#2a2a3e",
        labelcolor="white",
        framealpha=0.85,
    )

    ax.set_title(f"Vehicle Trajectories — {stem}", color="white", fontsize=11, pad=10)
    ax.set_xlabel("Easting (m, Web Mercator)", color="lightgray", fontsize=8)
    ax.set_ylabel("Northing (m, Web Mercator)", color="lightgray", fontsize=8)
    ax.tick_params(colors="lightgray", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    out_path = output_dir / f"{stem}_trajectory_map.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Exporter] Static map saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 2. GeoJSON export
# ---------------------------------------------------------------------------

def export_geojson(
    summary: list[dict],
    output_dir: Path,
    stem: str,
) -> Path:
    """
    Exports all trajectories as a GeoJSON FeatureCollection.
    Each track = one LineString Feature with properties:
        track_id, class_name, duration_sec, world_distance_m, world_velocity_kmph,
        start_lat, start_lon, end_lat, end_lon, n_points

    Compatible with Google Maps, Mapbox, QGIS, Leaflet, etc.
    """
    features = []

    for row in summary:
        trail = _parse_gps_trail(row)
        if len(trail) < 2:
            continue

        # GeoJSON coordinates are [lon, lat] (note: reversed from our (lat,lon) storage)
        coordinates = [[lon, lat] for lat, lon in trail]

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates,
            },
            "properties": {
                "track_id":           row.get("track_id"),
                "class_name":         row.get("class_name", "unknown"),
                "duration_sec":       row.get("duration_sec"),
                "world_distance_m":   row.get("world_distance_m"),
                "world_velocity_kmph":row.get("world_velocity_kmph"),
                "start_lat":          row.get("start_lat"),
                "start_lon":          row.get("start_lon"),
                "end_lat":            row.get("end_lat"),
                "end_lon":            row.get("end_lon"),
                "n_points":           len(trail),
                "color":              _get_color_hex(row.get("class_name", "unknown")),
            },
        }
        features.append(feature)

        # Also add start/end point markers
        for label, coord in [("start", trail[0]), ("end", trail[-1])]:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [coord[1], coord[0]],  # [lon, lat]
                },
                "properties": {
                    "track_id":   row.get("track_id"),
                    "class_name": row.get("class_name", "unknown"),
                    "marker":     label,
                    "color":      _get_color_hex(row.get("class_name", "unknown")),
                },
            })

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    out_path = output_dir / f"{stem}_trajectories.geojson"
    with open(out_path, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"[Exporter] GeoJSON saved:    {out_path}  ({len(features)} features)")
    return out_path


# ---------------------------------------------------------------------------
# 3. KML export
# ---------------------------------------------------------------------------

def export_kml(
    summary: list[dict],
    output_dir: Path,
    stem: str,
) -> Path:
    """
    Exports all trajectories as a KML file.
    Opens in Google Earth, Google My Maps, or any GIS tool.

    Each vehicle class gets its own named folder with a distinct line style.
    """
    kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    doc = ET.SubElement(kml, "Document")
    ET.SubElement(doc, "name").text = f"Vehicle Trajectories — {stem}"
    ET.SubElement(doc, "description").text = (
        "Aerial traffic analysis: vehicle trajectories mapped to GPS coordinates."
    )

    # Define line styles per class
    for cls, kml_color in CLASS_COLORS_KML.items():
        style = ET.SubElement(doc, "Style", id=f"style_{cls}")
        line_style = ET.SubElement(style, "LineStyle")
        ET.SubElement(line_style, "color").text = kml_color
        ET.SubElement(line_style, "width").text = "2"
        icon_style = ET.SubElement(style, "IconStyle")
        ET.SubElement(icon_style, "color").text = kml_color
        ET.SubElement(icon_style, "scale").text = "0.6"

    # Group tracks by class into folders
    class_folders: dict[str, ET.Element] = {}

    for row in summary:
        trail = _parse_gps_trail(row)
        if len(trail) < 2:
            continue

        cls      = row.get("class_name", "unknown")
        track_id = row.get("track_id")
        speed    = row.get("world_velocity_kmph")
        dist     = row.get("world_distance_m")
        dur      = row.get("duration_sec")

        if cls not in class_folders:
            folder = ET.SubElement(doc, "Folder")
            ET.SubElement(folder, "name").text = cls.capitalize()
            class_folders[cls] = folder
        folder = class_folders[cls]

        # --- Trajectory LineString ---
        pm = ET.SubElement(folder, "Placemark")
        ET.SubElement(pm, "name").text = f"Track {track_id} ({cls})"
        ET.SubElement(pm, "styleUrl").text = f"#style_{cls}"

        desc_parts = [f"Track ID: {track_id}", f"Class: {cls}"]
        if speed is not None:
            desc_parts.append(f"Avg Speed: {speed:.1f} km/h")
        if dist is not None:
            desc_parts.append(f"Distance: {dist:.1f} m")
        if dur is not None:
            desc_parts.append(f"Duration: {dur:.1f} s")
        ET.SubElement(pm, "description").text = "\n".join(desc_parts)

        ls = ET.SubElement(pm, "LineString")
        ET.SubElement(ls, "tessellate").text = "1"
        ET.SubElement(ls, "altitudeMode").text = "clampToGround"
        coords_text = " ".join(f"{lon},{lat},0" for lat, lon in trail)
        ET.SubElement(ls, "coordinates").text = coords_text

        # --- Start marker ---
        start_lat, start_lon = trail[0]
        pm_start = ET.SubElement(folder, "Placemark")
        ET.SubElement(pm_start, "name").text = f"Start — Track {track_id}"
        ET.SubElement(pm_start, "styleUrl").text = f"#style_{cls}"
        pt = ET.SubElement(pm_start, "Point")
        ET.SubElement(pt, "coordinates").text = f"{start_lon},{start_lat},0"

        # --- End marker ---
        end_lat, end_lon = trail[-1]
        pm_end = ET.SubElement(folder, "Placemark")
        ET.SubElement(pm_end, "name").text = f"End — Track {track_id}"
        ET.SubElement(pm_end, "styleUrl").text = f"#style_{cls}"
        pt2 = ET.SubElement(pm_end, "Point")
        ET.SubElement(pt2, "coordinates").text = f"{end_lon},{end_lat},0"

    tree = ET.ElementTree(kml)
    ET.indent(tree, space="  ")
    out_path = output_dir / f"{stem}_trajectories.kml"
    tree.write(out_path, xml_declaration=True, encoding="utf-8")

    track_count = sum(1 for row in summary if len(_parse_gps_trail(row)) >= 2)
    print(f"[Exporter] KML saved:        {out_path}  ({track_count} tracks)")
    return out_path


# ---------------------------------------------------------------------------
# 4. GPS trails CSV (flat, one row per detection point)
# ---------------------------------------------------------------------------

def export_gps_csv(
    gps_trail_rows: list[dict],
    output_dir: Path,
    stem: str,
) -> Path:
    """
    Saves a flat CSV:
        track_id, class_name, frame, cx, cy, wx_m, wy_m, latitude, longitude

    This is the Google Maps-compatible coordinate export —
    every row can be pasted directly as a waypoint.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required. pip install pandas")

    out_path = output_dir / f"{stem}_gps_trails.csv"
    df = pd.DataFrame(gps_trail_rows)
    df.to_csv(out_path, index=False)
    print(f"[Exporter] GPS trails CSV:   {out_path}  ({len(df)} rows)")
    return out_path


# ---------------------------------------------------------------------------
# Master export function
# ---------------------------------------------------------------------------

def export_all(
    summary: list[dict],
    gps_trail_rows: list[dict],
    output_dir: Path,
    stem: str,
    skip_map: bool = False,
) -> dict[str, Optional[Path]]:
    """
    Runs all four exports and returns a dict of {format: path}.

    Args:
        summary:         tracker.summary() list
        gps_trail_rows:  tracker.gps_trail_rows() list
        output_dir:      Path to outputs directory
        stem:            video filename stem (used in output filenames)
        skip_map:        Set True to skip static PNG map (useful in headless envs
                         without contextily/pyproj installed)
    """
    results: dict[str, Optional[Path]] = {}

    # GPS CSV (no extra deps — always runs)
    results["gps_csv"] = export_gps_csv(gps_trail_rows, output_dir, stem)

    # GeoJSON (no extra deps — always runs)
    results["geojson"] = export_geojson(summary, output_dir, stem)

    # KML (stdlib only — always runs)
    results["kml"] = export_kml(summary, output_dir, stem)

    # Static PNG map (needs matplotlib + contextily + pyproj)
    if skip_map:
        print("[Exporter] Static map skipped (--no-map flag).")
        results["static_map"] = None
    else:
        try:
            results["static_map"] = export_static_map(summary, output_dir, stem)
        except ImportError as e:
            print(f"[Exporter] Static map skipped — missing deps: {e}")
            results["static_map"] = None

    return results