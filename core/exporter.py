"""
core/exporter.py
Exports vehicle trajectory data in three formats:

  1. Static map image  — PNG satellite/OSM tile with coloured polylines per track
  2. GeoJSON           — drop into Google Maps, QGIS, Mapbox, etc.
  3. KML               — open in Google Earth or Google My Maps

All functions accept the tracker.summary() list and the stem (output filename base).

Usage (called from main.py):
    from core.exporter import export_all
    export_all(tracker.summary(), tracker.gps_trail_rows(), output_dir, stem)
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from config.settings import MIN_TRAIL_POINTS

# ---------------------------------------------------------------------------
# Color palette — one distinct color per vehicle class
# ---------------------------------------------------------------------------

CLASS_COLORS_HEX: dict[str, str] = {
    "car":        "#00FF00",
    "motorcycle": "#FFFF00",
    "rikshaw":    "#00FFFF",
    "HMV":        "#FF0000",
    "pedestrian": "#FF00FF",
    "unknown":    "#FFFFFF",
}

CLASS_COLORS_KML: dict[str, str] = {
    "car":        "ff00ff00",
    "motorcycle": "ff00ffff",
    "rikshaw":    "ffffff00",
    "HMV":        "ff0000ff",
    "pedestrian": "ffff00ff",
    "unknown":    "ffffffff",
}

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
    Handles both native list and stringified CSV form.
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
) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import contextily as ctx
        from pyproj import Transformer
    except ImportError as e:
        raise ImportError(
            f"Missing dependency for static map: {e}\n"
            "Install with: pip install matplotlib contextily pyproj"
        ) from e

    all_lats, all_lons = [], []
    track_data = []

    for row in summary:
        trail = _parse_gps_trail(row)
        if len(trail) < MIN_TRAIL_POINTS:
            continue
        cls = row.get("class_name", "unknown")
        track_data.append((row["track_id"], cls, trail))
        all_lats.extend(p[0] for p in trail)
        all_lons.extend(p[1] for p in trail)

    if not all_lats:
        print("[Exporter] No GPS trails to plot — skipping static map.")
        return None

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
        ax.scatter(xs[0],  ys[0],  color=color, s=18, zorder=4, alpha=0.9)
        ax.scatter(xs[-1], ys[-1], color=color, s=30, marker="^", zorder=4, alpha=0.9)

        mid = len(xs) // 2
        ax.annotate(f"ID{track_id}", (xs[mid], ys[mid]),
                    color="white", fontsize=5, ha="center", va="bottom", zorder=5)
        plotted_classes.add(cls)

    all_xs, all_ys = transformer.transform(all_lons, all_lats)
    pad_x = (max(all_xs) - min(all_xs)) * 0.15 + 10
    pad_y = (max(all_ys) - min(all_ys)) * 0.15 + 10
    ax.set_xlim(min(all_xs) - pad_x, max(all_xs) + pad_x)
    ax.set_ylim(min(all_ys) - pad_y, max(all_ys) + pad_y)

    try:
        ctx.add_basemap(ax, crs="EPSG:3857",
                        source=ctx.providers.OpenStreetMap.Mapnik,
                        alpha=0.6, zoom="auto")
    except Exception as e:
        print(f"[Exporter] Warning: could not fetch map tiles ({e}). Plotting without basemap.")

    legend_handles = [
        mpatches.Patch(color=_get_color_mpl(cls), label=cls)
        for cls in sorted(plotted_classes)
    ]
    legend_handles += [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markersize=6, label="Start"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
                   markersize=6, label="End"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7,
              facecolor="#2a2a3e", labelcolor="white", framealpha=0.85)

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
    features = []

    for row in summary:
        trail = _parse_gps_trail(row)
        if len(trail) < MIN_TRAIL_POINTS:
            continue

        coordinates = [[lon, lat] for lat, lon in trail]
        feature = {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coordinates},
            "properties": {
                "track_id":            row.get("track_id"),
                "class_name":          row.get("class_name", "unknown"),
                "duration_sec":        row.get("duration_sec"),
                "world_distance_m":    row.get("world_distance_m"),
                "world_velocity_kmph": row.get("world_velocity_kmph"),
                "start_lat":           row.get("start_lat"),
                "start_lon":           row.get("start_lon"),
                "end_lat":             row.get("end_lat"),
                "end_lon":             row.get("end_lon"),
                "n_points":            len(trail),
                "color":               _get_color_hex(row.get("class_name", "unknown")),
            },
        }
        features.append(feature)

        for label, coord in [("start", trail[0]), ("end", trail[-1])]:
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [coord[1], coord[0]]},
                "properties": {
                    "track_id":   row.get("track_id"),
                    "class_name": row.get("class_name", "unknown"),
                    "marker":     label,
                    "color":      _get_color_hex(row.get("class_name", "unknown")),
                },
            })

    geojson = {"type": "FeatureCollection", "features": features}
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
    kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    doc = ET.SubElement(kml, "Document")
    ET.SubElement(doc, "name").text = f"Vehicle Trajectories — {stem}"
    ET.SubElement(doc, "description").text = (
        "Aerial traffic analysis: vehicle trajectories mapped to GPS coordinates."
    )

    for cls, kml_color in CLASS_COLORS_KML.items():
        style = ET.SubElement(doc, "Style", id=f"style_{cls}")
        line_style = ET.SubElement(style, "LineStyle")
        ET.SubElement(line_style, "color").text = kml_color
        ET.SubElement(line_style, "width").text = "2"
        icon_style = ET.SubElement(style, "IconStyle")
        ET.SubElement(icon_style, "color").text = kml_color
        ET.SubElement(icon_style, "scale").text = "0.6"

    class_folders: dict[str, ET.Element] = {}

    for row in summary:
        trail = _parse_gps_trail(row)
        if len(trail) < MIN_TRAIL_POINTS:
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

        pm = ET.SubElement(folder, "Placemark")
        ET.SubElement(pm, "name").text = f"Track {track_id} ({cls})"
        ET.SubElement(pm, "styleUrl").text = f"#style_{cls}"

        desc_parts = [f"Track ID: {track_id}", f"Class: {cls}"]
        if speed is not None: desc_parts.append(f"Avg Speed: {speed:.1f} km/h")
        if dist  is not None: desc_parts.append(f"Distance: {dist:.1f} m")
        if dur   is not None: desc_parts.append(f"Duration: {dur:.1f} s")
        ET.SubElement(pm, "description").text = "\n".join(desc_parts)

        ls = ET.SubElement(pm, "LineString")
        ET.SubElement(ls, "tessellate").text = "1"
        ET.SubElement(ls, "altitudeMode").text = "clampToGround"
        ET.SubElement(ls, "coordinates").text = " ".join(
            f"{lon},{lat},0" for lat, lon in trail
        )

        for marker_name, (mlat, mlon) in [
            (f"Start — Track {track_id}", trail[0]),
            (f"End — Track {track_id}",   trail[-1]),
        ]:
            pm_m = ET.SubElement(folder, "Placemark")
            ET.SubElement(pm_m, "name").text     = marker_name
            ET.SubElement(pm_m, "styleUrl").text = f"#style_{cls}"
            pt = ET.SubElement(pm_m, "Point")
            ET.SubElement(pt, "coordinates").text = f"{mlon},{mlat},0"

    tree = ET.ElementTree(kml)
    ET.indent(tree, space="  ")
    out_path = output_dir / f"{stem}_trajectories.kml"
    tree.write(out_path, xml_declaration=True, encoding="utf-8")

    track_count = sum(1 for row in summary if len(_parse_gps_trail(row)) >= MIN_TRAIL_POINTS)
    print(f"[Exporter] KML saved:        {out_path}  ({track_count} tracks)")
    return out_path


# ---------------------------------------------------------------------------
# 4. GPS trails CSV
# ---------------------------------------------------------------------------

def export_gps_csv(
    gps_trail_rows: list[dict],
    output_dir: Path,
    stem: str,
) -> Path:
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
    results: dict[str, Optional[Path]] = {}

    results["gps_csv"] = export_gps_csv(gps_trail_rows, output_dir, stem)
    results["geojson"] = export_geojson(summary, output_dir, stem)
    results["kml"]     = export_kml(summary, output_dir, stem)

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