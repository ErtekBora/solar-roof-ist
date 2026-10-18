"""Write tests/fixtures/buildings_sample.parquet: the <=200 buildings nearest to the map centre.

Keeps the columns the loader reads plus the parquet schema metadata (including "height_ml").
Usage: python scripts/make_buildings_fixture.py
"""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import shapely
from pyproj import Transformer

ROOT = Path(__file__).parent.parent
SOURCE = ROOT / "data" / "buildings" / "erenkoy_suadiye_goztepe_ml.parquet"
TARGET = ROOT / "tests" / "fixtures" / "buildings_sample.parquet"
CENTER = (40.9720, 29.0760)  # lat, lon
MAX_ROWS = 200
COLUMNS = ["id", "geometry", "height", "num_floors", "height_ml_m", "is_underground", "class", "sources"]


def main():
    table = pq.read_table(SOURCE, columns=COLUMNS)
    to_m = Transformer.from_crs("EPSG:4326", "EPSG:32635", always_xy=True)
    cx, cy = to_m.transform(CENTER[1], CENTER[0])
    geoms = shapely.from_wkb(table.column("geometry").to_pylist(), on_invalid="ignore")
    dist = [float("inf") if g is None else shapely.Point(to_m.transform(*g.representative_point().coords[0])).distance(shapely.Point(cx, cy)) for g in geoms]
    keep = sorted(range(len(dist)), key=dist.__getitem__)[:MAX_ROWS]
    out = table.take(pa.array(sorted(keep)))
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, TARGET)
    print(f"{out.num_rows} rows -> {TARGET} ({TARGET.stat().st_size / 1024:.0f} KB), metadata keys: {list(out.schema.metadata)}")


if __name__ == "__main__":
    main()
