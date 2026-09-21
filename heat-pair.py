# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "datafusion>=54.0.0",
#     "xarray-sql>=0.3.2",
#     "xarray",
#     "zarr>=3",
#     "h3ronpy>=0.22.0",
#     "pyarrow>=25.0.0",
#     "obstore>=0.9.2",
#     "anywidget>=0.9",
#     "numpy==2.5.1",
#     "duckdb>=1.5.5",
# ]
# ///
"""Heat index and sustained heat, both at once: a pair map on one clock.

x-sql-marimo's sustained heat notebook (xsql-hrrr-heat-hex-forecast.py, the 48 h
forecast build: source.coop, plain Zarr v3, each 6-hourly run's first six leads laid
end to end into one hourly film) with the field switch taken out and the two fields
put side by side, the way zarr-grid-h3's pair maps sit two panes on one camera. LEFT:
NWS heat index this hour. RIGHT: sustained heat up to this hour, the exponentially
weighted mean of the heat index's excess over a threshold, computed in the browser
from the same frames. One clock plays both. Stephen's brief (2026-09-21): "something
like the sustained heat notebook where i look at both at once like the pair maps ...
should be pretty simple we're just playing both at the same time. hottest 7 day
window of 2026 summer usa".

The kernel (store, window stitch, land mask, res 6 fold in DataFusion with the h3ronpy
UDF, frame matrices) is the source notebook's, unchanged; its account of cost and
memory lives in that file's docstring and in x-sql-marimo/docs/hrrr-heat-hex-notes.md.
Only the widget, the opening window and the markdown differ. Both formulas are in the
first markdown cell.
"""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell
def _():
    import asyncio
    import gzip
    import json
    import math
    import struct

    import anywidget
    import duckdb
    import marimo as mo
    import numpy as np
    import obstore
    import pyarrow as pa
    import pyarrow.parquet as pq
    import traitlets
    import xarray as xr
    from h3ronpy.vector import coordinates_to_cells
    from obstore.store import S3Store
    from xarray_sql import XarrayContext

    return (
        S3Store,
        XarrayContext,
        anywidget,
        asyncio,
        coordinates_to_cells,
        duckdb,
        gzip,
        json,
        math,
        mo,
        np,
        obstore,
        pa,
        pq,
        struct,
        traitlets,
        xr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/github/github.com/kentstephen/weather-rnd/blob/main/heat-pair.py)

    # Heat index and sustained heat, side by side

    Two maps of the lower 48 on H3 hexagons, one camera, one clock. **Left: the heat
    index this hour.** **Right: sustained heat up to this hour**, what the last days
    of heat index have left behind in each cell. The window is 29 June to 5 July 2026,
    the week that ends on the July 4 weekend, when the eastern heat dome peaked
    (Atlantic City 106 °F on the 4th) before storms broke it up. Press play and both
    run together. Hover a cell on either map and it is ringed on both; click it and
    its two lines are drawn under the maps.

    ## Heat index

    The National Weather Service heat index, from each cell's hourly mean 2 m
    temperature $T$ (°F) and relative humidity $R$ (%). First Steadman's simple form:

    $$HI_s = 0.5\,\bigl[\,T + 61 + 1.2\,(T - 68) + 0.094\,R\,\bigr]$$

    If $(HI_s + T)/2 < 80$ °F that is the heat index. Otherwise the Rothfusz
    regression replaces it:

    $$
    \begin{aligned}
    HI ={}& -42.379 + 2.04901523\,T + 10.14333127\,R - 0.22475541\,TR \\
          & - 0.00683783\,T^2 - 0.05481717\,R^2 + 0.00122874\,T^2R \\
          & + 0.00085282\,TR^2 - 0.00000199\,T^2R^2
    \end{aligned}
    $$

    with the two NWS adjustments:

    $$HI \leftarrow HI - \frac{13 - R}{4}\sqrt{\frac{17 - |T - 95|}{17}} \quad \text{if } R < 13 \text{ and } 80 \le T \le 112$$

    $$HI \leftarrow HI + \frac{R - 85}{10}\cdot\frac{87 - T}{5} \quad \text{if } R > 85 \text{ and } 80 \le T \le 87$$

    The result is converted to °C, $(HI - 32)\cdot 5/9$, and that is what the left
    map paints.

    ## Sustained heat

    An exponentially weighted moving average of the heat index's excess over a
    threshold, per cell, hour by hour (the recurrence of hydrology's antecedent
    precipitation index, a first-order low-pass filter). With half-life $h$ hours
    and threshold $\theta$ (°C):

    $$a = 2^{-1/h}$$

    $$e_t = \max(0,\; HI_t - \theta)\,\cdot\,\max\!\Bigl(0,\; 1 - w\,\frac{s_t}{10}\Bigr)$$

    $$L_t = \bigl[\,a\,L_{t-1} + (1 - a)\,e_t\,\bigr]\cdot\Bigl(1 - r\,\min\!\bigl(1,\; \tfrac{p_t}{2.5}\bigr)\Bigr), \qquad L_{-1} = 0$$

    where $s_t$ is the 10 m wind speed (m/s), $p_t$ the rain rate (mm/h), $w$ the
    wind vent weight and $r$ the rain flush weight, both 0 to 1. With $w = r = 0$
    it is just $L_t = a\,L_{t-1} + (1 - a)\max(0, HI_t - \theta)$. $L$ is in °C
    above the threshold, so it compares with the heat index directly. Defaults:
    $h = 12$ h, $\theta = 27$ °C (where the NWS caution band starts), $r = 0.5$,
    $w = 0.3$. It starts from zero at the first hour of the window, so the first day
    is spin-up. This is a smoothing with stated parameters, not a published index;
    the four sliders under the right map are those parameters, and every move
    recomputes the whole film in the browser.

    ## Where the numbers come from

    [dynamical.org](https://dynamical.org/)'s Zarr build of NOAA's HRRR 48-hour
    forecast (3 km, hourly leads, CONUS, CC-BY 4.0), read anonymously from source.coop
    (`s3://us-west-2.opendata.source.coop/dynamical/noaa-hrrr-forecast-48-hour/`):
    2 m temperature and relative humidity, precipitation rate and 10 m wind, from
    every 00/06/12/18Z run inside the window, each run's first six lead hours laid
    end to end into one hourly film. The store is queried with
    [xarray-sql](https://github.com/alxmrs/xarray-sql); every pixel is labelled with
    its H3 res 6 cell inside the DataFusion `GROUP BY`, which averages the ~4 pixels
    in each cell per hour; numpy turns temperature and humidity into the heat index.
    The film crosses to the browser once as bytes; the accumulator, the clock, the
    picking and the charts run there. Counties (Overture Maps divisions) are the land
    mask and the name a clicked cell reports. A week is 28 runs x 5 variables, one
    shard each: minutes from home, seconds near the bucket.
    """)
    return


@app.cell
def _():
    # ------------------------------------------------------------------ the weather
    # dynamical.org's 48-hour HRRR forecast on source.coop: plain Zarr v3 (not
    # icechunk), dims (init_time, lead_time, y, x), 6-hourly inits from 2018-07-13
    # (11,941 on 2026-09-14), 49 leads, chunks (1, 49, 265, 300), one shard per init
    # (1, 49, 1060, 1800). One run is one shard per variable (the counties film
    # measured 2.2 s for 49 leads x CONUS of one variable from home), and a window
    # is STITCHED from the runs inside it (LEADS_PER_RUN leads each), so its cost is
    # runs x variables, the same for any dates. The analysis notebook's 90-day time
    # chunks, its chunk-bytes cache and its disk mirror have no counterpart here.
    FORECAST_BUCKET = "us-west-2.opendata.source.coop"
    FORECAST_PREFIX = "dynamical/noaa-hrrr-forecast-48-hour/v0.1.0.zarr"
    # Heat index needs temperature_2m and relative_humidity_2m (always read). Rain is
    # the accumulator's flush, wind its vent; each is another shard read (seconds).
    READ_RAIN = True  # precipitation_surface (kg m-2 s-1, an hourly-mean RATE -> mm/h)
    READ_WIND = True  # wind_u_10m + wind_v_10m -> speed (m/s); two variables
    # Leads kept from each 6-hourly run: 6 = a continuous hourly series of 0-5 h
    # forecasts (the runs abut, no overlap, no gap). More than 6 makes runs overlap
    # and two rows share a valid time (the frames cell keeps the last written, which
    # is arbitrary); fewer leaves gaps. Keep it at 6.
    LEADS_PER_RUN = 6
    # Opening window: an int is the last DAYS UTC days ending at the newest run; a
    # ("YYYY-MM-DD", "YYYY-MM-DD") tuple is a fixed window. Same presets as the
    # analysis notebook (summer 2026's heat domes); any week costs the same here:
    #   DAYS = ("2026-07-06", "2026-07-12")   # West dome, Salt Lake City 109 F Jul 12
    #   DAYS = ("2026-07-23", "2026-07-29")   # Plains dome, Rapid City 112 F Jul 26
    #   DAYS = 7                              # the last week
    # The week ending on the July 4 weekend (Stephen, 2026-09-21): the East dome, per
    # Wikipedia (2026 North American heat wave) at least 44 heat deaths Jul 1-4, 180
    # million people under major or extreme heat risk, Atlantic City 106 F on Jul 4;
    # storms broke it up Jul 4-6. Five days of build-up before the weekend, so the
    # sustained heat is past its spin-up by Friday.
    DAYS = ("2026-06-29", "2026-07-05")
    HOURLY_MAX_DAYS = 14  # 336 frames x 210k cells = 71 MB per field across the bridge

    # ------------------------------------------------------------------ the fold
    # Res 6: 4.2 HRRR pixels per cell (measured), 210,724 cells over CONUS land, all
    # of them catch a pixel; 35M (hour, cell) answers for a week, which is what sets
    # the kernel's memory (~5 GB peak with the fold cell's DataFusion knobs; 17 GB
    # without them) and the bytes to the browser (35 MB per field). The read is
    # identical at any res (same pixels). Res 5 (30,124 cells, 5M answers, well
    # under a GB, 5 MB per field, ~250 km2 hexes) is the one-constant retreat, and
    # was flown; the counties film with this accumulator is the fallback after that.
    # Res 7 is the pixel itself (a relabel): 1.47M land cells but only 879k land
    # pixels, so ~40% of the hexes hold no pixel centre and the map is full of holes
    # (Stephen ran it 2026-08-17: "a lot of gaps between the cells"), and a week is
    # 248M answers, ~35 GB of aggregate state. Res 6 is where it lives.
    RES = 6
    # DataFusion memory pool for the fold's final aggregate (one hash entry per
    # (hour, cell) answer, ~150 B). At res 6 a 3 GB pool holds the process near 5 GB
    # instead of 9.5, no time cost. AT RES 7 THE POOL MUST BE OFF: 248M answers a week
    # is ~30 GB of state, and under any pool that fits in RAM DataFusion spills nearly
    # all of it to disk and merge-sorts it back, which reads as the fold spinning
    # forever (2026-08-17, Stephen's res 7 demo run). None = unbounded, RAM decides.
    # The pool is for the two-field default only. With rain and/or wind on there are
    # three or four accumulators per (hour, cell) group and DataFusion cannot even
    # reserve the room it needs to sort a batch before spilling: 3 GB and 6 GB pools
    # both died with "Failed to reserve memory for sort during spill" (2026-08-17). So
    # the pool is off then and RAM decides; expect roughly 10-15 GB at res 6 for a
    # week with all four fields, on top of the read being 4-5 variables (~1 min in
    # a young chunk, 4-5 min in a full one).
    MEM_POOL_GB = 3 if (RES <= 6 and not READ_RAIN and not READ_WIND) else None

    # ------------------------------------------------------------------ the land mask
    # Overture's PMTiles build, same object, box, zoom and CONUS filter as the counties
    # film; the counties are the mask (a cell is CONUS land if its centre falls in a
    # county) and the click readout's name. THE RELEASE IS NOT PINNED: the extras bucket
    # keeps only the newest release's tiles (a pin to 2026-07-22.0 began to 404 once
    # 2026-08-19.0 landed), so the newest `tiles/<release>/` prefix in the bucket is the
    # release. Overture's STAC catalog (https://stac.overturemaps.org/catalog.json,
    # key "latest") says the same thing, but the listing is what is actually there.
    # A string here ("2026-08-19.0") pins it; OVERTURE_FALLBACK is used if the listing fails.
    OVERTURE_RELEASE = None
    OVERTURE_FALLBACK = "2026-08-19.0"
    PM_BUCKET = "overturemaps-extras-us-west-2"
    if OVERTURE_RELEASE is None:
        try:
            import obstore as _obstore
            from obstore.store import S3Store as _S3Store

            _found = _obstore.list_with_delimiter(
                _S3Store(PM_BUCKET, region="us-west-2", skip_signature=True), "tiles/"
            )["common_prefixes"]
            OVERTURE_RELEASE = max(p.rstrip("/").split("/")[-1] for p in _found)
        except Exception:  # noqa: BLE001
            OVERTURE_RELEASE = OVERTURE_FALLBACK
    PM_PATH = f"tiles/{OVERTURE_RELEASE}/divisions.pmtiles"
    COUNTY_Z = 8
    BOX = (-124.8, 24.4, -66.9, 49.5)
    NOT_CONUS = {"AK", "HI"}
    import tempfile as _tempfile

    # Same cache file as the counties film (tmp, not a project .cache; None disables).
    CACHE_DIR = str(_tempfile.gettempdir()) + "/x-sql-marimo"
    # Disk mirror of the forecast shards' byte ranges (store cell). None disables.
    MIRROR_DIR = CACHE_DIR + "/hrrr-forecast-mirror/v0.1.0.zarr"

    # ------------------------------------------------------------------ the film
    # Heat index: diverging blue <-> yellow/orange (protan-safe: no red leg, no
    # red-vs-green pair), pale at the pivot (the film's median), span to the wider of
    # p2/p98 unless PIVOT/SPAN pin it. Sustained heat: one-signed, so a luminance ramp,
    # dark to bright (matplotlib inferno's stops; on the colorblind-safe shortlist),
    # scaled in the browser to the p98 of the values it just computed.
    INDEX_STOPS = ["#08306b", "#2f79b5", "#9ecae1", "#f2f0e6", "#fee391", "#fdb034", "#d94801"]
    LOAD_STOPS = ["#000004", "#1b0c41", "#4a0c6b", "#781c6d", "#a52c60", "#cf4446", "#ed6925", "#fb9b06", "#f7d13d", "#fcffa4"]
    PIVOT = None  # degC heat index, or None for the film median
    SPAN = None  # degC either side, or None for the p2/p98 rule
    # Accumulator defaults (all live sliders in the HUD): sustained heat rises by the heat
    # index's excess over THRESHOLD (27 degC is where the NWS caution band starts)
    # and decays with HALF_LIFE hours; RAIN_FLUSH and WIND_VENT are 0..1 weights,
    # meaningful only when the field was read.
    THRESHOLD = 27.0
    HALF_LIFE = 12.0
    RAIN_FLUSH = 0.5
    WIND_VENT = 0.3
    FPS = 8
    MAP_HEIGHT = 560
    return (
        BOX,
        CACHE_DIR,
        COUNTY_Z,
        DAYS,
        FORECAST_BUCKET,
        FORECAST_PREFIX,
        FPS,
        HALF_LIFE,
        HOURLY_MAX_DAYS,
        INDEX_STOPS,
        LEADS_PER_RUN,
        LOAD_STOPS,
        MAP_HEIGHT,
        MEM_POOL_GB,
        MIRROR_DIR,
        NOT_CONUS,
        OVERTURE_RELEASE,
        PIVOT,
        PM_BUCKET,
        PM_PATH,
        RAIN_FLUSH,
        READ_RAIN,
        READ_WIND,
        RES,
        SPAN,
        THRESHOLD,
        WIND_VENT,
    )


@app.cell
def _(duckdb):
    # DuckDB does the geometry (tile-seam dissolve, polyfill); DataFusion does the fold.
    con = duckdb.connect()
    con.sql("INSTALL h3 FROM community; LOAD h3; INSTALL spatial; LOAD spatial;")
    return (con,)


@app.cell
def _(anywidget, traitlets):
    class HeatPair(anywidget.AnyWidget):
        """Two maplibre maps side by side, one camera, one clock, one film.

        LEFT paints the heat index of the hour, RIGHT the sustained heat up to that
        hour, each an opaque deck.gl H3HexagonLayer (highPrecision) interleaved into
        its own maplibre map (OpenFreeMap dark, text layers hidden). A move on either
        map is a jumpTo on the other. The clock is the browser's and both panes paint
        the same frame on the same tick. Hovering a cell on either pane rings it on
        both; a click picks it and the strip under the maps draws its two lines, each
        under its own pane. Each pane's name, ramp and CONUS mean sit in a bar ABOVE
        it; nothing is drawn over the maps.

        Kernel -> browser: `cells` (uint64 LE H3 ids, N, sorted), `cidx` (uint16 county
        index per cell), `names` (JSON list of "County, ST"), `frames` (uint8 F x N,
        frame-major: heat index in 0.5 degC steps offset -40, 255 = no data), `wx`
        (uint8 F x N or empty: wind m/s rounded in the high nibble, rain in 0.5 mm/h
        steps in the low nibble) and `config` (JSON). Browser -> kernel: `window`
        only, the date range + load button.

        The ACCUMULATOR runs in the browser over the frame matrix whenever a slider
        moves: L[f] = a * L[f-1] + (1 - a) * max(0, HI[f] - threshold),
        a = 2^(-1/half_life); rain multiplies L by (1 - flush * min(1, mm / 2.5)) that
        hour, wind scales the excess by max(0, 1 - vent * ws / 10). Stored as uint8
        (0.1 degC steps) for paint; the right pane's ramp is 0 .. p98 of the load just
        computed. Picking is h3-js `latLngToCell` on the unprojected pointer.

        esm.sh pins as in the source notebook (deck 9.3.10 with `?deps`, h3-js 4.5.0).
        """

        _esm = r"""
        import maplibregl from "https://esm.sh/maplibre-gl@5.24.0";
        import {MapboxOverlay} from "https://esm.sh/@deck.gl/mapbox@9.3.10?deps=@deck.gl/core@9.3.10,@deck.gl/extensions@9.3.10,@deck.gl/layers@9.3.10,@deck.gl/mesh-layers@9.3.10,@deck.gl/geo-layers@9.3.10,@deck.gl/aggregation-layers@9.3.10,@luma.gl/core@9.3.6,@luma.gl/engine@9.3.6,@luma.gl/shadertools@9.3.6,@luma.gl/webgl@9.3.6,@luma.gl/gltf@9.3.6,apache-arrow@18.1.0";
        import {H3HexagonLayer} from "https://esm.sh/@deck.gl/geo-layers@9.3.10?deps=@deck.gl/core@9.3.10,@deck.gl/extensions@9.3.10,@deck.gl/layers@9.3.10,@deck.gl/mesh-layers@9.3.10,@luma.gl/core@9.3.6,@luma.gl/engine@9.3.6,@luma.gl/gltf@9.3.6,@luma.gl/shadertools@9.3.6,@luma.gl/webgl@9.3.6,apache-arrow@18.1.0";
        import {latLngToCell} from "https://esm.sh/h3-js@4.5.0";

        const CSS = `
          .hp { --panel:rgba(15,18,22,.84); --ink:#dfe3e8; --dim:#8b929c; --accent:#e6c14a;
                font: 13px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink); background: #0f1216; outline: none; }
          .hp * { box-sizing: border-box; }
          .hp:fullscreen { display: flex; flex-direction: column; height: 100vh; width: 100vw; }
          .hp:fullscreen .hp-maps { flex: 1 1 auto; height: auto !important; }
          .hp .hp-num { font-variant-numeric: tabular-nums; }
          .hp .hp-two { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 2px; }
          .hp .hp-head { padding: .45rem .7rem .5rem; background: #12161b; }
          .hp .hp-name { display: flex; justify-content: space-between; align-items: baseline; gap: .6rem; }
          .hp .hp-ttl { font-weight: 600; font-size: 14px; }
          .hp .hp-mean { color: var(--dim); margin-left: auto; }
          .hp .hp-toggle { background: none; border: 0; color: var(--dim); cursor: pointer; font: inherit; padding: 0 0 0 .4rem; }
          .hp .hp-toggle:hover { color: var(--ink); }
          .hp.hp-collapsed .hp-strip { display: none; }  /* "hp-collapsed", never "hidden": marimo's Tailwind owns .hidden */
          .hp .hp-legend { display: flex; align-items: center; gap: .45rem; margin-top: .3rem; color: var(--dim); }
          .hp .hp-grad { height: .55rem; flex: 1; border: 1px solid rgba(255,255,255,.12); }
          .hp .hp-maps { grid-template-rows: minmax(0, 1fr); width: 100%; background: #0b0d10; }
          .hp .hp-pane { position: relative; overflow: hidden; min-height: 0; }
          .hp .hp-mapc { position: absolute; inset: 0; }
          .hp .hp-strip { background: #12161b; padding: .5rem .7rem .6rem; margin-top: 2px; }
          .hp .hp-transport { display: flex; align-items: center; gap: .55rem; }
          .hp .hp-stamp { font-size: 15px; min-width: 11.5rem; text-align: right; }
          .hp .hp-track { flex: 1 1 10rem; position: relative; padding-top: 6px; }
          .hp .hp-ticks { position: absolute; left: 0; right: 0; top: 0; height: 6px; }
          .hp .hp-ticks i { position: absolute; top: 0; width: 1px; height: 6px; background: var(--dim); }
          .hp input[type=range] { width: 100%; margin: 0; accent-color: var(--accent); }
          .hp button.hp-b, .hp select { background: #22282f; color: var(--ink); border: 1px solid #343b45; padding: .22rem .5rem; cursor: pointer; font: inherit; line-height: 1.2; min-width: 2rem; }
          .hp button.hp-b:hover, .hp select:hover { background: #2b323b; }
          .hp button:focus-visible, .hp select:focus-visible, .hp input:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
          .hp .hp-dim { color: var(--dim); }
          .hp .hp-cell { display: none; margin-top: .55rem; gap: 1.4rem; }
          .hp.hp-picked .hp-cell { display: grid; }
          .hp .hp-crow { display: flex; justify-content: space-between; align-items: baseline; gap: .6rem; }
          .hp .hp-crow .hp-v { font-size: 16px; }
          .hp .hp-chart { display: block; width: 100%; height: 96px; margin-top: .25rem; cursor: crosshair; }
          .hp .hp-clear { background: none; border: 0; color: var(--dim); cursor: pointer; font: inherit; padding: 0 .1rem; }
          .hp .hp-clear:hover { color: var(--ink); }
          .hp .hp-lower { margin-top: .55rem; padding-top: .5rem; border-top: 1px solid rgba(255,255,255,.08); gap: 1.4rem; }
          .hp .hp-p { display: grid; grid-template-columns: 6.2rem 1fr 3.6rem; align-items: center; gap: .4rem; margin-top: .2rem; }
          .hp .hp-p label { color: var(--dim); }
          .hp .hp-p.hp-off { display: none; }
          .hp .hp-win { display: flex; flex-wrap: wrap; align-items: center; gap: .3rem .4rem; }
          .hp .hp-win input[type=date] { background: #22282f; color: var(--ink); border: 1px solid #343b45; padding: .15rem .3rem; font: inherit; color-scheme: dark; min-width: 0; }
          .hp .hp-win .hp-note { flex-basis: 100%; }
          .hp .hp-win .hp-note.hp-bad { color: var(--accent); }
          .hp .hp-win button.hp-load:disabled { opacity: .55; cursor: default; }
          .hp .hp-foot { margin-top: .45rem; }
          @media (max-width: 720px) { .hp .hp-stamp { min-width: 0; } }
        `;

        function hexToRgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
        function buildLut(stops) {
          const rgb = stops.map(hexToRgb), lut = new Uint8Array(256 * 3);
          for (let i = 0; i < 256; i++) {
            const t = i / 255 * (rgb.length - 1), k = Math.min(rgb.length - 2, Math.floor(t)), f = t - k;
            for (let c = 0; c < 3; c++) lut[i * 3 + c] = Math.round(rgb[k][c] * (1 - f) + rgb[k + 1][c] * f);
          }
          return lut;
        }
        function bytesOf(v) {
          if (!v) return null;
          if (v instanceof DataView) return new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
          if (v instanceof ArrayBuffer) return new Uint8Array(v);
          if (v.buffer) return new Uint8Array(v.buffer, v.byteOffset ?? 0, v.byteLength);
          return null;
        }
        const HI_OF = q => q / 2 - 40;  // uint8 -> degC
        const FIELDS = ["index", "load"];

        function render({model, el}) {
          el.innerHTML = "";
          const mlCss = document.createElement("link");
          mlCss.rel = "stylesheet"; mlCss.href = "https://unpkg.com/maplibre-gl@5.24.0/dist/maplibre-gl.css";
          el.appendChild(mlCss);
          const root = document.createElement("div"); root.className = "hp";
          const head = (f, name, tip) => `<div class="hp-head" title="${tip}">
              <div class="hp-name"><span class="hp-ttl">${name}</span><span class="hp-mean hp-num hp-mean-${f}"></span>${f === "load" ? '<button class="hp-toggle" title="hide or show the panel under the maps (H); a click on a cell brings it back">hide panel</button>' : ""}</div>
              <div class="hp-legend"><span class="hp-num hp-lo-${f}"></span><div class="hp-grad hp-grad-${f}"></div><span class="hp-num hp-hi-${f}"></span></div>
            </div>`;
          root.innerHTML = `<style>${CSS}</style>
            <div class="hp-two hp-heads">
              ${head("index", "heat index, this hour", "NWS heat index from the hour's temperature and humidity")}
              ${head("load", "sustained heat, up to this hour", "exponentially weighted mean of the heat index above the threshold")}
            </div>
            <div class="hp-two hp-maps">
              <div class="hp-pane hp-pane-index"><div class="hp-mapc"></div></div>
              <div class="hp-pane hp-pane-load"><div class="hp-mapc"></div></div>
            </div>
            <div class="hp-strip">
              <div class="hp-transport">
                <button class="hp-b hp-prev" title="step back (left arrow)">‹</button>
                <button class="hp-b hp-play" title="play / pause (space)">▶</button>
                <button class="hp-b hp-next" title="step forward (right arrow)">›</button>
                <div class="hp-track"><div class="hp-ticks"></div><input class="hp-frame" type="range" min="0" max="0" value="0" step="1" aria-label="hour"></div>
                <div class="hp-stamp hp-num"></div>
                <select class="hp-fps" title="frames per second"><option>2</option><option>4</option><option>6</option><option>8</option><option>12</option><option>24</option></select>
                <button class="hp-b hp-full" title="fullscreen (F)">⛶</button>
              </div>
              <div class="hp-two hp-cell">
                <div><div class="hp-crow"><span class="hp-dim hp-cname"></span><span class="hp-num hp-v hp-cval-index"></span></div><canvas class="hp-chart hp-chart-index" height="96"></canvas></div>
                <div><div class="hp-crow"><span class="hp-dim">the same cell</span><span><span class="hp-num hp-v hp-cval-load"></span> <button class="hp-clear" title="clear">×</button></span></div><canvas class="hp-chart hp-chart-load" height="96"></canvas></div>
              </div>
              <div class="hp-two hp-lower">
                <div class="hp-win">
                  <input type="date" class="hp-d0" title="first UTC day, inclusive" aria-label="window start (UTC day)"><span class="hp-dim">to</span><input type="date" class="hp-d1" title="last UTC day, inclusive" aria-label="window end (UTC day)">
                  <button class="hp-b hp-load" title="fold this window (the kernel refetches one shard per run per variable)">load</button>
                  <span class="hp-dim hp-note"></span>
                </div>
                <div class="hp-params">
                  <div class="hp-p"><label>half-life</label><input type="range" class="hp-half" min="1" max="72" step="1"><span class="hp-num hp-halfv"></span></div>
                  <div class="hp-p"><label>threshold</label><input type="range" class="hp-thr" min="15" max="40" step="0.5"><span class="hp-num hp-thrv"></span></div>
                  <div class="hp-p hp-p-rain"><label>rain flush</label><input type="range" class="hp-rain" min="0" max="1" step="0.05"><span class="hp-num hp-rainv"></span></div>
                  <div class="hp-p hp-p-wind"><label>wind vent</label><input type="range" class="hp-wind" min="0" max="1" step="0.05"><span class="hp-num hp-windv"></span></div>
                </div>
              </div>
              <div class="hp-dim hp-foot"><span class="hp-sub"></span> <span class="hp-ruler"></span></div>
            </div>`;
          el.appendChild(root);
          const q = s => root.querySelector(s);
          const mapsEl = q(".hp-maps"), playBtn = q(".hp-play"), slider = q(".hp-frame"), ticks = q(".hp-ticks"),
                stampV = q(".hp-stamp"), fpsSel = q(".hp-fps"), ruler = q(".hp-ruler"), sub = q(".hp-sub"), cname = q(".hp-cname"),
                d0In = q(".hp-d0"), d1In = q(".hp-d1"), loadBtn = q(".hp-load"), noteEl = q(".hp-note"),
                halfIn = q(".hp-half"), thrIn = q(".hp-thr"), rainIn = q(".hp-rain"), windIn = q(".hp-wind"),
                halfV = q(".hp-halfv"), thrV = q(".hp-thrv"), rainV = q(".hp-rainv"), windV = q(".hp-windv");
          // one entry per pane: its map, its deck overlay and the elements that speak for it
          const P = {};
          FIELDS.forEach(f => { P[f] = {
            map: null, overlay: null, pane: q(".hp-pane-" + f), el: q(".hp-pane-" + f + " .hp-mapc"),
            grad: q(".hp-grad-" + f), lo: q(".hp-lo-" + f), hi: q(".hp-hi-" + f), mean: q(".hp-mean-" + f),
            cval: q(".hp-cval-" + f), chart: q(".hp-chart-" + f),
          }; });

          let hexes = [], hexIndex = new Map(), cidx = null, names = [], N = 0, F = 0;
          let frames = null, wx = null, load = null, loadHi = 1;
          let cfg = {}, frame = 0, playing = false, timer = null, selected = -1, hover = -1;
          let gen = 0, fgen = 0, lutI = null, lutL = null;
          let meansI = null, meansL = null;
          // OpenFreeMap dark, drawn by maplibre with deck INTERLEAVED into its context,
          // every deck layer slotted under the style's first road-name layer. Two maps,
          // one camera: a move on either is a jumpTo on the other.
          const STYLE = "https://tiles.openfreemap.org/styles/dark";
          const LABELS_SLOT = "highway_name_other";
          const CONUS = [[-125.2, 24.0], [-66.4, 49.9]];

          const fmtC = v => Number.isFinite(v) ? v.toFixed(1) + "°C" : "no data";
          const fmtOf = (field, v) => (field === "load" && Number.isFinite(v) ? "+" : "") + fmtC(v);
          const hiAt = (f, i) => { const qv = frames[f * N + i]; return qv === 255 ? NaN : HI_OF(qv); };
          const loadAt = (f, i) => { const qv = load[f * N + i]; return qv === 255 ? NaN : qv / 10; };
          const valAt = (field, f, i) => field === "load" ? loadAt(f, i) : hiAt(f, i);
          const params = () => ({
            half: parseFloat(halfIn.value) || 12, thr: parseFloat(thrIn.value) || 27,
            rain: cfg.has_rain ? (parseFloat(rainIn.value) || 0) : 0, wind: cfg.has_wind ? (parseFloat(windIn.value) || 0) : 0,
          });

          // THE ACCUMULATOR, over the whole film, into uint8 (0.1 degC steps, 255 = no data).
          function computeLoad() {
            if (!frames) return;
            const {half, thr, rain, wind} = params();
            const a = Math.pow(2, -1 / half), b = 1 - a;
            load = load && load.length === F * N ? load : new Uint8Array(F * N);
            const prev = new Float32Array(N), hist = new Uint32Array(256);
            meansL = new Float32Array(F);
            for (let f = 0; f < F; f++) {
              const base = f * N; let s = 0, n = 0;
              for (let i = 0; i < N; i++) {
                const k = base + i, qv = frames[k];
                if (qv === 255) { load[k] = 255; continue; }
                let ex = HI_OF(qv) - thr; if (ex < 0) ex = 0;
                if (wind) { const ws = wx[k] >> 4; ex *= Math.max(0, 1 - wind * ws / 10); }
                let L = a * prev[i] + b * ex;
                if (rain) { const mm = (wx[k] & 15) / 2; if (mm > 0) L *= 1 - rain * Math.min(1, mm / 2.5); }
                prev[i] = L;
                let lq = Math.round(L * 10); if (lq > 254) lq = 254;
                load[k] = lq; hist[lq]++; s += L; n++;
              }
              meansL[f] = n ? s / n : NaN;
            }
            // ramp top: p98 of the non-zero load, at least 1 degC
            let tot = 0; for (let i = 1; i < 255; i++) tot += hist[i];
            let acc = 0, top = 10;
            for (let i = 1; i < 255; i++) { acc += hist[i]; if (acc >= tot * 0.98) { top = i; break; } }
            loadHi = Math.max(1, top / 10);
            gen++;
          }
          function paramLabels() {
            const p = params();
            halfV.textContent = p.half + " h"; thrV.textContent = p.thr.toFixed(1) + "°C";
            rainV.textContent = p.rain.toFixed(2); windV.textContent = p.wind.toFixed(2);
          }
          let ptimer = null;
          const onParam = () => { paramLabels(); if (ptimer) clearTimeout(ptimer); ptimer = setTimeout(() => { computeLoad(); legend(); update(); }, 120); };
          halfIn.oninput = thrIn.oninput = rainIn.oninput = windIn.oninput = onParam;
          // a double click (or double tap) on a slider puts it back where it started
          const START = {half: ["half_life", 12], thr: ["threshold", 27], rain: ["rain_flush", 0.5], wind: ["wind_vent", 0.3]};
          [[halfIn, "half"], [thrIn, "thr"], [rainIn, "rain"], [windIn, "wind"]].forEach(([inp, k]) => {
            inp.title = "double click to reset";
            inp.addEventListener("dblclick", () => { inp.value = cfg[START[k][0]] ?? START[k][1]; onParam(); });
          });

          function loadCells() {
            const u8 = bytesOf(model.get("cells"));
            if (!u8 || !u8.length) return;
            const ids = new BigUint64Array(u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength));
            N = ids.length; hexes = new Array(N); hexIndex = new Map();
            for (let i = 0; i < N; i++) { const h = ids[i].toString(16); hexes[i] = h; hexIndex.set(h, i); }
            const c8 = bytesOf(model.get("cidx"));
            cidx = c8 && c8.length ? new Uint16Array(c8.buffer.slice(c8.byteOffset, c8.byteOffset + c8.byteLength)) : null;
            try { names = JSON.parse(model.get("names") || "[]"); } catch (e) { names = []; }
          }
          function loadFrames() {
            try { cfg = JSON.parse(model.get("config") || "{}"); } catch (e) { cfg = {}; }
            if (!document.fullscreenElement) mapsEl.style.height = (cfg.height || 560) + "px";
            lutI = buildLut(cfg.index_stops || ["#08306b", "#f2f0e6", "#d94801"]);
            lutL = buildLut(cfg.load_stops || ["#000004", "#fcffa4"]);
            const u8 = bytesOf(model.get("frames"));
            if (!u8 || !u8.length || !N) { frames = null; F = 0; return; }
            frames = new Uint8Array(u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength));
            F = Math.floor(frames.length / N);
            fgen++;
            const w8 = bytesOf(model.get("wx"));
            wx = w8 && w8.length === frames.length ? new Uint8Array(w8.buffer.slice(w8.byteOffset, w8.byteOffset + w8.byteLength)) : new Uint8Array(frames.length);
            meansI = new Float32Array(F);
            for (let f = 0; f < F; f++) { let s = 0, n = 0; for (let i = 0; i < N; i++) { const qv = frames[f * N + i]; if (qv !== 255) { s += HI_OF(qv); n++; } } meansI[f] = n ? s / n : NaN; }
            slider.max = String(Math.max(0, F - 1));
            if (frame >= F) frame = 0;
            fpsSel.value = String(cfg.fps || 8);
            if (!halfIn.dataset.set) {  // seed the sliders once from the kernel's defaults
              halfIn.value = cfg.half_life ?? 12; thrIn.value = cfg.threshold ?? 27;
              rainIn.value = cfg.rain_flush ?? 0.5; windIn.value = cfg.wind_vent ?? 0.3; halfIn.dataset.set = "1";
            }
            q(".hp-p-rain").classList.toggle("hp-off", !cfg.has_rain);
            q(".hp-p-wind").classList.toggle("hp-off", !cfg.has_wind);
            paramLabels();
            computeLoad();
            sub.textContent = cfg.subtitle || "";
            legend();
            syncWindow();
            const labels = cfg.labels || [];
            let html = "";
            for (let f = 1; f < labels.length; f++) {
              const d0 = labels[f - 1].slice(0, 10), d1 = labels[f].slice(0, 10);
              if (d0 !== d1) html += `<i style="left:${(f / (F - 1) * 100).toFixed(2)}%"></i>`;
            }
            ticks.innerHTML = F > 1 ? html : "";
          }
          function legend() {
            FIELDS.forEach(field => {
              const lut = field === "load" ? lutL : lutI, p = P[field];
              if (!lut) return;
              const stops = [];
              for (let i = 0; i <= 8; i++) { const j = Math.round(i / 8 * 255) * 3; stops.push(`rgb(${lut[j]},${lut[j+1]},${lut[j+2]}) ${i/8*100}%`); }
              p.grad.style.background = `linear-gradient(90deg, ${stops.join(",")})`;
              if (field === "load") { p.lo.textContent = "0"; p.hi.textContent = "+" + loadHi.toFixed(1) + "°C above " + params().thr.toFixed(1) + "°C"; }
              else { p.lo.textContent = fmtC(cfg.lo); p.hi.textContent = fmtC(cfg.hi); }
            });
          }

          // colour accessors: one attribute re-upload per pane per frame step (updateTriggers)
          const NODATA = t => { t[0] = 40; t[1] = 44; t[2] = 50; t[3] = 60; return t; };
          function fillIndex(d, {index, target}) {
            const qv = frames[frame * N + index];
            if (qv === 255) return NODATA(target);
            let t = (HI_OF(qv) - cfg.lo) / (cfg.hi - cfg.lo); t = t < 0 ? 0 : t > 1 ? 1 : t;
            const j = Math.round(t * 255) * 3;
            target[0] = lutI[j]; target[1] = lutI[j + 1]; target[2] = lutI[j + 2]; target[3] = 225;
            return target;
          }
          function fillLoad(d, {index, target}) {
            const qv = load ? load[frame * N + index] : 255;
            if (qv === 255) return NODATA(target);
            let t = (qv / 10) / loadHi; if (t > 1) t = 1;
            const j = Math.round(t * 255) * 3;
            target[0] = lutL[j]; target[1] = lutL[j + 1]; target[2] = lutL[j + 2]; target[3] = 60 + Math.round(175 * t);
            return target;
          }
          function layers(field) {
            // Dark ground; the ground is the maplibre style, only the film is deck. The
            // two rings (hovered cell, picked cell) are drawn on BOTH panes.
            const out = [];
            if (!(N && frames)) return out;
            out.push(new H3HexagonLayer({
              id: "cells-" + field,
              data: hexes,
              getHexagon: d => d,
              getFillColor: field === "load" ? fillLoad : fillIndex,
              updateTriggers: {getFillColor: field === "load" ? [frame, gen] : [frame, fgen]},
              filled: true, stroked: false, extruded: false, coverage: 1,
              highPrecision: true,
              pickable: false,
              beforeId: LABELS_SLOT,
            }));
            const ring = (id, i, color, w) => new H3HexagonLayer({
              id: id + "-" + field,
              data: [hexes[i]],
              getHexagon: d => d,
              filled: false, stroked: true, extruded: false, highPrecision: true,
              getLineColor: color, lineWidthUnits: "pixels", getLineWidth: w, lineWidthMinPixels: w,
              pickable: false,
              beforeId: LABELS_SLOT,
            });
            // each ring sits on a dark casing: gold alone is lost on the index ramp's warm end
            if (hover >= 0 && hover !== selected) { out.push(ring("hover-case", hover, [11, 13, 16, 255], 3.5)); out.push(ring("hover", hover, [255, 255, 255, 240], 1.5)); }
            if (selected >= 0) { out.push(ring("picked-case", selected, [11, 13, 16, 255], 5)); out.push(ring("picked", selected, [255, 214, 92, 255], 2.5)); }
            return out;
          }

          function stats() {
            FIELDS.forEach(field => {
              const m = field === "load" ? meansL : meansI, p = P[field];
              p.mean.textContent = m && F ? "CONUS mean " + fmtOf(field, m[frame]) : "";
              if (selected >= 0) p.cval.textContent = fmtOf(field, valAt(field, frame, selected));
            });
          }
          function drawChart(field) {
            const chart = P[field].chart;
            if (selected < 0 || !frames || F < 2) return;
            const w = chart.clientWidth || 300, h = chart.height;
            if (chart.width !== w) chart.width = w;
            const g = chart.getContext("2d");
            g.clearRect(0, 0, w, h);
            const L = 48, R = 4, T = 6, B = 14;
            const X = f => L + (w - L - R) * f / (F - 1);
            let lo = Infinity, hi = -Infinity;
            for (let f = 0; f < F; f++) { const v = valAt(field, f, selected); if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }
            if (!Number.isFinite(lo)) return;
            if (field === "load") lo = 0;
            if (hi - lo < 1) { hi += .5; lo = field === "load" ? 0 : lo - .5; }
            const Y = v => T + (h - T - B) * (1 - (v - lo) / (hi - lo));
            g.strokeStyle = "#262c35"; g.lineWidth = 1;
            g.beginPath(); g.moveTo(L, Y(lo)); g.lineTo(w - R, Y(lo)); g.moveTo(L, Y(hi)); g.lineTo(w - R, Y(hi)); g.stroke();
            if (field === "index") {  // the threshold, as a dashed line
              const thr = params().thr;
              if (thr > lo && thr < hi) { g.setLineDash([3, 3]); g.strokeStyle = "#8b929c"; g.beginPath(); g.moveTo(L, Y(thr)); g.lineTo(w - R, Y(thr)); g.stroke(); g.setLineDash([]); }
            }
            g.fillStyle = "#8b929c"; g.font = "11px system-ui, sans-serif"; g.textAlign = "right";
            g.fillText(fmtC(hi), L - 4, Y(hi) + 4); g.fillText(fmtC(lo), L - 4, Y(lo) + 4);
            g.font = "10px system-ui, sans-serif"; g.textAlign = "left"; g.fillText((cfg.labels?.[0] || "").slice(0, 10), L, h - 3);
            g.textAlign = "right"; g.fillText((cfg.labels?.[F - 1] || "").slice(0, 10), w - R, h - 3);
            g.strokeStyle = "#e6c14a"; g.lineWidth = 1.5; g.beginPath();
            let pen = false;
            for (let f = 0; f < F; f++) { const v = valAt(field, f, selected); if (!Number.isFinite(v)) { pen = false; continue; } pen ? g.lineTo(X(f), Y(v)) : g.moveTo(X(f), Y(v)); pen = true; }
            g.stroke();
            g.strokeStyle = "rgba(230,193,74,.55)"; g.lineWidth = 1; g.beginPath(); g.moveTo(X(frame), T); g.lineTo(X(frame), h - B); g.stroke();
            const cv = valAt(field, frame, selected);
            if (Number.isFinite(cv)) { g.fillStyle = "#ffffff"; g.beginPath(); g.arc(X(frame), Y(cv), 3, 0, 6.283); g.fill(); }
          }
          FIELDS.forEach(field => P[field].chart.addEventListener("click", ev => {
            if (F < 2) return;
            const r = P[field].chart.getBoundingClientRect(), L = 48, R = 4;
            const t = ((ev.clientX - r.left) - L) / (r.width - L - R);
            frame = Math.max(0, Math.min(F - 1, Math.round(t * (F - 1)))); update();
          }));

          // THE WINDOW CONTROL: the one thing that crosses back.
          let loading = false;
          const dayCount = () => {
            const a = Date.parse(d0In.value), b = Date.parse(d1In.value);
            return Number.isFinite(a) && Number.isFinite(b) ? Math.round(Math.abs(b - a) / 864e5) + 1 : 0;
          };
          function costOf() {
            const w = cfg.win || {};
            const n = dayCount();
            if (!n || !w.sec_per_day) return w.cost || "3 min";
            const s = Math.round(6 + w.sec_per_day * n);
            return s < 90 ? `${s} s` : `${Math.round(s / 60)} min`;
          }
          function checkWindow() {
            const w = cfg.win || {};
            const n = dayCount(), lim = w.hourly_max || 14;
            let bad = "";
            if (!n) bad = "pick both days";
            else if (n > lim) bad = `${n} days is over the ${lim}-day limit`;
            noteEl.classList.toggle("hp-bad", !!bad);
            if (loading) noteEl.textContent = `loading ${n} days, about ${costOf()}`;
            else noteEl.textContent = bad || `${n} UTC days, ${n * 4} runs x ${w.leads || 6} leads, read about ${costOf()}, limit ${lim} days`;
            loadBtn.disabled = loading || !!bad;
            return !bad;
          }
          function syncWindow() {
            const w = cfg.win;
            if (!w) return;
            d0In.min = d1In.min = w.first || ""; d0In.max = d1In.max = w.last || "";
            if (w.d0) d0In.value = w.d0;
            if (w.d1) d1In.value = w.d1;
            loading = false; loadBtn.textContent = "load";
            checkWindow();
          }
          d0In.onchange = d1In.onchange = checkWindow;
          loadBtn.onclick = () => {
            if (!checkWindow()) return;
            let d0 = d0In.value, d1 = d1In.value;
            if (d1 < d0) [d0, d1] = [d1, d0];
            loading = true; loadBtn.textContent = "loading"; frame = 0; checkWindow();
            model.set("window", JSON.stringify({d0, d1}));
            model.save_changes();
          };
          checkWindow();

          // THE PANEL UNDER THE MAPS folds away; a picked cell brings it back with its lines.
          const toggleBtn = q(".hp-toggle");
          function setCollapsed(c) {
            root.classList.toggle("hp-collapsed", c);
            toggleBtn.textContent = c ? "show panel" : "hide panel";
            if (!c) requestAnimationFrame(() => FIELDS.forEach(drawChart));  // the canvases had no width while folded
          }
          toggleBtn.onclick = () => { setCollapsed(!root.classList.contains("hp-collapsed")); root.focus({preventScroll: true}); };
          function select(i) {
            if (!(i >= 0 && i < N)) return;
            selected = i;
            root.classList.add("hp-picked");
            setCollapsed(false);
            const c = cidx && cidx[i] !== 65535 ? names[cidx[i]] : null;
            cname.textContent = c ? `cell in ${c}` : `cell ${hexes[i]}`;
            update();
          }
          const clearPick = () => { selected = -1; root.classList.remove("hp-picked"); update(); };
          q(".hp-clear").onclick = clearPick;
          function paint() { FIELDS.forEach(f => { const o = P[f].overlay; if (o) o.setProps({layers: layers(f)}); }); }
          function update() {
            paint();
            slider.value = String(frame);
            stampV.textContent = (cfg.labels && cfg.labels[frame]) ? cfg.labels[frame] : "";
            stats(); FIELDS.forEach(drawChart);
          }
          // ONE CLOCK: both panes paint the same frame on the same tick.
          function setPlaying(p) {
            playing = p; playBtn.textContent = p ? "❚❚" : "▶";
            if (timer) { clearInterval(timer); timer = null; }
            if (p && F > 1) timer = setInterval(() => { frame = (frame + 1) % F; update(); }, 1000 / (parseFloat(fpsSel.value) || 8));
          }
          const step = d => { if (F) { frame = (frame + d + F) % F; update(); } };
          const refocus = () => root.focus({preventScroll: true});  // so space stays the clock's, not the last button's
          playBtn.onclick = () => { setPlaying(!playing); refocus(); };
          q(".hp-prev").onclick = () => { step(-1); refocus(); };
          q(".hp-next").onclick = () => { step(1); refocus(); };
          slider.oninput = () => { frame = parseInt(slider.value) || 0; update(); };
          slider.addEventListener("dblclick", () => { frame = 0; update(); });
          fpsSel.onchange = () => { if (playing) setPlaying(true); refocus(); };
          const resizeMaps = () => FIELDS.forEach(f => { try { P[f].map && P[f].map.resize(); } catch (e) {} });
          q(".hp-full").onclick = () => { if (document.fullscreenElement) document.exitFullscreen(); else root.requestFullscreen?.(); refocus(); };
          root.addEventListener("fullscreenchange", () => {
            if (!document.fullscreenElement) mapsEl.style.height = (cfg.height || 560) + "px";
            setTimeout(resizeMaps, 50);
          });
          root.tabIndex = 0;
          root.addEventListener("keydown", ev => {
            const tag = ev.target.tagName, type = ev.target.type;
            if (tag === "SELECT" || (tag === "INPUT" && type !== "range")) return;
            if (ev.key === " ") { ev.preventDefault(); setPlaying(!playing); }
            else if (tag === "INPUT") return;  // arrows on a slider move the slider
            else if (ev.key === "ArrowLeft") { ev.preventDefault(); step(-1); }
            else if (ev.key === "ArrowRight") { ev.preventDefault(); step(1); }
            else if (ev.key === "f" || ev.key === "F") { q(".hp-full").click(); }
            else if (ev.key === "h" || ev.key === "H") { toggleBtn.click(); }
          });

          const rulerText = () => `${N.toLocaleString()} cells, ${F} hourly frames`;
          let syncing = false;
          function cellAt(field, ev) {
            const p = P[field];
            if (!p.map) return -1;
            const r = p.el.getBoundingClientRect();
            try {
              const ll = p.map.unproject([ev.clientX - r.left, ev.clientY - r.top]);
              return hexIndex.get(latLngToCell(ll.lat, ll.lng, cfg.res || 6)) ?? -1;
            } catch (e) { return -1; }
          }
          function makeMap(field) {
            const p = P[field], other = P[field === "load" ? "index" : "load"];
            const map = new maplibregl.Map({
              container: p.el, style: STYLE, bounds: CONUS, fitBoundsOptions: {padding: 10},
              minZoom: 2, maxZoom: 11,
              attributionControl: field === "load" ? {compact: true} : false,
            });
            p.map = map;
            p.overlay = new MapboxOverlay({
              interleaved: true,
              layers: layers(field),
              onError: e => { ruler.textContent = "deck: " + (e && e.message ? e.message : e); },
            });
            map.addControl(p.overlay);
            // no place names on this film: OpenFreeMap has no label-free dark style, so
            // every text layer is hidden once the style is in
            map.on("load", () => {
              try {
                (map.getStyle().layers || []).forEach(l => {
                  if (l.layout && l.layout["text-field"] !== undefined) map.setLayoutProperty(l.id, "visibility", "none");
                });
              } catch (e) { ruler.textContent = "labels: " + e.message; }
              update();
            });
            map.on("error", e => { if (e && e.error && e.error.message) ruler.textContent = "map: " + e.error.message; });
            // one camera
            map.on("move", () => {
              if (syncing || !other.map) return;
              syncing = true;
              try { other.map.jumpTo({center: map.getCenter(), zoom: map.getZoom(), bearing: map.getBearing(), pitch: map.getPitch()}); } catch (e) {}
              syncing = false;
            });
            new ResizeObserver(() => { try { map.resize(); } catch (e) {} }).observe(p.el);
            // click picks (h3-js on the unprojected point), hover rings the cell on both panes
            let down = null, raf = 0, last = null;
            p.pane.addEventListener("pointerdown", ev => { down = [ev.clientX, ev.clientY]; }, true);
            p.pane.addEventListener("pointerup", ev => {
              if (!down) return;
              const moved = Math.hypot(ev.clientX - down[0], ev.clientY - down[1]); down = null;
              if (moved > 4) return;
              const i = cellAt(field, ev);
              if (i >= 0 && i !== selected) select(i); else clearPick();
            }, true);
            p.pane.addEventListener("pointermove", ev => {
              last = ev;
              if (raf || ev.buttons) return;
              raf = requestAnimationFrame(() => { raf = 0; const i = cellAt(field, last); if (i !== hover) { hover = i; paint(); } });
            });
            p.pane.addEventListener("pointerleave", () => { if (hover !== -1) { hover = -1; paint(); } });
          }
          function boot() {
            loadCells(); loadFrames();
            FIELDS.forEach(makeMap);
            ruler.textContent = rulerText();
            update();
            if (cfg.autoplay) setPlaying(true);
          }
          model.on("change:cells", () => { loadCells(); loadFrames(); ruler.textContent = rulerText(); update(); });
          model.on("change:frames", () => { loadFrames(); ruler.textContent = rulerText(); update(); });
          model.on("change:config", () => { loadFrames(); update(); });
          try { boot(); } catch (e) { ruler.textContent = "boot: " + e.message; console.error(e); }
          return () => { setPlaying(false); FIELDS.forEach(f => { try { P[f].map && P[f].map.remove(); } catch (e) {} }); };
        }
        export default {render};
        """
        cells = traitlets.Bytes(b"").tag(sync=True)
        cidx = traitlets.Bytes(b"").tag(sync=True)
        names = traitlets.Unicode("[]").tag(sync=True)
        frames = traitlets.Bytes(b"").tag(sync=True)
        wx = traitlets.Bytes(b"").tag(sync=True)
        config = traitlets.Unicode("{}").tag(sync=True)
        # browser -> kernel, the one thing that crosses back: {"d0","d1"} JSON from the
        # HUD's load button ("" until the first load, meaning the default window).
        window = traitlets.Unicode("").tag(sync=True)

    return (HeatPair,)


@app.cell
async def _(
    BOX,
    CACHE_DIR,
    COUNTY_Z,
    NOT_CONUS,
    OVERTURE_RELEASE,
    PM_BUCKET,
    PM_PATH,
    S3Store,
    asyncio,
    con,
    gzip,
    math,
    np,
    obstore,
    pa,
    pq,
    struct,
):
    # THE COUNTIES, OUT OF ONE PMTILES OBJECT BY RANGED GET. The client and the MVT
    # decode are the interactive notebook's, ported by copy and trimmed of the LRU and
    # the coverage memo: everything here is fetched exactly once. The decode was
    # verified ring-exact against mapbox-vector-tile there before being trusted.
    import time as _ctime

    _ct0 = _ctime.perf_counter()
    # DISK CACHE in the OS temp dir: the dissolved counties never change for a pinned
    # Overture release and BOX, and this fetch + dissolve is ~7.3 s of the ~30 s before
    # the map. First run writes the parquet, every run after reads it (0.0 s).
    # CACHE_DIR = None turns it off.
    import pathlib as _pl

    _cache = (
        _pl.Path(CACHE_DIR) / f"counties-{OVERTURE_RELEASE}-z{COUNTY_Z}-{'-'.join(str(b) for b in BOX)}.parquet"
        if CACHE_DIR
        else None
    )
    _rows, _x0, _y0, _x1, _y1, _t_fetch = [], 0, 0, -1, -1, 0.0
    if _cache is not None and _cache.exists():
        counties = pq.read_table(_cache)
        _how = f"from {_cache}"
    else:
        _pm_store = S3Store(PM_BUCKET, region="us-west-2", skip_signature=True)

        async def _pm_range(a, b):
            """Inclusive byte range [a, b]. obstore's `end` is exclusive."""
            return bytes(
                memoryview(
                    await obstore.get_range_async(_pm_store, PM_PATH, start=a, end=b + 1)
                )
            )

        def _varint(buf, i):
            r = s = 0
            while True:
                c = buf[i]
                i += 1
                r |= (c & 0x7F) << s
                if not c & 0x80:
                    return r, i
                s += 7

        def _parse_dir(buf):
            """A PMTiles v3 directory: four varint columns, tile ids delta-encoded."""
            n, i = _varint(buf, 0)
            ids, last = [0] * n, 0
            for k in range(n):
                v, i = _varint(buf, i)
                last += v
                ids[k] = last
            runs = [0] * n
            for k in range(n):
                runs[k], i = _varint(buf, i)
            lens = [0] * n
            for k in range(n):
                lens[k], i = _varint(buf, i)
            offs = [0] * n
            for k in range(n):
                v, i = _varint(buf, i)
                offs[k] = (offs[k - 1] + lens[k - 1]) if v == 0 and k > 0 else v - 1
            return list(zip(ids, offs, lens, runs))

        def _tile_id(z, x, y):
            """z/x/y -> PMTiles v3 tile id: Hilbert order within a level, levels stacked."""
            acc = sum((1 << t) * (1 << t) for t in range(z))
            n = 1 << z
            d, s = 0, n >> 1
            while s > 0:
                rx = 1 if x & s else 0
                ry = 1 if y & s else 0
                d += s * s * ((3 * rx) ^ ry)
                if ry == 0:
                    if rx == 1:
                        x, y = s - 1 - x, s - 1 - y
                    x, y = y, x
                s >>= 1
            return acc + d

        def _find(entries, tid):
            """Binary search, falling back to the run that COVERS tid."""
            lo, hi = 0, len(entries) - 1
            while lo <= hi:
                m = (lo + hi) // 2
                if tid < entries[m][0]:
                    hi = m - 1
                elif tid > entries[m][0]:
                    lo = m + 1
                else:
                    return entries[m]
            if hi >= 0 and (entries[hi][3] == 0 or tid - entries[hi][0] < entries[hi][3]):
                return entries[hi]
            return None

        _hdr = await _pm_range(0, 126)
        assert _hdr[:7] == b"PMTiles" and _hdr[7] == 3, "not a PMTiles v3 archive"
        _rd_off, _rd_len, _, _, _ld_off, _, _td_off, _ = struct.unpack("<8Q", _hdr[8:72])
        assert COUNTY_Z <= _hdr[101], "COUNTY_Z above the pyramid"
        _root = _parse_dir(gzip.decompress(await _pm_range(_rd_off, _rd_off + _rd_len - 1)))
        _leaf = {}

        def _fields(buf):
            """Iterate (field_number, wire_type, value) over one protobuf message."""
            i, n = 0, len(buf)
            while i < n:
                key, i = _varint(buf, i)
                f, w = key >> 3, key & 0x7
                if w == 0:
                    v, i = _varint(buf, i)
                elif w == 2:
                    ln, i = _varint(buf, i)
                    v = buf[i : i + ln]
                    i += ln
                elif w == 5:
                    v = buf[i : i + 4]
                    i += 4
                elif w == 1:
                    v = buf[i : i + 8]
                    i += 8
                else:
                    raise ValueError(f"wire type {w}")
                yield f, w, v

        def _value(buf):
            """An MVT Value message: exactly one of its fields is set."""
            for f, _w, v in _fields(buf):
                if f == 1:
                    return v.decode("utf-8")
                if f == 2:
                    return struct.unpack("<f", v)[0]
                if f == 3:
                    return struct.unpack("<d", v)[0]
                if f in (4, 5):
                    return v
                if f == 6:
                    return (v >> 1) ^ -(v & 1)
                if f == 7:
                    return bool(v)
            return None

        def _mvt_rings(geom):
            """Packed geometry commands -> rings of (x, y) tile coords, closed."""
            rings, ring = [], None
            x = y = 0
            i, n = 0, len(geom)
            while i < n:
                cmd, i = _varint(geom, i)
                op, count = cmd & 0x7, cmd >> 3
                if op == 1:  # MoveTo: starts a ring
                    for _ in range(count):
                        dx, i = _varint(geom, i)
                        dy, i = _varint(geom, i)
                        x += (dx >> 1) ^ -(dx & 1)
                        y += (dy >> 1) ^ -(dy & 1)
                        ring = [(x, y)]
                        rings.append(ring)
                elif op == 2:  # LineTo
                    for _ in range(count):
                        dx, i = _varint(geom, i)
                        dy, i = _varint(geom, i)
                        x += (dx >> 1) ^ -(dx & 1)
                        y += (dy >> 1) ^ -(dy & 1)
                        ring.append((x, y))
                elif op == 7:  # ClosePath: repeat the first point
                    ring.append(ring[0])
                else:
                    raise ValueError(f"geometry op {op}")
            return rings

        def _area2(ring):
            """Twice the signed shoelace area: >0 marks an exterior ring (tile y is down)."""
            a = 0
            for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
                a += x0 * y1 - x1 * y0
            return a

        def _division_areas(tile_buf):
            """The division_area layer: ([(properties, [(exterior, holes), ...]), ...], extent)."""
            for f, _w, v in _fields(tile_buf):
                if f != 3:  # Tile.layers
                    continue
                name, extent = None, 4096
                keys, values, feats = [], [], []
                for lf, _lw, lv in _fields(v):
                    if lf == 1:
                        name = lv.decode("utf-8")
                    elif lf == 2:
                        feats.append(lv)
                    elif lf == 3:
                        keys.append(lv.decode("utf-8"))
                    elif lf == 4:
                        values.append(_value(lv))
                    elif lf == 5:
                        extent = lv
                if name != "division_area":
                    continue
                out = []
                for fv in feats:
                    tags, gtype, geom = [], 0, b""
                    for ff, _fw, fvv in _fields(fv):
                        if ff == 2:
                            i = 0
                            while i < len(fvv):
                                t, i = _varint(fvv, i)
                                tags.append(t)
                        elif ff == 3:
                            gtype = fvv
                        elif ff == 4:
                            geom = fvv
                    if gtype != 3:  # not a polygon feature
                        continue
                    props = {
                        keys[tags[i]]: values[tags[i + 1]] for i in range(0, len(tags), 2)
                    }
                    polys, cur = [], None
                    for ring in _mvt_rings(geom):
                        if _area2(ring) > 0:
                            cur = (ring, [])
                            polys.append(cur)
                        elif cur is not None:
                            cur[1].append(ring)
                    out.append((props, polys))
                return out, extent
            return [], 4096

        def _feature_wkb(polys, z, x, y, extent):
            """Tile-integer rings -> a lon/lat MultiPolygon WKB, closed-form Web Mercator."""
            n = 1 << z
            parts = []
            for ext, holes in polys:
                rings = []
                for r in (ext, *holes):
                    a = np.asarray(r, dtype=np.float64)
                    pts = np.empty_like(a)
                    pts[:, 0] = (x + a[:, 0] / extent) / n * 360.0 - 180.0
                    pts[:, 1] = np.degrees(
                        np.arctan(np.sinh(np.pi * (1.0 - 2.0 * (y + a[:, 1] / extent) / n)))
                    )
                    rings.append(struct.pack("<I", len(a)) + pts.tobytes())
                parts.append(struct.pack("<BII", 1, 3, len(rings)) + b"".join(rings))
            return struct.pack("<BII", 1, 6, len(parts)) + b"".join(parts)

        _sem = asyncio.Semaphore(32)

        async def _tile_pieces(z, x, y):
            """One tile, walked to through the directories, decoded, filtered to CONUS counties.

            A piece is one county's presence in one tile. The filter runs at decode: county
            subtype only, land only (is_land is always present in this tileset, measured in
            the interactive notebook), country US, region not in NOT_CONUS. `division_id`
            rather than `id`, because `id` names the AREA row and a division can own
            several; joining on the wrong one silently returns zero rows.
            """
            tid, ents = _tile_id(z, x, y), _root
            blob = None
            for _ in range(4):  # root + up to three leaf levels
                e = _find(ents, tid)
                if e is None:
                    break
                if e[3] == 0:
                    lk = (e[1], e[2])
                    if lk not in _leaf:
                        _leaf[lk] = _parse_dir(
                            gzip.decompress(
                                await _pm_range(_ld_off + e[1], _ld_off + e[1] + e[2] - 1)
                            )
                        )
                    ents = _leaf[lk]
                    continue
                async with _sem:
                    blob = await _pm_range(_td_off + e[1], _td_off + e[1] + e[2] - 1)
                break
            pieces = []
            if blob is not None:
                if blob[:2] == b"\x1f\x8b":  # tile_compression says gzip; trust the bytes
                    blob = gzip.decompress(blob)
                feats, extent = _division_areas(blob)
                for props, polys in feats:
                    if props.get("subtype") != "county":
                        continue
                    if props.get("is_land") is not True or not polys:
                        continue
                    if props.get("country") != "US":
                        continue
                    region = (props.get("region") or "").split("-", 1)[-1]
                    if region in NOT_CONUS:
                        continue
                    pieces.append(
                        {
                            "id": props.get("division_id") or props.get("id"),
                            "name": props.get("@name"),
                            "region": region,
                            "wkb": _feature_wkb(polys, z, x, y, extent),
                        }
                    )
            return pieces

        def _mtile(lon, lat, z):
            """lon/lat -> tile x, y at z, clamped to the grid."""
            n = 1 << z
            xx = min(n - 1, max(0, int((lon + 180.0) / 360.0 * n)))
            la = min(85.05, max(-85.05, lat))
            yy = (
                1.0
                - math.log(math.tan(math.radians(la)) + 1.0 / math.cos(math.radians(la)))
                / math.pi
            ) / 2.0
            return xx, min(n - 1, max(0, int(yy * n)))

        _x0, _y0 = _mtile(BOX[0], BOX[3], COUNTY_Z)
        _x1, _y1 = _mtile(BOX[2], BOX[1], COUNTY_Z)
        _parts = await asyncio.gather(
            *(
                _tile_pieces(COUNTY_Z, xx, yy)
                for yy in range(_y0, _y1 + 1)
                for xx in range(_x0, _x1 + 1)
            )
        )
        _rows = [p for tp in _parts for p in tp]
        _t_fetch = _ctime.perf_counter() - _ct0

        # THE SEAM DISSOLVE. Tile geometry arrives clipped, so one county is several pieces
        # and the clip edges are straight lines the stroke would draw across the map.
        # Union-ing per division removes every interior edge; the tile buffer (pieces
        # overlap slightly past each tile edge) is what makes the union clean.
        #
        # con.register, NOT the replacement scan the interactive notebook leans on: marimo
        # mangles underscore-prefixed cell locals to make them cell-private, so the frame
        # name never matches the SQL name and DuckDB reports the table as missing.
        _pieces = pa.table(
            {
                "id": pa.array([r["id"] for r in _rows]),
                "name": pa.array([r["name"] for r in _rows]),
                "region": pa.array([r["region"] for r in _rows]),
                "wkb": pa.array([r["wkb"] for r in _rows], pa.binary()),
            }
        )
        con.register("pm_pieces", _pieces)
        counties = con.sql("""
            SELECT id,
                   any_value(name)   AS name,
                   any_value(region) AS region,
                   CAST(ST_AsWKB(ST_Union_Agg(ST_GeomFromWKB(wkb))) AS BLOB) AS wkb
            FROM pm_pieces
            GROUP BY id
        """).to_arrow_table()
        con.unregister("pm_pieces")
        _how = "fetched"
        if _cache is not None:
            _cache.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(counties, _cache)

    county_stats = (
        f"{counties.num_rows:,} counties {_how} · "
        + (
            f"{(_x1 - _x0 + 1) * (_y1 - _y0 + 1)} tiles at z{COUNTY_Z} · {len(_rows):,} pieces · fetch {_t_fetch:.1f}s, with dissolve "
            if _rows
            else ""
        )
        + f"{_ctime.perf_counter() - _ct0:.1f}s"
    )
    return counties, county_stats


@app.cell
def _(FORECAST_BUCKET, FORECAST_PREFIX, MIRROR_DIR, READ_RAIN, READ_WIND, S3Store, np, xr):
    # THE STORE, opened once; only metadata, the init list and the 2-D lat/lon are read
    # here. Plain Zarr v3 through obstore (anonymous), like every other read in this
    # repo. NO DASK: chunks=None leaves it lazily indexed and xarray-sql cuts it into
    # blocks itself (fold cell). Which init to fold is the window cell's decision.
    import time as _stime

    from zarr.storage import ObjectStore as _ZStore

    _st0 = _stime.perf_counter()
    _zs = _ZStore(
        S3Store(FORECAST_BUCKET, prefix=FORECAST_PREFIX, region="us-west-2", skip_signature=True),
        read_only=True,
    )
    # DISK MIRROR of the byte ranges read (join/hrrr_mirror.py, the one the forecast
    # notebooks here use): a run's shard never changes once written, so a window costs
    # the wire once, ever, and a later kernel reads it from disk. Every chunk key
    # (var/c/<init>/0/0/0) but the newest init's is mirrored. MIRROR_DIR None disables,
    # and so does running the notebook on its own without join/ beside it (molab).
    try:
        import pathlib as _pathlib
        import sys as _sys

        _root = _pathlib.Path(__file__).resolve().parent if "__file__" in globals() else _pathlib.Path.cwd()
        _sys.path.insert(0, str(_root / "join"))
        import hrrr_mirror as _hrrr_mirror
    except ImportError:
        _hrrr_mirror = None
    if MIRROR_DIR and _hrrr_mirror is not None:
        import zarr as _zarr

        _n_init = _zarr.open_group(_zs, mode="r")["init_time"].shape[0]

        def _mirrorable(key, _n=_n_init):
            p = key.split("/")
            return len(p) == 6 and p[1] == "c" and p[2].isdigit() and int(p[2]) < _n - 1

        _zs = _hrrr_mirror.MirrorStore(_zs, MIRROR_DIR, _mirrorable)
    ds_all = xr.open_zarr(_zs, consolidated=True, chunks=None)
    VARS = ["temperature_2m", "relative_humidity_2m"]
    if READ_RAIN:
        VARS.append("precipitation_surface")
    if READ_WIND:
        VARS += ["wind_u_10m", "wind_v_10m"]
    all_inits = ds_all["init_time"].values.astype("datetime64[m]")
    lead_hours = ds_all["lead_time"].values.astype("timedelta64[h]")
    lat = ds_all["latitude"].values.astype("float64")
    lon = ds_all["longitude"].values.astype("float64")
    grid_y = ds_all["y"].values
    grid_x = ds_all["x"].values
    source_note = "HRRR 48 h forecast"
    store_stats = (
        f"{source_note} (source.coop) · {len(VARS)} variables · grid {lat.shape[1]}x{lat.shape[0]} px · "
        f"{all_inits.size:,} inits, {np.datetime_as_string(all_inits[0])} to {np.datetime_as_string(all_inits[-1])}Z "
        f"(6-hourly) · {lead_hours.size} leads · open {_stime.perf_counter() - _st0:.1f}s"
    )
    return (
        VARS,
        all_inits,
        ds_all,
        grid_x,
        grid_y,
        lat,
        lead_hours,
        lon,
        source_note,
        store_stats,
    )

@app.cell
def _(DAYS, HOURLY_MAX_DAYS, LEADS_PER_RUN, VARS, all_inits, ds_all, film, json, mo, np):
    # THE WINDOW: the HUD's `window` trait once "load" has been pressed ({"d0","d1"}),
    # else the DAYS constant. Read off the widget, not film.value (that packs every
    # synced trait, the frame bytes included). Over the limit stops with the reason.
    # The window is STITCHED: every run whose init falls inside it contributes its
    # first LEADS_PER_RUN leads, and the fold labels each row by valid time
    # (init + lead), so the film is one continuous hourly series.
    import datetime as _wdt

    _L = int(LEADS_PER_RUN)
    _last = all_inits[-1].astype("datetime64[D]").astype(_wdt.date)
    _first = all_inits[0].astype("datetime64[D]").astype(_wdt.date)
    _req = {}
    try:
        _req = json.loads(film.widget.window or "{}")
    except ValueError:
        _req = {}
    if _req.get("d0") and _req.get("d1"):
        _d0 = max(_first, min(_last, _wdt.date.fromisoformat(_req["d0"])))
        _d1 = max(_first, min(_last, _wdt.date.fromisoformat(_req["d1"])))
    elif isinstance(DAYS, tuple):
        _d0, _d1 = (max(_first, min(_last, _wdt.date.fromisoformat(d))) for d in DAYS)
    else:
        _d0, _d1 = _last - _wdt.timedelta(days=DAYS - 1), _last
    if _d1 < _d0:
        _d0, _d1 = _d1, _d0
    n_days = (_d1 - _d0).days + 1
    mo.stop(
        n_days > HOURLY_MAX_DAYS,
        mo.md(f"**{n_days} days is over the {HOURLY_MAX_DAYS}-day limit.** Shorten the window."),
    )
    t0 = np.datetime64(_d0.isoformat()).astype("datetime64[ns]")
    t1 = (np.datetime64(_d1.isoformat()) + np.timedelta64(23, "h")).astype("datetime64[ns]")
    _ks = np.nonzero((all_inits.astype("datetime64[ns]") >= t0) & (all_inits.astype("datetime64[ns]") <= t1))[0]
    mo.stop(_ks.size == 0, mo.md("**No runs in that window.** The archive runs from "
                                 f"{_first.isoformat()} to {_last.isoformat()}."))
    _k0, _k1 = int(_ks[0]), int(_ks[-1])
    n_runs = _k1 - _k0 + 1
    # the runs as a lazy 4-D cube (run, lead, y, x); lead as an int hour so it crosses
    # xarray-sql as a plain column, valid time is computed after the fold
    cube_all = (
        ds_all[VARS]
        .isel(init_time=slice(_k0, _k1 + 1), lead_time=slice(0, _L))
        .rename({"init_time": "it", "lead_time": "lh"})
        .assign_coords(lh=np.arange(_L, dtype="int32"))
    )
    all_times = (
        all_inits[_k0:_k1 + 1].astype("datetime64[ns]")[:, None] + np.arange(_L).astype("timedelta64[h]").astype("timedelta64[ns]")[None, :]
    ).ravel().astype("datetime64[m]")
    t1 = min(t1, all_times[-1].astype("datetime64[ns]"))
    # the read's cost, stated to the HUD: one shard per run per variable, ~1.5 s
    # each from home once DataFusion overlaps them (measured 2026-09-14: the East
    # dome week, 28 runs x 5 variables, 216.8 s fold end to end, 35.4M rows)
    _sec_per_day = 4 * len(VARS) * 1.5
    read_cost_s = int(round(6 + _sec_per_day * n_days))
    win_cfg = {
        "first": _first.isoformat(),
        "last": _last.isoformat(),
        "d0": _d0.isoformat(),
        "d1": _d1.isoformat(),
        "hourly_max": HOURLY_MAX_DAYS,
        "cost": f"{read_cost_s} s" if read_cost_s < 90 else f"{read_cost_s / 60:.0f} min",
        "sec_per_day": _sec_per_day,
        "leads": _L,
    }
    init_note = f"{n_runs} runs x leads 0-{_L - 1}"
    window_note = (
        f"{np.datetime_as_string(t0, unit='m').replace('T', ' ')}Z to "
        f"{np.datetime_as_string(t1, unit='m').replace('T', ' ')}Z"
    )
    return all_times, cube_all, init_note, n_days, n_runs, t0, t1, win_cfg, window_note

@app.cell
def _(
    RES,
    con,
    coordinates_to_cells,
    counties,
    grid_x,
    grid_y,
    lat,
    lon,
    mo,
    np,
    pa,
):
    # Res 7 and finer: stop with the reasons instead of starting a fold that never
    # ends. Coverage: a res 7 hex (5.2 km2) is smaller than the 3 km pixel, so only
    # 879k of the 1.47M land cells contain a pixel centre and ~40% of the map is
    # holes (flown 2026-08-17: "a lot of gaps between the cells"); res 8 is 7x
    # worse. Memory: a week at res 7 is 248M (hour, cell) answers, ~35 GB of
    # DataFusion aggregate state (res 6: 35M, ~5 GB with the pool).
    mo.stop(
        int(RES) >= 7,
        mo.md(
            f"**RES {RES} is too fine for this notebook: set RES to 6 or coarser.** "
            f"Two reasons. Coverage: a res {RES} hexagon is smaller than the 3 km HRRR pixel, so most "
            f"cells hold no pixel centre and the map is mostly holes (res 7: ~40% empty). Memory: a "
            f"week at res 7 is 248M hour-cell answers, ~35 GB of aggregate state in the kernel "
            f"(res 6: 35M, ~5 GB). Res 6 over a smaller BOX is the way to see finer weather."
        ),
    )
    # PIXEL -> CELL, ONCE, AND THE LAND MASK. Cell per pixel from the store's own
    # lat/lon; CONUS land = the res 6 cells whose centre falls in a county (DuckDB
    # polyfill, 'center' rule, so each cell has exactly one county, which is the click
    # readout's name). Only pixels in land cells enter the fold. The forecast store is
    # 24 chunks of 265x300 px per init (4 rows x 6 columns); the ones that touch no
    # land cell are named in a y/x range predicate (one term per block row, runs of
    # touching block columns) that xarray-sql pushes down as partition pruning, so
    # those chunks are never fetched. Coarser chunks than the analysis (45x45), so
    # the pruning buys less; the read is seconds either way.
    import time as _ptime

    _pt0 = _ptime.perf_counter()
    con.register("conus_divs", counties)
    _mapping = con.sql(
        """
        WITH parts AS (
            SELECT id, UNNEST(ST_Dump(ST_GeomFromWKB(wkb))).geom AS g FROM conus_divs
        ), filled AS (
            SELECT id, UNNEST(
                       h3_polygon_wkb_to_cells_experimental(ST_AsWKB(g), ?, 'center')
                   ) AS hex
            FROM parts
        )
        SELECT hex, any_value(id) AS id FROM filled GROUP BY hex ORDER BY hex
        """,
        params=[int(RES)],
    ).to_arrow_table()
    con.unregister("conus_divs")
    _t_fill = _ptime.perf_counter() - _pt0

    # THE CELL IDS ARE GENERATED IN THE SQL FOLD, by the h3ronpy UDF registered in
    # DataFusion (fold cell); this cell only decides WHICH pixels are CONUS land, so
    # the same h3ronpy call runs here once for the mask and the block predicate. The
    # lookup carries each land pixel's lat/lon for the UDF, not a precomputed cell.
    _ny, _nx = lat.shape
    _hex = np.asarray(coordinates_to_cells(lat.ravel(), lon.ravel(), int(RES)))
    cells = _mapping["hex"].to_numpy().astype(np.uint64)  # sorted: the film's row order
    _land = np.isin(_hex, cells).reshape(_ny, _nx)
    _flat = _land.ravel()
    pix2h = pa.table(
        {
            "y": pa.array(np.repeat(grid_y, _nx)[_flat]),
            "x": pa.array(np.tile(grid_x, _ny)[_flat]),
            "lat": pa.array(lat.ravel()[_flat]),
            "lon": pa.array(lon.ravel()[_flat]),
        }
    )
    # county index per cell (uint16, 65535 = none), names in county-table order
    _cid = counties["id"].to_pylist()
    _pos = {i: k for k, i in enumerate(_cid)}
    cell_county = np.fromiter((_pos.get(i, 65535) for i in _mapping["id"].to_pylist()), dtype=np.uint16, count=cells.size)
    county_names = [f"{n}, {r}" for n, r in zip(counties["name"].to_pylist(), counties["region"].to_pylist())]

    # the land-block predicate for partition pruning
    _BY, _BX = 265, 300  # the forecast store's inner chunk (y, x)
    _by, _bx = -(-_ny // _BY), -(-_nx // _BX)
    _terms, n_land_blocks = [], 0
    for _j in range(_by):
        _cols = [_i for _i in range(_bx) if _land[_j * _BY:(_j + 1) * _BY, _i * _BX:(_i + 1) * _BX].any()]
        if not _cols:
            continue
        _runs, _s, _prev = [], _cols[0], _cols[0]
        for _i in _cols[1:]:
            if _i != _prev + 1:
                _runs.append((_s, _prev))
                _s = _i
            _prev = _i
        _runs.append((_s, _prev))
        _ys = (float(grid_y[_j * _BY]), float(grid_y[min((_j + 1) * _BY, _ny) - 1]))
        _xr = []
        for _a, _b in _runs:
            _xs = (float(grid_x[_a * _BX]), float(grid_x[min((_b + 1) * _BX, _nx) - 1]))
            _xr.append(f"cube.x BETWEEN {min(_xs)} AND {max(_xs)}")
            n_land_blocks += _b - _a + 1
        _terms.append(f"(cube.y BETWEEN {min(_ys)} AND {max(_ys)} AND ({' OR '.join(_xr)}))")
    land_pred = " OR ".join(_terms)
    pix_stats = (
        f"{cells.size:,} res {RES} land cells (polyfill {_t_fill:.1f}s) · "
        f"{pix2h.num_rows:,} of {_hex.size:,} pixels on CONUS land · "
        f"{n_land_blocks} of {_by * _bx} store blocks touch land · {_ptime.perf_counter() - _pt0:.1f}s"
    )
    return cell_county, cells, county_names, land_pred, pix2h, pix_stats


@app.cell
def _():
    # Kernel-side memo across window loads: the last folded window and its table.
    HOLD = {"key": None, "cell_hour": None, "stats": ""}
    return (HOLD,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Where the wait in the next cell comes from

    The cell below is the one that reads. xarray-sql registers the Zarr lazily;
    nothing is fetched until the query runs, and then DataFusion pulls the
    land-touching chunks of every run in the window (24 chunks of 265 x 300 px per run
    per variable, each chunk holding all 49 leads even though six are kept) straight
    from source.coop through obstore, and joins and aggregates the rows as they
    arrive, one pipelined stream.

    Against the analysis notebook: there a window pays every filled hour of a 90-day
    time chunk (~30 s young, ~2 min full, two variables); here a window pays one shard
    per run per variable, whatever its dates (a week with all five variables is 28 x 5
    shards, the counties film measured 2.2 s per shard from home, so minutes; the
    panel states the estimate). The aggregate is hours x 210k cells at res 6, a few
    seconds. What would move it: running near the bucket (molab, us-west-2). Not:
    another S3 client, more concurrency, or a different query engine.
    """)
    return


@app.cell
def _(
    HOLD,
    MEM_POOL_GB,
    READ_RAIN,
    READ_WIND,
    RES,
    VARS,
    XarrayContext,
    coordinates_to_cells,
    cube_all,
    land_pred,
    pa,
    pix2h,
    t0,
    t1,
):
    # THE FOLD, ONE STATEMENT, STRAIGHT OFF THE CUBE: the join to pix2h on the grid
    # coordinates is the H3 fold (each pixel carries its cell), the group by averages
    # the ~4 pixels per cell per hour, and the land predicate prunes the blocks that
    # touch no land before any byte is fetched. Blocks are the store's own chunks
    # (one run x its kept leads x 265 x 300 px; 24 per run, 672 for a week), so each
    # shard is decoded once; the GROUP BY is (run, lead, cell) and the valid time is
    # put on afterwards in numpy (init + lead), which is the `t` the frames read. Wind
    # speed is averaged per pixel (sqrt(u^2+v^2)), not from the mean vector; rain rate
    # x 3600 is mm in the hour. Floats, not doubles, in the output: 35M rows at res 6.
    #
    # Memoised on the window (and the res): re-loading the same dates never refetches.
    import time as _ftime

    # RES and the cell count are in the key: a memo from another res served against a
    # new cell list indexes past its end (IndexError in the frames cell, 2026-08-17).
    _key = (int(RES), int(pix2h.num_rows), tuple(VARS), str(t0), str(t1))
    if HOLD["key"] == _key and HOLD["cell_hour"] is not None:
        cell_hour = HOLD["cell_hour"]
        fold_stats = HOLD["stats"] + " (memo)"
    else:
        _ft0 = _ftime.perf_counter()
        _cube = cube_all
        _L = int(_cube.sizes["lh"])
        _hours = int(_cube.sizes["it"]) * _L
        # A fair spill pool (MEM_POOL_GB, constants cell): the final aggregate holds one
        # entry per (hour, cell) answer until the last block lands (35M at res 6), and
        # over the pool it spills to the temp dir instead of growing; measured ~5 GB
        # process peak against 9.5 without, same ~28 s at res 6. A pool that is small
        # against the state (any pool at res 7) spills nearly everything and crawls.
        from datafusion import RuntimeEnvBuilder as _RTB, SessionConfig as _SC

        ctx = (
            XarrayContext(_SC(), _RTB().with_fair_spill_pool(int(MEM_POOL_GB * (1 << 30))))
            if MEM_POOL_GB
            else XarrayContext()
        )
        # Broadcast the pixel lookup to every block partition (CollectLeft) instead of
        # re-hashing the cube by (y, x): with the default Partitioned join a cell's
        # pixels scatter across partitions and every partition's partial aggregate
        # holds its own copy of nearly every (hour, cell) group (measured at res 6:
        # 17 GB against 9.5 GB). pix2h is ~10 MB, above the 1 MB / 128k-row defaults.
        ctx.sql("SET datafusion.optimizer.hash_join_single_partition_threshold = 268435456")
        ctx.sql("SET datafusion.optimizer.hash_join_single_partition_threshold_rows = 16777216")
        # THE H3 UDF: lat/lon -> cell as a whole-column h3ronpy call inside DataFusion,
        # the repo's fold (xsql-deforest-divisions.py registers the same one). The
        # join on (y, x) attaches each pixel-hour's lat/lon from the land lookup; the
        # UDF labels it; the GROUP BY averages the pixels that share a cell.
        from datafusion import udf as _udf

        ctx.register_udf(
            _udf(
                lambda la, lo, r: pa.array(
                    coordinates_to_cells(la.to_numpy(), lo.to_numpy(), r[0].as_py())
                ),
                [pa.float64(), pa.float64(), pa.int32()],
                pa.uint64(),
                "stable",
                name="h3_latlng_to_cell",
            )
        )
        ctx.from_arrow(pix2h, name="pix2h")
        ctx.from_dataset("cube", _cube, chunks={"it": 1, "lh": _L, "y": 265, "x": 300})
        _cols = [
            "CAST(avg(CAST(temperature_2m AS DOUBLE)) AS FLOAT) AS tc",
            "CAST(avg(CAST(relative_humidity_2m AS DOUBLE)) AS FLOAT) AS rh",
        ]
        if READ_RAIN:
            _cols.append("CAST(avg(CAST(precipitation_surface AS DOUBLE)) * 3600 AS FLOAT) AS mm")
        if READ_WIND:
            _cols.append(
                "CAST(avg(sqrt(CAST(wind_u_10m AS DOUBLE) * wind_u_10m + CAST(wind_v_10m AS DOUBLE) * wind_v_10m)) AS FLOAT) AS ws"
            )
        _raw = ctx.sql(f"""
            SELECT it, lh, h3_latlng_to_cell(lat, lon, CAST({int(RES)} AS INT)) AS hex, {", ".join(_cols)}
            FROM cube JOIN pix2h USING (y, x)
            WHERE temperature_2m = temperature_2m AND ({land_pred})
            GROUP BY 1, 2, 3
        """).to_arrow_table()
        # valid time = init + lead hour; (it, lh) -> t, the column the frames cell reads
        _t = _raw["it"].to_numpy().astype("datetime64[ns]") + _raw["lh"].to_numpy().astype("timedelta64[h]").astype("timedelta64[ns]")
        cell_hour = pa.table({"t": pa.array(_t), **{c: _raw[c] for c in _raw.column_names if c not in ("it", "lh")}})
        fold_stats = (
            f"{_hours} hours from {int(_cube.sizes['it'])} runs · {len(VARS)} variables · {cell_hour.num_rows:,} cell-hour rows · "
            f"fold {_ftime.perf_counter() - _ft0:.1f}s"
        )
        HOLD["key"], HOLD["cell_hour"], HOLD["stats"] = _key, cell_hour, fold_stats
    return cell_hour, fold_stats


@app.cell
def _(PIVOT, SPAN, cell_hour, cells, np):
    # THE FRAME MATRICES: F hours x N cells, in `cells` order (sorted ids; the widget
    # indexes by row). Heat index from the cell-mean temperature and humidity (NWS:
    # Steadman's simple formula, the Rothfusz regression once the mean of it and T
    # reaches 80 F, with the two RH adjustments), quantised to uint8 in 0.5 degC steps
    # from -40 (255 = no data). Wind and rain, if read, packed in one byte: wind m/s
    # rounded (0..15) in the high nibble, rain in 0.5 mm/h steps (0..7.5) in the low.
    # One ramp for the film's heat index: pivot at the median, span to the wider of
    # p2/p98, unless PIVOT/SPAN pin them.
    def _heat_index_c(tc, rh):
        T = tc * 9.0 / 5.0 + 32.0
        hi = 0.5 * (T + 61.0 + (T - 68.0) * 1.2 + rh * 0.094)
        m = (hi + T) / 2.0 >= 80.0
        T2, R2 = T[m], rh[m]
        h = (
            -42.379 + 2.04901523 * T2 + 10.14333127 * R2 - 0.22475541 * T2 * R2
            - 0.00683783 * T2 * T2 - 0.05481717 * R2 * R2 + 0.00122874 * T2 * T2 * R2
            + 0.00085282 * T2 * R2 * R2 - 0.00000199 * T2 * T2 * R2 * R2
        )
        a1 = (R2 < 13) & (T2 >= 80) & (T2 <= 112)
        h[a1] -= ((13 - R2[a1]) / 4.0) * np.sqrt((17 - np.abs(T2[a1] - 95.0)) / 17.0)
        a2 = (R2 > 85) & (T2 >= 80) & (T2 <= 87)
        h[a2] += ((R2[a2] - 85.0) / 10.0) * ((87.0 - T2[a2]) / 5.0)
        hi[m] = h
        return (hi - 32.0) * 5.0 / 9.0

    _t = cell_hour["t"].to_numpy()
    _fkeys = np.unique(_t)
    _fi = np.searchsorted(_fkeys, _t)
    _ci = np.searchsorted(cells, cell_hour["hex"].to_numpy().astype(np.uint64))
    F, N = _fkeys.size, cells.size
    _tc = np.full((F, N), np.nan, dtype=np.float32)
    _rh = np.full((F, N), np.nan, dtype=np.float32)
    _tc[_fi, _ci] = cell_hour["tc"].to_numpy()
    _rh[_fi, _ci] = cell_hour["rh"].to_numpy()
    _hi = _heat_index_c(_tc.astype(np.float64), _rh.astype(np.float64)).astype(np.float32)
    _ok = np.isfinite(_hi)
    hi_q = np.full((F, N), 255, dtype=np.uint8)
    hi_q[_ok] = np.clip(np.rint((_hi[_ok] + 40.0) * 2.0), 0, 254).astype(np.uint8)
    wx_q = np.zeros((F, N), dtype=np.uint8)
    has_rain = "mm" in cell_hour.column_names
    has_wind = "ws" in cell_hour.column_names
    if has_rain:
        _mm = np.zeros((F, N), dtype=np.float32)
        _mm[_fi, _ci] = cell_hour["mm"].to_numpy()
        wx_q |= np.clip(np.rint(np.nan_to_num(_mm) * 2.0), 0, 15).astype(np.uint8)
    if has_wind:
        _ws = np.zeros((F, N), dtype=np.float32)
        _ws[_fi, _ci] = cell_hour["ws"].to_numpy()
        wx_q |= (np.clip(np.rint(np.nan_to_num(_ws)), 0, 15).astype(np.uint8) << 4)
    frame_labels = [np.datetime_as_string(t, unit="m").replace("T", " ") + "Z" for t in _fkeys]

    _vals = _hi[_ok]
    _mid = float(np.median(_vals)) if PIVOT is None else float(PIVOT)
    _span = (
        float(max(_mid - np.percentile(_vals, 2), np.percentile(_vals, 98) - _mid))
        if SPAN is None
        else float(SPAN)
    )
    ramp_lo, ramp_mid, ramp_hi = _mid - _span, _mid, _mid + _span
    frame_stats = (
        f"{F} frames x {N:,} cells · heat index {np.nanmin(_hi):.1f} to {np.nanmax(_hi):.1f} °C · "
        f"ramp {ramp_lo:.1f} / {ramp_mid:.1f} / {ramp_hi:.1f} · "
        f"{hi_q.nbytes / 1e6:.0f} MB per field to the browser"
    )
    return (
        frame_labels,
        frame_stats,
        has_rain,
        has_wind,
        hi_q,
        ramp_hi,
        ramp_lo,
        ramp_mid,
        wx_q,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Using the maps.** Space plays, arrows step, drag the slider to scrub, `F` or ⛶
    goes fullscreen. `H` or "hide panel" (top right) folds the panel under the maps away;
    clicking a cell brings it back with that cell's lines. Drag or zoom either map and the other follows. Hover rings the
    cell on both maps; click a cell for its two lines over the window (the dashed line
    on the left chart is the threshold), click it again or empty ground to clear; click
    a chart to jump to that hour. The sliders under the right map are the accumulator.
    The window under the left map takes UTC days, inclusive, up to 14; load refetches
    every run inside it and refolds.
    """)
    return


@app.cell
def _(HeatPair, cell_county, cells, county_names, json, mo):
    # THE WIDGET, BUILT ONCE with the cell ids and nothing else. Frames and config are
    # set from the wiring cell below, so a window change never rebuilds the map.
    film = mo.ui.anywidget(
        HeatPair(
            cells=cells.astype("<u8").tobytes(),
            cidx=cell_county.astype("<u2").tobytes(),
            names=json.dumps(county_names),
        )
    )
    film
    return (film,)


@app.cell
def _(
    FPS,
    HALF_LIFE,
    INDEX_STOPS,
    LOAD_STOPS,
    MAP_HEIGHT,
    RAIN_FLUSH,
    RES,
    THRESHOLD,
    WIND_VENT,
    film,
    fold_stats,
    frame_labels,
    frame_stats,
    has_rain,
    has_wind,
    hi_q,
    init_note,
    json,
    n_days,
    ramp_hi,
    ramp_lo,
    ramp_mid,
    source_note,
    win_cfg,
    window_note,
    wx_q,
):
    # THE WIRING: re-runs on every window change and only pushes JSON + bytes at the
    # existing widget. Config, then wx, then frames: the JS recomputes on frames.
    film.config = json.dumps(
        {
            "labels": frame_labels,
            "lo": ramp_lo,
            "mid": ramp_mid,
            "hi": ramp_hi,
            "index_stops": INDEX_STOPS,
            "load_stops": LOAD_STOPS,
            "threshold": THRESHOLD,
            "half_life": HALF_LIFE,
            "rain_flush": RAIN_FLUSH,
            "wind_vent": WIND_VENT,
            "has_rain": has_rain,
            "has_wind": has_wind,
            "res": RES,
            "fps": FPS,
            "height": MAP_HEIGHT,
            "title": f"{source_note}, stitched",
            "subtitle": f"{source_note}, {window_note}, {n_days} days, hourly, {init_note}.",
            "meta": f"{fold_stats} · {frame_stats}",
            "win": win_cfg,
            "autoplay": False,
        }
    )
    film.wx = wx_q.tobytes() if (has_rain or has_wind) else b""
    film.frames = hi_q.tobytes()
    return


@app.cell
def _(county_stats, fold_stats, frame_stats, mo, pix_stats, store_stats):
    mo.md(
        "<br>".join(
            f"<span style='color:#8b929c;font-size:.85em'>{s}</span>"
            for s in (store_stats, county_stats, pix_stats, fold_stats, frame_stats)
        )
    )
    return


if __name__ == "__main__":
    app.run()
