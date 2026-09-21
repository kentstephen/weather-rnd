"""Headless flight of heat-pair.py: marimo run + playwright. Screenshots to shots/,
console to stdout. Usage: uv run python fly_heat.py [notebook.py] [shots_dir]"""
import pathlib, subprocess, sys, time
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).parent
NB = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "heat-pair.py"
SHOTS = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "shots"
SHOTS.mkdir(exist_ok=True)
PORT = 2739
srv = subprocess.Popen([sys.executable, "-m", "marimo", "run", str(NB), "--headless", "--no-token", "--port", str(PORT)],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
time.sleep(3)
logs = []


def txt(pg, sel):
    try:
        return pg.locator(sel).first.inner_text()
    except Exception as e:  # noqa: BLE001
        return f"<{sel}: {e.__class__.__name__}>"


def setr(pg, sel, v):
    pg.locator(sel).fill(str(v)); pg.locator(sel).dispatch_event("input"); time.sleep(2.5)


try:
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1500, "height": 1000})
        pg.on("console", lambda m: logs.append(f"[{m.type}] {m.text}"))
        pg.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))
        pg.goto(f"http://127.0.0.1:{PORT}", wait_until="load")
        t = time.perf_counter()
        pg.wait_for_selector(".hp-pane-load canvas", timeout=3_000_000)
        deadline = time.time() + 3000
        while time.time() < deadline:
            v = txt(pg, ".hp-mean-load")
            if v.startswith("CONUS"):
                break
            time.sleep(5)
        else:
            print("WAIT FAILED; ruler:", txt(pg, ".hp-ruler"))
            pg.screenshot(path=str(SHOTS / "hp-00-fail.png")); print("\n".join(logs[-40:])); raise SystemExit(1)
        time.sleep(10)
        print(f"booted in {time.perf_counter()-t:.1f}s; {txt(pg, '.hp-foot')}")
        pg.locator(".hp").scroll_into_view_if_needed()
        for f in (21, 69, 117, 165):
            setr(pg, ".hp-frame", f)
            print(f"{txt(pg, '.hp-stamp')}: index {txt(pg, '.hp-mean-index')} | sustained {txt(pg, '.hp-mean-load')} | ramp top {txt(pg, '.hp-hi-load')}")
            pg.locator(".hp").screenshot(path=str(SHOTS / f"hp-01-frame-{f}.png"))
        # hover on the left pane: the ring must show on both
        box = pg.locator(".hp-pane-index").bounding_box()
        x, y = box["x"] + box["width"] * 0.48, box["y"] + box["height"] * 0.42
        pg.mouse.move(x, y); time.sleep(1.5)
        pg.locator(".hp").screenshot(path=str(SHOTS / "hp-02-hover.png"))
        pg.mouse.click(x, y); time.sleep(2)
        for sel in (".hp-cname", ".hp-cval-index", ".hp-cval-load"):
            print(sel, "=", txt(pg, sel))
        pg.locator(".hp").screenshot(path=str(SHOTS / "hp-03-pick.png"))
        # hide the panel, then a click on a cell must bring it back
        pg.locator(".hp-clear").click(); time.sleep(0.5)
        pg.locator(".hp-toggle").click(); time.sleep(1)
        print("panel hidden:", not pg.locator(".hp-strip").is_visible())
        pg.locator(".hp").screenshot(path=str(SHOTS / "hp-03b-hidden.png"))
        pg.mouse.click(x + 30, y + 20); time.sleep(2)
        print("panel back after a click:", pg.locator(".hp-strip").is_visible(), "|", txt(pg, ".hp-cname"))
        pg.locator(".hp").screenshot(path=str(SHOTS / "hp-03c-back.png"))
        # camera sync: wheel-zoom the RIGHT pane, read both maps' zoom off the canvases
        rb = pg.locator(".hp-pane-load").bounding_box()
        pg.mouse.move(rb["x"] + rb["width"] * 0.5, rb["y"] + rb["height"] * 0.45)
        for _ in range(6):
            pg.mouse.wheel(0, -400); time.sleep(0.4)
        time.sleep(6)
        pg.locator(".hp").screenshot(path=str(SHOTS / "hp-04-zoomed.png"))
        # half-life to 48 h
        setr(pg, ".hp-half", 48)
        pg.locator(".hp-half").dblclick(); time.sleep(2.5)
        print(f"after double click: half-life {txt(pg, '.hp-halfv')}")
        setr(pg, ".hp-half", 48)
        print(f"half-life 48: sustained {txt(pg, '.hp-mean-load')} | ramp top {txt(pg, '.hp-hi-load')}")
        pg.locator(".hp").screenshot(path=str(SHOTS / "hp-05-half48.png"))
        STEP = """async b => { const raf = () => new Promise(r => requestAnimationFrame(r)); await raf();
            const t = performance.now(); for (let i = 0; i < 10; i++) { b.click(); await raf(); await raf(); } return (performance.now() - t) / 10; }"""
        print(f"in-page ms per hour step, both panes: {pg.locator('.hp-next').evaluate(STEP):.0f}")
        b.close()
finally:
    srv.terminate()
    out = srv.stdout.read() if srv.stdout else ""
    print("--- console ---"); print("\n".join(l for l in logs if ("deck" in l.lower() or "error" in l.lower()) and "preload" not in l and "GL Driver" not in l)[:3000])
    print("--- server tail ---"); print(out[-2500:])
