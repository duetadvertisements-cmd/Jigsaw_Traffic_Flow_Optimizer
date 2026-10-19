from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import deque

ROADS = [
    {"name": "Amapara Main Road (North)", "short": "North",
     "counts": [100, 130, 240]},
    {"name": "Lakhe Nagar Road (South)", "short": "South",
     "counts": [90, 110, 220]},
    {"name": "Great Eastern Rd (West)", "short": "West",
     "counts": [140, 220, 120]},
    {"name": "Great Eastern Rd (East)", "short": "East",
     "counts": [150, 240, 130]},
]
SCENARIOS = ["Balanced traffic", "Busy east\u2013west", "Heavy north\u2013south"]
CSV_SCENARIO_NAMES = ["Balanced", "Busy east-west", "Heavy north-south"]
DURATIONS = [("1 minute demo", 60), ("5 minute demo", 300), ("15 minute model", 900)]
SPEEDS = [("Slow", 0.5), ("Normal", 1.0), ("Fast", 2.0)]
CAR_COLORS = ["#f45b5b", "#ffbd4a", "#55a7ff", "#f4f7fb", "#39d6a0", "#ba8cff"]
PHASE_NAMES = [
    "Southbound",  # North approach: cars travel down
    "Northbound",  # South approach: cars travel up
    "Eastbound",   # West approach: cars travel right
    "Westbound",   # East approach: cars travel left
]

YELLOW_MAX = 3.0
YELLOW_MIN = 3.0

SCORE_VEHICLE_WEIGHT = 2
SMART_MIN_GREEN = 10
SMART_MAX_GREEN = 90
SWITCH_MARGIN = 1.05
MODES = ("fixed", "smart")


def rng(n: float) -> float:
    x = math.sin(n * 127.1 + 311.7) * 43758.5453
    return x - math.floor(x)


def t_fmt(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


class Stats:
    def __init__(self):
        self.clear = 0
        self.wait = 0.0
        self.served = 0
        self.max = 0
        self.switches = 0 

    @property
    def avg_wait(self) -> float:
        return self.wait / self.served if self.served else 0.0


class Simulation:

    def __init__(self, scenario: int = 1, duration: int = 300, smooth: bool = True):
        self.smooth = smooth
        self.seed = scenario
        self.limit = duration
        self.setup(scenario, duration)

    def setup(self, scenario=None, duration=None):
        if scenario is not None:
            self.seed = scenario
        if duration is not None:
            self.limit = duration
        self.elapsed = 0.0
        self.spawn = [0.0] * 4
        self.total_cars = 0
        self.queues = {m: [deque() for _ in range(4)] for m in MODES}
        self.stats = {m: Stats() for m in MODES}
        self.dis_acc = {m: START_UP_ACC for m in MODES}
        self.departures = {m: deque() for m in MODES}
        self.phase = {m: 0 for m in MODES}
        self.phase_start = {m: 0.0 for m in MODES}
        self.greens = {"fixed": [60] * 4}                
        self.green_secs = {m: [0.0] * 4 for m in MODES}   
        self.stage = {m: "green" for m in MODES}          
        self.stage_start = {m: 0.0 for m in MODES}         
        self.next_phase = {m: 0 for m in MODES}           
        self.yellow_passed = {m: False for m in MODES}     

    @property
    def finished(self) -> bool:
        return self.elapsed >= self.limit

    def step(self, dt: float):
        self.elapsed += dt
        self._arrivals(dt)
        self._update_mode("fixed", dt)
        self._update_mode("smart", dt)

    def _arrivals(self, dt):
        for r in range(4):
            rate = ROADS[r]["counts"][self.seed] / 900
            self.spawn[r] += dt * rate * (0.82 + 0.32 * math.sin(self.elapsed / 13 + r * 1.3))
            while self.spawn[r] >= 1:
                self.spawn[r] -= 1
                color = CAR_COLORS[int(rng(self.total_cars + self.seed) * 6)]
                car = (self.elapsed, color, self.total_cars)
                self.total_cars += 1
                for mode in MODES:
                    self.queues[mode][r].append(car)

    def lane_scores(self, mode="smart"):
        out = []
        for arr in self.queues[mode]:
            n = len(arr)
            avg = (n * self.elapsed - sum(c[0] for c in arr)) / n if n else 0.0
            out.append(n * SCORE_VEHICLE_WEIGHT + avg)
        return out

    def _wants_switch(self, mode, p):
        since = self.elapsed - self.phase_start[mode]
        if mode == "fixed":
            return (p + 1) % 4 if since >= self.greens["fixed"][p] else None
        if since < SMART_MIN_GREEN:
            return None
        scores = self.lane_scores("smart")
        best = max(((p + k) % 4 for k in (1, 2, 3)), key=lambda i: scores[i])
        if since >= SMART_MAX_GREEN or scores[best] > scores[p] * SWITCH_MARGIN:
            return best
        return None

    
    def _signal(self, mode):
        p = self.phase[mode]

        if self.stage[mode] == "green":
            nxt = self._wants_switch(mode, p)
            if nxt is None:
                return "green"

            self.stage[mode] = "yellow"
            self.stage_start[mode] = self.elapsed
            self.next_phase[mode] = nxt
            self.yellow_passed[mode] = False
            self.stats[mode].switches += 1

            return "yellow"

        yellow_elapsed = self.elapsed - self.stage_start[mode]

        if yellow_elapsed < YELLOW_MIN:
            return "yellow"
        if (
            self.yellow_passed[mode]
            or not self.queues[mode][p]
            or yellow_elapsed >= YELLOW_MAX
        ):
            self.phase[mode] = self.next_phase[mode]
            self.phase_start[mode] = self.elapsed
            self.stage[mode] = "green"
            self.dis_acc[mode] = START_UP_ACC
            return "green"

        return "yellow"


    def _try_depart(self, mode, p, state):
        q = self.queues[mode][p]
        if not q or (state == "yellow" and self.yellow_passed[mode]):
            return False
        if state == "yellow":
            self.yellow_passed[mode] = True
        self._depart(mode, p, q.popleft())
        return True

    def _update_mode(self, mode, dt):
        q = self.queues[mode]
        st = self.stats[mode]
        state = self._signal(mode)
        p = self.phase[mode]
        if state == "green":
            self.green_secs[mode][p] += dt

        if not self.smooth:
            discharge = dt / 2.4 if state in ("green", "yellow") else 0.0
            while discharge >= 1 and self._try_depart(mode, p, state):
                discharge -= 1
            if discharge > 0 and q[p] and rng(math.floor(self.elapsed * 10) + p + (5 if mode == "smart" else 0)) < discharge:
                self._try_depart(mode, p, state)
        else:
            acc = min(self.dis_acc[mode] + dt / DISCHARGE_HEADWAY, 1.0)
            if acc >= 1.0 and q[p] and self.elapsed - q[p][0][0] >= FREE_FLOW_DELAY[p]:
                if self._try_depart(mode, p, state):
                    acc = 0.0
            self.dis_acc[mode] = acc

        dep = self.departures[mode]
        while dep and self.elapsed - dep[0][1] > DEPART_SECONDS:
            dep.popleft()
        st.max = max(st.max, max(len(a) for a in q))

    def _depart(self, mode, road, car):
        st = self.stats[mode]
        st.clear += 1
        st.served += 1
        st.wait += self.elapsed - car[0]
        self.departures[mode].append((road, self.elapsed, car[1], car[2]))

    def light_state(self, mode) -> str:
        return self.stage[mode]

    def phase_text(self, mode) -> str:
        name = PHASE_NAMES[self.phase[mode]]
        if self.stage[mode] == "yellow":
            return f"{name} \u00b7 yellow, next {PHASE_NAMES[self.next_phase[mode]]}"
        since = self.elapsed - self.phase_start[mode]
        if mode == "fixed":
            return f"{name} \u00b7 {max(0, math.ceil(self.greens['fixed'][self.phase[mode]] - since))}s"
        return f"{name} \u00b7 green {int(since)}s"

    def queue_now(self, mode) -> int:
        return max(len(a) for a in self.queues[mode])

    def cars_waiting(self, mode) -> int:
        return sum(len(a) for a in self.queues[mode])

    def live_avg_wait(self, mode) -> float:
        st, q = self.stats[mode], self.queues[mode]
        n = st.served + self.cars_waiting(mode)
        if not n:
            return 0.0
        waiting = sum(self.elapsed - car[0] for a in q for car in a)
        return (st.wait + waiting) / n

    def summary_text(self) -> str:
        f, s = self.stats["fixed"], self.stats["smart"]
        return (
            f"Elapsed {t_fmt(self.elapsed)}. Fixed: {f.clear} cleared, {self.live_avg_wait('fixed'):.1f} s average wait, "
            f"queue {self.queue_now('fixed')} (peak {f.max}). Smart: {s.clear} cleared, "
            f"{self.live_avg_wait('smart'):.1f} s average wait, queue {self.queue_now('smart')} (peak {s.max}). "
            f"Live demo estimates only."
        )

    def result_rows(self):
        f, s = self.stats["fixed"], self.stats["smart"]

        return [
            ["Metric", "Value", "Fixed Timer", "Smart Signal"],
            ["Scenario", CSV_SCENARIO_NAMES[self.seed], "", ""],
            ["Elapsed seconds", round(self.elapsed, 2), "", ""],
            ["Vehicles cleared", "", f.clear, s.clear],
            ["Average wait seconds (cleared vehicles)", "",
            round(self.live_avg_wait("fixed"), 2),
            round(self.live_avg_wait("smart"), 2)],
            ["Longest queue", "", f.max, s.max],
            ["Signal switches", "", f.switches, s.switches],
            *[
                [r["name"] + " vehicles per 15 min",
                r["counts"][self.seed], "", ""]
                for r in ROADS
            ],
        ]


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh, quoting=csv.QUOTE_ALL, lineterminator="\r\n").writerows(rows)


def run_headless(scenario, duration, csv_path=None, smooth=True):
    sim = Simulation(scenario, duration, smooth)
    while not sim.finished:
        sim.step(1 / 60)
    f, s = sim.stats["fixed"], sim.stats["smart"]
    print(f"Scenario: {SCENARIOS[scenario]}   Simulated time: {t_fmt(sim.elapsed)}")
    print(f"{'':22}{'Fixed timer':>14}{'Smart signal':>14}")
    print(f"{'Vehicles cleared':22}{f.clear:>14}{s.clear:>14}")
    print(f"{'Avg wait (s)':22}{f.avg_wait:>14.1f}{s.avg_wait:>14.1f}")
    print(f"{'Longest queue':22}{f.max:>14}{s.max:>14}")
    print(f"{'Signal switches':22}{f.switches:>14}{s.switches:>14}")
    shorts = [r["short"] for r in ROADS]
    print("Green seconds per road  fixed:", dict(zip(shorts, map(round, sim.green_secs["fixed"]))))
    print("                        smart:", dict(zip(shorts, map(round, sim.green_secs["smart"]))))
    if csv_path:
        write_csv(csv_path, sim.result_rows())
        print("Saved", csv_path)


W, H, CX, CY, ROAD_W = 700, 450, 350, 225, 150 

BG, PANEL, LINE = "#071426", "#0d2038", "#244563"
TEXT, MUTED = "#eaf4ff", "#a7bed7"
BLUE, GREEN, RED, YEL = "#2690ff", "#31d17c", "#ff5a65", "#ffca54"
FONT = "Helvetica"
STEP = 1 / 60           
MAX_QUEUE_DRAWN = 12    
MAX_DEPARTING = 12       
MAX_SPRITES = 4 * MAX_QUEUE_DRAWN + MAX_DEPARTING
DEPART_SECONDS = 7.0    
APPROACH_SPEED = 200     
SLOT = 30                
ARM_LEN = (245, 245, 370, 370) 
QUEUE_FRONT = 121        
CREEP_MIN, CREEP_GAIN = 20, 2.5      
DEPART_START, DEPART_ACCEL, DEPART_MAX = 40.0, 150.0, 130.0   
DISCHARGE_HEADWAY = 2.4  
START_UP_ACC = 0.7       
FREE_FLOW_DELAY = tuple((a - QUEUE_FRONT) / APPROACH_SPEED for a in ARM_LEN)  
SIGNAL_POS = [(CX + 86, CY - 104), (CX - 105, CY + 85), (CX - 105, CY - 104), (CX + 86, CY + 85)]
LAMP_COLORS = {"red": "#ff4e5b", "yellow": "#ffcb4f", "green": "#34dc83"}
LAMP_OFF = "#36454c"
CAR_PARTS = [(-13, -8, 26, 16, "#111b26"), (-12, -7, 24, 14, None), (-6, -5, 9, 10, "#ccecff"),
             (8, -5, 3, 4, "#182332"), (8, 2, 3, 4, "#182332")]  


class JunctionView:

    def __init__(self, parent, mode, scale):
        import tkinter as tk

        self.mode, self.s = mode, scale
        self.vis = {}       
        self.last_t = 0.0
        self.canvas = tk.Canvas(parent, width=round(W * scale), height=round(H * scale),
                                bg="#263b3b", highlightthickness=0)
        self._draw_static()
        self._build_dynamic()

    def _rect(self, x, y, w, h, fill, tag):
        s = self.s
        self.canvas.create_rectangle(x * s, y * s, (x + w) * s, (y + h) * s, fill=fill, outline="", tags=tag)

    def _rect2(self, x0, y0, x1, y1, fill, tag):
        s = self.s
        self.canvas.create_rectangle(x0 * s, y0 * s, x1 * s, y1 * s, fill=fill, outline="", tags=tag)

    def _circle(self, x, y, r, fill, tag):
        s = self.s
        self.canvas.create_oval((x - r) * s, (y - r) * s, (x + r) * s, (y + r) * s, fill=fill, outline="", tags=tag)

    def _text(self, x, y, text, size, tag, color="#f3f7fc"):
        self.canvas.create_text(x * self.s, y * self.s, text=text, fill=color, anchor="center",
                                font=(FONT, -max(8, round(size * self.s)), "bold"), tags=tag)

    def _build_dynamic(self):
        c, s = self.canvas, self.s
        self.lamps, self.lamp_state = [], [None] * 4
        for px, py in SIGNAL_POS:
            self._rect(px, py, 19, 43, "#101820", "sig")
            ids = []
            for j in range(3):
                lx, ly = px + 9.5, py + 8 + j * 13
                ids.append(c.create_oval((lx - 5) * s, (ly - 5) * s, (lx + 5) * s, (ly + 5) * s,
                                         fill=LAMP_OFF, outline="", tags="sig"))
            self.lamps.append(ids)
        self.sprites = []         
        for _ in range(MAX_SPRITES):
            self.sprites.append([c.create_rectangle(0, 0, 0, 0, fill=col or "#888", outline="",
                                                    state="hidden", tags="car") for *_, col in CAR_PARTS])
        self.sprite_key = [None] * MAX_SPRITES  
        self.sprite_color = [None] * MAX_SPRITES
        self.sprite_visible = [False] * MAX_SPRITES
        self.sprites_used = 0
        c.create_rectangle((CX - 63) * s, (CY - 25) * s, (CX + 63) * s, (CY + 25) * s, fill="#0e1a27",
                           outline=GREEN if self.mode == "smart" else BLUE, width=2, tags="ov")
        self._text(CX, CY, "SMART SIGNAL" if self.mode == "smart" else "FIXED TIMER", 15, "ov", color="#ffffff")

    def _place(self, n, x, y, color, vertical):
        c, s = self.canvas, self.s
        ids = self.sprites[n]
        key = (round(x, 1), round(y, 1), vertical)
        if self.sprite_key[n] != key:
            self.sprite_key[n] = key
            for item, (rx, ry, rw, rh, _) in zip(ids, CAR_PARTS):
                if vertical:  
                    c.coords(item, (x - ry - rh) * s, (y + rx) * s, (x - ry) * s, (y + rx + rw) * s)
                else:
                    c.coords(item, (x + rx) * s, (y + ry) * s, (x + rx + rw) * s, (y + ry + rh) * s)
        if self.sprite_color[n] != color:
            self.sprite_color[n] = color
            c.itemconfigure(ids[1], fill=color)
        if not self.sprite_visible[n]:
            self.sprite_visible[n] = True
            for item in ids:
                c.itemconfigure(item, state="normal")

    def _hide_from(self, first):
        c = self.canvas
        for n in range(first, self.sprites_used):
            if self.sprite_visible[n]:
                self.sprite_visible[n] = False
                for item in self.sprites[n]:
                    c.itemconfigure(item, state="hidden")

    def _draw_static(self):
        s = self.s
        self._rect(0, 0, W, H, "#49644d", "bg")
        for y in range(0, H, 105):
            for x in range(0, W, 125):
                if 270 < x < 430 or 150 < y < 300:
                    continue
                self._rect(x + 5, y + 5, 90, 70, "#53646a" if (x + y) % 2 else "#64736c", "bg")
                self._rect(x + 15, y + 15, 70, 50, "#77848a", "bg")
                for k in range(3):
                    self._circle(x + 100 + (k % 2) * 12, y + 80 + (k % 3) * 8, 10, "#1d7a4e", "bg")
        self._rect(CX - ROAD_W / 2, 0, ROAD_W, H, "#303b48", "bg")
        self._rect(0, CY - ROAD_W / 2, W, ROAD_W, "#303b48", "bg")

        dash = (max(1, round(14 * s)), max(1, round(13 * s)))
        self.canvas.create_line(CX * s, 0, CX * s, H * s, fill="#d8d8c8", width=2, dash=dash, tags="bg")
        self.canvas.create_line(0, CY * s, W * s, CY * s, fill="#d8d8c8", width=2, dash=dash, tags="bg")

        c = "#f4f5ed"
        for k in range(8):
            self._rect(CX - 75 + k * 19, CY - 91, 10, 6, c, "bg")
            self._rect(CX - 75 + k * 19, CY + 85, 10, 6, c, "bg")
            self._rect(CX - 91, CY - 75 + k * 19, 6, 10, c, "bg")
            self._rect(CX + 85, CY - 75 + k * 19, 6, 10, c, "bg")
        self._rect(CX - 76, CY - 103, 152, 4, c, "bg")
        self._rect(CX - 76, CY + 99, 152, 4, c, "bg")
        self._rect(CX - 103, CY - 76, 4, 152, c, "bg")
        self._rect(CX + 99, CY - 76, 4, 152, c, "bg")

        self._text(CX, 13, "\u2191 Amapara Main Road", 13, "bg")
        self._text(CX, 433, "\u2193 Lakhe Nagar Road", 13, "bg")
        self._text(120, CY - 92, "\u2190 Great Eastern Road (West)", 13, "bg")
        self._text(570, CY - 92, "Great Eastern Road (East) \u2192", 13, "bg")

    @staticmethod
    def _xy(road, pos):

        if road == 0:
            return CX - 37, CY - pos, True
        if road == 1:
            return CX + 37, CY + pos, True
        if road == 2:
            return CX - pos, CY - 37, False
        return CX + pos, CY + 37, False

    def draw(self, sim):
        c, mode = self.canvas, self.mode
        p = sim.phase[mode]
        state = sim.light_state(mode)


        for i in range(4):
            want = state if i == p else "red"
            if self.lamp_state[i] != want:
                self.lamp_state[i] = want
                for lamp, col in zip(self.lamps[i], ("red", "yellow", "green")):
                    c.itemconfigure(lamp, fill=LAMP_COLORS[col] if col == want else LAMP_OFF)


        dt = sim.elapsed - self.last_t
        self.last_t = sim.elapsed
        if dt < 0:                      
            self.vis.clear()
            dt = 0.0
        dt = min(dt, 0.25)
        vis = self.vis


        dep_ids = set()
        for road, _t, color, cid in sim.departures[mode]:
            dep_ids.add(cid)
            v = vis.get(cid)
            if v is None:
                vis[cid] = [road, float(QUEUE_FRONT), color, DEPART_START]
            elif v[3] is None:
                v[3] = DEPART_START
        front = [None] * 4            
        for cid in dep_ids:
            v = vis[cid]
            v[3] = min(DEPART_MAX, v[3] + DEPART_ACCEL * dt)   
            v[1] -= v[3] * dt
            if v[1] > 0 and (front[v[0]] is None or v[1] > front[v[0]]):
                front[v[0]] = v[1]


        live = set()
        for r, arr in enumerate(sim.queues[mode]):
            prev = front[r]
            for i, car in enumerate(arr):
                cid = car[2]
                live.add(cid)
                v = vis.get(cid)
                if v is None:                      
                    v = vis[cid] = [r, float(ARM_LEN[r]), car[1], None]
                target = QUEUE_FRONT + SLOT * i
                speed = min(APPROACH_SPEED, max(CREEP_MIN, CREEP_GAIN * (v[1] - target)))
                pos = max(target, v[1] - speed * dt)   
                if prev is not None and pos < prev + SLOT:
                    pos = prev + SLOT              
                v[1] = prev = pos

        for cid in [k for k in vis if k not in live and k not in dep_ids]:
            del vis[cid]

      
        n = 0
        for road, pos, color, _speed in vis.values():
            if n >= MAX_SPRITES or abs(pos) > ARM_LEN[road] + 13:
                continue
            x, y, vertical = self._xy(road, pos)
            self._place(n, x, y, color, vertical)
            n += 1

        self._hide_from(n)
        self.sprites_used = max(self.sprites_used, n)


class App:
    def __init__(self, root, scale=None, smooth=True):
        import tkinter as tk
        from tkinter import ttk

        self.tk, self.ttk, self.root = tk, ttk, root
        root.title("Traffic Flow Optimizer - Animated Junction")
        root.configure(bg=BG)
        if scale is None:
            sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
            scale = max(0.45, min(1.0, (sw - 90) / 1400, (sh - 480) / 450))
        self.scale = scale

        self.sim = Simulation(1, 300, smooth)
        self.running = False
        self.paused = False
        self.acc = 0.0
        self._cache = {}
        self.last = time.perf_counter()

        self._style()
        self._header()
        main = tk.Frame(root, bg=BG)
        main.pack(fill="both", expand=True, padx=10, pady=8)
        main.columnconfigure(0, weight=1, uniform="a")
        main.columnconfigure(1, weight=1, uniform="a")
        self._controls(main)
        self.views, self.w = {}, {}
        self._junction(main, "fixed", 0)
        self._junction(main, "smart", 1)
        self._lower(main)

        self.setup()
        self.frame()

    
    def _style(self):
        ttk = self.ttk
        st = ttk.Style(self.root)
        st.theme_use("clam")
        st.configure("Primary.TButton", background=BLUE, foreground="white", borderwidth=0,
                     padding=(14, 7), font=(FONT, 10, "bold"))
        st.map("Primary.TButton", background=[("active", "#4aa3ff")])
        st.configure("Secondary.TButton", background="#243b55", foreground="white", borderwidth=0,
                     padding=(14, 7), font=(FONT, 10, "bold"))
        st.map("Secondary.TButton", background=[("active", "#32506f")])
        st.configure("Dark.TCombobox", fieldbackground="#102b49", background="#102b49", foreground=TEXT,
                     arrowcolor=TEXT, bordercolor="#426184", lightcolor="#102b49", darkcolor="#102b49")
        st.map("Dark.TCombobox", fieldbackground=[("readonly", "#102b49")], foreground=[("readonly", TEXT)],
               selectbackground=[("readonly", "#102b49")], selectforeground=[("readonly", TEXT)])
        self.root.option_add("*TCombobox*Listbox.background", "#102b49")
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", BLUE)

    def _panel(self, parent):
        return self.tk.Frame(parent, bg=PANEL, highlightbackground=LINE, highlightthickness=1)

    def _label(self, parent, text="", fg=TEXT, size=10, bold=False, bg=PANEL, **kw):
        return self.tk.Label(parent, text=text, fg=fg, bg=bg, font=(FONT, size, "bold" if bold else "normal"), **kw)

    def _header(self):
        tk = self.tk
        head = tk.Frame(self.root, bg="#091b31", highlightbackground=LINE, highlightthickness=1)
        head.pack(fill="x")
        left = tk.Frame(head, bg="#091b31")
        left.pack(side="left", padx=16, pady=10)
        # (emoji omitted from the title: some Linux Tk builds crash on them)
        self._label(left, "Traffic Flow Optimizer", size=20, bold=True, bg="#091b31").pack(anchor="w")
        self._label(left, "Animated Smart Signal Simulation \u00b7 Amapara Chowk, Raipur",
                    fg="#bdd5f2", bg="#091b31").pack(anchor="w")
        self._label(head, "HCL Jigsaw 7.0 \u00b7 Grade 8", fg="#9dc5f4", bg="#091b31").pack(side="right", padx=16)

    def _controls(self, main):
        tk, ttk = self.tk, self.ttk
        bar = self._panel(main)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        inner = tk.Frame(bar, bg=PANEL)
        inner.pack(fill="x", padx=10, pady=10)

        def field(label, items, default, cmd):
            f = tk.Frame(inner, bg=PANEL)
            f.pack(side="left", padx=(0, 12))
            self._label(f, label, fg=MUTED, size=9).pack(anchor="w")
            cb = ttk.Combobox(f, state="readonly", style="Dark.TCombobox", width=20,
                              values=[i[0] if isinstance(i, tuple) else i for i in items])
            cb.current(default)
            cb.pack()
            cb.bind("<<ComboboxSelected>>", lambda e: (cb.selection_clear(), cmd()))
            return cb

        self.cb_scenario = field("Traffic scenario", SCENARIOS, 1, self.on_scenario)
        self.cb_duration = field("Simulation length", DURATIONS, 1, self.on_duration)
        self.cb_speed = field("Car animation speed", SPEEDS, 1, lambda: None)
        self.btn_start = ttk.Button(inner, text="\u25b6 Run simulation", style="Primary.TButton", command=self.on_start)
        self.btn_start.pack(side="left", padx=(8, 6), pady=(14, 0))
        self.btn_pause = ttk.Button(inner, text="\u2161 Pause", style="Secondary.TButton", command=self.on_pause)
        self.btn_pause.pack(side="left", padx=6, pady=(14, 0))
        ttk.Button(inner, text="\u21ba Reset", style="Secondary.TButton", command=self.on_reset).pack(
            side="left", padx=6, pady=(14, 0))

    def _junction(self, main, mode, col):
        tk = self.tk
        pnl = self._panel(main)
        pnl.grid(row=1, column=col, sticky="nsew", padx=(0, 4) if col == 0 else (4, 0))
        head = tk.Frame(pnl, bg=PANEL)
        head.pack(fill="x", padx=12, pady=8)
        self._label(head, "Fixed timer signal" if mode == "fixed" else "Smart adaptive signal",
                    size=13, bold=True).pack(side="left")
        pill_bg, pill_fg = ("#193b60", "#9ed0ff") if mode == "fixed" else ("#143b35", "#75efae")
        self._label(head, "60 s green per road" if mode == "fixed" else "Traffic-score green",
                    fg=pill_fg, bg=pill_bg, size=9, padx=8, pady=2).pack(side="right")

        view = JunctionView(pnl, mode, self.scale)
        view.canvas.pack()
        self.views[mode] = view

        foot = tk.Frame(pnl, bg=PANEL)
        foot.pack(fill="x", padx=12, pady=6)
        w = {}
        w["time"] = self._label(foot, "Time: 00:00", fg=MUTED, size=9)
        w["time"].pack(side="left")
        w["count"] = self._label(foot, "Cleared: 0", fg=MUTED, size=9)
        w["count"].pack(side="right")
        w["phase"] = self._label(foot, "", fg=MUTED, size=9)
        w["phase"].pack(side="right", padx=14)

        cards = tk.Frame(pnl, bg=PANEL)
        cards.pack(fill="x", padx=12, pady=(0, 12))
        for i, (key, title) in enumerate((("clear", "Vehicles cleared"), ("wait", "Avg wait (live)"),
                                          ("queue", "Longest queue now / peak"))):
            card = tk.Frame(cards, bg="#0a1a2d", highlightbackground=LINE, highlightthickness=1)
            card.grid(row=0, column=i, sticky="ew", padx=3)
            cards.columnconfigure(i, weight=1, uniform="c")
            self._label(card, title, fg=MUTED, size=8, bg="#0a1a2d").pack(anchor="w", padx=8, pady=(6, 0))
            w[key] = self._label(card, "0", size=16, bold=True, bg="#0a1a2d")
            w[key].pack(anchor="w", padx=8, pady=(0, 6))
        self.w[mode] = w

    def _lower(self, main):
        tk = self.tk
        row = tk.Frame(main, bg=BG)
        row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        for i in range(3):
            row.columnconfigure(i, weight=1, uniform="l")

       
        p1 = self._panel(row)
        p1.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self._label(p1, "Live results comparison", size=12, bold=True).pack(anchor="w", padx=12, pady=(10, 6))
        self.bars = {}
        for key, text, color in (("fc", "Cleared \u00b7 fixed", BLUE), ("sc", "Cleared \u00b7 smart", GREEN),
                                 ("fw", "Wait \u00b7 fixed", BLUE), ("sw", "Wait \u00b7 smart", GREEN)):
            line = tk.Frame(p1, bg=PANEL)
            line.pack(fill="x", padx=12, pady=3)
            self._label(line, text, size=9, width=14, anchor="w").pack(side="left")
            val = self._label(line, "0", size=9, bold=True, width=6, anchor="e")
            val.pack(side="right")
            bar = tk.Canvas(line, height=12, bg="#203a55", highlightthickness=0)
            bar.pack(side="left", fill="x", expand=True, padx=6)
            self.bars[key] = (bar, val, color)
        self.summary = self._label(p1, "Press Run simulation to begin. The same arrivals feed both junctions.",
                                   fg=MUTED, size=9, justify="left", wraplength=360)
        self.summary.pack(anchor="w", padx=12, pady=(8, 10))

        
        p2 = self._panel(row)
        p2.grid(row=0, column=1, sticky="nsew", padx=4)
        self._label(p2, "Road counts & green time", size=12, bold=True).pack(anchor="w", padx=12, pady=(10, 6))
        table = tk.Frame(p2, bg=PANEL)
        table.pack(fill="x", padx=12)
        for c, h in enumerate(("Road", "Cars / 15 min*", "Fixed green", "Smart score")):
            self._label(table, h, fg=MUTED, size=9, anchor="w" if c == 0 else "e").grid(row=0, column=c, sticky="ew", padx=3, pady=3)
        table.columnconfigure(0, weight=1)
        self.road_counts, self.smart_scores = [], []
        for i, r in enumerate(ROADS, start=1):
            self._label(table, r["name"], size=9, anchor="w").grid(row=i, column=0, sticky="ew", padx=3, pady=3)
            cnt = self._label(table, "", size=9, anchor="e")
            cnt.grid(row=i, column=1, sticky="ew", padx=3)
            self._label(table, "60 s", size=9, anchor="e").grid(row=i, column=2, sticky="ew", padx=3)
            sg = self._label(table, "0", size=9, anchor="e")
            sg.grid(row=i, column=3, sticky="ew", padx=3)
            self.road_counts.append(cnt)
            self.smart_scores.append(sg)
        self._label(p2, "*Synthetic example counts, not real observations. Smart score = vehicles \u00d7 2 + average wait (s), live.",
                    fg=MUTED, size=8, justify="left", wraplength=340).pack(anchor="w", padx=12, pady=(8, 10))

       
        p3 = self._panel(row)
        p3.grid(row=0, column=2, sticky="nsew", padx=(4, 0))
        self._label(p3, "Signal logic", size=12, bold=True).pack(anchor="w", padx=12, pady=(10, 6))
        for line in ("1. Score each road: vehicles waiting \u00d7 2 + average wait (s).",
                     "2. Keep the current road green unless another road needs it more.",
                     "3. Green lasts at least 10 s and at most 90 s.",
                     "4. After 10 s, switch if another road scores over 5% higher; force a switch by 90 s.",
                     "5. To switch: the light turns yellow, then the next road goes green."):
            self._label(p3, line, size=9, justify="left", anchor="w", wraplength=340).pack(anchor="w", padx=12, pady=1)
        self.ttk.Button(p3, text="\u2193 Download results CSV", style="Secondary.TButton",
                        command=self.on_csv).pack(anchor="w", padx=12, pady=(0, 10))

    
    def _scenario(self):
        return self.cb_scenario.current()

    def _duration(self):
        return DURATIONS[self.cb_duration.current()][1]

    def _speed(self):
        return SPEEDS[self.cb_speed.current()][1]

    def setup(self):
        self.acc = 0.0
        self.sim.setup(self._scenario(), self._duration())
        for i, r in enumerate(ROADS):
            self.road_counts[i].config(text=str(r["counts"][self.sim.seed]))
        self.render()

    def on_start(self):
        if not self.running and self.sim.finished:
            self.setup()
        self.running, self.paused = True, False
        self.btn_start.config(text="\u25b6 Running\u2026")
        self.btn_pause.config(text="\u2161 Pause")

    def on_pause(self):
        self.paused = not self.paused
        self.btn_pause.config(text="\u25b6 Resume" if self.paused else "\u2161 Pause")

    def on_reset(self):
        self.running = self.paused = False
        self.setup()
        self.btn_start.config(text="\u25b6 Run simulation")
        self.btn_pause.config(text="\u2161 Pause")

    def on_scenario(self):
        self.running = False
        self.setup()
        self.btn_start.config(text="\u25b6 Run simulation")

    def on_duration(self):
        self.sim.limit = self._duration()

    def on_csv(self):
        from tkinter import filedialog, messagebox

        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="traffic-flow-results.csv",
                                            filetypes=[("CSV files", "*.csv")])
        if path:
            write_csv(path, self.sim.result_rows())
            messagebox.showinfo("Saved", f"Results saved to\n{path}")

    
    def frame(self):
        t0 = time.perf_counter()
        dt = min(t0 - self.last, 0.25)       
        self.last = t0
        if self.running and not self.paused:
          
            self.acc += dt * self._speed()
            steps = 0
            while self.acc >= STEP and steps < 60:
                self.sim.step(STEP)
                self.acc -= STEP
                steps += 1
                if self.sim.finished:
                    self.running = False
                    self.acc = 0.0
                    self.btn_start.config(text="\u25b6 Run again")
                    break
            if steps:
                self.render()
        
        delay = max(1, 16 - int((time.perf_counter() - t0) * 1000))
        self.root.after(delay, self.frame)

    def _set(self, widget, text):
        
        if self._cache.get(widget) != text:
            self._cache[widget] = text
            widget.config(text=text)

    def _set_bar(self, key, frac, text):
        bar, val, color = self.bars[key]
        w = bar.winfo_width()
        px = int(w * max(0.0, min(1.0, frac)))
        if self._cache.get(bar) != (px, w):
            self._cache[bar] = (px, w)
            bar.delete("all")
            bar.create_rectangle(0, 0, px, 12, fill=color, outline="")
        self._set(val, text)

    def render(self):
        sim = self.sim
        for mode in MODES:
            self.views[mode].draw(sim)
            st, w = sim.stats[mode], self.w[mode]
            self._set(w["time"], f"Time: {int(sim.elapsed) // 60:02d}:{int(sim.elapsed) % 60:02d}")
            self._set(w["phase"], f"Signal: {sim.phase_text(mode)}")
            self._set(w["count"], f"Cleared: {st.clear}")
            self._set(w["clear"], str(st.clear))
            self._set(w["wait"], f"{sim.live_avg_wait(mode):.1f} s")
            self._set(w["queue"], f"{sim.queue_now(mode)} / {st.max}")

        f, s = sim.stats["fixed"], sim.stats["smart"]
        maxc = max(1, f.clear, s.clear)
        fw, sw = sim.live_avg_wait("fixed"), sim.live_avg_wait("smart")
        maxw = max(1, fw, sw)
        self._set_bar("fc", f.clear / maxc, str(f.clear))
        self._set_bar("sc", s.clear / maxc, str(s.clear))
        self._set_bar("fw", fw / maxw, f"{fw:.1f} s")
        self._set_bar("sw", sw / maxw, f"{sw:.1f} s")
        self._set(self.summary, sim.summary_text())
        for i, sc in enumerate(sim.lane_scores("smart")):
            self._set(self.smart_scores[i], f"{sc:.0f}")



def main():
    ap = argparse.ArgumentParser(description="Traffic Flow Optimizer - fixed vs smart signal simulation")
    ap.add_argument("--headless", action="store_true", help="run without a window and print the results")
    ap.add_argument("--scenario", type=int, choices=(0, 1, 2), default=1,
                    help="0 balanced, 1 busy east-west (default), 2 heavy north-south")
    ap.add_argument("--duration", type=int, default=300, help="simulated seconds (default 300)")
    ap.add_argument("--csv", metavar="PATH", help="(headless) also write results to this CSV file")
    ap.add_argument("--original-discharge", action="store_true",
                    help="release cars in random bursts like the original HTML (for exact comparison)")
    ap.add_argument("--scale", type=float, help="(GUI) junction drawing scale, e.g. 0.7 (default: auto-fit screen)")
    args = ap.parse_args()

    if args.headless:
        run_headless(args.scenario, args.duration, args.csv, not args.original_discharge)
        return

    try:
        import tkinter as tk
    except ImportError:
        sys.exit("tkinter is not installed. Install it (e.g. 'sudo apt install python3-tk') "
                 "or use --headless.")
    root = tk.Tk()
    app = App(root, args.scale, not args.original_discharge)
    
    app.cb_scenario.current(args.scenario)
    for i, (_, secs) in enumerate(DURATIONS):
        if secs == args.duration:
            app.cb_duration.current(i)
    app.setup()
    root.mainloop()


if __name__ == "__main__":
    main()