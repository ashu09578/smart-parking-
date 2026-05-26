"""
=============================================================
  SMART PARKING ALLOCATION SYSTEM
  Python Console-Based Project
  Uses: OOP, BFS, CSP, Priority Queue, Graph Navigation
=============================================================

AI & CO CONCEPT MAP:
  CO1 - Problem formulation using PEAS model, state/action representation
  CO2 - Uninformed search (BFS) for nearest slot navigation
  CO3 - Constraint Satisfaction Problem (CSP) for slot allocation rules

PEAS Model for this system:
  Performance  : Minimize waiting time, maximize slot utilization
  Environment  : Parking lot with multiple slots, vehicles of varied types
  Actuators    : Slot assignment, billing, navigation display
  Sensors      : Vehicle type, arrival time, slot availability
"""

# ─── Standard Library Imports Only ────────────────────────────────────────────
import heapq                          # For priority queue (emergency vehicles)
import time                           # For timestamps
import os                             # For screen clear
from datetime import datetime         # For entry/exit times
from dataclasses import dataclass, field  # Clean class definitions
from typing import Optional, List, Dict, Tuple, Set
from collections import deque         # For BFS queue


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 – CONSTANTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Hourly rates per vehicle type (in ₹)
RATES: Dict[str, float] = {
    "Bike":      10.0,
    "Car":       20.0,
    "SUV":       30.0,
    "Emergency":  0.0   # Emergency vehicles park free
}

# Which slot types each vehicle is allowed to use
# CO3 – CSP: These are the domain constraints
ALLOWED_SLOTS: Dict[str, List[str]] = {
    "Bike":      ["Bike"],
    "Car":       ["Car"],
    "SUV":       ["SUV", "Car"],   # SUV can use Car slot if needed
    "Emergency": ["Emergency"]
}

# Visual icons for display
ICONS = {"Bike": "🏍", "Car": "🚗", "SUV": "🚙", "Emergency": "🚑"}
SLOT_STATUS_ICON = {"free": "🟢", "occupied": "🔴", "reserved": "🟡"}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 – DATA CLASSES (CO1: State Representation)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Vehicle:
    """
    Represents a vehicle entering the parking lot.
    CO1: Each vehicle is an agent 'state' in the environment.
    """
    number:     str            # e.g., "TS09AB1234"
    vtype:      str            # Bike | Car | SUV | Emergency
    entry_time: datetime = field(default_factory=datetime.now)
    is_disabled: bool = False  # Priority flag for differently-abled users

    def __str__(self) -> str:
        icon = ICONS.get(self.vtype, "🚗")
        tag  = " [♿]" if self.is_disabled else ""
        return f"{icon} {self.number} ({self.vtype}){tag}"


@dataclass
class ParkingSlot:
    """
    Represents a single parking slot.
    CO1: Each slot is a node in the environment state space.
    """
    slot_id:      str          # e.g., "A1", "B3"
    slot_type:    str          # Bike | Car | SUV | Emergency
    status:       str = "free" # free | occupied | reserved
    vehicle:      Optional[Vehicle] = None
    reserved_for: Optional[str] = None  # vehicle number that reserved it

    def is_available_for(self, vtype: str) -> bool:
        """
        CO3 – CSP Constraint Check:
        Returns True only if this slot satisfies all constraints for the given vehicle type.
        """
        if self.status != "free":
            return False
        if vtype not in ALLOWED_SLOTS:
            return False
        return self.slot_type in ALLOWED_SLOTS[vtype]

    def __str__(self) -> str:
        icon = SLOT_STATUS_ICON.get(self.status, "⬜")
        v    = f" ← {self.vehicle.number}" if self.vehicle else ""
        r    = f" (Reserved: {self.reserved_for})" if self.reserved_for else ""
        return f"  {icon} [{self.slot_id}] {self.slot_type:<10}{v}{r}"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 – GRAPH FOR NAVIGATION (CO2: BFS Search)
# ══════════════════════════════════════════════════════════════════════════════

class ParkingGraph:
    """
    Models the parking lot as a graph.
    Nodes = slots/positions; Edges = walkable paths between them.
    CO2: BFS finds the shortest path from entry to a target slot.
    Time Complexity of BFS: O(V + E)  where V=nodes, E=edges
    """

    def __init__(self):
        # Adjacency list: { node: [neighbour, ...] }
        self.graph: Dict[str, List[str]] = {}

    def add_edge(self, u: str, v: str):
        """Add a bidirectional edge (undirected graph)."""
        self.graph.setdefault(u, []).append(v)
        self.graph.setdefault(v, []).append(u)

    def bfs(self, start: str, goal: str) -> List[str]:
        """
        BFS (Breadth-First Search) – CO2 Uninformed Search.
        Returns the shortest path from 'start' to 'goal'.
        If no path exists, returns an empty list.
        """
        if start not in self.graph or goal not in self.graph:
            return []
        if start == goal:
            return [start]

        visited: Set[str]         = {start}
        queue:   deque            = deque([[start]])   # each item is a path

        while queue:
            path = queue.popleft()
            node = path[-1]

            for neighbour in self.graph.get(node, []):
                if neighbour == goal:
                    return path + [neighbour]           # ✅ Found!
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(path + [neighbour])

        return []  # No path found

    def display_path(self, path: List[str]):
        """Pretty-print the navigation path in the console."""
        if not path:
            print("  ❌ No navigation path found.")
            return
        print("  🗺  Navigation Path:")
        print("  " + " ──► ".join(path))
        print(f"  Total steps: {len(path) - 1}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 – PARKING LOT (Core System)
# ══════════════════════════════════════════════════════════════════════════════

class ParkingLot:
    """
    Central class managing slots, vehicles, billing, and navigation.
    Combines: CSP allocation, BFS navigation, Priority Queue for emergencies.
    """

    def __init__(self, name: str = "SMART PARK - BLOCK A"):
        self.name          = name
        self.slots:    Dict[str, ParkingSlot] = {}
        self.revenue:  float                  = 0.0
        self.exit_log: List[Dict]             = []   # Completed parking records
        self.graph:    ParkingGraph           = ParkingGraph()

        # Priority queue for emergency vehicles: (priority_score, vehicle)
        # Lower score = higher priority  (heapq is a min-heap)
        self._priority_queue: List[Tuple[int, str]] = []

        self._build_default_layout()
        self._build_graph()

    # ── 4a. Layout Setup ─────────────────────────────────────────────────────

    def _build_default_layout(self):
        """
        Creates the initial parking layout.
        Sample Layout:
          Row A: A1–A3 Bike slots
          Row B: B1–B4 Car slots
          Row C: C1–C2 SUV slots
          Row E: E1    Emergency slot
        """
        layout = [
            ("A1","Bike"), ("A2","Bike"), ("A3","Bike"),
            ("B1","Car"),  ("B2","Car"),  ("B3","Car"),  ("B4","Car"),
            ("C1","SUV"),  ("C2","SUV"),
            ("E1","Emergency"),
        ]
        for slot_id, slot_type in layout:
            self.slots[slot_id] = ParkingSlot(slot_id, slot_type)
        print(f"  ✅ Default layout loaded: {len(self.slots)} slots ready.")

    def _build_graph(self):
        """
        Builds the navigation graph for the parking lot.
        ENTRY node connects to all row-starting slots.
        Slots within the same row are connected linearly.
        CO2: This graph is traversed by BFS.
        """
        g = self.graph
        # ENTRY → row heads
        g.add_edge("ENTRY", "A1")
        g.add_edge("ENTRY", "B1")
        g.add_edge("ENTRY", "C1")
        g.add_edge("ENTRY", "E1")
        # Row A connections
        g.add_edge("A1", "A2"); g.add_edge("A2", "A3")
        # Row B connections
        g.add_edge("B1", "B2"); g.add_edge("B2", "B3"); g.add_edge("B3", "B4")
        # Row C connections
        g.add_edge("C1", "C2")
        # Cross-row shortcut (aisle)
        g.add_edge("A3", "B1"); g.add_edge("B4", "C1")

    # ── 4b. Smart Slot Allocation (CSP + Priority Queue) ────────────────────

    def allocate_slot(self, vehicle: Vehicle) -> Optional[ParkingSlot]:
        """
        SMART ALLOCATION ENGINE
        ─────────────────────────
        CO1: Action selection based on current state
        CO3: Constraint Satisfaction – only valid slots are considered
        Uses heapq priority queue for emergency / disabled vehicles.

        Priority scoring (lower = higher priority):
          Emergency vehicle → 0
          Differently-abled → 1
          Normal vehicle    → 2
        """
        print(f"\n  [AI LOG] Allocating slot for {vehicle} ...")

        # Determine priority score for the vehicle
        if vehicle.vtype == "Emergency":
            priority = 0
        elif vehicle.is_disabled:
            priority = 1
        else:
            priority = 2

        print(f"  [AI LOG] Priority Score = {priority} (0=highest)")

        # Collect valid candidate slots (CO3: CSP constraint check)
        candidates: List[ParkingSlot] = []
        for slot in self.slots.values():
            if slot.is_available_for(vehicle.vtype):
                candidates.append(slot)
                print(f"  [AI LOG] Candidate slot: {slot.slot_id} ✓")
            else:
                reason = (
                    "occupied/reserved" if slot.status != "free"
                    else f"type mismatch ({slot.slot_type} ≠ {vehicle.vtype})"
                )
                print(f"  [AI LOG] Rejected slot: {slot.slot_id} — {reason}")

        if not candidates:
            print("  [AI LOG] ❌ No valid slot found after CSP check.")
            return None

        # Use BFS to pick the nearest valid slot (CO2: BFS Search)
        best_slot  = None
        best_path  = None
        best_steps = float('inf')

        for slot in candidates:
            path = self.graph.bfs("ENTRY", slot.slot_id)
            if path and len(path) < best_steps:
                best_steps = len(path)
                best_slot  = slot
                best_path  = path

        # Fallback: if graph has no path, just pick first candidate
        if best_slot is None:
            best_slot = candidates[0]
            best_path = [best_slot.slot_id]

        print(f"  [AI LOG] ✅ Nearest slot selected: {best_slot.slot_id} "
              f"({best_steps - 1} steps from ENTRY)")

        # Show navigation path
        self.graph.display_path(best_path)

        # Assign the vehicle to the slot
        best_slot.status  = "occupied"
        best_slot.vehicle = vehicle

        # Push to priority queue for tracking
        heapq.heappush(self._priority_queue, (priority, vehicle.number))

        return best_slot

    # ── 4c. Vehicle Exit & Billing ───────────────────────────────────────────

    def exit_vehicle(self, vehicle_number: str) -> Optional[Dict]:
        """
        Finds the vehicle, calculates bill, frees the slot.
        CO1: State transition — occupied → free
        """
        for slot in self.slots.values():
            if slot.vehicle and slot.vehicle.number == vehicle_number:
                vehicle    = slot.vehicle
                exit_time  = datetime.now()
                duration_h = max(
                    (exit_time - vehicle.entry_time).total_seconds() / 3600,
                    0.25  # Minimum billing: 15 minutes
                )
                rate   = RATES.get(vehicle.vtype, 20.0)
                amount = round(duration_h * rate, 2)
                self.revenue += amount

                record = {
                    "slot":       slot.slot_id,
                    "vehicle":    vehicle.number,
                    "type":       vehicle.vtype,
                    "entry":      vehicle.entry_time.strftime("%H:%M:%S"),
                    "exit":       exit_time.strftime("%H:%M:%S"),
                    "duration_h": round(duration_h, 2),
                    "rate":       rate,
                    "amount":     amount,
                }
                self.exit_log.append(record)

                # Free the slot (state change)
                slot.status  = "free"
                slot.vehicle = None

                return record
        return None

    # ── 4d. Reservation System ───────────────────────────────────────────────

    def reserve_slot(self, slot_id: str, vehicle_number: str) -> bool:
        """Reserve a free slot for a vehicle number."""
        slot = self.slots.get(slot_id)
        if not slot:
            print(f"  ❌ Slot {slot_id} does not exist.")
            return False
        if slot.status != "free":
            print(f"  ❌ Slot {slot_id} is already {slot.status}.")
            return False
        slot.status       = "reserved"
        slot.reserved_for = vehicle_number
        return True

    def cancel_reservation(self, slot_id: str) -> bool:
        """Cancel a reservation and free the slot."""
        slot = self.slots.get(slot_id)
        if not slot or slot.status != "reserved":
            print(f"  ❌ Slot {slot_id} has no active reservation.")
            return False
        slot.status       = "free"
        slot.reserved_for = None
        return True

    # ── 4e. Display Methods ───────────────────────────────────────────────────

    def display_all_slots(self):
        """Show the full parking lot status."""
        print("\n" + "═" * 52)
        print(f"  🅿  {self.name} — PARKING LAYOUT")
        print("═" * 52)
        rows: Dict[str, List[ParkingSlot]] = {}
        for slot in self.slots.values():
            row = slot.slot_id[0]
            rows.setdefault(row, []).append(slot)

        for row_key in sorted(rows):
            print(f"\n  Row {row_key}:")
            for slot in sorted(rows[row_key], key=lambda s: s.slot_id):
                print(slot)
        print("═" * 52)

        free     = sum(1 for s in self.slots.values() if s.status == "free")
        occupied = sum(1 for s in self.slots.values() if s.status == "occupied")
        reserved = sum(1 for s in self.slots.values() if s.status == "reserved")
        print(f"  🟢 Free: {free}  🔴 Occupied: {occupied}  🟡 Reserved: {reserved}")
        print("═" * 52)

    def display_stats(self):
        """Admin: parking statistics."""
        print("\n" + "═" * 52)
        print("  📊 PARKING STATISTICS")
        print("═" * 52)
        total    = len(self.slots)
        occupied = sum(1 for s in self.slots.values() if s.status == "occupied")
        print(f"  Total Slots    : {total}")
        print(f"  Occupied       : {occupied}")
        print(f"  Utilization    : {occupied/total*100:.1f}%")
        print(f"  Total Revenue  : ₹{self.revenue:.2f}")
        print(f"  Completed Trips: {len(self.exit_log)}")
        if self.exit_log:
            avg = sum(r['amount'] for r in self.exit_log) / len(self.exit_log)
            print(f"  Avg Bill       : ₹{avg:.2f}")
        print("═" * 52)

    def display_parked_vehicles(self):
        """Admin: all currently parked vehicles."""
        occupied_slots = [s for s in self.slots.values() if s.status == "occupied"]
        print("\n" + "═" * 52)
        print("  🚗 CURRENTLY PARKED VEHICLES")
        print("═" * 52)
        if not occupied_slots:
            print("  (No vehicles currently parked)")
        for slot in occupied_slots:
            v = slot.vehicle
            elapsed = (datetime.now() - v.entry_time).total_seconds() / 3600
            print(f"  [{slot.slot_id}] {v.number:<12} {v.vtype:<10} "
                  f"Entry: {v.entry_time.strftime('%H:%M:%S')}  "
                  f"({elapsed:.2f}h)")
        print("═" * 52)

    # ── 4f. Admin: Add / Remove Slots ────────────────────────────────────────

    def add_slot(self, slot_id: str, slot_type: str) -> bool:
        if slot_id in self.slots:
            print(f"  ❌ Slot {slot_id} already exists.")
            return False
        self.slots[slot_id] = ParkingSlot(slot_id, slot_type)
        # Connect new slot to graph (simple: connect to ENTRY)
        self.graph.add_edge("ENTRY", slot_id)
        print(f"  ✅ Slot {slot_id} ({slot_type}) added.")
        return True

    def remove_slot(self, slot_id: str) -> bool:
        slot = self.slots.get(slot_id)
        if not slot:
            print(f"  ❌ Slot {slot_id} not found.")
            return False
        if slot.status == "occupied":
            print(f"  ❌ Cannot remove occupied slot {slot_id}.")
            return False
        del self.slots[slot_id]
        print(f"  ✅ Slot {slot_id} removed.")
        return True


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 – BILLING & PAYMENT SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def display_bill(record: Dict):
    """Prints a formatted billing receipt."""
    print("\n" + "═" * 52)
    print("  🧾  PARKING BILL / RECEIPT")
    print("═" * 52)
    print(f"  Vehicle No.   : {record['vehicle']}")
    print(f"  Vehicle Type  : {record['type']}")
    print(f"  Slot          : {record['slot']}")
    print(f"  Entry Time    : {record['entry']}")
    print(f"  Exit Time     : {record['exit']}")
    print(f"  Duration      : {record['duration_h']} hour(s)")
    print(f"  Rate          : ₹{record['rate']}/hr")
    print(f"  ─────────────────────────────────────────")
    print(f"  TOTAL AMOUNT  : ₹{record['amount']}")
    print("═" * 52)
    simulate_payment(record['amount'], record['vehicle'])


def simulate_payment(amount: float, vehicle_no: str):
    """Simulates cashless payment options."""
    if amount == 0:
        print("  ✅ Emergency vehicle — No charge.")
        return
    print("\n  💳 SELECT PAYMENT METHOD:")
    print("  [1] UPI (PhonePe / GPay / Paytm)")
    print("  [2] Credit / Debit Card")
    print("  [3] Net Banking")
    print("  [4] Cash")
    choice = input("  Enter choice (1–4): ").strip()
    methods = {"1": "UPI", "2": "Card", "3": "Net Banking", "4": "Cash"}
    method  = methods.get(choice, "UPI")
    print(f"\n  Processing ₹{amount} via {method} ...")
    time.sleep(0.5)
    print(f"  ✅ Payment of ₹{amount} received! Thank you, {vehicle_no}.")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 – INPUT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_vehicle_type() -> str:
    """Validated vehicle type input."""
    types = {"1": "Bike", "2": "Car", "3": "SUV", "4": "Emergency"}
    print("  Vehicle Type:")
    for k, v in types.items():
        print(f"    [{k}] {ICONS[v]} {v}")
    while True:
        ch = input("  Select (1-4): ").strip()
        if ch in types:
            return types[ch]
        print("  ❌ Invalid choice. Try again.")


def get_vehicle_number() -> str:
    """Basic vehicle number validation."""
    while True:
        num = input("  Vehicle Number (e.g. TS09AB1234): ").strip().upper()
        if len(num) >= 4:
            return num
        print("  ❌ Too short. Enter a valid vehicle number.")


def pause():
    input("\n  Press Enter to continue...")


def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 – SAMPLE TEST DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_sample_data(lot: ParkingLot):
    """
    Pre-loads sample vehicles to demonstrate the system.
    Useful for testing without manual input.
    """
    print("\n  [TEST] Loading sample data...")
    samples = [
        ("TS01AA0001", "Bike",      False),
        ("TS02BB0002", "Car",       False),
        ("TS03CC0003", "Emergency", False),
        ("TS04DD0004", "SUV",       True),   # Differently-abled
    ]
    for num, vtype, disabled in samples:
        v = Vehicle(number=num, vtype=vtype, is_disabled=disabled)
        slot = lot.allocate_slot(v)
        if slot:
            print(f"  [TEST] {v.number} → Slot {slot.slot_id}")
        else:
            print(f"  [TEST] {v.number} → No slot available")
    print("  [TEST] Sample data loaded.\n")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 – PROJECT SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def display_project_summary():
    """
    Displays a final academic project summary.
    Shows which CO concepts were applied and where.
    """
    print("\n" + "═" * 60)
    print("  📋  SMART PARKING ALLOCATION SYSTEM — PROJECT SUMMARY")
    print("═" * 60)
    summary = [
        ("Project Name",   "Smart Parking Allocation System"),
        ("Language",       "Python 3 (Standard Library Only)"),
        ("Paradigm",       "Object-Oriented + AI Search"),
        ("Classes",        "Vehicle, ParkingSlot, ParkingLot, ParkingGraph"),
        ("Data Structures","dict, list, set, deque, heapq"),
    ]
    for label, value in summary:
        print(f"  {label:<20}: {value}")

    print("\n  ── CO Concept Mapping ──────────────────────────────────")
    cos = [
        ("CO1", "Problem Formulation",
         "PEAS model, Vehicle/Slot as state, allocate_slot as action"),
        ("CO2", "Uninformed Search (BFS)",
         "ParkingGraph.bfs() finds shortest path ENTRY→Slot O(V+E)"),
        ("CO3", "CSP & Priority Queue",
         "is_available_for() enforces constraints; heapq for emergency"),
    ]
    for co, concept, where in cos:
        print(f"\n  [{co}] {concept}")
        print(f"       → {where}")

    print("\n  ── Features Implemented ────────────────────────────────")
    features = [
        "✅ Vehicle Registration (Bike/Car/SUV/Emergency)",
        "✅ Smart Slot Allocation with AI trace logs",
        "✅ BFS Navigation with path display",
        "✅ CSP Constraint checking (type, status, reserved)",
        "✅ Priority Queue for Emergency / Disabled vehicles",
        "✅ Reservation & Cancellation",
        "✅ Billing with cashless payment simulation",
        "✅ Admin Panel (stats, revenue, add/remove slots)",
        "✅ Sample test data loader",
        "✅ Error handling throughout",
    ]
    for f in features:
        print(f"  {f}")
    print("═" * 60)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 – MENUS
# ══════════════════════════════════════════════════════════════════════════════

def user_menu(lot: ParkingLot):
    """Main user-facing menu."""
    while True:
        print("\n" + "═" * 52)
        print(f"  🅿  {lot.name} — USER MENU")
        print("═" * 52)
        print("  [1] 🚗 Register & Park a Vehicle")
        print("  [2] 🚪 Exit / Checkout Vehicle")
        print("  [3] 🗺  View Parking Availability")
        print("  [4] 📌 Reserve a Slot")
        print("  [5] ❌ Cancel Reservation")
        print("  [6] 🔙 Back to Main Menu")
        print("═" * 52)
        ch = input("  Choose an option: ").strip()

        if ch == "1":
            # Register vehicle and allocate slot
            print("\n  ── VEHICLE REGISTRATION ──")
            number   = get_vehicle_number()
            vtype    = get_vehicle_type()
            disabled = input("  Differently-abled user? (y/n): ").strip().lower() == "y"
            vehicle  = Vehicle(number=number, vtype=vtype, is_disabled=disabled)
            slot     = lot.allocate_slot(vehicle)
            if slot:
                print(f"\n  ✅ {vehicle} assigned to Slot [{slot.slot_id}]")
            else:
                print(f"\n  ❌ Sorry! No available slot for {vtype}.")
            pause()

        elif ch == "2":
            # Exit and bill
            print("\n  ── VEHICLE EXIT ──")
            num = input("  Enter Vehicle Number: ").strip().upper()
            record = lot.exit_vehicle(num)
            if record:
                display_bill(record)
            else:
                print(f"  ❌ Vehicle {num} not found in parking lot.")
            pause()

        elif ch == "3":
            lot.display_all_slots()
            pause()

        elif ch == "4":
            # Reserve a slot
            print("\n  ── RESERVE A SLOT ──")
            lot.display_all_slots()
            slot_id = input("  Enter Slot ID to reserve: ").strip().upper()
            num     = get_vehicle_number()
            if lot.reserve_slot(slot_id, num):
                print(f"  ✅ Slot {slot_id} reserved for {num}.")
            pause()

        elif ch == "5":
            # Cancel reservation
            print("\n  ── CANCEL RESERVATION ──")
            slot_id = input("  Enter Slot ID: ").strip().upper()
            if lot.cancel_reservation(slot_id):
                print(f"  ✅ Reservation for Slot {slot_id} cancelled.")
            pause()

        elif ch == "6":
            break
        else:
            print("  ❌ Invalid option.")


def admin_menu(lot: ParkingLot):
    """Admin panel menu."""
    PIN = "1234"
    pin = input("\n  🔐 Enter Admin PIN: ").strip()
    if pin != PIN:
        print("  ❌ Wrong PIN. Access denied.")
        pause()
        return

    while True:
        print("\n" + "═" * 52)
        print("  🛠  ADMIN PANEL")
        print("═" * 52)
        print("  [1] 📋 View All Parked Vehicles")
        print("  [2] 📊 Parking Statistics")
        print("  [3] 💰 Total Revenue")
        print("  [4] ➕ Add a New Slot")
        print("  [5] ➖ Remove a Slot")
        print("  [6] 🧪 Load Sample Test Data")
        print("  [7] 📋 Project Summary Report")
        print("  [8] 🔙 Back to Main Menu")
        print("═" * 52)
        ch = input("  Choose an option: ").strip()

        if ch == "1":
            lot.display_parked_vehicles()
            pause()
        elif ch == "2":
            lot.display_stats()
            pause()
        elif ch == "3":
            print(f"\n  💰 Total Revenue Collected: ₹{lot.revenue:.2f}")
            pause()
        elif ch == "4":
            print("\n  ── ADD NEW SLOT ──")
            sid  = input("  Slot ID (e.g. D1): ").strip().upper()
            stype = get_vehicle_type()
            lot.add_slot(sid, stype)
            pause()
        elif ch == "5":
            print("\n  ── REMOVE SLOT ──")
            sid = input("  Slot ID to remove: ").strip().upper()
            lot.remove_slot(sid)
            pause()
        elif ch == "6":
            load_sample_data(lot)
            pause()
        elif ch == "7":
            display_project_summary()
            pause()
        elif ch == "8":
            break
        else:
            print("  ❌ Invalid option.")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 – MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Application entry point.
    Initializes the parking lot and launches the main menu loop.
    """
    clear()
    print("═" * 60)
    print("""
  ███████╗███╗   ███╗ █████╗ ██████╗ ████████╗
  ██╔════╝████╗ ████║██╔══██╗██╔══██╗╚══██╔══╝
  ███████╗██╔████╔██║███████║██████╔╝   ██║
  ╚════██║██║╚██╔╝██║██╔══██║██╔══██╗   ██║
  ███████║██║ ╚═╝ ██║██║  ██║██║  ██║   ██║
  ╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝
      SMART PARKING ALLOCATION SYSTEM
      Python | OOP | BFS | CSP | Priority Queue
    """)
    print("═" * 60)
    print("\n  Initializing system...")
    lot = ParkingLot()
    pause()

    while True:
        clear()
        print("\n" + "═" * 52)
        print(f"  🅿  {lot.name}")
        print("═" * 52)
        print("  [1] 👤 User Portal")
        print("  [2] 🛠  Admin Panel")
        print("  [3] 📋 Project Summary")
        print("  [4] 🚪 Exit System")
        print("═" * 52)
        ch = input("  Main Menu Choice: ").strip()

        if ch == "1":
            user_menu(lot)
        elif ch == "2":
            admin_menu(lot)
        elif ch == "3":
            display_project_summary()
            pause()
        elif ch == "4":
            display_project_summary()
            print("\n  👋 Thank you for using Smart Parking System. Goodbye!\n")
            break
        else:
            print("  ❌ Invalid choice.")


# ── Run the application ────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
