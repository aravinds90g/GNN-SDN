# 🔧 Fix for SDN Blocking Both Devices

## Problem

The SDN controller was blocking **BOTH** normal and malicious devices instead of just the malicious one.

## Root Cause

The graph labeling logic in `graph_builder.py` was labeling a node as malicious if it had **ANY** malicious edge (either incoming or outgoing). This caused the **server node** to be labeled as malicious because it receives malicious traffic from the malicious device.

Since both the normal and malicious devices communicate with the server, and the server was labeled as malicious, the GNN would detect multiple malicious nodes and block both devices.

## Solution

Changed the graph labeling logic to **only label source IPs** of malicious traffic as malicious:

### Changes Made

1. **Use Directed Graph** (`graph_builder.py`)
   - Changed from `nx.Graph()` to `nx.DiGraph()` to track source→destination direction
   - Track malicious source IPs explicitly

2. **Fix Node Labeling** (`graph_builder.py`)
   - Updated `extract_node_labels()` to only check **outgoing edges** (for directed graphs)
   - A node is malicious ONLY if it is the **source** of malicious traffic
   - Destination nodes (like the server) are NOT labeled as malicious

3. **Updated Graph Conversion** (`graph_builder.py`)
   - Handle both directed and undirected graphs in `to_pytorch_geometric()`

## Files Modified

- [`archive/graph_builder.py`](file:///mnt/3A7069D670699981/Aravind/FinalYearProject/archive/graph_builder.py)
  - Line 26-36: Added `malicious_sources` tracking
  - Line 38-95: Changed to directed graph and track malicious sources
  - Line 158-202: Fixed node labeling to only check outgoing edges
  - Line 204-241: Updated PyTorch Geometric conversion

## Testing

Run the test script to verify the fix:

```bash
cd /mnt/3A7069D670699981/Aravind/FinalYearProject/archive
python test_labeling.py
```

Expected output:
```
✅ PASS: Normal device (192.168.1.201) correctly labeled as NORMAL
✅ PASS: Malicious device (192.168.1.202) correctly labeled as MALICIOUS
✅ PASS: Server (192.168.1.100) correctly labeled as NORMAL
✅ ALL TESTS PASSED
```

## How It Works Now

### Before (Incorrect):
```
Normal Device (192.168.1.201) → Server (192.168.1.100)  [Label: 0]
Malicious Device (192.168.1.202) → Server (192.168.1.100)  [Label: 1]

Node Labels:
- 192.168.1.201: NORMAL ✅
- 192.168.1.202: MALICIOUS ✅
- 192.168.1.100: MALICIOUS ❌ (has malicious edge)

Result: Both devices blocked ❌
```

### After (Correct):
```
Normal Device (192.168.1.201) → Server (192.168.1.100)  [Label: 0]
Malicious Device (192.168.1.202) → Server (192.168.1.100)  [Label: 1]

Node Labels (checking OUTGOING edges only):
- 192.168.1.201: NORMAL ✅ (outgoing edge is normal)
- 192.168.1.202: MALICIOUS ✅ (outgoing edge is malicious)
- 192.168.1.100: NORMAL ✅ (no outgoing malicious edges)

Result: Only malicious device blocked ✅
```

## Verification Steps

1. **Clear old data:**
   ```bash
   rm backend/data/network_flows.csv
   ```

2. **Restart all components:**
   - Terminal 1: `cd backend && npm start`
   - Terminal 2: `cd archive && source venv/bin/activate && python sdn/run_controller.py`
   - Terminal 3: `cd archive && source venv/bin/activate && python real_time_detection.py`

3. **Let ESP8266 devices send data for 2-3 minutes**

4. **Check detection output** (Terminal 3):
   - Should show only 1 malicious device detected
   - Should only block 192.168.1.202 (malicious device)
   - Should NOT block 192.168.1.201 (normal device)

5. **Verify normal device still works:**
   - Check Serial Monitor on normal ESP8266
   - Should continue receiving `200 OK` responses
   - Should NOT see `403 Forbidden` errors

6. **Verify malicious device is blocked:**
   - Check Serial Monitor on malicious ESP8266
   - Should start receiving `403 Forbidden` errors after detection
   - Backend should reject its data

## Expected Behavior

- ✅ Normal device (ESP8266_NORMAL) continues to operate
- ✅ Malicious device (ESP8266_MALICIOUS) gets detected and blocked
- ✅ Server node is NOT labeled as malicious
- ✅ Only 1 device blocked (not both)

## Troubleshooting

If both devices are still being blocked:

1. **Check graph type:**
   ```python
   # In real_time_detection.py, after building graph
   print(f"Graph type: {type(data)}")
   print(f"Is directed: {isinstance(builder.G, nx.DiGraph)}")
   ```

2. **Check node labels:**
   ```python
   # Print all node labels
   for i, (node, label) in enumerate(zip(builder.idx_to_node.values(), data.y)):
       print(f"Node {i}: {node} -> {'MALICIOUS' if label == 1 else 'NORMAL'}")
   ```

3. **Check malicious sources:**
   ```python
   print(f"Malicious sources: {builder.malicious_sources}")
   ```

If you see the server IP in malicious sources or labeled as malicious, the fix didn't apply correctly.
