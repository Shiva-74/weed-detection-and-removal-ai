
---

## 🔧 Hardware Implementation

### Complete Hardware Bill of Materials (BOM)

#### **Core Computing Unit**
| Component | Model | Cost (USD) | Purpose |
|-----------|-------|------------|---------|
| Edge AI Computer | NVIDIA Jetson Orin Nano 8GB | $499 | AI inference processing |
| Alternative (Budget) | NVIDIA Jetson Nano 4GB | $99 | Entry-level deployment |
| MicroSD Card | SanDisk 128GB UHS-I | $20 | Operating system storage |
| Power Supply | 5V 4A USB-C / 19V Barrel Jack | $15 | Jetson power |

#### **Vision System**
| Component | Model | Cost (USD) | Purpose |
|-----------|-------|------------|---------|
| Camera | Raspberry Pi Camera V2 (IMX219) | $25 | Image capture |
| High-Performance | Arducam IMX477 HQ Camera | $50 | 4K resolution imaging |
| Camera Mount | Pan-Tilt Servo Bracket | $15 | Adjustable viewing angle |
| Lens | Wide-angle 160° (optional) | $30 | Increased field of view |

#### **Spray System Components**
| Component | Model | Cost (USD) | Purpose |
|-----------|-------|------------|---------|
| Microcontroller | ESP32 Dev Board | $8 | Actuator control |
| Solenoid Valves | 12V DC Electric Valve (×4) | $40 | Precision spray control |
| Relay Module | 4-Channel 5V Relay | $7 | Switch solenoid valves |
| Peristaltic Pump | 12V DC Dosing Pump | $25 | Chemical delivery |
| Spray Nozzles | Adjustable Mist Nozzles (×4) | $20 | Fine spray pattern |
| Chemical Tank | 5L Agricultural Sprayer Tank | $30 | Herbicide storage |
| Pressure Regulator | 0-100 PSI Adjustable | $15 | Consistent spray pressure |

#### **Positioning & Sensors**
| Component | Model | Cost (USD) | Purpose |
|-----------|-------|------------|---------|
| GPS Module | NEO-6M or NEO-M8N | $15 | Location tracking |
| IMU Sensor | MPU6050 (6-axis) | $5 | Orientation/stability |
| Ultrasonic Sensor | HC-SR04 | $3 | Distance measurement |

#### **Power System**
| Component | Model | Cost (USD) | Purpose |
|-----------|-------|------------|---------|
| Battery | 12V 20Ah LiFePO4 | $80 | Portable power |
| Solar Panel (Optional) | 50W Monocrystalline | $60 | Renewable charging |
| Voltage Regulator | 12V to 5V DC-DC Buck | $10 | Power conversion |
| Charge Controller | MPPT Solar Controller | $25 | Battery management |

#### **Connectivity & Misc**
| Component | Model | Cost (USD) | Purpose |
|-----------|-------|------------|---------|
| WiFi Module | Built into Jetson | $0 | Local network |
| 4G LTE Module (Optional) | SIM7600 USB Dongle | $35 | Remote connectivity |
| Enclosure | Waterproof IP65 Box | $40 | Weather protection |
| Cooling Fan | 40mm 5V PWM Fan | $8 | Jetson cooling |
| Cables & Connectors | Assorted | $30 | Wiring |

### **Total Cost Breakdown**
- **Budget Configuration**: ~$450 (Jetson Nano + basic components)
- **Standard Configuration**: ~$850 (Jetson Orin Nano)
- **Premium Configuration**: ~$1,200 (4K camera + solar + 4G)

---

### 🔌 Hardware Assembly Guide

#### **Step 1: Jetson Setup**
