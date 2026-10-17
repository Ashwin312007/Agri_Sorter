# Pi ultrasonic and camera hardware design

Hardware design summary: 40 kHz ultrasonic transmit and receive, LM741 conditioning, ADS1115 ADC, and Raspberry Pi Camera Module 3.

## 1. Open question: the transducer pinout

The GU16D01L01-A27 is listed as an "integrated transmitter and receiver," and the wiring below assumes two leads per unit. If it has V+ and V- terminals, it's a module with power pins instead. Check which one you have before wiring:

- **Two leads only (piezo transducer):** use the netlists in sections 3 and 4 as written.
- **Module with VCC/GND/signal pins:** the module handles its own drive and receive. The LM741 stages become optional conditioning, and the connections change.

Also confirm the seller ships two units, one for transmit and one for receive.

## 2. Parts list

| Part | Qty | Purpose |
| --- | --- | --- |
| GU16D01L01-A27 (40 kHz) | 2 | One transmitter, one receiver (pending section 1) |
| LM741 op-amp (DIP-8) | 2 | Transmit driver (U1) and receive amplifier (U2) |
| ADS1115 ADC breakout | 1 | 16-bit, I2C, digitises the envelope |
| Raspberry Pi Camera Module 3 | 1 | Image input (CSI port) |
| 1N5817 Schottky diode | 1 | Envelope detector |
| 1N4148 diode | 1 | Optional clamp for the ADC input |
| Resistors: 1 kΩ (×4), 10 kΩ (×4), 100 kΩ (×1), 470 kΩ (×1) |  | Feedback, bias, envelope discharge |
| Capacitors: 100 nF ceramic (×6), 10 nF (×1) |  | Coupling, bypass, envelope |
| Breadboard, jumper wires, USB supply or bench PSU |  | Prototyping |

**Power note:** the LM741 needs about ±5 V for proper operation. Options:

- A ±5 V DC-DC module or ICL7660 charge pump.
- A single-supply op-amp such as the TLV9062 or MCP6002, which runs from 5 V and is the cleaner design.
- A single 5 V supply with a 2.5 V bias, using the LM741 as-is. This works only partially: reduced output swing, poor linearity, compressed transmit amplitude. Use it for testing only, and flag it in any PR.

## 3. Transmit stage (U1)

Pi GPIO generates a 40 kHz PWM or burst signal at 3.3 V logic. U1 amplifies it to drive the transmitter.

Netlist (non-inverting amplifier, gain ≈ 11):

- GPIO → 1 kΩ → U1 pin 3 (IN+)
- U1 pin 3 → 10 kΩ → 2.5 V bias node
- U1 pin 2 (IN−) → 1 kΩ → GND
- U1 pin 2 → 10 kΩ → U1 pin 6 (OUT)
- U1 pin 6 → 100 nF → transmitter terminal A
- Transmitter terminal B → GND

**Drive limit:** the LM741 slew rate (0.5 V/µs) limits output at 40 kHz to about 0.5 V/µs ÷ (2π × 40 kHz) ≈ 2 V peak. Don't drive the transducer directly from a GPIO pin. For more amplitude, add a push-pull transistor stage after U1.

## 4. Receive stage (U2)

The echo is small and at 40 kHz. The chain amplifies it, rectifies it, and smooths it into a slow envelope the ADS1115 can sample.

Netlist:

- Receiver terminal A → 100 nF → U2 pin 3 (IN+)
- Receiver terminal B → GND
- U2 pin 3 → 470 kΩ → 2.5 V bias node
- U2 pin 2 (IN−) → 1 kΩ → 2.5 V bias node
- U2 pin 2 → 10 kΩ → U2 pin 6 (OUT)
- U2 pin 6 → 1N5817 anode; cathode → node E
- Node E → 10 nF → GND, with 100 kΩ in parallel with that 10 nF
- Node E → ADS1115 A0

**Why the envelope detector:** the ADS1115 samples at most 860 times per second, far below 40 kHz, so it can't read the carrier directly. The diode and RC network extract the echo's amplitude. Keep gain at or below about 20, because the LM741's 1 MHz gain-bandwidth limits the amplifier at 40 kHz.

## 5. Bias and power

- Build the 2.5 V bias from two 10 kΩ resistors (5 V to node B, node B to GND), with 100 nF from node B to GND. Use it for both stages.
- LM741 pins 7 (V+) and 4 (V-): V+ to 5 V and V- to GND on a single supply, or ±5 V on a split supply.
- Pins 1, 5 (offset null) and 8 (NC): leave unconnected.
- Place 100 nF from V+ to GND on each LM741, close to the chip.
- Tie all grounds together: Pi GND, supply ground, ADS1115 ground, transducer ground.

## 6. ADS1115

| ADS1115 pin | Connect to |
| --- | --- |
| VDD | Pi 3.3 V |
| GND | GND |
| SDA | Pi GPIO2 (SDA) |
| SCL | Pi GPIO3 (SCL) |
| ADDR | GND (address 0x48) |
| A0 | Node E (envelope) |
| A1–A3 | Unconnected |

Keep all ADC inputs between 0 V and 3.3 V. The envelope rides on the 2.5 V bias, so check it stays in range with a scope.

## 7. Camera

Connect the Camera Module 3 to the Pi's CSI port with the ribbon cable. Enable the camera with `sudo raspi-config`, and capture frames with Picamera2.

## 8. Pi setup

- Enable I2C and the camera interface: `sudo raspi-config`.
- Confirm the ADS1115 appears at 0x48 with `i2cdetect -y 1`.
- Pi GPIO is 3.3 V only.

## 9. ML model

The model is not yet defined. Before writing the pipeline, specify:

- What it predicts or classifies
- Its input format (echo amplitude over time, camera frame size and channels, or both)
- Whether it runs on the Pi or on a laptop

## 10. Bench checks before connecting the transducer

1. Check the transducer's rated drive voltage and pinout in the GU16D01L01-A27 datasheet.
2. Power the LM741 stages alone. Confirm the output rests near 2.5 V.
3. Feed a small 40 kHz signal into the transmit input. Check U1's output on a scope and keep the amplitude low.
4. With the receiver connected, confirm node E stays between 0 V and 3.3 V.
5. Read ADS1115 A0 from the Pi and confirm the reading changes when you tap the receiver or send a burst.

All component values are starting points. Confirm them on the bench.
