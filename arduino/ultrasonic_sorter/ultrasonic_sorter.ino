/*
 * Agri_Sorter - Ultrasonic HC-SR04 Firmware for Arduino Uno/Nano
 * Non-blocking 40Hz loop with 3-sample median filtering and JSON serial output.
 * 
 * Hardware Wiring:
 *   - TRIG_PIN:   Digital Pin 9
 *   - ECHO_PIN:   Digital Pin 10
 *   - STATUS_LED: Digital Pin 13 (Built-in LED)
 *   - Baud Rate:  115200 baud
 */

#include <Arduino.h>

const int TRIG_PIN = 9;
const int ECHO_PIN = 10;
const int STATUS_LED = 13;

const unsigned long LOOP_INTERVAL_MS = 25; // 40 Hz sampling rate
const unsigned long TIMEOUT_US = 30000;    // 30 ms echo timeout (~5.1 meters)

// Default baseline distance from sensor to empty conveyor belt (mm)
unsigned long baseline_distance_mm = 240;
const unsigned long FRUIT_THRESHOLD_MM = 30; // Height threshold above belt for fruit detection

// Circular buffer for 3-sample median filtering
unsigned long sample_buf[3] = {0, 0, 0};
uint8_t sample_idx = 0;
bool buffer_filled = false;

unsigned long last_loop_ms = 0;
String serial_cmd_buf = "";

// Trigger single HC-SR04 pulse and measure echo duration (microseconds)
unsigned long read_raw_us() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  unsigned long duration = pulseIn(ECHO_PIN, HIGH, TIMEOUT_US);
  return duration;
}

// 3-value median helper
unsigned long get_median_3(unsigned long a, unsigned long b, unsigned long c) {
  if ((a <= b && b <= c) || (c <= b && b <= a)) return b;
  if ((b <= a && a <= c) || (c <= a && a <= b)) return a;
  return c;
}

void process_serial_command(const String& cmd) {
  String trimmed = cmd;
  trimmed.trim();
  trimmed.toUpperCase();

  if (trimmed == "PING") {
    Serial.println("{\"res\":\"PONG\"}");
  } else if (trimmed == "CALIB") {
    // Perform 3 clean measurements to calibrate baseline
    unsigned long s1 = read_raw_us();
    delay(10);
    unsigned long s2 = read_raw_us();
    delay(10);
    unsigned long s3 = read_raw_us();
    unsigned long med_us = get_median_3(s1, s2, s3);
    if (med_us > 0) {
      baseline_distance_mm = (med_us * 343UL) / 2000UL;
    }
    Serial.print("{\"res\":\"CALIB_OK\",\"baseline_mm\":");
    Serial.print(baseline_distance_mm);
    Serial.println("}");
  } else if (trimmed.length() > 0) {
    Serial.print("{\"err\":\"UNKNOWN_COMMAND\",\"cmd\":\"");
    Serial.print(trimmed);
    Serial.println("\"}");
  }
}

void setup() {
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(STATUS_LED, OUTPUT);
  digitalWrite(TRIG_PIN, LOW);
  digitalWrite(STATUS_LED, LOW);

  Serial.begin(115200);
  while (!Serial && millis() < 2000) {
    // Wait up to 2 seconds for USB CDC on boards like Leonardo/Nano 33
  }

  // Pre-seed filter buffer
  for (int i = 0; i < 3; i++) {
    sample_buf[i] = read_raw_us();
    delay(5);
  }
  buffer_filled = true;
  last_loop_ms = millis();
}

void loop() {
  // Check incoming serial commands without blocking
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (serial_cmd_buf.length() > 0) {
        process_serial_command(serial_cmd_buf);
        serial_cmd_buf = "";
      }
    } else {
      if (serial_cmd_buf.length() < 64) {
        serial_cmd_buf += c;
      }
    }
  }

  unsigned long current_ms = millis();
  if (current_ms - last_loop_ms >= LOOP_INTERVAL_MS) {
    last_loop_ms = current_ms;

    // Take measurement
    unsigned long raw_us = read_raw_us();
    sample_buf[sample_idx] = raw_us;
    sample_idx = (sample_idx + 1) % 3;

    // 3-sample median filter to reject outlier acoustic noise
    unsigned long tof_us = get_median_3(sample_buf[0], sample_buf[1], sample_buf[2]);

    // Speed of sound: 343 m/s = 0.343 mm/us
    // Distance (mm) = (tof_us * 0.343) / 2 = (tof_us * 343) / 2000
    unsigned long dist_mm = (tof_us > 0) ? ((tof_us * 343UL) / 2000UL) : baseline_distance_mm;

    // Fruit detection logic: object detected if distance is significantly less than belt baseline
    int fruit = 0;
    if (dist_mm > 0 && dist_mm + FRUIT_THRESHOLD_MM <= baseline_distance_mm) {
      fruit = 1;
      digitalWrite(STATUS_LED, HIGH);
    } else {
      fruit = 0;
      digitalWrite(STATUS_LED, LOW);
    }

    // JSON formatted output: {"tof_us": 4210, "dist_mm": 72, "fruit": 1}
    Serial.print("{\"tof_us\": ");
    Serial.print(tof_us);
    Serial.print(", \"dist_mm\": ");
    Serial.print(dist_mm);
    Serial.print(", \"fruit\": ");
    Serial.print(fruit);
    Serial.println("}");
  }
}
