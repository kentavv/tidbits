// Demo Arduino code for the OPT-301 non-contact thermometer
// Maiji OTI-301 T420 D1 Non-Contact Digital Infrared Temperature Sensor Module No Algorithms IIC Interface Sensor Module 
// https://amzn.to/2vG9Np1 (Amazon Affiliate link)
// References used:
//   https://drive.google.com/file/d/1XMRDCNzY3fn0q6lkGejs5D-pqksfRuZ8/view?usp=sharing
//   https://drive.google.com/file/d/1mmdkHkNbwC5VgQxGMrq07XolajNvgBkN/view?usp=sharing

#include <Wire.h>

void setup() {
  Serial.begin(9600);
  Wire.begin();
}

void loop() {
  const int addr = 0x10; // I2C addresses at 7-bit, must shift documented 0x20 address, so 0x10.

  Wire.beginTransmission(addr);
  Wire.write(0x80);
  int rv = Wire.endTransmission(false);

  if (rv != 0) {
    Serial.print("Error while transmitting: ");
    Serial.println(rv);
  } else {
    const int n = 6;
    uint8_t dat[n];

    Wire.requestFrom(addr, n);

    int i;
    for (i = 0; i < n && Wire.available(); i++) {
      dat[i] = Wire.read();
    }

    if (i != n) {
      Serial.println("Incomplete data");
    } else {
      float amb = (dat[2] * 65536L + dat[1] * 256L + dat[0]) / 200.;
      float obj = (dat[5] * 65536L + dat[4] * 256L + dat[3]) / 200.;

      Serial.print("Temperatures (C): Ambient: ");
      Serial.print(amb);
      Serial.print("  Object: ");
      Serial.println(obj);
    }
  }
}
