/*
  ESP32 Edge AI - Land Cover Classification
  ESP32 Dev Module + MicroTFLite + SSD1306 OLED
*/

#include <MicroTFLite.h>

#include "model_data.h"
#include "test_images.h"

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// Must match the class order used during training
const char* CLASS_NAMES[] = {
  "asphalt",
  "bare_soil",
  "bright_surface",
  "other",
  "shadow",
  "vegetation",
  "water"
};

const int NUM_CLASSES = 7;

// Increase this if ModelInit() fails
constexpr int kTensorArenaSize = 60 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];


void setup() {

  Serial.begin(115200);
  delay(1000);

  // OLED I2C
  Wire.begin(21, 22);

  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("SSD1306 allocation failed!");
    while (true);
  }

  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("Loading model...");
  display.display();

  Serial.println("Initializing TensorFlow Lite Micro...");

  // Initialize model
  if (!ModelInit(model_data, tensor_arena, kTensorArenaSize)) {
    Serial.println("Model initialization failed!");

    display.clearDisplay();
    display.setCursor(0, 0);
    display.println("Model init FAILED");
    display.display();

    while (true);
  }

  Serial.println("Model initialization done.");

  // Optional debugging information
  ModelPrintMetadata();
  ModelPrintInputTensorDimensions();
  ModelPrintOutputTensorDimensions();

  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("Model ready!");
  display.display();

  delay(1500);
}


void classifyAndShow(const int8_t* imageData, const char* label) {

  Serial.println();
  Serial.print("Classifying: ");
  Serial.println(label);

  /*
    Your test image contains 6912 int8 values.

    We set each value into the model input tensor.
    If the model has a different input size, ModelSetInput()
    will reveal the issue through the library's error handling.
  */

  for (int i = 0; i < 6912; i++) {

    if (!ModelSetInput((float)imageData[i], i, true)) {
      Serial.print("Failed to set input at index ");
      Serial.println(i);
      return;
    }
  }

  // Run inference
  if (!ModelRunInference()) {
    Serial.println("Inference failed!");
    return;
  }

  // Find class with highest output
  int bestIndex = 0;
  float bestScore = ModelGetOutput(0);

  for (int i = 1; i < NUM_CLASSES; i++) {

    float score = ModelGetOutput(i);

    if (score > bestScore) {
      bestScore = score;
      bestIndex = i;
    }
  }

  Serial.print("Predicted: ");
  Serial.println(CLASS_NAMES[bestIndex]);

  Serial.print("Score: ");
  Serial.println(bestScore);

  // OLED
  display.clearDisplay();

  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println(label);

  display.println();

  display.setTextSize(2);
  display.println(CLASS_NAMES[bestIndex]);

  display.setTextSize(1);
  display.print("Score: ");
  display.println(bestScore, 3);

  display.display();
}


void loop() {

  classifyAndShow(
    test_image_vegetation,
    "Test: vegetation"
  );

  delay(4000);

  classifyAndShow(
    test_image_bright,
    "Test: bright surface"
  );

  delay(4000);
}