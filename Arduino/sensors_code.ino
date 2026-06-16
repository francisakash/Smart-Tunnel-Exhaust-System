// node mcu #1

#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <DHT.h>

#define DHTPIN D2
#define DHTTYPE DHT11
#define MQ2_PIN A0
#define RELAY_PIN D1

DHT dht(DHTPIN, DHTTYPE);

const char* ssid = "ComedKares-Students";
const char* password = "comedkares@12345";
const char* mqtt_server = "192.168.0.186";

WiFiClient espClient;
PubSubClient client(espClient);

int alcoholLevel = 0;

// =========================
// MQTT CALLBACK
// =========================
void callback(char* topic, byte* payload, unsigned int length) {

  String msg = "";

  for (unsigned int i = 0; i < length; i++) {
    msg += (char)payload[i];
  }

  Serial.println();
  Serial.println("===== MQTT MESSAGE =====");
  Serial.print("Topic: ");
  Serial.println(topic);

  Serial.print("Payload: ");
  Serial.println(msg);

  if (String(topic) == "tunnel/alcohol") {

    alcoholLevel = msg.toInt();

    Serial.print("Alcohol Received: ");
    Serial.println(alcoholLevel);
  }

  Serial.println("========================");
}

// =========================
// MQTT RECONNECT
// =========================
void reconnect() {

  while (!client.connected()) {

    Serial.print("Connecting to MQTT...");

    if (client.connect("NodeMCU1")) {

      Serial.println(" CONNECTED!");

      if (client.subscribe("tunnel/alcohol")) {
        Serial.println("Subscribed to tunnel/alcohol");
      } else {
        Serial.println("Subscribe Failed!");
      }

    } else {

      Serial.print(" FAILED, rc=");
      Serial.println(client.state());

      delay(2000);
    }
  }
}

// =========================
// SETUP
// =========================
void setup() {

  Serial.begin(115200);
  delay(2000);

  Serial.println();
  Serial.println("================================");
  Serial.println("NodeMCU #1 STARTING");
  Serial.println("================================");

  dht.begin();

  pinMode(RELAY_PIN, OUTPUT);

  // Relay OFF
  digitalWrite(RELAY_PIN, HIGH);

  Serial.println("Connecting WiFi...");

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {

    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("WiFi Connected!");

  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  client.setServer(mqtt_server, 1883);
  client.setBufferSize(512);
  client.setCallback(callback);

  Serial.println("Setup Complete");
}

// =========================
// LOOP
// =========================
void loop() {

  if (!client.connected()) {
    reconnect();
  }

  client.loop();

  float temp = dht.readTemperature();
  float hum = dht.readHumidity();

  if (isnan(temp) || isnan(hum)) {

    Serial.println("DHT FAILED!");

    delay(2000);
    return;
  }

  int smoke = analogRead(MQ2_PIN);

  // Create JSON
  StaticJsonDocument<256> doc;

  doc["temperature"] = temp;
  doc["humidity"] = hum;
  doc["smoke"] = smoke;

  char buffer[256];

  serializeJson(doc, buffer);

  bool publishResult =
      client.publish("tunnel/sensors", buffer);

  // Fan Logic
  bool fanOn = false;

  if (temp > 35)
    fanOn = true;

  if (smoke > 600)
    fanOn = true;

  if (alcoholLevel > 700)
    fanOn = true;

  digitalWrite(RELAY_PIN, fanOn ? LOW : HIGH);

  // Serial Monitor Output
  Serial.println();
  Serial.println("============== STATUS ==============");

  Serial.print("Publish Status: ");
  Serial.println(publishResult ? "SUCCESS" : "FAILED");

  Serial.print("Temperature: ");
  Serial.println(temp);

  Serial.print("Humidity: ");
  Serial.println(hum);

  Serial.print("Smoke: ");
  Serial.println(smoke);

  Serial.print("Alcohol: ");
  Serial.println(alcoholLevel);

  Serial.print("Fan: ");
  Serial.println(fanOn ? "ON" : "OFF");

  Serial.print("JSON: ");
  Serial.println(buffer);

  Serial.println("====================================");

  // MQTT-friendly 5 second wait
  for (int i = 0; i < 500; i++) {

    client.loop();
    delay(10);
  }
}



// node mcu #2
/*
#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define MQ3_PIN A0

// =========================
// OLED CONFIG
// =========================
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// =========================
// WIFI CONFIG
// =========================
const char* ssid = "ComedKares-Students";
const char* password = "comedkares@12345";

// =========================
// MQTT BROKER IP (YOUR PC)
// =========================
const char* mqtt_server = "192.168.0.186";

WiFiClient espClient;
PubSubClient client(espClient);

// =========================
// DATA FROM NODEMCU #1
// =========================
float temperature = 0;
float humidity = 0;
int smokeLevel = 0;

// =========================
// LOCAL MQ3
// =========================
int alcoholLevel = 0;

unsigned long lastPublish = 0;

// =========================
// OLED UPDATE
// =========================
void updateOLED() {

  display.clearDisplay();

  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);

  display.print("Temp:");
  display.print(temperature, 1);
  display.println(" C");

  display.print("Hum :");
  display.print(humidity, 1);
  display.println(" %");

  display.print("Smoke:");
  display.println(smokeLevel);

  display.print("Alcohol:");
  display.println(alcoholLevel);

  display.println();

  if (temperature > 35)
    display.println("HIGH TEMP!");

  if (smokeLevel > 600)
    display.println("SMOKE ALERT!");

  if (alcoholLevel > 700)
    display.println("ALCOHOL ALERT!");

  display.display();
}

// =========================
// MQTT CALLBACK
// =========================
void callback(char* topic, byte* payload, unsigned int length) {

  String message = "";

  for (unsigned int i = 0; i < length; i++) {
    message += (char)payload[i];
  }

  Serial.println();
  Serial.println("===== MQTT MESSAGE =====");
  Serial.println(message);

  StaticJsonDocument<512> doc;

  DeserializationError error =
      deserializeJson(doc, message);

  if (error) {

    Serial.print("JSON Parse Failed: ");
    Serial.println(error.c_str());

    return;
  }

  temperature = doc["temperature"];
  humidity    = doc["humidity"];
  smokeLevel  = doc["smoke"];

  Serial.println("===== SENSOR VALUES =====");

  Serial.print("Temperature: ");
  Serial.println(temperature);

  Serial.print("Humidity: ");
  Serial.println(humidity);

  Serial.print("Smoke: ");
  Serial.println(smokeLevel);

  updateOLED();
}

// =========================
// MQTT RECONNECT
// =========================
void reconnect() {

  while (!client.connected()) {

    Serial.print("Connecting MQTT...");

    if (client.connect("NodeMCU2")) {

      Serial.println(" CONNECTED");

      delay(500);

      bool result = client.subscribe("tunnel/sensors");

      Serial.print("Subscribe Result = ");
      Serial.println(result);

      client.loop();

      Serial.println("Subscribed to tunnel/sensors");

    } else {

      Serial.print(" FAILED, rc=");
      Serial.println(client.state());

      delay(2000);
    }
  }
}

// =========================
// SETUP
// =========================
void setup() {

  Serial.begin(115200);

  // OLED
  Wire.begin(D2, D1);

  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {

    Serial.println("OLED Failed");

    while (true);
  }

  display.clearDisplay();
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("Starting...");
  display.display();

  // WiFi
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("Connecting WiFi");
  display.display();

  while (WiFi.status() != WL_CONNECTED) {

    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("WiFi Connected");

  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);

  Serial.println("Callback Registered");

  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("WiFi Connected");
  display.println(WiFi.localIP());
  display.println("MQTT Ready");
  display.display();
}

// =========================
// LOOP
// =========================
void loop() {

  Serial.println("LOOP RUNNING");

  if (!client.connected()) {

    Serial.println("MQTT DISCONNECTED");

    reconnect();
  }

  bool mqttStatus = client.loop();

  Serial.print("client.loop() = ");
  Serial.println(mqttStatus);

  // Read MQ3
  alcoholLevel = analogRead(MQ3_PIN);

  // Every 5 seconds
  if (millis() - lastPublish > 5000) {

    lastPublish = millis();

    bool pubResult = client.publish(
      "tunnel/alcohol",
      String(alcoholLevel).c_str()
    );

    Serial.print("Alcohol Publish = ");
    Serial.println(pubResult);

    // Alerts
    if (temperature > 35)
      client.publish("tunnel/alerts", "HIGH_TEMP");

    if (smokeLevel > 700)
      client.publish("tunnel/alerts", "SMOKE");

    if (alcoholLevel > 800)
      client.publish("tunnel/alerts", "ALCOHOL");

    updateOLED();

    Serial.println();
    Serial.println("========== STATUS ==========");

    Serial.print("Temperature: ");
    Serial.println(temperature);

    Serial.print("Humidity: ");
    Serial.println(humidity);

    Serial.print("Smoke: ");
    Serial.println(smokeLevel);

    Serial.print("Alcohol: ");
    Serial.println(alcoholLevel);

    Serial.println("============================");
  }

  delay(100);
}
*/