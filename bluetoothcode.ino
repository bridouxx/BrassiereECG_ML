#include <ArduinoBLE.h>

const int ecgPin = A0;  
BLEService ecgService("180D");
BLECharacteristic ecgCharacteristic("2A37", BLERead | BLENotify, 100);

// Paramètres pour le buffer
const int bufferSize = 20;  // On stocke 20 valeurs avant envoi
float timeBuffer[bufferSize];
int ecgBuffer[bufferSize];
int bufferIndex = 0;
unsigned long startMillis;
unsigned long lastReadMillis = 0;

void setup() {
    Serial.begin(115200);
    while (!Serial);

    if (!BLE.begin()) {
        Serial.println("Erreur d'initialisation BLE !");
        while (1);
    }

    BLE.setLocalName("ECG_Monitor");
    BLE.setAdvertisedService(ecgService);
    ecgService.addCharacteristic(ecgCharacteristic);
    BLE.addService(ecgService);
    BLE.advertise();

    Serial.println("BLE prêt, en attente de connexion...");
    startMillis = millis();  // Enregistre le temps de départ
}

void loop() {
    BLEDevice central = BLE.central();
    if (central) {
        Serial.print("Connecté à : ");
        Serial.println(central.address());
        

        while (central.connected()) {
            unsigned long currentMillis = millis();

            // Vérifie si 2ms se sont écoulées pour la lecture ECG (500Hz)
            if (currentMillis - lastReadMillis >= 2) {
                lastReadMillis = currentMillis;

                // Lire ECG et stocker dans le buffer
                timeBuffer[bufferIndex] = (currentMillis - startMillis) / 1000.0;  // Temps en secondes
                ecgBuffer[bufferIndex] = analogRead(ecgPin);
                bufferIndex++;

                // Si le buffer est plein, envoyer les données
                if (bufferIndex >= bufferSize) {
                    bufferIndex = 0;
                    char dataBuffer[201];  // Taille pour stocker les données groupées
                    
                    snprintf(dataBuffer, sizeof(dataBuffer),
                        "%.3f,%d;%.3f,%d;%.3f,%d;%.3f,%d;%.3f,%d;%.3f,%d;%.3f,%d;%.3f,%d;%.3f,%d;%.3f,%d;"
                        "%.3f,%d;%.3f,%d;%.3f,%d;%.3f,%d;%.3f,%d;%.3f,%d;%.3f,%d;%.3f,%d;%.3f,%d;%.3f,%d;",
                        timeBuffer[0], ecgBuffer[0], timeBuffer[1], ecgBuffer[1],
                        timeBuffer[2], ecgBuffer[2], timeBuffer[3], ecgBuffer[3],
                        timeBuffer[4], ecgBuffer[4], timeBuffer[5], ecgBuffer[5],
                        timeBuffer[6], ecgBuffer[6], timeBuffer[7], ecgBuffer[7],
                        timeBuffer[8], ecgBuffer[8], timeBuffer[9], ecgBuffer[9],
                        timeBuffer[10], ecgBuffer[10], timeBuffer[11], ecgBuffer[11],
                        timeBuffer[12], ecgBuffer[12], timeBuffer[13], ecgBuffer[13],
                        timeBuffer[14], ecgBuffer[14], timeBuffer[15], ecgBuffer[15],
                        timeBuffer[16], ecgBuffer[16], timeBuffer[17], ecgBuffer[17],
                        timeBuffer[18], ecgBuffer[18], timeBuffer[19], ecgBuffer[19]);

                    ecgCharacteristic.writeValue(dataBuffer);
                    Serial.println(dataBuffer);
                }
           
           delayMicroseconds(2000); 
           }
        }
        Serial.println("Déconnecté");
    }
}

