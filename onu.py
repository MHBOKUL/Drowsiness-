#include <AFMotor.h>

#define sen4 A3

AF_DCMotor motor1(4, MOTOR12_8KHZ);
AF_DCMotor motor2(2, MOTOR12_8KHZ);

bool sleeping = false;

void setup() {
  Serial.begin(9600);

  motor1.setSpeed(254);
  motor2.setSpeed(254);

  pinMode(sen4, INPUT);

  forwordM();
  Bipoff();
}

void loop() {

  if (Serial.available()) {

    String input = Serial.readStringUntil('\n');
    input.trim();

 

    if (input == "1") {
      int speed=254;
      for (int i = 0; i < 10; i++) {
        Bip();
        delay(300);
         motor1.setSpeed(speed);
        speed=speed-25;
        Bipoff();
        delay(300);
      }

      stopM();
      
      for (int i = 0; i < 10; i++) {
        Bip();
        delay(200);
       
        Bipoff();
        delay(200);
      }

      delay(5000);
    }
  }

    else {
      
      
      if(analogRead(sen4)<500){
        stopM();
        Bip();
        delay(5000);
      }
      else{
        motor1.setSpeed(255);
        forwordM();
        Bipoff();
      }

    }
}

void forwordM() {
  motor1.run(FORWARD);
}

void stopM() {
  motor1.run(RELEASE);
}

void Bip() {
  motor2.run(FORWARD);
}

void Bipoff() {
  motor2.run(RELEASE);
}