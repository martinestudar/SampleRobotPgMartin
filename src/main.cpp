#include "ClassRobo.h"
#include "Aria.h"
#include <iostream>
#include "Config.h"
#include "Colisionavoidancethread.h"
#include "Wallfollowerthread.h"
#include "Sonarthread.h"
#include "Laserthread.h"
#include "ColisionAvoidanceNeuralNetworkThread.h"

PioneerRobot *robo;
NeuralNetwork *neuralNetwork;

int main(int argc, char **argv)
{
    int sucesso;

    robo = new PioneerRobot(ConexaoSimulacao, "", &sucesso);

    ArLog::log(ArLog::Normal, "Criando as theads...");
    ColisionAvoidanceThread colisionAvoidanceThread(robo);
    // WallFollowerThread wallFollowerThread(robo);
    SonarThread sonarReadingThread(robo);
    // LaserThread laserReadingThread(robo);

    ArLog::log(ArLog::Normal, "Sonar Readings thread ...");
    sonarReadingThread.runAsync();

    // ArLog::log(ArLog::Normal, "Laser Readings thread ...");
    // laserReadingThread.runAsync();

    ArLog::log(ArLog::Normal, "Colision Avoidance thread ...");
    colisionAvoidanceThread.runAsync();

    // ArLog::log(ArLog::Normal, "Wall Following thread ...");
    // wallFollowerThread.runAsync();

    robo->robot.waitForRunExit();

    Aria::exit(0);
}
