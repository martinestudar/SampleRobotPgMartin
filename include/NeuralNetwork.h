#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "Sigmoid.h"
#include "ExpectedMovement.h"

// --- CORREÇÃO: Ajustei os números para bater com a quantidade real de linhas que você criou ---
#define PadroesTreinamento 18
#define PadroesValidacao 12

#define Sucesso 0.04            // 0.0004
#define NumeroCiclos 100000     // Exibir o progresso do treinamento a cada NumeroCiclos ciclos

//Sigmoide
#define TaxaAprendizado 0.3     //0.3 converge super rápido e com uma boa precisão (sigmoide na oculta).
#define Momentum 0.9            // Dificulta a convergencia da rede em minimos locais
#define MaximoPesoInicial 0.5

// --- Definições de Saída ---
// Direcao de rotacao
#define OUT_DR_DIREITA    0.25    
#define OUT_DR_ESQUERDA   0.5   
#define OUT_DR_FRENTE     0.75

// Direcao de movimento
#define OUT_DM_FRENTE     0.3       
#define OUT_DM_RE         0.7
#define OUT_DM_PARAR      0.5  // Sua adição correta

// Angulo de rotacao
#define OUT_AR_SEM_ROTACAO  0.2    
#define OUT_AR_LATERAL      0.4    
#define OUT_AR_DIAGONAL     0.6    
#define OUT_AR_FRONTAL      0.8    

#define ALCANCE_MAX_SENSOR 5000

// Camada de entrada
#define NodosEntrada 8
// Camada oculta
#define NodosOcultos 9
// Camada de saída
#define NodosSaida 3

class NeuralNetwork {
public:
    int i, j, p, q, r;
    int IntervaloTreinamentosPrintTela;
    int IndiceRandom[PadroesTreinamento];
    long CiclosDeTreinamento;
    float Rando;
    float Error;
    float AcumulaPeso;

    int esquerda = 0;
    int diagonal_esquerda_lateral = 0;
    int diagonal_esquerda_frontal = 0;
    int frente_esquerda = 0;
    int direita = 0;
    int diagonal_direita_lateral = 0;
    int diagonal_direita_frontal = 0;
    int frente_direita = 0;

    // Camada oculta
    float Oculto[NodosOcultos];
    float PesosCamadaOculta[NodosEntrada + 1][NodosOcultos];
    float OcultoDelta[NodosOcultos];
    float AlteracaoPesosOcultos[NodosEntrada + 1][NodosOcultos];
    ActivationFunction* activationFunctionCamadasOcultas;

    // Camada de saída
    float Saida[NodosSaida];
    float SaidaDelta[NodosSaida];
    float PesosSaida[NodosOcultos + 1][NodosSaida];
    float AlterarPesosSaida[NodosOcultos + 1][NodosSaida];
    ActivationFunction* activationFunctionCamadaSaida;

    float ValoresSensores[1][NodosEntrada] = {{0, 0, 0, 0, 0, 0, 0, 0}};

    // --- DADOS DE TREINAMENTO (18 Padrões) ---
    const float Input[PadroesTreinamento][NodosEntrada] = {
        // ---- Cenários sem obstáculos (seguir em frente)
        {5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000},
        {4800, 5000, 5000, 5000, 5000, 5000, 5000, 4800},
        {4500, 4600, 4800, 5000, 5000, 4800, 4600, 4500},

        // ---- Obstáculo frontal (desviar suave)
        {5000, 5000, 3000, 1500, 1500, 3000, 5000, 5000},
        {4800, 4800, 2500, 1200, 1200, 2500, 4800, 4800},
        {4500, 4500, 2000, 1000, 1000, 2000, 4500, 4500},

        // ---- Obstáculo forte à esquerda (girar direita)
        { 800, 1200, 3000, 5000, 5000, 5000, 4800, 4500},
        { 600, 1000, 2500, 4500, 5000, 4800, 4600, 4300},
        { 500,  800, 2000, 4000, 4500, 4800, 4700, 4600},

        // ---- Obstáculo forte à direita (girar esquerda)
        {4500, 4800, 5000, 5000, 5000, 3000, 1200,  800},
        {4300, 4600, 4800, 5000, 5000, 2500, 1000,  600},
        {4200, 4500, 4700, 5000, 5000, 2000,  900,  500},

        // ---- Obstáculo diagonal esquerdo
        {1500, 3000, 4500, 5000, 5000, 4500, 3000, 1500},
        {1200, 2500, 4200, 4800, 4800, 4200, 2500, 1200},
        {1000, 2000, 4000, 4500, 4500, 4000, 2000, 1000},

        // ---- Obstáculo diagonal direito
        {1500, 3000, 4500, 5000, 5000, 4500, 3000, 1500},
        {1200, 2400, 4200, 4800, 4800, 4200, 2400, 1200},
        {1000, 2000, 3800, 4300, 4300, 3800, 2000, 1000}
    };
    
    float InputNormalizado[PadroesTreinamento][NodosEntrada];

    //Exemplo de output esperado para os dados de treinamento acima
    const float Objetivo[PadroesTreinamento][NodosSaida] = {
            // --- Sem obstáculos -> seguir frente
            {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
            {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
            {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},

            // --- Obstáculo frontal -> Ré + diagonal
            {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL,    OUT_DM_RE},
            {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL,    OUT_DM_RE},
            {OUT_DR_DIREITA,  OUT_AR_DIAGONAL,    OUT_DM_RE},

            // --- Obstáculo forte à esquerda -> virar à direita
            {OUT_DR_DIREITA,  OUT_AR_LATERAL,     OUT_DM_FRENTE},
            {OUT_DR_DIREITA,  OUT_AR_LATERAL,     OUT_DM_FRENTE},
            {OUT_DR_DIREITA,  OUT_AR_LATERAL,     OUT_DM_FRENTE},

            // --- Obstáculo forte à direita -> virar à esquerda
            {OUT_DR_ESQUERDA, OUT_AR_LATERAL,     OUT_DM_FRENTE},
            {OUT_DR_ESQUERDA, OUT_AR_LATERAL,     OUT_DM_FRENTE},
            {OUT_DR_ESQUERDA, OUT_AR_LATERAL,     OUT_DM_FRENTE},

            // --- Obstáculo diagonal esquerdo -> diagonalisar para direita
            {OUT_DR_DIREITA,  OUT_AR_DIAGONAL,    OUT_DM_FRENTE},
            {OUT_DR_DIREITA,  OUT_AR_DIAGONAL,    OUT_DM_FRENTE},
            {OUT_DR_DIREITA,  OUT_AR_DIAGONAL,    OUT_DM_FRENTE},

            // --- Obstáculo diagonal direito -> diagonalisar para esquerda
            {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL,    OUT_DM_FRENTE},
            {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL,    OUT_DM_FRENTE},
            {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL,    OUT_DM_FRENTE},
    };
    
    // --- DADOS DE VALIDAÇÃO (12 Padrões) ---
    const float InputValidacao[PadroesValidacao][NodosEntrada] = {
            // 1. Livre total
            {5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000},
            // 2. Livre com pequena inclinação diagonal
            {4600, 4800, 5000, 5000, 5000, 5000, 4800, 4600},
            // 3. Livre porém médio (ambiente aberto)
            {4000, 4000, 4200, 4300, 4300, 4200, 4000, 4000},
            // 4. Obstáculo frontal médio
            {5000, 5000, 3200, 1400, 1400, 3200, 5000, 5000},
            // 5. Obstáculo frontal forte
            {5000, 5000, 2200, 900,  900, 2200, 5000, 5000},
            // 6. Obstáculo frontal fortíssimo (quase colisão)
            {3000, 3000, 1500, 500,  500, 1500, 3000, 3000},
            // 7. Obstáculo forte à esquerda
            {600,  900, 2500, 4500, 4800, 5000, 5000, 5000},
            // 8. Obstáculo médio à esquerda
            {900, 1500, 3000, 4800, 5000, 5000, 5000, 5000},
            // 9. Obstáculo leve à esquerda
            {1200, 2500, 4000, 5000, 5000, 5000, 5000, 5000},
            // 10. Obstáculo forte à direita
            {5000, 5000, 5000, 5000, 4800, 2500, 800,  500},
            // 11. Obstáculo médio à direita
            {5000, 5000, 5000, 5000, 3000, 1400, 600,  500},
            // 12. Obstáculo leve à direita
            {5000, 5000, 5000, 5000, 2000, 1200, 900,  700}
    };
    
    float InputValidacaoNormalizado[PadroesValidacao][NodosEntrada];
    
    const float ObjetivoValidacao[PadroesValidacao][NodosSaida] = {
            {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE}, // 1
            {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE}, // 2
            {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE}, // 3
            {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL,    OUT_DM_RE},     // 4
            {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL,    OUT_DM_RE},     // 5
            {OUT_DR_ESQUERDA, OUT_AR_FRONTAL,     OUT_DM_RE},     // 6
            {OUT_DR_DIREITA,  OUT_AR_LATERAL,     OUT_DM_FRENTE}, // 7
            {OUT_DR_DIREITA,  OUT_AR_DIAGONAL,    OUT_DM_FRENTE}, // 8
            {OUT_DR_DIREITA,  OUT_AR_DIAGONAL,    OUT_DM_FRENTE}, // 9
            {OUT_DR_ESQUERDA, OUT_AR_LATERAL,     OUT_DM_FRENTE}, // 10
            {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL,    OUT_DM_FRENTE}, // 11
            {OUT_DR_ESQUERDA, OUT_AR_DIAGONAL,    OUT_DM_FRENTE}  // 12
    };
    
    //--

public:
    NeuralNetwork();
    void treinarRedeNeural();
    void inicializacaoPesos();
    int treinoInicialRede();
    void PrintarValores();
    ExpectedMovement testarValor();
    ExpectedMovement definirAcao(int sensor0, int sensor1, int sensor2, int sensor3, int sensor4, int sensor5, int sensor6, int sensor7);
    void validarRedeNeural();
    void treinarValidar();
    void normalizarEntradas();
    void setupCamadas() ;
};

#endif // NEURALNETWORK_H
